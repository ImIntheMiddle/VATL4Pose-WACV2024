"""The active-learning loop of one video, with a round state that survives a crash.

A round of :class:`ActiveLearningLoop` is the procedure of the paper. One method does each step:

1. :meth:`~ActiveLearningLoop._infer`: the model predicts the heatmaps (and the embeddings, if the
   selector clusters them) of all people of the video.
2. :meth:`~ActiveLearningLoop._evaluate`: AP and OSPA of the predictions against the ground truth,
   and the same after the labeled people's predictions are replaced by their labels. A video in
   which some person has no ground truth (boxes without keypoints) has no AP and no OSPA: the
   metrics are ``None`` in every round and ``run`` logs one warning. The other steps do not
   read the ground truth, so the selection, the oracle and the retraining are the same.
3. :meth:`~ActiveLearningLoop._extras`: the inputs that WPU (the autoencoder) and VL4Pose (the link
   Gaussians) need.
4. :meth:`~ActiveLearningLoop._select`: the strategy chooses the people to annotate.
5. :meth:`~ActiveLearningLoop._ask_oracle`: the oracle answers, or the loop stores the query and
   stops with :class:`Pending` until the human has annotated.
6. :meth:`~ActiveLearningLoop._retrain`: the estimator learns from the new answers and the people it
   has not solved yet.
7. :meth:`~ActiveLearningLoop._finetune_ae`: the autoencoder is tuned on the labeled people.
8. :meth:`~ActiveLearningLoop._save`: the state of the finished round is written.

Round ``r`` evaluates and then queries ``query_sizes[r] - n_labeled`` people with
``query_sizes = [int(N * ratio) for ratio in query_ratio]``. The last round, numbered
``len(query_ratio)``, only evaluates the final model: no query and no retraining. A last ratio
below 1.0 ends the run at that fraction (labeling under a budget); the legacy code queried all the
remaining people in an extra round.

Decisions of this implementation:

* The last round writes the poses of everybody under the final model: ``predictions.json`` and
  ``predictions_ann.json`` (see :mod:`.results`). They come from the inference of its step 1, the
  one that step 2 evaluates, so the files hold the coordinates behind the last values of
  ``performances`` and ``performances_ann``. They need no ground truth.
* A query of size 0 (``int`` rounding on a small video) skips steps 3 to 7; the round still
  evaluates and is saved. Its mean query OKS does not exist, so it is ``None`` and the strategy's
  ``moks_previous`` stays at the last value that exists (0.0 before any).
* A query that takes every unlabeled person is forced by ``Strategy.query``, which scores nothing;
  step 3 is skipped too, so no auxiliary network is trained and no autoencoder is passed on.
* The query is stored (see :mod:`.state`) before the oracle is asked the first time, so that a run
  killed after the request never selects the round again.
* Every list of queried indices is sorted before the oracle sees it, so the order of the answers,
  of the fine-tuning set and of the files does not depend on the selector's order.
* The auxiliary network of VL4Pose is new in every round, trained on the labeled people with this
  round's features; nothing of it is saved. Without labeled people (round 0) it keeps its random
  initial weights.
* The autoencoder of round ``r + 1`` is a function of the pretrained checkpoint, the labels of the
  labeled people and the seed. A resume rebuilds it with the same function instead of storing it.
* The retraining set, the mean query OKS, the OKS of the labeled people, the autoencoder tuning
  (the hybrid features of the labels) and the training of the auxiliary network use the labels the
  oracle gave. With the simulated oracle they are the ground truth.
* An answer that marks no joint of a person visible (a hidden person, or a wrong box) labels the
  person, so it is not asked again, but the person is left out of the steps that read labels:
  the retraining set, the mean query OKS, the autoencoder and the auxiliary network. A round in
  which no queried person shows a joint does not retrain, and its mean query OKS is ``None``.
* ``run_dir/progress.json`` names the step the loop works on (the steps of the ``timings`` of a
  round) and, while the estimator trains, the epoch; an interface reads it to show progress. A
  stopped run leaves its last step in the file.
* A ``run`` that finds the stored query of its round (the oracle had no answers yet, or the run
  was killed after the request) asks the oracle for the same people with the same predictions,
  without inference or selection, whatever the model is then. The round then ends as if it had run
  without a pause.

The processes that run videos in parallel share nothing; the per-video lock is the job of the caller.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..data.cache import VideoCache
from ..data.datasets import FinetuneDataset
from ..data.geometry import HEATMAP_SIZE, heatmaps_to_coords, to_input_tensor
from ..eval import evaluate_predictions, oks_per_sample, pose_score, xyxy_to_xywh
from ..explain import record as round_details
from ..models.estimator import PoseEstimator
from ..models.wholebody_ae import WholeBodyAE, compute_hybrid, hybrid_valid_mask
from ..oracle import Oracle
from ..strategies import RoundContext, Strategy, local_peak_mean
from ..strategies.baselines.vl4pose import EXTRAS_KEY as VL4POSE_KEY
from ..train import Finetuner
from ..train._common import write_atomic
from . import extras
from .results import build_predictions, build_result, write_predictions, write_result
from .runs import video_seed
from .state import (
    LoopState,
    PendingQuery,
    RoundRecord,
    load_pending,
    load_state,
    save_pending,
    save_state,
)

logger = logging.getLogger(__name__)

AE_KEY = "ae"
"""Key of ``RoundContext.extras`` that holds the whole-body autoencoder."""

STEPS = ("infer", "eval", "select", "oracle", "retrain", "ae")
PROGRESS_SCHEMA = "vatl4pose.progress/1"
_REPORT_INTERVAL = 0.5  # seconds between two epoch updates of progress.json
"""The timed steps of a round; ``total`` is added to them in the record."""

_PEAK_CHUNK = 256
# Settings that do not change what a round computes; a resume may differ in them.
_RESUME_FREE = ("infer_batch_size", "save_final_model", "save_estimator", "stopping", "save_round_details")


@dataclass(frozen=True)
class LoopConfig:
    """Settings of the loop; the defaults are the ones of the paper.

    Attributes:
        query_ratio: Share of the video that is labeled after each query, strictly increasing in
            ``(0, 1]``. ``len(query_ratio)`` queries and one more evaluation make the rounds.
            The run ends once the last ratio is labeled: a last ratio below 1.0 stops at that
            fraction, which labels under a budget on data of your own. The legacy code queried
            all remaining people in an extra round after the last ratio.
        infer_batch_size: People per forward pass of the inference.
        save_final_model: Write the state dict of the final model to ``final_estimator.pth``.
        save_estimator: Write the state dict of the estimator to ``estimator.pth`` after every
            round, so that a run stopped early leaves the model of its last finished round.
        stopping: Judge the stopping criterion of the paper (SC_All) in every round with a query:
            whether the OKS of every labeled person with a visible joint, the people queried in
            the round included, is at least ``retrain.finish_acc`` under the model before it
            learns from the round (``>=``, as the 2023 code). The round records the judgment
            (:attr:`RoundRecord.stopping`); the loop does not stop by itself, so that the
            annotator decides.
        save_round_details: Write ``details/round_XX.npz`` in every round
            (:mod:`vatl4pose.explain.record`); the selection and the results do not change.
        seed: Base seed; the seed of a video is ``video_seed(seed, video_id)``.
        ae_finetune_epochs: Epochs of the autoencoder tuning after every retraining.
        ae_finetune_lr: Learning rate (Adam) of that tuning.
        ae_finetune_batch: Batch size of that tuning.
        ae_z_dim: Latent size of the autoencoder checkpoint (4 for the released ones), at least 1.
        vl4pose_epochs: Training epochs of the auxiliary network of VL4Pose in every round.
        vl4pose_lr: Learning rate (Adam) of the auxiliary network.
        vl4pose_batch_size: Batch size of its training and of its prediction.
    """

    query_ratio: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
    infer_batch_size: int = 256
    save_final_model: bool = False
    save_estimator: bool = False
    stopping: bool = False
    save_round_details: bool = False
    seed: int = 166
    ae_finetune_epochs: int = 20
    ae_finetune_lr: float = 8e-5
    ae_finetune_batch: int = 10
    ae_z_dim: int = 4
    vl4pose_epochs: int = 20
    vl4pose_lr: float = 1e-3
    vl4pose_batch_size: int = 64

    def __post_init__(self) -> None:
        ratio = tuple(float(r) for r in self.query_ratio)
        object.__setattr__(self, "query_ratio", ratio)
        if not ratio or not all(0.0 < r <= 1.0 for r in ratio):
            raise ValueError(
                f"query_ratio must be a non-empty sequence of values in (0, 1], got {ratio}"
            )
        if any(b <= a for a, b in zip(ratio, ratio[1:])):
            raise ValueError(f"query_ratio must increase strictly, got {ratio}")
        for name in ("infer_batch_size", "ae_finetune_batch", "vl4pose_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.vl4pose_epochs < 1:
            raise ValueError(
                f"vl4pose_epochs must be at least 1, got {self.vl4pose_epochs}"
            )
        if self.ae_z_dim < 1:
            raise ValueError(f"ae_z_dim must be at least 1, got {self.ae_z_dim}")
        if self.ae_finetune_epochs < 0:
            raise ValueError(
                f"ae_finetune_epochs must be at least 0, got {self.ae_finetune_epochs}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass
class Pending:
    """The oracle waits for human annotations; run again once they are in place.

    Attributes:
        round_index: The round that needs its answers.
        message: What the oracle asks the human to do and where.
    """

    round_index: int
    message: str


@dataclass
class _Pool:
    """The inference pass over all ``N`` people of the video."""

    heatmaps: torch.Tensor  # (N, K, 64, 48) float32
    coords: np.ndarray  # (N, K, 2) float32, image pixels
    maxvals: np.ndarray  # (N, K) float32
    embeddings: np.ndarray | None  # (N, C) float32

    def __post_init__(self) -> None:
        n, k = self.coords.shape[:2]
        assert self.coords.shape == (n, k, 2) and self.coords.dtype == np.float32, (
            f"coords: expected float32 (N, K, 2), got {self.coords.dtype} {self.coords.shape}"
        )
        assert self.maxvals.shape == (n, k) and self.maxvals.dtype == np.float32, (
            f"maxvals: expected float32 ({n}, {k}), got {self.maxvals.dtype} {self.maxvals.shape}"
        )
        assert self.heatmaps.dtype == torch.float32 and tuple(self.heatmaps.shape) == (
            n,
            k,
            *HEATMAP_SIZE,
        ), (
            f"heatmaps: expected float32 ({n}, {k}, {HEATMAP_SIZE[0]}, {HEATMAP_SIZE[1]}), got "
            f"{self.heatmaps.dtype} {tuple(self.heatmaps.shape)}"
        )
        assert self.embeddings is None or (
            self.embeddings.ndim == 2
            and self.embeddings.shape[0] == n
            and self.embeddings.dtype == np.float32
        ), "embeddings: expected float32 (N, C) or None"

    @property
    def predictions(self) -> np.ndarray:
        """``(N, K, 3)`` float64 x, y and peak value, what the oracle may prefill from."""
        return np.concatenate([self.coords, self.maxvals[..., None]], axis=-1).astype(
            np.float64
        )

    @property
    def pose_score(self) -> np.ndarray:
        """``(N,)`` float64 score of every pose: mean plus 1.25 times the maximum of its peaks."""
        return pose_score(self.maxvals)


@dataclass
class _Evaluation:
    """The evaluation of a round; the four metrics are ``None`` without ground truth for everybody."""

    percentage: float
    performance: dict[str, float] | None
    performance_ann: dict[str, float] | None
    ospa: float | None
    ospa_ann: float | None
    combine_weight: float | None


@dataclass
class _Query:
    """What a round has decided before the oracle answers.

    Attributes:
        queried: Sorted int64 sample indices to annotate, at least one.
        predictions: ``(N, K, 3)`` float64 predictions (x, y, peak value) of the model as it was
            when the round inferred; the oracle may prefill from them and the OKS of the queried
            and the labeled people is measured on them.
        score_means: Mean raw score of every scorer that scored.
    """

    queried: np.ndarray
    predictions: np.ndarray
    score_means: dict[str, float]

    def __post_init__(self) -> None:
        assert (
            self.predictions.ndim == 3
            and self.predictions.shape[2] == 3
            and self.predictions.dtype == np.float64
        ), (
            f"predictions: expected float64 (N, K, 3), got {self.predictions.dtype} "
            f"{self.predictions.shape}"
        )
        n = self.predictions.shape[0]
        assert (
            self.queried.ndim == 1
            and self.queried.dtype == np.int64
            and self.queried.size > 0
            and self.queried[0] >= 0
            and self.queried[-1] < n
            and bool(np.all(np.diff(self.queried) > 0))
        ), (
            f"queried: expected sorted distinct int64 indices in [0, {n}), got {self.queried}"
        )

    @property
    def coords(self) -> np.ndarray:
        """``(N, K, 2)`` float32 predicted joints, exactly the ones that were decoded."""
        return self.predictions[:, :, :2].astype(np.float32)


class _Clock:
    """Wall-clock seconds per step of one round; waits for the GPU so that a step owns its work."""

    def __init__(self, device: torch.device, on_step: Callable[[str], None]) -> None:
        self._device = device
        self._on_step = on_step
        self.times = dict.fromkeys(STEPS, 0.0)
        self._start = time.perf_counter()
        self._carried = 0.0

    def carry(self, timings: dict[str, float]) -> None:
        """Add the work that a stopped run did on this round before it waited for the oracle."""
        for step in STEPS:
            self.times[step] += timings[step]
        self._carried = timings["total"]

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @contextmanager
    def measure(self, step: str) -> Iterator[None]:
        self._synchronize()
        self._on_step(step)
        began = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.times[step] += time.perf_counter() - began

    def timings(self) -> dict[str, float]:
        """The seconds of every step and ``total``, the time the round has worked."""
        self._synchronize()
        return {
            **self.times,
            "total": self._carried + time.perf_counter() - self._start,
        }


def _loop_seed(base: int, *parts: int | str) -> int:
    """A 63-bit seed from ``base`` and ``parts`` through SHA-256; equal on every run and machine."""
    text = "/".join(["vatl4pose.loop", str(base), *map(str, parts)])
    return (
        int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") >> 1
    )


class ActiveLearningLoop:
    """Runs the rounds of one video and keeps the state that lets a stopped run continue.

    Args:
        cache: Cache of the video.
        estimator: The pose model; it must be the model of ``finetuner``.
        strategy: The query strategy. The scorers ``wpu`` and ``vl4pose`` make the loop provide the
            autoencoder and the link Gaussians; a selector with ``needs_embeddings`` makes it
            compute the embeddings.
        oracle: The source of the labels.
        finetuner: Trains ``estimator`` after every query. Its seed must be
            ``video_seed(cfg.seed, cache.video_id)``, so that one seed drives every part of a run.
        ae_ckpt: Pretrained whole-body autoencoder; required by strategies with the ``wpu`` scorer.
        cfg: Settings of the loop.
        run_dir: Directory of the run; ``state/``, ``result.json``, ``predictions.json``,
            ``predictions_ann.json``, ``final_estimator.pth`` and ``estimator.pth`` go here. The CLI
            holds the per-video lock.
        device: Device of the inference and of the autoencoder.
        on_round: Called as ``on_round(round_index, record)`` after the state of every finished round
            is saved, for logging. It is not called for a round that stops with :class:`Pending`, and
            a resumed run calls it only for the rounds it finishes itself.

    Raises:
        ValueError: If the finetuner does not train ``estimator`` or has another seed, the
            strategy needs embeddings (a selector with ``needs_embeddings``, or the VL4Pose scorer)
            and the estimator defines none (HRNet), or the strategy needs an autoencoder and
            ``ae_ckpt`` is ``None``.
    """

    def __init__(
        self,
        cache: VideoCache,
        estimator: PoseEstimator,
        strategy: Strategy,
        oracle: Oracle,
        finetuner: Finetuner,
        ae_ckpt: pathlib.Path | None,
        cfg: LoopConfig,
        run_dir: pathlib.Path,
        device: torch.device,
        on_round: Callable[[int, RoundRecord], None] | None = None,
    ) -> None:
        self.cache = cache
        self.estimator = estimator
        self.strategy = strategy
        self.oracle = oracle
        self.finetuner = finetuner
        self.ae_ckpt = ae_ckpt
        self.cfg = cfg
        self.run_dir = pathlib.Path(run_dir)
        self.state_dir = self.run_dir / "state"
        self.device = device
        self._on_round = on_round
        self._reported: tuple[int, str, float] = (-1, "", 0.0)
        self.spec = cache.spec
        self.seed = video_seed(cfg.seed, cache.video_id)
        if finetuner.model is not estimator:
            raise ValueError(
                "the estimator is not the model of the finetuner: the loop would predict with "
                "one network and train another"
            )
        if finetuner.seed != self.seed:
            raise ValueError(
                f"the finetuner has seed {finetuner.seed}, but the seed of video "
                f"{cache.video_id!r} is video_seed({cfg.seed}, {cache.video_id!r}) = {self.seed}; "
                "build the Finetuner with that seed so that augmentation and epoch order of every "
                "round belong to this run"
            )

        scorers = strategy.config.scorers
        self._uses_ae = "wpu" in scorers
        self._uses_vl4pose = "vl4pose" in scorers
        # the selector declares what it reads; the VL4Pose scorer is fed from the embeddings
        self._uses_embeddings = self._uses_vl4pose or strategy.needs_embeddings
        if self._uses_embeddings and not estimator.has_embedding:
            needs = (
                "the VL4Pose scorer reads the backbone features"
                if self._uses_vl4pose
                else f"its selector {strategy.config.selector!r} reads the embeddings"
            )
            raise ValueError(
                f"estimator {estimator.arch!r} does not define an embedding, but strategy "
                f"{strategy.config.name!r} needs one ({needs}). Use the estimator simplebaseline "
                "or fastpose, or a strategy without embeddings (strategy=random, hp, mpe, tpc, "
                "thc, wpu or thc_wpu)"
            )
        if self._uses_ae:
            if ae_ckpt is None:
                raise ValueError(
                    f"strategy {strategy.config.name!r} uses the 'wpu' scorer, which needs the "
                    "whole-body autoencoder; set ae.checkpoint to its checkpoint path "
                    "(ae_ckpt in the Python API)"
                )
            self._pretrained_ae()  # fails now, not in round 0, if the checkpoint does not fit

        arrays = cache.arrays
        self._arrays = arrays
        self.n = len(cache)
        # AP and OSPA compare with the ground truth of every person, or they are not computed
        self._has_ground_truth = bool(arrays.has_gt.all())
        self._query_sizes = [int(self.n * ratio) for ratio in cfg.query_ratio]
        self._boxes_xywh = xyxy_to_xywh(arrays.bbox_gt)
        self._sigmas = np.asarray(self.spec.oks_sigmas, dtype=np.float64)
        self._identity: dict[str, Any] = {
            "dataset": cache.dataset,
            "video_id": cache.video_id,
            "estimator": estimator.arch,
            "strategy": strategy.config.name,
            "params": json.loads(
                json.dumps(dict(strategy.config.params), allow_nan=False)
            ),
            "n_samples": self.n,
        }

        # per run, set by _restore()
        self._config: dict[str, Any]
        self._state: LoopState
        self._rng: np.random.Generator
        self._labeled: np.ndarray
        self._labels: np.ndarray
        self._ae: WholeBodyAE | None = None

    # ------------------------------------------------------------------ the run

    def run(self) -> dict[str, Any] | Pending:
        """Run the rounds that are not finished and write ``result.json``.

        The state in ``run_dir/state`` decides where to start: without it at round 0, with it at the
        round after the last finished one. A round that was cut short starts again from its
        beginning, so after a :class:`Pending` return (call ``run`` again once the answers exist)
        the same loop can run again. After an exception the finetuner may be half trained: build a
        new loop and a new finetuner for the next call. A run without state seeds the global torch
        generator with the seed of the video; a resume restores the generator state of the last
        finished round.

        Returns:
            The result document, also written to ``run_dir/result.json``, or :class:`Pending` when
            the oracle waits for annotations.

        Raises:
            ValueError: If ``run_dir`` holds the state of a run with other settings.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._restore()
        if not self._has_ground_truth:
            missing = self.n - int(self._arrays.has_gt.sum())
            logger.warning(
                "video %r: %d of %d people have no ground-truth keypoints, so AP and OSPA are not "
                "computed and the metrics in result.json are null. The loop selects, asks the "
                "oracle and retrains as usual.",
                self._identity["video_id"],
                missing,
                self.n,
            )
        for round_index in range(
            self._state.rounds_done, len(self.cfg.query_ratio) + 1
        ):
            pending = self._run_round(round_index)
            if pending is not None:
                return pending
        return self._finish()

    def _run_round(self, round_index: int) -> Pending | None:
        clock = _Clock(self.device, lambda step: self._report(round_index, step))
        stored = load_pending(self.state_dir, round_index)
        if stored is None:
            evaluation, query = self._prepare(round_index, clock)
        else:
            evaluation, query = self._reuse(stored, clock)
        queried: list[int] | None = None
        answers: np.ndarray | None = None
        moks: float | None = None
        stopping: dict[str, Any] | None = None
        if self._has_query(round_index):
            queried = []
            if query is not None:
                # stored before the first request, so that a kill cannot lose the query
                if stored is None:
                    self._store_pending(round_index, evaluation, query, clock)
                with clock.measure("oracle"):
                    outcome = self._ask_oracle(round_index, query)
                if isinstance(outcome, Pending):
                    return outcome
                answers = outcome
                queried = query.queried.tolist()
                with clock.measure("retrain"):
                    moks, stopping = self._retrain(round_index, query, answers)
                if self._uses_ae and self._has_query(round_index + 1):
                    with clock.measure("ae"):
                        self._finetune_ae()
        record = RoundRecord(
            percentage=evaluation.percentage,
            performance=evaluation.performance,
            performance_ann=evaluation.performance_ann,
            ospa=evaluation.ospa,
            ospa_ann=evaluation.ospa_ann,
            combine_weight=evaluation.combine_weight,
            score_means={
                name: query.score_means[name] if query and query.score_means else None
                for name in self.strategy.config.scorers
            },
            queried=queried,
            answers=None if answers is None else answers.tolist(),
            moks_queried=moks,
            stopping=stopping,
            timings=clock.timings(),
        )
        self._save(record)
        if record.performance is None or record.performance_ann is None:
            logger.info(
                "%s round %d: %.1f %% labeled, no ground truth, %.1f s",
                self._identity["video_id"],
                round_index,
                record.percentage,
                record.timings["total"],
            )
        else:
            logger.info(
                "%s round %d: %.1f %% labeled, AP %.3f (with labels %.3f), %.1f s",
                self._identity["video_id"],
                round_index,
                record.percentage,
                record.performance["AP"],
                record.performance_ann["AP"],
                record.timings["total"],
            )
        if self._on_round is not None:
            self._on_round(round_index, record)
        return None

    def _prepare(
        self, round_index: int, clock: _Clock
    ) -> tuple[_Evaluation, _Query | None]:
        """Steps 1 to 4: infer, evaluate and (if the round queries somebody) select.

        The final round, which has no query, also writes the poses of everybody from its inference
        (:meth:`_write_predictions`).

        Returns:
            The evaluation and the query; the query is ``None`` in the final round and for a query
            of size 0.
        """
        with clock.measure("infer"):
            pool = self._infer()
        with clock.measure("eval"):
            evaluation = self._evaluate(pool)
        if not self._has_query(round_index):
            self._write_predictions(pool)
        size = 0
        if self._has_query(round_index):
            size = self._query_sizes[round_index] - int(self._labeled.sum())
        if size == 0:
            if self.cfg.save_round_details:
                self._write_details(round_index, pool, {})
            return evaluation, None
        with clock.measure("select"):
            # a query that takes every unlabeled person is forced: the strategy scores nothing
            forced = size >= self.n - int(self._labeled.sum())
            extra_inputs = {} if forced else self._extras(round_index, pool)
            picked, score_means = self._select(round_index, pool, size, extra_inputs)
        self._check_scores(score_means, size)
        return evaluation, _Query(picked, pool.predictions, score_means)

    def _check_scores(self, score_means: dict[str, float], size: int) -> None:
        """Raise unless the scores are ``{}`` (a forced query) or those of the strategy's scorers."""
        forced = size >= self.n - int(self._labeled.sum())
        scorers = self.strategy.config.scorers
        expected = [] if forced else sorted(scorers)
        if sorted(score_means) != expected:
            raise ValueError(
                f"strategy {self.strategy.config.name!r} gave scores for {sorted(score_means)}; a "
                f"{'forced' if forced else 'regular'} query must give {expected}"
            )

    def _reuse(self, stored: PendingQuery, clock: _Clock) -> tuple[_Evaluation, _Query]:
        """The query of a round that was waiting for the oracle, instead of steps 1 to 4.

        The generator continues from where the selection left it, and the clock continues from
        the work done before the pause.
        """
        self._check_config(
            stored.config, f"the pending query of round {stored.round_index}"
        )
        queried = np.asarray(stored.queried, dtype=np.int64)
        expected = (self.n, self.spec.num_joints, 3)
        if (
            queried.ndim != 1
            or queried.size == 0
            or queried.min() < 0
            or queried.max() >= self.n
            or stored.predictions.shape != expected
        ):
            raise ValueError(
                f"the pending query of round {stored.round_index} does not fit this video: "
                f"{queried.size} queried people, predictions of shape {stored.predictions.shape} "
                f"instead of {expected}; remove {self.state_dir} to start the run again"
            )
        self._check_scores(stored.score_means, queried.size)
        self._rng.bit_generator.state = stored.rng
        clock.carry(stored.timings)
        return (
            _Evaluation(**stored.evaluation),
            _Query(queried, stored.predictions, stored.score_means),
        )

    def _store_pending(
        self,
        round_index: int,
        evaluation: _Evaluation,
        query: _Query,
        clock: _Clock,
    ) -> None:
        """Keep the query of a round whose oracle has no answers yet (see :mod:`.state`)."""
        save_pending(
            self.state_dir,
            PendingQuery(
                round_index=round_index,
                config=self._config,
                queried=query.queried.tolist(),
                predictions=query.predictions,
                evaluation=dataclasses.asdict(evaluation),
                score_means=query.score_means,
                rng=self._rng.bit_generator.state,
                timings=clock.timings(),
            ),
        )

    # ------------------------------------------------------------------ start, finish

    def _has_query(self, round_index: int) -> bool:
        return round_index < len(self.cfg.query_ratio)

    def _fingerprint(self) -> dict[str, Any]:
        """The settings that decide what the rounds compute, as the JSON a state file holds."""
        loop = {
            k: v
            for k, v in dataclasses.asdict(self.cfg).items()
            if k not in _RESUME_FREE
        }
        retrain = dataclasses.asdict(self.finetuner.cfg)
        del retrain["num_workers"]
        strategy = self.strategy.config
        config = {
            **self._identity,
            **loop,
            "scorers": list(strategy.scorers),
            "combiner": strategy.combiner,
            "selector": strategy.selector,
            "retrain": retrain,
            "seed_of_video": self.seed,
        }
        return json.loads(json.dumps(config, allow_nan=False))

    def _check_config(self, stored: dict[str, Any], what: str) -> None:
        """Raise if ``what`` was written by a run with other settings than this one."""
        config = self._config
        differences = [
            f"{key}: {what} has {stored.get(key)!r}, this run has {config.get(key)!r}"
            for key in sorted(set(config) | set(stored))
            if config.get(key) != stored.get(key)
        ]
        if differences:
            raise ValueError(
                f"{self.state_dir} belongs to a run with other settings ("
                + "; ".join(differences)
                + "). Use another run directory, or remove the state to start again."
            )

    def _restore(self) -> None:
        """Load the state of the finished rounds, or start a new one; always from the files."""
        self._labeled = np.zeros(self.n, dtype=bool)
        self._labels = np.zeros((self.n, self.spec.num_joints, 3), dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)
        self._config = self._fingerprint()
        loaded = load_state(self.state_dir)
        if loaded is None:
            self._state = LoopState(
                config=self._config, rng=self._rng.bit_generator.state
            )
            torch.manual_seed(self.seed)
        else:
            self._check_config(loaded.state.config, "the state")
            self._state = loaded.state
            self._rng.bit_generator.state = loaded.state.rng
            torch.set_rng_state(loaded.torch_rng)
            self.finetuner.load_state_dict(loaded.finetuner)
            for record in loaded.state.rounds:
                if record.queried:
                    self._labeled[record.queried] = True
                    self._labels[record.queried] = np.asarray(
                        record.answers, dtype=np.float64
                    )
        if self._uses_ae and self._has_query(self._state.rounds_done):
            self._ae = self._build_ae()

    def _finish(self) -> dict[str, Any]:
        result = build_result(
            self._identity, self.strategy.config.scorers, self._state.rounds
        )
        if self.cfg.save_final_model:
            self._write_estimator("final_estimator.pth")
        write_result(self.run_dir, result)
        return result

    # ------------------------------------------------------------------ step 1: inference

    @torch.no_grad()
    def _infer(self) -> _Pool:
        """Heatmaps, decoded joints and (if a selector needs them) embeddings of all people."""
        self.estimator.eval()
        n, k = self.n, self.spec.num_joints
        heatmaps = torch.empty((n, k, *HEATMAP_SIZE), dtype=torch.float32)
        coords = np.empty((n, k, 2), dtype=np.float32)
        maxvals = np.empty((n, k), dtype=np.float32)
        embeddings: list[np.ndarray] = []
        step = self.cfg.infer_batch_size
        for start in range(0, n, step):
            index = np.arange(start, min(start + step, n))
            x = to_input_tensor(self.cache.crops(index)).to(self.device)
            if self._uses_embeddings:
                output, embedding = self.estimator.forward_with_embedding(x)
                embeddings.append(embedding.cpu().numpy())
            else:
                output = self.estimator(x)
            assert output.dtype == torch.float32, (
                f"the model returned {output.dtype} heatmaps"
            )
            output = output.cpu()
            heatmaps[start : start + len(index)] = output
            decoded, scores = heatmaps_to_coords(
                output, torch.from_numpy(self._arrays.bbox_crop[index])
            )
            coords[index] = decoded.numpy()
            maxvals[index] = scores.numpy()
        return _Pool(
            heatmaps=heatmaps,
            coords=coords,
            maxvals=maxvals,
            embeddings=np.concatenate(embeddings) if embeddings else None,
        )

    # ------------------------------------------------------------------ step 2: evaluation

    def _report(
        self, round_index: int, step: str, done: int | None = None, total: int | None = None
    ) -> None:
        """Write the step of the loop, and the epoch while it trains, to ``progress.json``.

        A new step is always written; the epochs of a step at most every half second and at the
        last epoch, so that a short epoch does not wait for the file system.
        """
        now = time.monotonic()
        if (
            done is not None
            and done != total
            and (round_index, step) == self._reported[:2]
            and now - self._reported[2] < _REPORT_INTERVAL
        ):
            return
        self._reported = (round_index, step, now)
        document = {
            "schema": PROGRESS_SCHEMA,
            "round": round_index,
            "step": step,
            "done": done,
            "total": total,
            "time": time.time(),
        }
        write_atomic(
            self.run_dir / "progress.json",
            lambda f: f.write(json.dumps(document).encode()),
        )

    def _labeled_with_joints(self) -> np.ndarray:
        """Sample indices of the labeled people whose label has a visible joint.

        An annotator answers a person who shows no joint (hidden, or a wrong box) with no visible
        joint. Such a person counts as labeled, so it is not asked again, but it gives no training
        target, no autoencoder or auxiliary network input and no OKS.
        """
        return np.flatnonzero(self._labeled & (self._labels[:, :, 2] > 0).any(axis=1))

    def _oks(
        self, coords: np.ndarray, index: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """OKS of the predicted joints ``coords`` of the people ``index`` against ``labels``."""
        return oks_per_sample(
            coords[index], labels, self._boxes_xywh[index], self._sigmas
        )

    def _evaluate(self, pool: _Pool) -> _Evaluation:
        """AP and OSPA against the ground truth, with and without the labels of the labeled people.

        The four metrics are ``None`` unless every person of the video has ground truth.
        """
        arrays = self._arrays
        score = pool.pose_score

        def measure(coords: np.ndarray) -> tuple[dict[str, float], float]:
            metrics = evaluate_predictions(
                coords,
                score,
                arrays.keypoints_gt,
                self._boxes_xywh,
                arrays.image_id,
                self.spec,
            )
            return {key: float(v) for key, v in metrics.ap.items()}, float(metrics.ospa)

        labeled = np.flatnonzero(self._labeled)
        performance = performance_ann = ospa = ospa_ann = None
        if self._has_ground_truth:
            performance, ospa = measure(pool.coords)
            replaced = pool.coords.astype(np.float64)
            with_joints = self._labeled_with_joints()
            replaced[with_joints] = self._labels[with_joints, :, :2]
            performance_ann, ospa_ann = measure(replaced)

        unlabeled = np.flatnonzero(~self._labeled)
        combine_weight = None
        if unlabeled.size:
            peaks = [
                local_peak_mean(pool.heatmaps[unlabeled[i : i + _PEAK_CHUNK]])
                for i in range(0, unlabeled.size, _PEAK_CHUNK)
            ]
            combine_weight = float(np.concatenate(peaks).mean())
        return _Evaluation(
            100.0 * int(labeled.size) / self.n,
            performance,
            performance_ann,
            ospa,
            ospa_ann,
            combine_weight,
        )

    def _write_predictions(self, pool: _Pool) -> None:
        """Write ``predictions.json`` and ``predictions_ann.json`` from the final inference.

        ``pool`` is the inference that :meth:`_evaluate` measured, so the coordinates are the ones
        of the last ``performances``. The second file replaces the labeled people with a visible
        joint by their labels, as ``performances_ann`` does.
        """
        arrays = self._arrays
        estimated = pool.predictions
        annotated = estimated.copy()
        labeled = self._labeled_with_joints()
        annotated[labeled] = self._labels[labeled]
        score = pool.pose_score
        write_predictions(
            self.run_dir,
            build_predictions(
                estimated, score, arrays.ann_id, arrays.image_id, arrays.track_id
            ),
            build_predictions(
                annotated, score, arrays.ann_id, arrays.image_id, arrays.track_id
            ),
        )

    # ------------------------------------------------------------------ steps 3 and 4: query

    def _extras(self, round_index: int, pool: _Pool) -> dict[str, Any]:
        """The inputs of the scorers that need more than the heatmaps (see :mod:`.extras`)."""
        inputs: dict[str, Any] = {}
        if self._uses_ae:
            assert self._ae is not None, "the autoencoder of this round was not built"
            inputs[AE_KEY] = self._ae
        if self._uses_vl4pose:
            assert pool.embeddings is not None
            labeled = self._labeled_with_joints()
            inputs[VL4POSE_KEY] = extras.fit_link_gaussians(
                pool.embeddings,
                labeled,
                self._labels[labeled],
                self._arrays.bbox_gt,
                self.spec,
                epochs=self.cfg.vl4pose_epochs,
                lr=self.cfg.vl4pose_lr,
                batch_size=self.cfg.vl4pose_batch_size,
                init_seed=_loop_seed(self.seed, "auxnet-init", round_index),
                train_seed=_loop_seed(self.seed, "auxnet-train", round_index),
            )
        return inputs

    def _moks_previous(self) -> float:
        """Mean query OKS of the latest round that queried somebody; 0.0 before any."""
        for record in reversed(self._state.rounds):
            if record.moks_queried is not None:
                return record.moks_queried
        return 0.0

    def _select(
        self, round_index: int, pool: _Pool, size: int, extra_inputs: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Ask the strategy for ``size`` people; returns them sorted and the mean raw scores."""
        arrays = self._arrays
        ctx = RoundContext(
            round_index=round_index,
            spec=self.spec,
            heatmaps=pool.heatmaps,
            coords=pool.coords,
            maxvals=pool.maxvals,
            bbox_crop=arrays.bbox_crop,
            prev_index=arrays.prev_index,
            next_index=arrays.next_index,
            labeled=self._labeled.copy(),
            embeddings=None
            if pool.embeddings is None
            else pool.embeddings.astype(np.float64),
            moks_previous=self._moks_previous(),
            rng=self._rng,
            extras=extra_inputs,
        )
        selected, raw = self.strategy.query(ctx, size)
        if self.cfg.save_round_details:
            forced = size >= self.n - int(self._labeled.sum())
            self._write_details(
                round_index,
                pool,
                round_details.selection_details(
                    self.strategy, ctx, raw, selected, forced
                ),
            )
        unlabeled = ctx.unlabeled_index
        means = {name: float(score[unlabeled].mean()) for name, score in raw.items()}
        return np.sort(selected), means

    def _write_details(
        self, round_index: int, pool: _Pool, selection: dict[str, np.ndarray]
    ) -> None:
        """Write the details of the round; ``selection`` is empty for a round without a query."""
        details = round_details.prediction_details(
            round_index, self._arrays, pool.coords, pool.maxvals, self._labeled
        )
        round_details.write_round_details(
            self.run_dir, round_index, {**details, **selection}
        )

    # ------------------------------------------------------------------ step 5: oracle

    def _ask_oracle(self, round_index: int, query: _Query) -> np.ndarray | Pending:
        """The labels of the queried people as ``(k, K, 3)``, or :class:`Pending` if not there yet."""
        outcome = self.oracle.request(query.queried, round_index, query.predictions)
        if not outcome.ready:
            return Pending(round_index=round_index, message=outcome.message)
        assert outcome.keypoints is not None
        answers = outcome.keypoints
        expected = (len(query.queried), self.spec.num_joints, 3)
        if answers.shape != expected:
            raise ValueError(
                f"the oracle answered with shape {answers.shape} for a request of "
                f"{len(query.queried)} people; expected {expected}"
            )
        blind = query.queried[~(answers[:, :, 2] > 0).any(axis=1)]
        if blind.size:
            logger.info(
                "round %d: the oracle marked no joint visible for the samples %s; they count "
                "as labeled but give no training target, autoencoder feature or OKS",
                round_index,
                blind.tolist(),
            )
        return answers

    # ------------------------------------------------------------------ step 6: retraining

    def _retrain(
        self, round_index: int, query: _Query, answers: np.ndarray
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Label the people, train the estimator on them, return their mean query OKS.

        The training set is the queried people plus the labeled people whose OKS (under the model
        before this training) is at most ``finish_acc + 0.05``; the number of epochs follows from
        the mean OKS of the queried people. People without a visible joint in their label are
        labeled but left out of both (see :meth:`_labeled_with_joints`). If no queried person has
        a visible joint, the estimator is not trained and the mean OKS is ``None``. With
        ``loop.stopping`` the second value is the judgment of :meth:`_stopping` on the OKS of the
        labeled and the queried people, all under the model before this training.
        """
        picked, coords = query.queried, query.coords
        before = self._labeled_with_joints()
        oks_before = self._oks(coords, before, self._labels[before])
        seen = (answers[:, :, 2] > 0).any(axis=1)
        self._labeled[picked] = True
        self._labels[picked] = answers
        oks_queried = (
            self._oks(coords, picked[seen], answers[seen]) if seen.any() else np.empty(0)
        )
        stopping = (
            self._stopping(np.concatenate([oks_before, oks_queried]), coords)
            if self.cfg.stopping
            else None
        )
        if not seen.any():
            logger.warning(
                "round %d: no queried person has a visible joint, so the estimator is not trained",
                round_index,
            )
            return None, stopping
        moks = float(oks_queried.mean())
        training = Finetuner.retrain_indices(
            before, oks_before, picked[seen], self.finetuner.cfg.finish_acc
        )
        epochs = self.finetuner.epochs_for_round(int(self._labeled.sum()), self.n, moks)
        dataset = FinetuneDataset(
            self.cache,
            training,
            self._labels[training],
            self.spec,
            self.finetuner.cfg.augment,
            self.seed,
            round_index,
        )
        self.finetuner.run(
            dataset,
            epochs,
            on_epoch=lambda done, total: self._report(round_index, "retrain", done, total),
        )
        return moks, stopping

    def _stopping(self, oks: np.ndarray, coords: np.ndarray) -> dict[str, Any] | None:
        """SC_All of the paper over the OKS of the annotated people; ``None`` without anybody.

        ``oracle`` is the same count over every person of the video (see :meth:`_oracle_stopping`).
        """
        if oks.size == 0:
            return None
        theta = self.finetuner.cfg.finish_acc
        above = int((oks >= theta).sum())
        return {
            "theta": theta,
            "people": int(oks.size),
            "above": above,
            "lowest": float(oks.min()),
            "met": above == oks.size,
            "oracle": self._oracle_stopping(coords, theta),
        }

    def _oracle_stopping(
        self, coords: np.ndarray, theta: float
    ) -> dict[str, Any] | None:
        """The oracle stop of the paper (``actual`` in Table 5): does every person reach ``theta``?

        The OKS of the predicted joints ``coords`` of all people of the video against the ground
        truth, the people not labeled yet included. The loop never reads it; it only records it,
        so that ``vatl4pose-summarize`` can name the round in which the video was solved. ``None``
        unless every person has ground truth.
        """
        if not self._has_ground_truth:
            return None
        oks = self._oks(coords, np.arange(self.n), self._arrays.keypoints_gt)
        above = int((oks >= theta).sum())
        return {
            "people": int(oks.size),
            "above": above,
            "lowest": float(oks.min()),
            "met": above == oks.size,
        }

    # ------------------------------------------------------------------ step 7: autoencoder

    def _pretrained_ae(self) -> WholeBodyAE:
        assert self.ae_ckpt is not None
        return extras.load_pretrained_ae(
            self.ae_ckpt, self.spec, self.cfg.ae_z_dim, self.device
        )

    def _build_ae(self) -> WholeBodyAE:
        """The pretrained autoencoder, tuned on the hybrid features of the labels with a visible joint.

        The features are those of the answers of the oracle, with the label visibility as weights
        (with the simulated oracle the answers are the ground truth); a joint with no visibility
        is masked out of the loss. The pretrained feature statistics stay fixed. The result depends
        on the checkpoint, the labels and the seed only, so a resume gets the autoencoder the
        uninterrupted run had.
        """
        ae = self._pretrained_ae()
        labeled = self._labeled_with_joints()
        if labeled.size:
            labels = self._labels[labeled]
            features = compute_hybrid(
                self._arrays.bbox_json_xywh[labeled], labels, self.spec
            )
            extras.finetune_wholebody_ae(
                ae,
                features,
                hybrid_valid_mask(labels[..., 2], self.spec),
                epochs=self.cfg.ae_finetune_epochs,
                lr=self.cfg.ae_finetune_lr,
                batch_size=self.cfg.ae_finetune_batch,
                seed=_loop_seed(self.seed, "ae", int(labeled.size)),
            )
        return ae

    def _finetune_ae(self) -> None:
        """Tune a new autoencoder on the labeled people for the next round's scores."""
        self._ae = self._build_ae()

    # ------------------------------------------------------------------ step 8: state

    def _save(self, record: RoundRecord) -> None:
        self._state.rounds.append(record)
        self._state.rng = self._rng.bit_generator.state
        save_state(
            self.state_dir,
            self._state,
            torch.get_rng_state(),
            self.finetuner.state_dict(),
        )
        if self.cfg.save_estimator:
            self._write_estimator("estimator.pth")

    def _write_estimator(self, name: str) -> None:
        """Write the state dict of the estimator to ``run_dir / name``."""
        weights = {k: v.detach().cpu() for k, v in self.estimator.state_dict().items()}
        write_atomic(self.run_dir / name, lambda f: torch.save(weights, f))
