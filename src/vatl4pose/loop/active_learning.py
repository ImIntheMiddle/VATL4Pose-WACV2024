"""The active-learning loop of one video, with a round state that survives a crash.

A round of :class:`ActiveLearningLoop` is the procedure of the paper. One method does each step:

1. :meth:`~ActiveLearningLoop._infer`: the model predicts the heatmaps (and the embeddings, if the
   selector clusters them) of all people of the video.
2. :meth:`~ActiveLearningLoop._evaluate`: AP and OSPA of the predictions against the ground truth,
   and the same after the labeled people's predictions are replaced by their labels.
3. :meth:`~ActiveLearningLoop._extras`: the inputs that WPU (the autoencoder) and VL4Pose (the link
   Gaussians) need.
4. :meth:`~ActiveLearningLoop._select`: the strategy chooses the people to annotate.
5. :meth:`~ActiveLearningLoop._ask_oracle`: the oracle answers, or the loop stops with
   :class:`Pending` until the human has annotated.
6. :meth:`~ActiveLearningLoop._retrain`: the estimator learns from the new answers and the people it
   has not solved yet.
7. :meth:`~ActiveLearningLoop._finetune_ae`: the autoencoder is tuned on the labeled people.
8. :meth:`~ActiveLearningLoop._save`: the state of the finished round is written.

Round ``r`` evaluates and then queries ``query_sizes[r] - n_labeled`` people with
``query_sizes = [int(N * ratio) for ratio in query_ratio]``. The last round, numbered
``len(query_ratio)``, only evaluates the final model: no query and no retraining.

Decisions where the plan left a choice:

* A query of size 0 (``int`` rounding on a small video) skips steps 3 to 7; the round still
  evaluates and is saved. Its mean query OKS does not exist, so it is ``None`` and the strategy's
  ``moks_previous`` stays at the last value that exists (0.0 before any).
* Every list of queried indices is sorted before the oracle sees it, so the order of the answers,
  of the fine-tuning set and of the files does not depend on the selector's order.
* The auxiliary network of VL4Pose is new in every round, trained on the labeled people with this
  round's features; nothing of it is saved. Without labeled people (round 0) it keeps its random
  initial weights.
* The autoencoder of round ``r + 1`` is a function of the pretrained checkpoint, the labeled set
  and the seed. A resume rebuilds it with the same function instead of storing it.
* The retraining set, the mean query OKS and the OKS of the labeled people use the labels the
  oracle gave. With the simulated oracle they are the ground truth.

The processes that run videos in parallel share nothing; the per-video lock is the job of the caller.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..data.cache import VideoCache
from ..data.datasets import FinetuneDataset
from ..data.geometry import HEATMAP_SIZE, heatmaps_to_coords, to_input_tensor
from ..eval import evaluate_predictions, oks_per_sample
from ..models.estimator import PoseEstimator
from ..models.wholebody_ae import WholeBodyAE
from ..oracle import Oracle
from ..strategies import RoundContext, Strategy, local_peak_mean
from ..strategies.baselines.vl4pose import EXTRAS_KEY as VL4POSE_KEY
from ..train import Finetuner
from . import extras
from .results import build_result, write_result
from .runs import video_seed
from .state import LoopState, RoundRecord, load_state, save_state, write_atomic

logger = logging.getLogger(__name__)

AE_KEY = "ae"
"""Key of ``RoundContext.extras`` that holds the whole-body autoencoder."""

STEPS = ("infer", "eval", "select", "oracle", "retrain", "ae")
"""The timed steps of a round; ``total`` is added to them in the record."""

# The selectors that cluster the backbone embeddings. The VL4Pose scorer also reads them. A
# selector of another package that needs embeddings has to be added here; without them it raises
# its own error in the first query.
_EMBEDDING_SELECTORS = frozenset({"kmeans", "coreset", "coreset_duw"})
_PEAK_CHUNK = 256
# Settings that do not change what a round computes; a resume may differ in them.
_RESUME_FREE = ("infer_batch_size", "save_final_model")


@dataclass(frozen=True)
class LoopConfig:
    """Settings of the loop; the defaults are the ones of the paper.

    Attributes:
        query_ratio: Share of the video that is labeled after each query, non-decreasing in
            ``(0, 1]``. ``len(query_ratio)`` queries and one more evaluation make the rounds.
        infer_batch_size: People per forward pass of the inference.
        save_final_model: Write the state dict of the final model to ``final_estimator.pth``.
        seed: Base seed; the seed of a video is ``video_seed(seed, video_id)``.
        ae_finetune_epochs: Epochs of the autoencoder tuning after every retraining.
        ae_finetune_lr: Learning rate (Adam) of that tuning.
        ae_finetune_batch: Batch size of that tuning.
        ae_z_dim: Latent size of the autoencoder checkpoint (4 for the released ones).
        vl4pose_epochs: Training epochs of the auxiliary network of VL4Pose in every round.
        vl4pose_lr: Learning rate (Adam) of the auxiliary network.
        vl4pose_batch_size: Batch size of its training and of its prediction.
    """

    query_ratio: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
    infer_batch_size: int = 256
    save_final_model: bool = False
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
        if any(b < a for a, b in zip(ratio, ratio[1:])):
            raise ValueError(f"query_ratio must not decrease, got {ratio}")
        for name in ("infer_batch_size", "ae_finetune_batch", "vl4pose_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.vl4pose_epochs < 1:
            raise ValueError(
                f"vl4pose_epochs must be at least 1, got {self.vl4pose_epochs}"
            )
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

    @property
    def predictions(self) -> np.ndarray:
        """``(N, K, 3)`` float64 x, y and peak value, what the oracle may prefill from."""
        return np.concatenate([self.coords, self.maxvals[..., None]], axis=-1).astype(
            np.float64
        )

    @property
    def pose_score(self) -> np.ndarray:
        """``(N,)`` float64 score of every pose: mean plus 1.25 times the maximum of its peaks."""
        peaks = self.maxvals.astype(np.float64)
        return peaks.mean(axis=1) + 1.25 * peaks.max(axis=1)


@dataclass
class _Evaluation:
    percentage: float
    performance: dict[str, float]
    performance_ann: dict[str, float]
    ospa: float
    ospa_ann: float
    combine_weight: float | None


class _Clock:
    """Wall-clock seconds per step of one round; waits for the GPU so that a step owns its work."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.times = dict.fromkeys(STEPS, 0.0)
        self._start = time.perf_counter()

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @contextmanager
    def measure(self, step: str) -> Iterator[None]:
        self._synchronize()
        began = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.times[step] += time.perf_counter() - began

    def timings(self) -> dict[str, float]:
        """The seconds of every step and ``total``, the time since the round began."""
        self._synchronize()
        return {**self.times, "total": time.perf_counter() - self._start}


def _derived_seed(base: int, *parts: int | str) -> int:
    """A 63-bit seed from ``base`` and ``parts`` through SHA-256; equal on every run and machine."""
    text = "/".join(["vatl4pose.loop", str(base), *map(str, parts)])
    return (
        int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") >> 1
    )


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """``x1, y1, x2, y2`` to ``x, y, w, h`` with width and height counting both end pixels."""
    return np.hstack((boxes[:, :2], boxes[:, 2:4] - boxes[:, :2] + 1.0))


class ActiveLearningLoop:
    """Runs the rounds of one video and keeps the state that lets a stopped run continue.

    Args:
        cache: Cache of the video.
        estimator: The pose model; it must be the model of ``finetuner``.
        strategy: The query strategy. Scorers ``wpu`` and ``vl4pose`` and the selectors
            ``kmeans``, ``coreset`` and ``coreset_duw`` make the loop provide what they need.
        oracle: The source of the labels.
        finetuner: Trains ``estimator`` after every query. Its seed must be
            ``video_seed(cfg.seed, cache.video_id)``, so that one seed drives every part of a run.
        ae_ckpt: Pretrained whole-body autoencoder; required by strategies with the ``wpu`` scorer.
        cfg: Settings of the loop.
        run_dir: Directory of the run; ``state/``, ``result.json`` and ``final_estimator.pth`` go
            here. The CLI holds the per-video lock.
        device: Device of the inference and of the autoencoder.

    Raises:
        ValueError: If the finetuner does not train ``estimator`` or has another seed, or the
            strategy needs an autoencoder and ``ae_ckpt`` is ``None``.
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
        self._uses_embeddings = (
            self._uses_vl4pose or strategy.config.selector in _EMBEDDING_SELECTORS
        )
        if self._uses_ae:
            if ae_ckpt is None:
                raise ValueError(
                    f"strategy {strategy.config.name!r} uses the 'wpu' scorer, which needs the "
                    "whole-body autoencoder; pass its checkpoint as ae_ckpt"
                )
            self._pretrained_ae()  # fails now, not in round 0, if the checkpoint does not fit

        arrays = cache.arrays
        self._arrays = arrays
        self.n = len(cache)
        self._query_sizes = [int(self.n * ratio) for ratio in cfg.query_ratio]
        self._boxes_xywh = _xyxy_to_xywh(arrays.bbox_gt)
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
        for round_index in range(
            self._state.rounds_done, len(self.cfg.query_ratio) + 1
        ):
            pending = self._run_round(round_index)
            if pending is not None:
                return pending
        return self._finish()

    def _run_round(self, round_index: int) -> Pending | None:
        clock = _Clock(self.device)
        with clock.measure("infer"):
            pool = self._infer()
        with clock.measure("eval"):
            evaluation = self._evaluate(pool)
        queried: list[int] | None = None
        answers: np.ndarray | None = None
        moks: float | None = None
        score_means: dict[str, float] = {}
        if self._has_query(round_index):
            size = self._query_sizes[round_index] - int(self._labeled.sum())
            if size > 0:
                with clock.measure("select"):
                    extra_inputs = self._extras(round_index, pool)
                    picked, score_means = self._select(
                        round_index, pool, size, extra_inputs
                    )
                with clock.measure("oracle"):
                    outcome = self._ask_oracle(round_index, picked, pool)
                if isinstance(outcome, Pending):
                    return outcome
                answers = outcome
                with clock.measure("retrain"):
                    moks = self._retrain(round_index, pool, picked, answers)
                if self._uses_ae and self._has_query(round_index + 1):
                    with clock.measure("ae"):
                        self._finetune_ae()
                queried = picked.tolist()
            else:
                queried = []
        record = RoundRecord(
            percentage=evaluation.percentage,
            performance=evaluation.performance,
            performance_ann=evaluation.performance_ann,
            ospa=evaluation.ospa,
            ospa_ann=evaluation.ospa_ann,
            combine_weight=evaluation.combine_weight,
            score_means={
                name: score_means.get(name) for name in self.strategy.config.scorers
            },
            queried=queried,
            answers=None if answers is None else answers.tolist(),
            moks_queried=moks,
            timings=clock.timings(),
        )
        self._save(record)
        logger.info(
            "%s round %d: %.1f %% labeled, AP %.3f (with labels %.3f), %.1f s",
            self._identity["video_id"],
            round_index,
            record.percentage,
            record.performance["AP"],
            record.performance_ann["AP"],
            record.timings["total"],
        )
        return None

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

    def _restore(self) -> None:
        """Load the state of the finished rounds, or start a new one; always from the files."""
        config = self._fingerprint()
        self._labeled = np.zeros(self.n, dtype=bool)
        self._labels = np.zeros((self.n, self.spec.num_joints, 3), dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)
        loaded = load_state(self.state_dir)
        if loaded is None:
            self._state = LoopState(config=config, rng=self._rng.bit_generator.state)
            torch.manual_seed(self.seed)
        else:
            differences = [
                f"{key}: the state has {loaded.state.config.get(key)!r}, this run has "
                f"{config.get(key)!r}"
                for key in sorted(set(config) | set(loaded.state.config))
                if config.get(key) != loaded.state.config.get(key)
            ]
            if differences:
                raise ValueError(
                    f"{self.state_dir} belongs to a run with other settings ("
                    + "; ".join(differences)
                    + "). Use another run directory, or remove the state to start again."
                )
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
            weights = {
                k: v.detach().cpu() for k, v in self.estimator.state_dict().items()
            }
            write_atomic(
                self.run_dir / "final_estimator.pth", lambda f: torch.save(weights, f)
            )
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

    def _oks(self, pool: _Pool, index: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """OKS of the predictions of the people ``index`` against ``labels``, float64."""
        return oks_per_sample(
            pool.coords[index], labels, self._boxes_xywh[index], self._sigmas
        )

    def _evaluate(self, pool: _Pool) -> _Evaluation:
        """AP and OSPA against the ground truth, with and without the labels of the labeled people."""
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

        performance, ospa = measure(pool.coords)
        labeled = np.flatnonzero(self._labeled)
        replaced = pool.coords.astype(np.float64)
        replaced[labeled] = self._labels[labeled, :, :2]
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

    # ------------------------------------------------------------------ steps 3 and 4: query

    def _extras(self, round_index: int, pool: _Pool) -> dict[str, Any]:
        """The inputs of the scorers that need more than the heatmaps (see :mod:`.extras`)."""
        inputs: dict[str, Any] = {}
        if self._uses_ae:
            inputs[AE_KEY] = self._ae
        if self._uses_vl4pose:
            assert pool.embeddings is not None
            labeled = np.flatnonzero(self._labeled)
            inputs[VL4POSE_KEY] = extras.fit_link_gaussians(
                pool.embeddings,
                labeled,
                self._labels[labeled],
                self._arrays.bbox_gt,
                self.spec,
                epochs=self.cfg.vl4pose_epochs,
                lr=self.cfg.vl4pose_lr,
                batch_size=self.cfg.vl4pose_batch_size,
                init_seed=_derived_seed(self.seed, "auxnet-init", round_index),
                train_seed=_derived_seed(self.seed, "auxnet-train", round_index),
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
        unlabeled = ctx.unlabeled_index
        means = {name: float(score[unlabeled].mean()) for name, score in raw.items()}
        return np.sort(selected), means

    # ------------------------------------------------------------------ step 5: oracle

    def _ask_oracle(
        self, round_index: int, picked: np.ndarray, pool: _Pool
    ) -> np.ndarray | Pending:
        """The labels of ``picked`` as ``(k, K, 3)``, or :class:`Pending` if they are not there yet."""
        outcome = self.oracle.request(picked, round_index, pool.predictions)
        if not outcome.ready:
            return Pending(round_index=round_index, message=outcome.message)
        assert outcome.keypoints is not None
        expected = (len(picked), self.spec.num_joints, 3)
        if outcome.keypoints.shape != expected:
            raise ValueError(
                f"the oracle answered with shape {outcome.keypoints.shape} for a request of "
                f"{len(picked)} people; expected {expected}"
            )
        return outcome.keypoints

    # ------------------------------------------------------------------ step 6: retraining

    def _retrain(
        self, round_index: int, pool: _Pool, picked: np.ndarray, answers: np.ndarray
    ) -> float:
        """Label the people, train the estimator on them, return their mean query OKS.

        The training set is the queried people plus the labeled people whose OKS (under the model
        before this training) is at most ``finish_acc + 0.05``; the number of epochs follows from
        the mean OKS of the queried people.
        """
        before = np.flatnonzero(self._labeled)
        oks_before = self._oks(pool, before, self._labels[before])
        moks = float(self._oks(pool, picked, answers).mean())
        self._labeled[picked] = True
        self._labels[picked] = answers
        training = Finetuner.retrain_indices(
            before, oks_before, picked, self.finetuner.cfg.finish_acc
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
        self.finetuner.run(dataset, epochs)
        return moks

    # ------------------------------------------------------------------ step 7: autoencoder

    def _pretrained_ae(self) -> WholeBodyAE:
        assert self.ae_ckpt is not None
        return extras.load_pretrained_ae(
            self.ae_ckpt, self.spec, self.cfg.ae_z_dim, self.device
        )

    def _build_ae(self) -> WholeBodyAE:
        """The pretrained autoencoder, tuned on the ground-truth features of the labeled people.

        The result depends on the checkpoint, the labeled set and the seed only, so a resume
        gets the autoencoder the uninterrupted run had.
        """
        ae = self._pretrained_ae()
        labeled = np.flatnonzero(self._labeled)
        if labeled.size:
            extras.finetune_wholebody_ae(
                ae,
                self._arrays.hybrid_gt[labeled],
                epochs=self.cfg.ae_finetune_epochs,
                lr=self.cfg.ae_finetune_lr,
                batch_size=self.cfg.ae_finetune_batch,
                seed=_derived_seed(self.seed, "ae", int(labeled.size)),
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
