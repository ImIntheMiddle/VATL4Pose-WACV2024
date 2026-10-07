"""What the commands share: from a composed config to videos, directories and one run of the loop.

Directory layout below ``paths.output_root``::

    cache/<dataset>[/<split>]/<video>.h5               prepare; the split is part of the path
                                                        because the id "00" of JRDB-Pose names a
                                                        different video in train, val and test
    cache/<dataset>[/<split>]/.locks/<video>/.lock      the lock of the cache build of one video
    <run.name>/<estimator>/<strategy>/<video>/          atl: result.json, state/, log.jsonl,
                                                        settings.json, .done.json, .lock, oracle/
    <run.name>/tune/<estimator>/<strategy>/lam_<lam>/<video>/   tune: the same files per lambda
    <run.name>/summary/<estimator>[_<ids>]/             summarize
    train/<run.name>/                                   train: best.pth, last.pth, log.jsonl
    eval/<name>.json                                    eval
    hydra/<date>/<time>_<command>_<host>_<pid>/         Hydra's own files of one process

Every directory that one process writes has one writer: the per-video lock (``VideoLock``) guards
the cache build and the loop run, so a second process on the same video stops with the path of the
lock file in its error.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import pathlib
import signal
import types
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig

from vatl4pose.config import (
    AESection,
    DatasetSection,
    EstimatorSection,
    LoopSection,
    OracleSection,
    PathsSection,
    RetrainSection,
    RunSection,
    StrategySection,
    WandbSection,
    estimator_config,
    keypoint_spec,
    loop_config,
    resolve_path,
    retrain_config,
    section,
    strategy_config,
)
from vatl4pose.data import (
    VideoAnnotation,
    read_coco_custom_video,
    read_jrdb_pose_video,
    read_posetrack21_video,
)
from vatl4pose.data.cache import VideoCache, annotation_sha256
from vatl4pose.eval import compute_alc
from vatl4pose.loop import (
    ActiveLearningLoop,
    LoopConfig,
    Pending,
    VideoLock,
    is_done,
    mark_done,
    parse_shard,
    read_id_list,
    select_shard,
    video_seed,
)
from vatl4pose.loop.results import RESULT_NAME, git_commit
from vatl4pose.models.estimator import build_estimator
from vatl4pose.oracle import CocoFileOracle, SimulatedOracle
from vatl4pose.strategies import StrategyConfig, build_strategy
from vatl4pose.train import Finetuner, RetrainConfig

from ._runlog import open_round_log, wandb_run_name
from ._settings import check_settings, fingerprint, write_settings

LOCKS_DIR = ".locks"
LOGGERS = ("json", "wandb")
ORACLES = ("simulated", "coco_file")


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
"""The signals that end a command like Ctrl-C does: ``timeout``, schedulers and ``kill -TERM``."""


def _stop(signum: int, frame: Optional[types.FrameType]) -> None:
    raise SystemExit(128 + signum)


def hydra_command(task: Callable[[DictConfig], None]) -> Callable[[], None]:
    """The console-script entry of ``task``: Hydra composes ``base.yaml`` and calls ``task(cfg)``.

    The entry turns ``SIGTERM`` and ``SIGHUP`` into ``SystemExit`` (exit status ``128 + signal``),
    so that both unwind the task the way ``SIGINT`` does through ``KeyboardInterrupt``. The
    ``with`` blocks of the task then run, a stopped ``vatl4pose-atl`` releases the lock of its
    video and the next start resumes it. ``SIGKILL`` cannot be caught; it leaves the lock behind,
    and the user removes it by hand (see ``VideoLock``). A signal that the parent set to be
    ignored stays ignored, so a run started with ``nohup`` survives the logout.
    """
    main = hydra.main(
        version_base="1.3", config_path="pkg://vatl4pose.configs", config_name="base"
    )(task)

    @functools.wraps(main)
    def stoppable(*args: Any, **kwargs: Any) -> None:
        for signum in STOP_SIGNALS:
            if signal.getsignal(signum) is not signal.SIG_IGN:
                signal.signal(signum, _stop)
        main(*args, **kwargs)

    return stoppable


# ------------------------------------------------------------------ run settings


def run_name(run: RunSection) -> str:
    """``run.name``, which must be one directory name."""
    name = run.name
    if not name:
        raise ValueError(
            "run.name is not set: use an experiment config (experiment=wacv2024_posetrack21) "
            "or give the name of the directory below paths.output_root, e.g. run.name=my_run"
        )
    if pathlib.PurePath(name).name != name or name in (".", ".."):
        raise ValueError(f"run.name must be one directory name, got {name!r}")
    return name


def torch_device(run: RunSection) -> torch.device:
    """The device of ``run.device``; a CUDA device that does not exist is an error."""
    try:
        device = torch.device(run.device)
    except RuntimeError as error:
        raise ValueError(
            f"run.device must be cpu, cuda or cuda:N, got {run.device!r} ({error})"
        ) from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"run.device must be cpu, cuda or cuda:N, got {run.device!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"run.device={run.device} but CUDA is not available to this process; "
            "check nvidia-smi and CUDA_VISIBLE_DEVICES, or use run.device=cpu"
        )
    return device


def require_unsharded(run: RunSection, command: str) -> None:
    """Raise unless ``run.shard`` is ``0/1``; ``command`` does not split its work."""
    if parse_shard(run.shard) != (0, 1):
        raise ValueError(
            f"run.shard={run.shard} is set, but {command} does not split its videos; "
            "only prepare and atl take a shard"
        )


# ------------------------------------------------------------------ videos


def read_ids(ids: Optional[list[str]], ids_file: Optional[str], what: str) -> list[str]:
    """The video ids of the list ``ids``, or else of the file ``ids_file``.

    A list given by the user beats the file that a dataset or experiment config names, so that
    ``dataset.ids=[000342]`` selects one video of an experiment. Raises if neither is given, a
    file is missing, an id occurs twice or the list is empty.
    """
    if ids is None and ids_file is None:
        raise ValueError(
            f"{what}: give the video ids as ids=[...] or as the file ids_file"
        )
    if ids is not None:
        found = [str(i) for i in ids]
        repeated = sorted(i for i, n in Counter(found).items() if n > 1)
        if repeated:
            raise ValueError(
                f"{what}: video ids occur more than once: {', '.join(repeated)}"
            )
    else:
        assert ids_file is not None
        path = pathlib.Path(ids_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{what}: id list {path} does not exist")
        found = read_id_list(path)
    if not found:
        raise ValueError(f"{what}: the list of video ids is empty")
    return found


def dataset_ids(dataset: DatasetSection) -> list[str]:
    """The ids of the videos of the dataset config, in list order."""
    return read_ids(dataset.ids, dataset.ids_file, "dataset.ids / dataset.ids_file")


def shard_ids(ids: list[str], run: RunSection) -> list[str]:
    """The videos of the shard ``run.shard``: the sorted ids taken round-robin."""
    shard, count = parse_shard(run.shard)
    return select_shard(ids, shard, count)


def dataset_root(paths: PathsSection, dataset: DatasetSection) -> pathlib.Path:
    """The absolute dataset directory with its symbolic links resolved.

    The annotation readers resolve the root before they build the image paths of the cache, so the
    COCO-file oracle must compare against the root in the same form.
    """
    return resolve_path(paths.data_root, dataset.root).resolve()


def read_video(
    paths: PathsSection,
    dataset: DatasetSection,
    video_id: str,
    split: Optional[str],
) -> VideoAnnotation:
    """Read the annotation of one video; ``split`` is ``activelearning/<split>`` or ``None``."""
    root = dataset_root(paths, dataset)
    if dataset.name in ("posetrack21", "jrdb_pose"):
        if not split:
            raise ValueError(
                f"dataset {dataset.name!r} needs a split (the directory below activelearning), "
                "got none"
            )
        if dataset.name == "posetrack21":
            if dataset.panorama:
                raise ValueError(
                    "dataset.panorama=true, but PoseTrack21 has no 360-degree frames; "
                    "panorama is an option of jrdb_pose and coco_custom"
                )
            return read_posetrack21_video(root, video_id, split, keypoint_spec(dataset))
        return read_jrdb_pose_video(
            root, video_id, split, keypoint_spec(dataset), dataset.panorama
        )
    if dataset.name == "coco_custom":
        if split:
            raise ValueError(
                f"coco_custom has no splits, but the split {split!r} is set; "
                "use an id list for the videos you want"
            )
        return read_coco_custom_video(
            root, video_id, keypoint_spec(dataset), dataset.panorama
        )
    raise ValueError(f"unknown dataset {dataset.name!r}")


def output_root(paths: PathsSection) -> pathlib.Path:
    """The absolute ``paths.output_root``."""
    return resolve_path(".", paths.output_root)


def cache_dir(
    paths: PathsSection, dataset: DatasetSection, split: Optional[str]
) -> pathlib.Path:
    """Directory of the caches of one dataset and split (see the module docstring)."""
    base = output_root(paths) / "cache" / dataset.name
    return base / split if split else base


def cache_file(
    paths: PathsSection,
    dataset: DatasetSection,
    split: Optional[str],
    video_id: str,
) -> pathlib.Path:
    """The cache file of one video."""
    return cache_dir(paths, dataset, split) / f"{video_id}.h5"


def run_dir_of(
    paths: PathsSection,
    run: RunSection,
    estimator: EstimatorSection,
    strategy_name: str,
    video_id: str,
) -> pathlib.Path:
    """Output directory of the atl run of one video."""
    return (
        output_root(paths) / run_name(run) / estimator.arch / strategy_name / video_id
    )


# ------------------------------------------------------------------ the loop of one video


@dataclass(frozen=True)
class Job:
    """Everything ``atl`` and ``tune`` need to run the loop on any video of a dataset."""

    paths: PathsSection
    run: RunSection
    dataset: DatasetSection
    estimator: EstimatorSection
    ae: AESection
    oracle: OracleSection
    logger: str
    wandb: WandbSection
    strategy: StrategyConfig
    loop: LoopConfig
    retrain: RetrainConfig
    device: torch.device


def logger_name(cfg: DictConfig) -> str:
    """``cfg.logger``, which must be ``json`` or ``wandb``."""
    logger = cfg.logger
    if logger not in LOGGERS:
        raise ValueError(f"logger must be one of {LOGGERS}, got {logger!r}")
    return str(logger)


def load_job(cfg: DictConfig) -> Job:
    """Read the sections the loop needs and check the choices that no section type can."""
    run = section(cfg, "run", RunSection)
    oracle = section(cfg, "oracle", OracleSection)
    if oracle.name not in ORACLES:
        raise ValueError(f"oracle.name must be one of {ORACLES}, got {oracle.name!r}")
    logger = logger_name(cfg)
    return Job(
        paths=section(cfg, "paths", PathsSection),
        run=run,
        dataset=section(cfg, "dataset", DatasetSection),
        estimator=section(cfg, "estimator", EstimatorSection),
        ae=section(cfg, "ae", AESection),
        oracle=oracle,
        logger=logger,
        wandb=section(cfg, "wandb", WandbSection),
        strategy=strategy_config(section(cfg, "strategy", StrategySection)),
        loop=loop_config(section(cfg, "loop", LoopSection)),
        retrain=retrain_config(section(cfg, "retrain", RetrainSection)),
        device=torch_device(run),
    )


@dataclass(frozen=True)
class VideoOutcome:
    """What happened to one video: ``done``, ``skipped`` (already done) or ``pending``."""

    status: str
    message: str


def _wandb_config(job: Job, video_id: str, split: Optional[str]) -> dict[str, Any]:
    return {
        "dataset": job.dataset.name,
        "split": split,
        "video_id": video_id,
        "run_name": job.run.name,
        "estimator": dataclasses.asdict(job.estimator),
        "ae": dataclasses.asdict(job.ae),
        "strategy": dataclasses.asdict(job.strategy),
        "oracle": job.oracle.name,
        "loop": dataclasses.asdict(job.loop),
        "retrain": dataclasses.asdict(job.retrain),
        "seed": job.run.seed,
        "device": str(job.device),
        "git_commit": git_commit(),
    }


def _build_loop(
    job: Job,
    video: VideoAnnotation,
    cache: VideoCache,
    run_dir: pathlib.Path,
    on_round: Callable[..., None],
) -> ActiveLearningLoop:
    seed = video_seed(job.loop.seed, cache.video_id)
    spec = cache.spec
    est_cfg = estimator_config(job.estimator, spec, job.paths.weights_root)
    # the initial weights of a network without a checkpoint follow the seed of the video
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        estimator = build_estimator(est_cfg, job.device)
    finetuner = Finetuner(
        estimator, estimator.state_dict(), job.retrain, job.device, seed
    )
    oracle: Any
    if job.oracle.name == "coco_file":
        oracle = CocoFileOracle(cache, run_dir, dataset_root(job.paths, job.dataset))
    else:
        oracle = SimulatedOracle(cache)
    ae_ckpt = (
        None
        if job.ae.checkpoint is None
        else resolve_path(job.paths.weights_root, job.ae.checkpoint)
    )
    return ActiveLearningLoop(
        cache,
        estimator,
        build_strategy(job.strategy),
        oracle,
        finetuner,
        ae_ckpt,
        job.loop,
        run_dir,
        job.device,
        on_round=on_round,
    )


def run_video(
    job: Job,
    video_id: str,
    split: Optional[str],
    run_dir: pathlib.Path,
    tag: Optional[str] = None,
) -> VideoOutcome:
    """Run the loop on one video under its lock, unless it is done.

    Takes ``<run_dir>/.lock`` (a second process on the same ``run_dir`` fails with the path of the
    lock in its error), compares the settings with ``settings.json`` of the directory (see
    :mod:`._settings`; a difference raises, also for a finished video), returns ``skipped`` if
    ``.done.json`` exists, otherwise runs or resumes the loop, and writes ``.done.json`` when
    ``result.json`` is complete. The first start of a run writes ``settings.json``. A loop whose
    oracle waits for annotations returns ``pending`` and leaves its state for the next call.

    Args:
        job: The settings of the run.
        video_id: The video.
        split: The ``activelearning`` sub-directory of the video, ``None`` for ``coco_custom``.
        run_dir: Output directory of the run of this video.
        tag: Added to the W&B run name, to tell runs of one video apart (the lambda of ``tune``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with VideoLock(run_dir):
        # a run directory belongs to one set of settings, also when its video is finished
        settings = fingerprint(job, video_id, split)
        known = check_settings(run_dir, settings)
        if is_done(run_dir):
            return VideoOutcome("skipped", f"{video_id}: already done ({run_dir})")
        video = read_video(job.paths, job.dataset, video_id, split)
        spec = keypoint_spec(job.dataset)
        path = cache_file(job.paths, job.dataset, split, video_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"the cache {path} does not exist; build it with vatl4pose-prepare using the "
                "same dataset, split and paths.output_root"
            )
        cache = VideoCache(path, spec, annotation_sha256(video), job.dataset.panorama)
        if not known:
            write_settings(run_dir, settings)
        name = wandb_run_name(
            datetime.date.today(), job.dataset.name, job.strategy.name, video_id, tag
        )
        with open_round_log(
            job.logger,
            job.wandb,
            run_dir,
            name,
            job.run.name or "",
            _wandb_config(job, video_id, split) if job.logger == "wandb" else {},
            output_root(job.paths),
            video_id,
        ) as on_round:
            loop = _build_loop(job, video, cache, run_dir, on_round)
            outcome = loop.run()
        if isinstance(outcome, Pending):
            return VideoOutcome(
                "pending",
                f"Pending: {video_id} round {outcome.round_index}: {outcome.message}",
            )
        mark_done(
            run_dir,
            {
                "dataset": job.dataset.name,
                "video_id": video_id,
                "strategy": job.strategy.name,
                "n_samples": len(cache),
                "finished_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
        )
        return VideoOutcome("done", f"{video_id}: done ({run_dir / RESULT_NAME})")


# ------------------------------------------------------------------ results


METRIC_FIELDS = ("performances_ann", "ospa_ann")
"""The fields of ``result.json`` that :func:`metric_curve` reads."""


def load_result(run_dir: pathlib.Path) -> dict[str, Any]:
    """Read ``<run_dir>/result.json``.

    Raises:
        ValueError: If the file lacks ``performances_ann`` or ``ospa_ann``, which every summary
            and the tuning read (they are ``null`` lists for a video without ground truth, not
            missing). The message names the file.
    """
    path = run_dir / RESULT_NAME
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    missing = [field for field in METRIC_FIELDS if field not in result]
    if missing:
        raise ValueError(
            f"{path}: the result lacks {', '.join(missing)}; it is not a result.json of "
            "vatl4pose-atl"
        )
    return result


def has_ground_truth(result: dict[str, Any]) -> bool:
    """``False`` for the result of a video without ground truth, where every metric is ``null``.

    The loop gives ``null`` to the metrics in every round when some person of the video has no
    ground-truth keypoints. This reads the two fields that :func:`metric_curve` reads
    (``performances_ann`` and ``ospa_ann``); :func:`load_result` has checked that they exist.

    Raises:
        KeyError: If ``result`` lacks one of the two fields.
        ValueError: If the metrics are ``null`` in some rounds only; no run writes such a file.
    """
    missing = [value is None for field in METRIC_FIELDS for value in result[field]]
    if missing and all(missing):
        return False
    if any(missing):
        raise ValueError(
            "the metrics of the result are null in some rounds and set in others; the loop "
            "writes either all of them or none"
        )
    return True


def metric_curve(result: dict[str, Any], metric: str) -> list[float]:
    """The value of ``metric`` after every round, on the scale ``[0, 1]``.

    ``AP`` and ``AR`` keys are read from ``performances_ann`` (the predictions of the labeled people
    replaced by their labels); ``OSPA`` is read from ``ospa_ann``.

    Raises:
        ValueError: If the video of the result has no ground-truth keypoints (the metrics are
            ``null``), so that there is no curve.
    """
    if not has_ground_truth(result):
        raise ValueError(
            "the result has no metrics because its video has no ground-truth keypoints "
            "(every metric is null); there is no curve of " + repr(metric)
        )
    if metric == "OSPA":
        return [float(v) for v in result["ospa_ann"]]
    return [float(round_[metric]) for round_ in result["performances_ann"]]


def alc_percent(result: dict[str, Any], metric: str) -> Optional[float]:
    """100 times the ALC of ``metric``, or ``None`` if the curve has no defined value.

    The area covers the range of labeled people of the result, so a run that ends below 100 %
    labeled people has an ALC of at most that range (a perfect curve up to 20 % gives 20).

    A value of ``-1`` (the metric has no ground truth in its area range) makes the curve undefined,
    as in the 2023 summaries.
    """
    curve = metric_curve(result, metric)
    if any(v < 0.0 for v in curve):
        return None
    return 100.0 * compute_alc(result["percentages"], curve)
