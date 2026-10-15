"""Structured configs of the commands and their conversion to the objects of the library.

The YAML tree in :mod:`vatl4pose.configs` is composed by Hydra on top of the schema :class:`Config`,
so a misspelled key or a value of the wrong type stops a command at start-up. A command reads only
the sections it needs (:func:`section`); a section that stays ``???`` (missing) matters only to the
commands that read it.

Paths. ``paths.data_root``, ``paths.weights_root`` and ``paths.output_root`` are the three places
the commands read and write, and no command depends on the working directory except through a
relative path the user typed. A dataset root is relative to ``paths.data_root``. Every weight file
(``estimator.weights``, ``ae.checkpoint``, ``train.init_weights``) is relative to
``paths.weights_root`` unless it is absolute; ``${abspath:...}`` makes a composed path absolute.

``lam``. The weight of the uncertainty term of the DUW selector is a top-level value, so that an
experiment can set it without touching the strategies that have no such term: the strategy files
with the ``coreset_duw`` selector read it as ``strategy.params.lam``.

Packaged id lists live in ``vatl4pose/data/splits``; ``${splits:<file>}`` gives the absolute path of
one.
"""

import dataclasses
import os
import pathlib
import socket
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, II, DictConfig, OmegaConf

from vatl4pose.data.augment import AugmentConfig
from vatl4pose.keypoints import KeypointSpec, get_spec, spec_from_mapping
from vatl4pose.loop import LoopConfig
from vatl4pose.models.estimator import EstimatorConfig
from vatl4pose.strategies import StrategyConfig
from vatl4pose.train import AETrainConfig, InitialTrainConfig, RetrainConfig

SCHEMA_NAME = "vatl4pose_schema"
"""Name of the schema in Hydra's config store; the defaults list of ``base.yaml`` names it."""

SPLITS_DIR = pathlib.Path(__file__).resolve().parent / "data" / "splits"
"""Directory of the packaged id lists."""

DATASETS = ("posetrack21", "jrdb_pose", "coco_custom")
"""The dataset kinds a dataset config can name."""


# ------------------------------------------------------------------ the schema


@dataclass
class PathsSection:
    """The three places the commands read and write; relative values are relative to the shell."""

    data_root: str = "data"
    weights_root: str = "pretrained_models"
    output_root: str = "outputs"


@dataclass
class RunSection:
    """What identifies one invocation.

    Attributes:
        name: Experiment name; the directory below ``paths.output_root`` that holds the runs.
            The experiment configs set it. Required by every command that writes runs.
        seed: Base seed; the seed of a video follows from it and the video id.
        device: ``cpu``, ``cuda`` or ``cuda:N``. Never chosen automatically; pin a GPU with
            ``CUDA_VISIBLE_DEVICES`` and ``run.device=cuda``.
        shard: ``i/n``; the process handles the videos ``sorted(ids)[i::n]``. Only ``prepare`` and
            ``atl`` take another value than ``0/1``.
    """

    name: Optional[str] = None
    seed: int = 166
    device: str = "cpu"
    shard: str = "0/1"


@dataclass
class DatasetSection:
    """The videos of one dataset.

    Attributes:
        name: ``posetrack21``, ``jrdb_pose`` or ``coco_custom`` (your own data, see
            :func:`vatl4pose.data.annotations.read_coco_custom_video` for the layout).
        root: Dataset directory, relative to ``paths.data_root`` unless absolute.
        split: Sub-directory of ``activelearning`` for ``posetrack21`` and ``jrdb_pose``; must be
            ``null`` for ``coco_custom``.
        ids_file: Text file with one video id per line (``#`` starts a comment line).
        ids: The video ids themselves; if set they replace ``ids_file``, so that
            ``dataset.ids=[000342]`` selects one video of an experiment. An id with leading zeros
            stays a string.
        keypoints: The name of a preset keypoint spec (``coco17``, ``posetrack21``,
            ``jrdb_pose``) or a mapping with the seven fields of
            :class:`~vatl4pose.keypoints.KeypointSpec`. Required for ``coco_custom``.
            ``posetrack21`` and ``jrdb_pose`` use their own preset when it is ``null``; set it to
            swap the keypoint spec (flip pairs, OKS sigmas, angle triplets) of the paper datasets,
            for example in an A/B run. The annotation files must have the joint count of the
            spec, and a cache is bound to the spec (build it again with the new spec, in another
            ``paths.output_root``).
        panorama: Whether the frames are 360-degree panoramas whose right edge continues at the
            left edge. The JRDB-Pose frames are, so ``jrdb_pose.yaml`` sets it to ``true``; set it
            for ``coco_custom`` to treat your own 360-degree video the same way (the readers place
            the joints of a person on the seam next to the box and the crops wrap at the seam, see
            :mod:`vatl4pose.data.annotations`). ``posetrack21`` has no panorama. A cache is bound
            to the value (build it again in another ``paths.output_root`` after a change).
    """

    name: str = MISSING
    root: str = MISSING
    split: Optional[str] = None
    ids_file: Optional[str] = None
    ids: Optional[list[str]] = None
    keypoints: Any = None
    panorama: bool = False


@dataclass
class EstimatorSection:
    """The pose estimator: ``simplebaseline``, ``fastpose`` or ``hrnet_w32``.

    ``weights`` has no default, so that a run never starts from random weights by mistake: give a
    file (relative to ``paths.weights_root``) or ``null`` (random initial weights, or the ImageNet
    backbone with ``imagenet_init=true``).
    """

    arch: str = MISSING
    weights: Optional[str] = MISSING
    imagenet_init: bool = False


@dataclass
class AESection:
    """The pretrained whole-body autoencoder of WPU, relative to ``paths.weights_root``."""

    checkpoint: Optional[str] = None


@dataclass
class StrategySection:
    """A query strategy: the names of its parts, see :class:`~vatl4pose.strategies.StrategyConfig`."""

    name: str = MISSING
    scorers: list[str] = field(default_factory=list)
    combiner: str = MISSING
    selector: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleSection:
    """The source of the labels: ``simulated`` (ground truth) or ``coco_file`` (an annotator)."""

    name: str = "simulated"


@dataclass
class AugmentSection:
    """Mirror of :class:`~vatl4pose.data.augment.AugmentConfig`."""

    scale_factor: float = 0.3
    rotation_factor: float = 40.0
    rotation_prob: float = 0.6
    flip: bool = True
    flip_prob: float = 0.5


@dataclass
class LoopSection:
    """Mirror of :class:`~vatl4pose.loop.LoopConfig`; ``seed`` follows ``run.seed``."""

    query_ratio: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    )
    infer_batch_size: int = 256
    save_final_model: bool = False
    save_estimator: bool = False
    stopping: bool = False
    save_round_details: bool = False
    seed: int = II("run.seed")
    ae_finetune_epochs: int = 20
    ae_finetune_lr: float = 8e-5
    ae_finetune_batch: int = 10
    ae_z_dim: int = 4
    vl4pose_epochs: int = 20
    vl4pose_lr: float = 1e-3
    vl4pose_batch_size: int = 64


@dataclass
class RetrainSection:
    """Mirror of :class:`~vatl4pose.train.RetrainConfig`, the retraining of every round."""

    optimizer: str = "adamw"
    lr: float = 2.5e-4
    weight_decay: float = 0.7
    lr_gamma: float = 0.99
    base: float = 25.0
    alpha: float = 250.0
    batch_size: int = 32
    continual: bool = True
    finish_acc: float = 1.0
    num_workers: int = 4
    augment: AugmentSection = field(default_factory=AugmentSection)


@dataclass
class LrStep:
    """One step of the autoencoder learning rate: from the epoch with this index (from 0) on."""

    epoch: int = MISSING
    lr: float = MISSING


@dataclass
class AETrainSection:
    """Mirror of :class:`~vatl4pose.train.AETrainConfig`."""

    z_dim: int = 4
    lr: float = 1e-3
    lr_steps: list[LrStep] = field(
        default_factory=lambda: [LrStep(12, 2e-4), LrStep(40, 5e-5)]
    )
    epochs: int = 80
    batch_size: int = 10000
    patience: int = 30


@dataclass
class TrainSection:
    """The initial training of the estimator or the autoencoder (``vatl4pose-train``).

    Attributes:
        target: ``estimator`` or ``ae``.
        train_split, train_ids_file: Videos of the training set (``activelearning/<split>`` and the
            id list). ``coco_custom`` has no split: give the id lists only.
        val_split, val_ids_file: The same for the validation set.
        lr, milestones, epochs, batch_size, val_every, select_metric, num_workers, augment:
            Mirror of :class:`~vatl4pose.train.InitialTrainConfig` (``target=estimator``).
        init_weights: Start from this checkpoint (relative to ``paths.weights_root``).
        ae: Mirror of :class:`~vatl4pose.train.AETrainConfig` (``target=ae``).
    """

    target: Optional[str] = None
    train_split: Optional[str] = None
    train_ids_file: Optional[str] = None
    val_split: Optional[str] = None
    val_ids_file: Optional[str] = None
    lr: float = 1e-3
    milestones: list[int] = field(default_factory=lambda: [90, 120])
    epochs: int = 200
    batch_size: int = 32
    val_every: int = 1
    select_metric: str = "AP"
    init_weights: Optional[str] = None
    num_workers: int = 4
    augment: AugmentSection = field(default_factory=AugmentSection)
    ae: AETrainSection = field(default_factory=AETrainSection)


@dataclass
class PrepareSection:
    """``vatl4pose-prepare`` and ``vatl4pose-prepare-annotations``.

    ``overwrite`` rebuilds a cache that does not match its video, or replaces an annotation file
    that differs from the one made from the official release. ``num_workers`` is the number of
    threads of the cache build, or of the image-header reads of the annotation preparation.
    """

    overwrite: bool = False
    num_workers: int = 8


@dataclass
class EvalSection:
    """``vatl4pose-eval``: ``output`` is the JSON file (default below ``paths.output_root``/eval).

    An existing output file is replaced only with ``overwrite=true``.
    """

    batch_size: int = 64
    num_workers: int = 4
    output: Optional[str] = None
    overwrite: bool = False


@dataclass
class TuneSection:
    """``vatl4pose-tune``: the weights ``lams`` tried on the validation videos.

    The videos are ``split`` and ``ids_file`` (the experiments set them to the validation
    videos). ``query_ratio`` replaces ``loop.query_ratio`` during the tuning (the tuning of the
    2023 code used a shorter schedule); ``metric`` is the key of ``performances_ann`` whose ALC is
    compared.
    """

    lams: list[float] = field(default_factory=list)
    metric: str = "AP .95"
    split: Optional[str] = None
    ids_file: Optional[str] = None
    query_ratio: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4, 1.0]
    )


@dataclass
class SummarizeSection:
    """``vatl4pose-summarize``: ``ids_file`` restricts the videos (default: those of the dataset)."""

    ids_file: Optional[str] = None


@dataclass
class WandbSection:
    """Settings of ``logger=wandb``; ``mode`` is ``online``, ``offline`` or ``disabled``."""

    project: str = "vatl4pose"
    entity: Optional[str] = None
    mode: str = "online"


@dataclass
class Config:
    """The configuration every command composes; see the module docstring.

    ``logger`` is ``json`` (``log.jsonl`` in the run directory of every video) or ``wandb``
    (the same, and the metrics and timings of every round to W&B).
    """

    paths: PathsSection = field(default_factory=PathsSection)
    run: RunSection = field(default_factory=RunSection)
    dataset: DatasetSection = field(default_factory=DatasetSection)
    estimator: EstimatorSection = field(default_factory=EstimatorSection)
    ae: AESection = field(default_factory=AESection)
    strategy: StrategySection = field(default_factory=StrategySection)
    oracle: OracleSection = field(default_factory=OracleSection)
    loop: LoopSection = field(default_factory=LoopSection)
    retrain: RetrainSection = field(default_factory=RetrainSection)
    lam: float = 0.01
    train: TrainSection = field(default_factory=TrainSection)
    prepare: PrepareSection = field(default_factory=PrepareSection)
    eval: EvalSection = field(default_factory=EvalSection)
    tune: TuneSection = field(default_factory=TuneSection)
    summarize: SummarizeSection = field(default_factory=SummarizeSection)
    logger: str = "json"
    wandb: WandbSection = field(default_factory=WandbSection)


# ------------------------------------------------------------------ registration


def split_file(name: str) -> str:
    """Absolute path of the packaged id list ``name``; raises ``FileNotFoundError`` if it is absent."""
    path = SPLITS_DIR / name
    if not path.is_file():
        known = sorted(p.name for p in SPLITS_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"no packaged id list {name!r} in {SPLITS_DIR}; available: {', '.join(known)}"
        )
    return str(path)


def _abspath(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


def register_configs() -> None:
    """Register the schema and the resolvers ``splits``, ``abspath``, ``hostname`` and ``pid`` (safe to repeat)."""
    ConfigStore.instance().store(name=SCHEMA_NAME, node=Config)
    OmegaConf.register_new_resolver("splits", split_file, replace=True)
    OmegaConf.register_new_resolver("abspath", _abspath, replace=True)
    OmegaConf.register_new_resolver(
        "hostname", socket.gethostname, use_cache=False, replace=True
    )
    OmegaConf.register_new_resolver("pid", os.getpid, use_cache=False, replace=True)


register_configs()


# ------------------------------------------------------------------ from a composed config to objects

T = TypeVar("T")


def section(cfg: DictConfig, name: str, kind: type[T]) -> T:
    """The section ``cfg[name]`` as an instance of the schema dataclass ``kind``.

    Interpolations are resolved. A value that is still ``???`` raises
    ``omegaconf.errors.MissingMandatoryValue`` that names it.
    """
    value = OmegaConf.to_object(cfg[name])
    if not isinstance(value, kind):
        raise TypeError(
            f"cfg.{name} is {type(value).__name__}, expected {kind.__name__}"
        )
    return value


def resolve_path(base: str, value: str) -> pathlib.Path:
    """``value`` as an absolute path; a relative one is taken relative to ``base``."""
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(base).expanduser() / path
    return pathlib.Path(os.path.abspath(path))


def keypoint_spec(dataset: DatasetSection) -> KeypointSpec:
    """The keypoint spec of a dataset config.

    ``dataset.keypoints`` names a preset (``coco17``, ``posetrack21``, ``jrdb_pose``) or holds the
    seven fields of a spec. ``posetrack21`` and ``jrdb_pose`` use their own preset when it is not
    set (the paper layouts); ``coco_custom`` has no default and needs it.
    """
    if dataset.name not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset.name!r}; expected one of {', '.join(DATASETS)}"
        )
    keypoints = dataset.keypoints
    if keypoints is None and dataset.name in ("posetrack21", "jrdb_pose"):
        return get_spec(dataset.name)
    if isinstance(keypoints, str):
        return get_spec(keypoints)
    if isinstance(keypoints, dict):
        return spec_from_mapping(keypoints)
    raise ValueError(
        "dataset.keypoints must name a preset (coco17, posetrack21, jrdb_pose) or hold the "
        f"seven fields of a keypoint spec, got {keypoints!r}"
    )


def _augment(section_: AugmentSection) -> AugmentConfig:
    return AugmentConfig(**dataclasses.asdict(section_))


def loop_config(section_: LoopSection) -> LoopConfig:
    """:class:`~vatl4pose.loop.LoopConfig` of a loop section (the list becomes a tuple)."""
    values = dataclasses.asdict(section_)
    values["query_ratio"] = tuple(values["query_ratio"])
    return LoopConfig(**values)


def retrain_config(section_: RetrainSection) -> RetrainConfig:
    """:class:`~vatl4pose.train.RetrainConfig` of a retrain section."""
    values = {f.name: getattr(section_, f.name) for f in dataclasses.fields(section_)}
    values["augment"] = _augment(section_.augment)
    return RetrainConfig(**values)


def initial_train_config(
    section_: TrainSection, weights_root: str
) -> InitialTrainConfig:
    """:class:`~vatl4pose.train.InitialTrainConfig` of a train section (``target=estimator``)."""
    init = (
        None
        if section_.init_weights is None
        else resolve_path(weights_root, section_.init_weights)
    )
    return InitialTrainConfig(
        lr=section_.lr,
        milestones=tuple(section_.milestones),
        epochs=section_.epochs,
        batch_size=section_.batch_size,
        val_every=section_.val_every,
        select_metric=section_.select_metric,
        init_weights=init,
        augment=_augment(section_.augment),
        num_workers=section_.num_workers,
    )


def ae_train_config(section_: AETrainSection) -> AETrainConfig:
    """:class:`~vatl4pose.train.AETrainConfig` of an autoencoder section."""
    return AETrainConfig(
        z_dim=section_.z_dim,
        lr=section_.lr,
        lr_steps=tuple((step.epoch, step.lr) for step in section_.lr_steps),
        epochs=section_.epochs,
        batch_size=section_.batch_size,
        patience=section_.patience,
    )


def strategy_config(section_: StrategySection) -> StrategyConfig:
    """:class:`~vatl4pose.strategies.StrategyConfig` of a strategy section."""
    return StrategyConfig(
        name=section_.name,
        scorers=tuple(section_.scorers),
        combiner=section_.combiner,
        selector=section_.selector,
        params=dict(section_.params),
    )


def estimator_config(
    section_: EstimatorSection, spec: KeypointSpec, weights_root: str
) -> EstimatorConfig:
    """:class:`~vatl4pose.models.estimator.EstimatorConfig` for the joints of ``spec``."""
    weights = (
        None
        if section_.weights is None
        else resolve_path(weights_root, section_.weights)
    )
    return EstimatorConfig(
        arch=section_.arch,  # type: ignore[arg-type]  # validated by build_estimator
        weights=weights,
        num_joints=spec.num_joints,
        imagenet_init=section_.imagenet_init,
    )
