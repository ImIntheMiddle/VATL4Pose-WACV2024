"""Training of the initial pose estimator, the starting weights of the active-learning loop.

The released weights of 2023 came from two scripts that are not part of this package. This module
retrains them. The recipe is that of the 2023 SimpleBaseline, FastPose and HRNet-W32 runs:

* Adam on all parameters at one learning rate (default 1e-3, no weight decay), and a
  :class:`~torch.optim.lr_scheduler.MultiStepLR` that multiplies the rate by 0.1 at the epochs in
  ``milestones`` (default 90 and 120). The scheduler steps once per epoch, after its batches.
* The loss is ``0.5 * MSE(output * weight, target * weight)`` (:func:`heatmap_loss`), the loss of
  the per-round fine-tuning.
* A training item is the crop of the ground-truth box of one person with a random scale, rotation
  and flip (:class:`~vatl4pose.data.datasets.PersonTrainDataset`), the augmentation of the
  fine-tuning. The frames are read from disk, with no cache.
* After an epoch the model is evaluated on the validation people. The best epoch by
  :attr:`InitialTrainConfig.select_metric` becomes ``best.pth``.

The validation uses the evaluation of the active-learning loop: the person crops of
:func:`~vatl4pose.data.geometry.crop_person` from the ground-truth boxes, the decoding of
:func:`~vatl4pose.data.geometry.heatmaps_to_coords`, the pose score ``mean + 1.25 * max`` of the
peak values, and :func:`~vatl4pose.eval.evaluate_predictions` with the OKS sigmas of the keypoint
spec. The in-training validation of the 2023 JRDB-Pose script reported an AP of 0.0 in all 84
logged epochs, so it could not select the model. The PoseTrack21 script compares a dict with an
int (``gt_AP > best_score``) and cannot have run as it stands.

Two parts of the 2023 scripts are not ported. After epoch 140 they restart the learning rate and
switch on a "DPG" augmentation. The restart ran; the augmentation read the image size before it was
assigned and raised an error, so the epoch after it failed. The scripts also multiplied the batch
size by the number of GPUs and trained with ``DataParallel``; this module trains on one device with
the batch size given.

Reproducibility: the augmentation of an item is drawn by the dataset from ``(seed, epoch, item)``,
the order of the items in an epoch comes from
:func:`~vatl4pose.train._common.shuffle_order`, and the initial weights of a network without a
checkpoint are drawn with a generator seeded from ``seed``. None of them reads or changes the
global random state, and none depends on the number of loader workers.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vatl4pose.data.annotations import VideoAnnotation
from vatl4pose.data.augment import AugmentConfig
from vatl4pose.data.datasets import (
    PersonCropDataset,
    PersonTrainDataset,
    collect_people,
)
from vatl4pose.data.geometry import heatmaps_to_coords
from vatl4pose.eval import (
    METRIC_KEYS,
    RoundMetrics,
    evaluate_predictions,
    pose_score,
    xyxy_to_xywh,
)
from vatl4pose.keypoints import KeypointSpec
from vatl4pose.models.estimator import EstimatorConfig, PoseEstimator, build_estimator
from vatl4pose.train._common import (
    FixedOrderSampler,
    derived_seed,
    shuffle_order,
    write_atomic,
)
from vatl4pose.train._files import prepare_output_dir
from vatl4pose.train.finetune import heatmap_loss

# Factor of the learning rate at every milestone (``LR_FACTOR`` of the 2023 configs).
LR_GAMMA = 0.1
# The three files of a run; a run does not start in a directory that holds one of them.
_OUTPUT_FILES = ("best.pth", "last.pth", "log.jsonl")


@dataclass(frozen=True)
class InitialTrainConfig:
    """Settings of the initial training of the estimator.

    Attributes:
        lr: Learning rate of Adam, the same for every parameter.
        milestones: Numbers of finished epochs at which the rate is multiplied by 0.1: with 90 the
            epoch with index 90, the 91st, is the first at the lower rate. Milestones beyond
            ``epochs`` are never reached and do no harm: the JRDB-Pose recipe of 2023 trained 100
            epochs with the milestones 90 and 120.
        epochs: Number of training epochs.
        batch_size: Items per training and per validation batch. The 2023 configs used 70 to 240
            per GPU, whatever the memory of the GPU allowed, and multiplied it by the GPU count.
        val_every: Validate after every ``val_every``-th epoch; the last epoch is always validated.
        select_metric: Key of :data:`vatl4pose.eval.METRIC_KEYS` whose largest value picks
            ``best.pth``; the first epoch with the largest value wins.
        init_weights: Checkpoint to start from, for example the PoseTrack21 weights before the
            training on JRDB-Pose. It is loaded like ``EstimatorConfig.weights`` and replaces it,
            so the estimator config must not set both.
        augment: Augmentation of the training crops.
        num_workers: Worker processes of the data loaders; 0 loads in the main process.
    """

    lr: float = 1e-3
    milestones: tuple[int, ...] = (90, 120)
    epochs: int = 200
    batch_size: int = 32
    val_every: int = 1
    select_metric: str = "AP"
    init_weights: pathlib.Path | None = None
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    num_workers: int = 4

    def __post_init__(self) -> None:
        if not self.lr > 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        milestones = self.milestones
        if (
            not isinstance(milestones, tuple)
            or not all(isinstance(m, int) and m >= 1 for m in milestones)
            or any(a >= b for a, b in zip(milestones, milestones[1:]))
        ):
            raise ValueError(
                f"milestones must be a tuple of strictly increasing ints >= 1, got {milestones!r}"
            )
        for name in ("epochs", "batch_size", "val_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if self.select_metric not in METRIC_KEYS:
            raise ValueError(
                f"select_metric must be one of {list(METRIC_KEYS)}, got {self.select_metric!r}"
            )
        if self.init_weights is not None and not isinstance(
            self.init_weights, pathlib.Path
        ):
            raise ValueError(
                f"init_weights must be a pathlib.Path or None, got {self.init_weights!r}"
            )
        if not isinstance(self.augment, AugmentConfig):
            raise ValueError(
                f"augment must be an AugmentConfig, got {type(self.augment).__name__}"
            )
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(
                f"num_workers must be an int >= 0, got {self.num_workers!r}"
            )


def ground_truth_arrays(
    videos: Sequence[VideoAnnotation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The labels of all people of ``videos`` in the form that the evaluation takes.

    Args:
        videos: Videos with the same keypoint spec and at least one person in total.

    Returns:
        ``(keypoints, boxes_xywh, image_id)`` with one row per person, video after video:
        ``(N, K, 3)`` float64 x, y and visibility; ``(N, 4)`` float64 boxes ``x, y, w, h`` with the
        width and height counting both end pixels, the box of the 2023 evaluation; ``(N,)``
        int64 image ids. Two frames of different videos get different ids, also when the files
        number their images alike.
    """
    _, people = collect_people(videos)
    ids: dict[tuple[int, int], int] = {}
    image_id = np.array(
        [
            ids.setdefault((v, person.image_id), len(ids))
            for v, video in enumerate(videos)
            for person in video.people
        ],
        dtype=np.int64,
    )
    keypoints = np.stack([person.keypoints for person in people])
    xyxy = np.stack([person.bbox_xyxy for person in people])
    boxes = xyxy_to_xywh(xyxy)
    assert keypoints.dtype == boxes.dtype == np.float64
    assert image_id.shape == (len(people),)
    return keypoints, boxes, image_id


@torch.no_grad()
def evaluate_estimator(
    model: PoseEstimator,
    videos: Sequence[VideoAnnotation],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> RoundMetrics:
    """AP and OSPA of ``model`` on all people of ``videos``, the evaluation of the loop.

    Every person is cropped from its ground-truth box, the heatmaps are decoded into image
    coordinates, and the pose score is ``mean + 1.25 * max`` of the peak values. The prediction
    carries the ground-truth box. The mode of ``model`` (training or evaluation) is restored.

    Args:
        model: The estimator, on ``device``.
        videos: Videos with the same keypoint spec and at least one person in total.
        batch_size: Crops per forward pass.
        num_workers: Worker processes that read and crop the frames; 0 reads in this process.
        device: Device of ``model``.

    Returns:
        The ten AP and AR values and the OSPA, computed with the OKS sigmas of the spec.

    Raises:
        ValueError: If the videos differ in their keypoint spec, hold no person, or have a
            number of joints other than that of ``model``.
    """
    spec, _ = collect_people(videos)
    if model.num_joints != spec.num_joints:
        raise ValueError(
            f"the model has {model.num_joints} joints, spec {spec.name!r} has {spec.num_joints}"
        )
    dataset = PersonCropDataset(videos)
    keypoints_gt, boxes_xywh, image_id = ground_truth_arrays(videos)
    n, k = len(dataset), spec.num_joints
    coords = np.empty((n, k, 2), dtype=np.float32)
    maxvals = np.empty((n, k), dtype=np.float32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        # the loader draws its worker seeds from this generator, not from the global one
        generator=torch.Generator().manual_seed(0),
    )
    was_training = model.training
    model.eval()
    try:
        for index, x, bbox_crop in loader:
            output = model(x.to(device, non_blocking=True))
            assert output.dtype == torch.float32, (
                f"the model returned {output.dtype} heatmaps"
            )
            decoded, scores = heatmaps_to_coords(output.cpu(), bbox_crop)
            coords[index.numpy()] = decoded.numpy()
            maxvals[index.numpy()] = scores.numpy()
    finally:
        model.train(was_training)
    return evaluate_predictions(
        coords, pose_score(maxvals), keypoints_gt, boxes_xywh, image_id, spec
    )


def _split(name: str, videos: Sequence[VideoAnnotation]) -> tuple[KeypointSpec, int]:
    """``(spec, number of people)`` of one split; errors name the split."""
    try:
        spec, people = collect_people(videos)
    except ValueError as exc:
        raise ValueError(f"{name} videos: {exc}") from exc
    return spec, len(people)


def _to_cpu(value: Any) -> Any:
    """Copy of a nested dict, list or tuple whose tensors are on the CPU and detached."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def _save(obj: Any, path: pathlib.Path) -> None:
    """``torch.save`` of ``obj`` through :func:`write_atomic`."""
    write_atomic(path, lambda handle: torch.save(obj, handle))


def _train_epoch(
    model: PoseEstimator,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n: int,
    epoch: int,
) -> float:
    """One pass over ``loader``; the mean loss weighted by batch size."""
    model.train()
    total = torch.zeros((), dtype=torch.float64, device=device)
    seen = 0
    for inputs, targets, weight in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        loss = heatmap_loss(model(inputs), targets, weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach().double() * inputs.shape[0]
        seen += inputs.shape[0]
    assert seen == n, (seen, n)
    mean = float(total) / n
    if not math.isfinite(mean):
        raise FloatingPointError(
            f"the loss is {mean} in epoch {epoch}; training has diverged"
        )
    return mean


def train_estimator(
    train_videos: Sequence[VideoAnnotation],
    val_videos: Sequence[VideoAnnotation],
    est_cfg: EstimatorConfig,
    cfg: InitialTrainConfig,
    out_dir: pathlib.Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train an estimator on ``train_videos`` and validate it on ``val_videos`` after every epoch.

    Writes three files to ``out_dir``:

    * ``log.jsonl``: one line per validation, with ``epoch`` (the number of finished epochs, from
      1), ``lr`` (the learning rate of that epoch), ``loss`` (the mean training loss of that
      epoch), the ten keys of :data:`vatl4pose.eval.METRIC_KEYS`, and ``OSPA``.
    * ``best.pth``: the state dict of the epoch with the largest ``cfg.select_metric``; it loads
      with ``EstimatorConfig(weights=...)``, so it is the start of the active-learning loop.
    * ``last.pth``: after every epoch, a dict with ``model``, ``optimizer``, ``scheduler``,
      ``epoch`` (finished epochs) and ``seed``, everything that continuing the run needs. Resuming
      from it is not implemented.

    Args:
        train_videos: Videos of the training split, same keypoint spec as ``val_videos``.
        val_videos: Videos of the validation split.
        est_cfg: The network to train. Its ``num_joints`` must equal that of the keypoint spec.
        cfg: Training settings.
        out_dir: Directory of the three files; created if missing, and it must not hold any of
            them already.
        device: Device of the training.
        seed: Seed of the run, non-negative.

    Returns:
        A summary: ``epochs``, ``best_epoch``, ``select_metric``, ``best_value``, ``last_loss``,
        ``n_train`` and ``n_val`` (numbers of people), and the paths ``best``, ``last`` and ``log``.

    Raises:
        ValueError: If ``seed`` is negative, a split has no person, the splits or the estimator
            do not fit one keypoint spec, or both ``est_cfg.weights`` and ``cfg.init_weights``
            are set.
        FileExistsError: If ``out_dir`` already holds one of the output files.
        FloatingPointError: If the loss of an epoch is not finite.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    train_spec, n_train = _split("train", train_videos)
    val_spec, n_val = _split("val", val_videos)
    if train_spec != val_spec:
        raise ValueError(
            "train and val videos must have the same keypoint spec, got "
            f"{train_spec.name!r} and {val_spec.name!r}"
        )
    if est_cfg.num_joints != train_spec.num_joints:
        raise ValueError(
            f"est_cfg.num_joints is {est_cfg.num_joints}, spec {train_spec.name!r} has "
            f"{train_spec.num_joints} joints"
        )
    if cfg.init_weights is not None:
        if est_cfg.weights is not None:
            raise ValueError(
                f"set only one of est_cfg.weights ({est_cfg.weights}) and "
                f"cfg.init_weights ({cfg.init_weights})"
            )
        est_cfg = dataclasses.replace(est_cfg, weights=cfg.init_weights)
    best_path, last_path, log_path = prepare_output_dir(out_dir, _OUTPUT_FILES)

    # the initial weights of a network without a checkpoint come from a generator of the seed;
    # the global CPU state is restored afterwards, and the CUDA state is not touched
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derived_seed("initial", "model", seed))
        model = build_estimator(est_cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.milestones), gamma=LR_GAMMA
    )

    dataset = PersonTrainDataset(train_videos, cfg.augment, seed)
    sampler = FixedOrderSampler(n_train)
    workers = cfg.num_workers
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        # the loader draws its worker seeds from this generator, not from the global one
        generator=torch.Generator().manual_seed(
            derived_seed("initial", "loader", seed)
        ),
    )

    best_value, best_epoch, loss = -math.inf, 0, math.nan
    for epoch in range(cfg.epochs):
        lr = optimizer.param_groups[0]["lr"]
        dataset.set_epoch(epoch)
        sampler.order = shuffle_order(n_train, seed, 0, epoch).tolist()
        loss = _train_epoch(model, loader, optimizer, device, n_train, epoch)
        scheduler.step()
        finished = epoch + 1
        if finished % cfg.val_every == 0 or finished == cfg.epochs:
            metrics = evaluate_estimator(
                model,
                val_videos,
                batch_size=cfg.batch_size,
                num_workers=workers,
                device=device,
            )
            value = float(metrics.ap[cfg.select_metric])
            line = {
                "epoch": finished,
                "lr": lr,
                "loss": loss,
                **{key: float(metrics.ap[key]) for key in METRIC_KEYS},
                "OSPA": float(metrics.ospa),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            if value > best_value:
                best_value, best_epoch = value, finished
                _save(_to_cpu(model.state_dict()), best_path)
        _save(
            {
                "model": _to_cpu(model.state_dict()),
                "optimizer": _to_cpu(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "epoch": finished,
                "seed": seed,
            },
            last_path,
        )
    return {
        "epochs": cfg.epochs,
        "best_epoch": best_epoch,
        "select_metric": cfg.select_metric,
        "best_value": best_value,
        "last_loss": loss,
        "n_train": n_train,
        "n_val": n_val,
        "best": str(best_path),
        "last": str(last_path),
        "log": str(log_path),
    }
