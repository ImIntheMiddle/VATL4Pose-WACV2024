"""Pretraining of the whole-body autoencoder behind the WPU uncertainty.

The autoencoder reconstructs the standardised hybrid feature of a person (:func:`compute_hybrid`).
It trains on the ground-truth poses of the training split, and the active-learning loop later tunes
it on the labeled people of one video. The optimiser is that of the 2023 script:

* AdamW (PyTorch defaults, so a weight decay of 0.01) at ``lr``, lowered to the rates of
  ``lr_steps`` at their epochs (counted from 0); batches of ``batch_size`` rows in a new random
  order every epoch.
* The loss is the mean squared error over the *valid* entries of the batch
  (:func:`~vatl4pose.models.wholebody_ae.masked_mse`). An entry is invalid when its joint is not
  visible, or when it is an angle with an invisible joint (:func:`hybrid_valid_mask`).
* The features are standardised. The mean and the standard deviation of every dimension come from
  the valid entries of the training features (:func:`feature_statistics`) and are stored in the
  model. A dimension with no valid entry, or with a standard deviation of 0, makes the training
  raise. The only exceptions are the dimensions that the spec never makes valid (the offsets of its
  redundant joints and the angles that use them), which keep the mean 0 and the standard deviation 1.
* After every epoch the loss on the validation features is computed, in batches of 8000 rows, as the
  mean over all valid validation entries, so it does not depend on the batch size. The training loss
  of the log is the mean of the batch means.
* The run stops when the validation loss has not improved for ``patience`` epochs.

The features are the hybrid features of the ground truth, one row per person: the raw ``bbox``
of the annotation (height only), the keypoints with the visibility as the weight of every joint.
The fine-tuning in the loop builds its features and masks the same way, from the labels of the
oracle. Every row of the training set is its own person; the 2023 script reused one feature
dictionary, so its rows were copies of the last person.

Reproducibility: the order of the rows in an epoch comes from
:func:`~vatl4pose.train._common.shuffle_order`, and the initial weights are drawn with a generator
seeded from ``seed``. Neither reads or changes the global random state.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vatl4pose.data.annotations import VideoAnnotation
from vatl4pose.data.datasets import collect_people
from vatl4pose.keypoints import KeypointSpec
from vatl4pose.models.wholebody_ae import (
    WholeBodyAE,
    compute_hybrid,
    feature_dim_name,
    hybrid_valid_mask,
    masked_mse,
)
from vatl4pose.train._common import derived_seed, shuffle_order, write_atomic
from vatl4pose.train._files import prepare_output_dir

# Rows per batch of the validation loss; the 2023 script used 8000.
VALID_BATCH_SIZE = 8000
_OUTPUT_FILES = ("best.pth", "log.jsonl")


@dataclass(frozen=True)
class AETrainConfig:
    """Settings of the autoencoder pretraining.

    Attributes:
        z_dim: Size of the latent code. The released checkpoints of the paper use 4.
        lr: Learning rate of AdamW at the start.
        lr_steps: ``(epoch, lr)`` pairs with strictly increasing epochs. At the start of the epoch
            with this index, counted from 0, the learning rate becomes ``lr``.
        epochs: Maximal number of epochs.
        batch_size: Rows per training batch.
        patience: Number of epochs without a new lowest validation loss after which the run stops.
    """

    z_dim: int = 4
    lr: float = 1e-3
    lr_steps: tuple[tuple[int, float], ...] = ((12, 2e-4), (40, 5e-5))
    epochs: int = 80
    batch_size: int = 10000
    patience: int = 30

    def __post_init__(self) -> None:
        for name in ("z_dim", "epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if not self.lr > 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        steps = self.lr_steps
        if (
            not isinstance(steps, tuple)
            or not all(
                isinstance(step, tuple)
                and len(step) == 2
                and isinstance(step[0], int)
                and step[0] >= 1
                and step[1] > 0.0
                for step in steps
            )
            or any(a[0] >= b[0] for a, b in zip(steps, steps[1:]))
        ):
            raise ValueError(
                "lr_steps must be a tuple of (epoch, lr) pairs with strictly increasing "
                f"epochs >= 1 and lr > 0, got {steps!r}"
            )


def hybrid_features(
    videos: Sequence[VideoAnnotation], spec: KeypointSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid features of the ground truth of all people of ``videos``, with their validity masks.

    Args:
        videos: Videos of the keypoint spec ``spec`` with at least one person in total.
        spec: Keypoint spec that defines the angles of the feature.

    Returns:
        ``(features, valid)``: the ``(N, spec.ae_input_dim)`` float64 raw features, the people
        video after video, and the ``(N, spec.ae_input_dim)`` bool mask of the entries that hold a
        measured value (:func:`~vatl4pose.models.wholebody_ae.hybrid_valid_mask` of the label
        visibilities). A person without a visible joint (a ``coco_custom`` box without keypoints)
        has no feature: :func:`~vatl4pose.data.datasets.collect_people` raises for it.

    Raises:
        ValueError: If the videos hold no person, or their keypoint spec differs from ``spec``,
            or the feature of a person cannot be computed.
    """
    video_spec, people = collect_people(videos)
    if video_spec != spec:
        raise ValueError(
            "videos and spec must have the same keypoint spec, got "
            f"{video_spec.name!r} and {spec.name!r}"
        )
    keypoints = np.stack([person.keypoints for person in people])
    features = compute_hybrid(
        np.stack([person.bbox_json_xywh for person in people]), keypoints, spec
    )
    valid = hybrid_valid_mask(keypoints[..., 2], spec)
    assert features.shape == valid.shape == (len(people), spec.ae_input_dim)
    assert features.dtype == np.float64
    return features, valid


def feature_statistics(
    features: torch.Tensor, valid: torch.Tensor, spec: KeypointSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of every feature dimension over its valid entries.

    The standard deviation is the population one (divisor: the number of valid entries). A
    dimension that ``spec`` never makes valid (:func:`hybrid_valid_mask` of all-visible joints
    leaves it invalid) has no entries to measure and gets the mean 0 and the standard deviation 1.

    Args:
        features: ``(N, spec.ae_input_dim)`` raw features.
        valid: Boolean mask of the same shape.
        spec: Keypoint spec of the features.

    Returns:
        ``(mean, std)``, each ``float32`` of shape ``(spec.ae_input_dim,)``.

    Raises:
        ValueError: If another dimension has no valid entry or a standard deviation that is not
            positive and finite. The message names the dimension.
    """
    dim = spec.ae_input_dim
    assert features.is_floating_point() and features.shape == valid.shape, (
        features.shape,
        valid.shape,
    )
    assert valid.dtype == torch.bool and features.shape[1] == dim, (
        valid.dtype,
        features.shape,
    )
    never_valid = torch.from_numpy(
        ~hybrid_valid_mask(np.ones((1, spec.num_joints)), spec)[0]
    )
    x = features.double()
    weight = valid.double()
    count = weight.sum(dim=0)
    empty = torch.nonzero((count == 0) & ~never_valid).flatten()
    if empty.numel():
        d = int(empty[0])
        raise ValueError(
            f"dimension {d} ({feature_dim_name(d, spec)}) has no valid entry in the training "
            "features, so its mean and standard deviation are undefined"
        )
    safe_count = torch.where(count > 0, count, torch.ones_like(count))
    mean = (x * weight).sum(dim=0) / safe_count
    variance = (weight * (x - mean) ** 2).sum(dim=0) / safe_count
    std = variance.sqrt()
    mean = torch.where(never_valid, torch.zeros_like(mean), mean)
    std = torch.where(never_valid, torch.ones_like(std), std)
    bad = torch.nonzero(
        ~((std > 0) & torch.isfinite(std) & torch.isfinite(mean))
    ).flatten()
    if bad.numel():
        d = int(bad[0])
        raise ValueError(
            f"dimension {d} ({feature_dim_name(d, spec)}): the standard deviation over its "
            f"{int(count[d])} valid training entries is {float(std[d])} and the mean "
            f"{float(mean[d])}; standardising would divide by it"
        )
    return mean.float(), std.float()


def _split_features(
    name: str, videos: Sequence[VideoAnnotation], spec: KeypointSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Features of one split as ``(N, input_dim)`` float32 and bool; errors name the split."""
    try:
        features, valid = hybrid_features(videos, spec)
    except ValueError as exc:
        raise ValueError(f"{name} videos: {exc}") from exc
    if not valid.any():
        raise ValueError(f"{name} videos: no valid feature entry; no joint is visible")
    return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(valid)


def train_wholebody_ae(
    train_videos: Sequence[VideoAnnotation],
    val_videos: Sequence[VideoAnnotation],
    spec: KeypointSpec,
    cfg: AETrainConfig,
    out_dir: pathlib.Path,
    device: torch.device,
    seed: int,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train a whole-body autoencoder on the ground-truth poses of ``train_videos``.

    Writes two files to ``out_dir``:

    * ``log.jsonl``: one line per epoch, with ``epoch`` (the number of finished epochs, from 1),
      ``lr`` (the learning rate of that epoch), ``train_loss`` and ``valid_loss``.
    * ``best.pth``: the state dict of the epoch with the lowest ``valid_loss``, with the feature
      statistics of the training split in its buffers. It loads with
      :func:`~vatl4pose.models.wholebody_ae.load_wholebody_ae`.

    Args:
        train_videos: Videos of the training split, with keypoint spec ``spec``.
        val_videos: Videos of the validation split, with keypoint spec ``spec``.
        spec: Keypoint spec; its ``ae_input_dim`` is the input size of the network.
        cfg: Training settings.
        out_dir: Directory of the two files; created if missing, and it must not hold any of
            them already.
        device: Device of the training.
        seed: Seed of the run, non-negative.
        on_epoch: Called with the log line of an epoch right after it is written to ``log.jsonl``,
            for example to send it to W&B.

    Returns:
        A summary: ``epochs_run``, ``best_epoch`` (from 1), ``best_valid_loss``,
        ``stopped_early`` (the patience ran out), ``n_train`` and
        ``n_valid`` (numbers of people), and the paths ``best`` and ``log``.

    Raises:
        ValueError: If ``seed`` is negative, a split has no person or another keypoint spec, a
            feature cannot be computed, or a dimension of the training features has no valid entry
            or a standard deviation of 0 (see :func:`feature_statistics`).
        FileExistsError: If ``out_dir`` already holds one of the output files.
        FloatingPointError: If a loss is not finite.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    x_train, mask_train = _split_features("train", train_videos, spec)
    x_valid, mask_val = _split_features("val", val_videos, spec)
    mean, std = feature_statistics(x_train, mask_train, spec)
    best_path, log_path = prepare_output_dir(out_dir, _OUTPUT_FILES)
    n_train, n_valid = x_train.shape[0], x_valid.shape[0]

    # the global CPU state is restored afterwards, and the CUDA state is not touched
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derived_seed("ae", "model", seed))
        model = WholeBodyAE(spec.ae_input_dim, cfg.z_dim)
    model.set_normalization(mean, std)
    model = model.to(device)
    # standardised once; the statistics are buffers, so training does not change them
    z_train = model.standardize(x_train.to(device), mask_train.to(device))
    z_valid = model.standardize(x_valid.to(device), mask_val.to(device))
    mask_train, mask_val = mask_train.to(device), mask_val.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    rate_at = dict(cfg.lr_steps)

    best_loss, best_epoch, waited, epochs_run = math.inf, 0, 0, 0
    stopped_early = False
    for epoch in range(cfg.epochs):
        if epoch in rate_at:
            for group in optimizer.param_groups:
                group["lr"] = rate_at[epoch]
        lr = optimizer.param_groups[0]["lr"]

        model.train()
        order = shuffle_order(n_train, seed, 0, epoch).to(device)
        batch_losses = []
        for start in range(0, n_train, cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            batch = z_train[rows]
            loss = masked_mse(model(batch), batch, mask_train[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_loss = sum(batch_losses) / len(batch_losses)

        model.eval()
        with torch.no_grad():
            error_sum, error_count = 0.0, 0
            for start in range(0, n_valid, VALID_BATCH_SIZE):
                batch = z_valid[start : start + VALID_BATCH_SIZE]
                mask = mask_val[start : start + VALID_BATCH_SIZE]
                error_sum += float(((model(batch) - batch) ** 2)[mask].double().sum())
                error_count += int(mask.sum())
        valid_loss = error_sum / error_count
        if not (math.isfinite(train_loss) and math.isfinite(valid_loss)):
            raise FloatingPointError(
                f"the loss is {train_loss} (train) and {valid_loss} (valid) in epoch {epoch}; "
                "training has diverged"
            )

        epochs_run = epoch + 1
        line = {
            "epoch": epochs_run,
            "lr": lr,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        if on_epoch is not None:
            on_epoch(line)
        if valid_loss < best_loss:
            best_loss, best_epoch, waited = valid_loss, epochs_run, 0
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            write_atomic(best_path, lambda handle: torch.save(state, handle))
        else:
            waited += 1
            if waited >= cfg.patience:
                stopped_early = True
                break
    return {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "stopped_early": stopped_early,
        "n_train": n_train,
        "n_valid": n_valid,
        "best": str(best_path),
        "log": str(log_path),
    }
