"""Pretraining of the whole-body autoencoder behind the WPU uncertainty.

The autoencoder reconstructs the hybrid feature of a person (:func:`compute_hybrid`). It trains
on the ground-truth poses of the training split, and the active-learning loop later tunes it on the
labeled people of one video. The recipe is that of the 2023 script:

* AdamW (PyTorch defaults, so a weight decay of 0.01) at ``lr``, lowered to the rates of
  ``lr_steps`` at their epochs (counted from 0); batches of ``batch_size`` rows in a new random
  order every epoch; loss ``MSE(reconstruction, feature)``.
* After every epoch the loss on the validation features is computed, in batches of 8000 rows. Both
  losses of the log are the mean of the batch means, as in the 2023 script.
* ``best.pth`` is the state dict of the epoch with the lowest validation loss. The run stops when
  the validation loss has not improved for ``patience`` epochs.
* The features are used as they are, with no scaling. The decoder ends in a sigmoid, so a feature
  below 0 or above 1 is reconstructed with an error that no training removes.

The features are the hybrid features of the ground truth, one row per person: the raw ``bbox``
of the annotation (height only), the keypoints with the visibility as the weight of every joint.
The fine-tuning in the loop builds its features the same way, from the labels of the oracle.
The 2023 script built them with one dictionary that it reused for every person, so all rows of its
training set were copies of the last person (an aliasing bug). Here every row is its own person.

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
from torch.nn import functional as F

from vatl4pose.data.annotations import VideoAnnotation
from vatl4pose.data.datasets import collect_people
from vatl4pose.keypoints import KeypointSpec
from vatl4pose.models.wholebody_ae import WholeBodyAE, compute_hybrid
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
            with this index, counted from 0, the learning rate becomes ``lr``. The 2023 script
            set 2e-4 for the epoch with index 12 and 5e-5 for the one with index 40.
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
) -> np.ndarray:
    """Hybrid features of the ground truth of all people of ``videos``, one row per person.

    Args:
        videos: Videos of the keypoint spec ``spec`` with at least one person in total.
        spec: Keypoint spec that defines the angles of the feature.

    Returns:
        ``(N, spec.ae_input_dim)`` float64 array, the people video after video. A person without a
        visible joint (a ``coco_custom`` box without keypoints) has no feature:
        :func:`~vatl4pose.data.datasets.collect_people` raises for it.

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
    features = compute_hybrid(
        np.stack([person.bbox_json_xywh for person in people]),
        np.stack([person.keypoints for person in people]),
        spec,
    )
    assert features.shape == (len(people), spec.ae_input_dim)
    assert features.dtype == np.float64
    return features


def _split_features(
    name: str, videos: Sequence[VideoAnnotation], spec: KeypointSpec
) -> torch.Tensor:
    """Features of one split as ``(N, input_dim)`` float32; errors name the split."""
    try:
        features = hybrid_features(videos, spec)
    except ValueError as exc:
        raise ValueError(f"{name} videos: {exc}") from exc
    return torch.from_numpy(features.astype(np.float32))


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
    * ``best.pth``: the state dict of the epoch with the lowest ``valid_loss``. It loads with
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
        ValueError: If ``seed`` is negative, a split has no person or another keypoint spec, or a
            feature cannot be computed.
        FileExistsError: If ``out_dir`` already holds one of the output files.
        FloatingPointError: If a loss is not finite.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    x_train = _split_features("train", train_videos, spec).to(device)
    x_valid = _split_features("val", val_videos, spec).to(device)
    best_path, log_path = prepare_output_dir(out_dir, _OUTPUT_FILES)
    n_train, n_valid = x_train.shape[0], x_valid.shape[0]

    # the global CPU state is restored afterwards, and the CUDA state is not touched
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derived_seed("ae", "model", seed))
        model = WholeBodyAE(spec.ae_input_dim, cfg.z_dim).to(device)
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
            batch = x_train[order[start : start + cfg.batch_size]]
            loss = F.mse_loss(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_loss = sum(batch_losses) / len(batch_losses)

        model.eval()
        with torch.no_grad():
            valid_losses = [
                float(F.mse_loss(model(batch), batch))
                for batch in x_valid.split(VALID_BATCH_SIZE)
            ]
        valid_loss = sum(valid_losses) / len(valid_losses)
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
