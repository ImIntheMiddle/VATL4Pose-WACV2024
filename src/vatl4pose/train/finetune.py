"""Per-round fine-tuning of the pose estimator on the labeled people of one video.

After every query the estimator is trained again on the people the oracle has labeled. Two
schemes exist:

* continual (default): the model, the optimizer and the learning-rate schedule carry over from
  round to round, and a round trains ``int(alpha * (1 - moks_queried))`` epochs;
* non-continual: every round starts from the initial weights with a fresh optimizer and schedule,
  and trains ``int(base * n_labeled / n_total + alpha * (1 - moks_queried))`` epochs.

Here ``moks_queried`` is the mean OKS that the model reached on the people queried in this round
before they were labeled: when the model already solved them, little training is needed.

The training set of a round is the people labeled earlier whose OKS is at most
``finish_acc + 0.05`` plus the people queried now (:meth:`Finetuner.retrain_indices`). With the
default ``finish_acc = 1`` every labeled person is kept.

The loss is ``0.5 * MSE(output * weight, target * weight)`` (:func:`heatmap_loss`). The optimizer is
AdamW with the learning-rate groups of the estimator (head x10, backbone x1, upsampling x5), the
base learning rate decays by ``lr_gamma`` after every epoch. Reproducibility: the augmentation of
an item is drawn by the dataset from ``(seed, round, epoch, item)``, and the order of the items
in an epoch from :func:`shuffle_order`; neither depends on the global random state or on the number
of loader workers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vatl4pose.data.augment import AugmentConfig
from vatl4pose.data.datasets import FinetuneDataset
from vatl4pose.data.geometry import init_loader_worker
from vatl4pose.models.estimator import PoseEstimator
from vatl4pose.train._common import FixedOrderSampler, derived_seed, shuffle_order

# A labeled person is retrained while its OKS is at most ``finish_acc + FINISH_MARGIN``.
FINISH_MARGIN = 0.05
# SGD keeps the settings of the 2023 code: one group, momentum 0.9 and its own weight decay.
_SGD_MOMENTUM = 0.9
_SGD_WEIGHT_DECAY = 5e-4
_STATE_KEYS = ("model", "optimizer", "scheduler", "seed", "epochs_trained")


@dataclass(frozen=True)
class RetrainConfig:
    """Settings of the fine-tuning of every round.

    Attributes:
        optimizer: ``adamw`` uses the learning-rate groups of the estimator and ``weight_decay``.
            ``adam`` and ``sgd`` follow the 2023 code: one group at ``lr`` for all parameters;
            Adam without weight decay, SGD with momentum 0.9 and weight decay 5e-4. They ignore
            ``weight_decay``. Both datasets of the paper use ``adamw``.
        lr: Base learning rate.
        weight_decay: Weight decay of AdamW.
        lr_gamma: Factor of the exponential decay of the learning rate after every epoch.
        base: Epochs per labeled share of the video in the non-continual scheme.
        alpha: Epochs for the part of the queried people that the model has not solved yet.
        batch_size: Training batch size.
        continual: Keep the model, optimizer and schedule across rounds when true; start every
            round from the initial weights when false.
        finish_acc: OKS from which a labeled person counts as solved, in ``[0, 1]``. People
            labeled earlier are retrained only while their OKS is at most
            ``finish_acc + 0.05``.
        num_workers: Worker processes of the data loader; 0 loads in the main process.
        augment: Augmentation of the training crops.
    """

    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 2.5e-4
    weight_decay: float = 0.7
    lr_gamma: float = 0.99
    base: float = 25.0
    alpha: float = 250.0
    batch_size: int = 32
    continual: bool = True
    finish_acc: float = 1.0
    num_workers: int = 4
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def __post_init__(self) -> None:
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ValueError(
                f"optimizer must be 'adamw', 'adam' or 'sgd', got {self.optimizer!r}"
            )
        if not self.lr > 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not self.weight_decay >= 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 < self.lr_gamma <= 1.0:
            raise ValueError(f"lr_gamma must be in (0, 1], got {self.lr_gamma}")
        if not self.base >= 0.0:
            raise ValueError(f"base must be >= 0, got {self.base}")
        if not self.alpha >= 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be an int >= 1, got {self.batch_size!r}")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(
                f"num_workers must be an int >= 0, got {self.num_workers!r}"
            )
        if not 0.0 <= self.finish_acc <= 1.0:
            raise ValueError(f"finish_acc must be in [0, 1], got {self.finish_acc}")
        if not isinstance(self.augment, AugmentConfig):
            raise ValueError(
                f"augment must be an AugmentConfig, got {type(self.augment).__name__}"
            )


def heatmap_loss(
    output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``0.5 * MSE(output * weight, target * weight)``, the mean over all elements of the batch.

    Args:
        output: ``(B, K, H, W)`` predicted heatmaps.
        target: ``(B, K, H, W)`` target heatmaps.
        weight: ``(B, K, 1, 1)`` joint weights; 0 removes a joint from the loss, its elements
            still count in the denominator of the mean.

    Returns:
        A scalar tensor.
    """
    assert (
        output.is_floating_point() and output.dtype == target.dtype == weight.dtype
    ), (
        output.dtype,
        target.dtype,
        weight.dtype,
    )
    assert output.ndim == 4 and output.shape == target.shape, (
        tuple(output.shape),
        tuple(target.shape),
    )
    assert tuple(weight.shape) == (*output.shape[:2], 1, 1), tuple(weight.shape)
    return 0.5 * F.mse_loss(output * weight, target * weight)


class Finetuner:
    """Trains one estimator round after round on the labeled people.

    Attributes:
        model: The estimator; trained in place.
        optimizer: Current optimizer; replaced at every reset of the non-continual scheme.
        scheduler: Current exponential learning-rate schedule, stepped once per epoch.
        epochs_trained: Epochs trained by this finetuner so far, over all rounds and resets.
    """

    def __init__(
        self,
        estimator: PoseEstimator,
        initial_state: dict[str, torch.Tensor],
        cfg: RetrainConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        """
        Args:
            estimator: The network to fine-tune, already holding ``initial_state``.
            initial_state: State dict the non-continual scheme returns to at every round. It is
                copied, so later training of ``estimator`` does not change it.
            cfg: Settings of the fine-tuning.
            device: Device of the training; the estimator is moved to it.
            seed: Seed of the run, non-negative.

        Raises:
            ValueError: If ``seed`` is negative or ``initial_state`` does not fit the estimator.
        """
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        expected = estimator.state_dict()
        if set(initial_state) != set(expected) or any(
            initial_state[k].shape != expected[k].shape
            or initial_state[k].dtype != expected[k].dtype
            for k in expected
        ):
            raise ValueError(
                "initial_state does not fit the estimator: "
                f"{len(set(initial_state) - set(expected))} unexpected keys, "
                f"{len(set(expected) - set(initial_state))} missing keys, "
                "shapes and dtypes must agree"
            )
        self.model = estimator.to(device)
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.epochs_trained = 0
        self._initial_state = {
            k: v.detach().to("cpu", copy=True) for k, v in initial_state.items()
        }
        self.optimizer, self.scheduler = self._build_optimizer()

    def _build_optimizer(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ExponentialLR]:
        cfg = self.cfg
        optimizer: torch.optim.Optimizer
        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.param_groups(cfg.lr), weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                momentum=_SGD_MOMENTUM,
                weight_decay=_SGD_WEIGHT_DECAY,
            )
        return optimizer, torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.lr_gamma
        )

    def epochs_for_round(
        self, n_labeled: int, n_total: int, moks_queried: float
    ) -> int:
        """Number of epochs to train in this round.

        Args:
            n_labeled: People labeled after this round's query (queried ones included).
            n_total: People of the video.
            moks_queried: Mean OKS of this round's queried people before their labels were used.

        Returns:
            ``int(alpha * (1 - moks_queried))`` when continual, otherwise
            ``int(base * n_labeled / n_total + alpha * (1 - moks_queried))``; both truncate.
        """
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")
        if not 0 <= n_labeled <= n_total:
            raise ValueError(f"n_labeled must be in [0, {n_total}], got {n_labeled}")
        if not 0.0 <= moks_queried <= 1.0:
            raise ValueError(f"moks_queried must be in [0, 1], got {moks_queried}")
        cfg = self.cfg
        if cfg.continual:
            return int(cfg.alpha * (1 - moks_queried))
        return int(cfg.base * n_labeled / n_total + cfg.alpha * (1 - moks_queried))

    @staticmethod
    def retrain_indices(
        labeled_before: np.ndarray,
        oks_before: np.ndarray,
        queried: np.ndarray,
        finish_acc: float,
    ) -> np.ndarray:
        """Training set of a round: unsolved labeled people plus the people queried now.

        Args:
            labeled_before: ``(L,)`` integer sample indices labeled before this round.
            oks_before: ``(L,)`` float64 finite OKS of those people, row for row.
            queried: ``(Q,)`` integer sample indices queried in this round.
            finish_acc: OKS from which a person counts as solved, in ``[0, 1]``.

        Returns:
            Sorted unique ``(M,)`` int64 sample indices: the people of ``labeled_before`` with
            ``oks <= finish_acc + 0.05`` and all of ``queried``.
        """
        if labeled_before.ndim != 1 or not np.issubdtype(
            labeled_before.dtype, np.integer
        ):
            raise ValueError(
                f"labeled_before: expected a 1-D integer array, got "
                f"{labeled_before.dtype} {labeled_before.shape}"
            )
        if queried.ndim != 1 or not np.issubdtype(queried.dtype, np.integer):
            raise ValueError(
                f"queried: expected a 1-D integer array, got {queried.dtype} {queried.shape}"
            )
        if oks_before.shape != labeled_before.shape or oks_before.dtype != np.float64:
            raise ValueError(
                f"oks_before: expected a float64 array of shape {labeled_before.shape}, got "
                f"{oks_before.dtype} {oks_before.shape}"
            )
        if not np.isfinite(oks_before).all():
            raise ValueError("oks_before: every OKS must be finite")
        if not 0.0 <= finish_acc <= 1.0:
            raise ValueError(f"finish_acc must be in [0, 1], got {finish_acc}")
        kept = labeled_before[oks_before <= finish_acc + FINISH_MARGIN]
        return np.unique(np.concatenate([kept, queried]).astype(np.int64))

    def _reset(self) -> None:
        """Return to the initial weights with a fresh optimizer and schedule."""
        self.model.load_state_dict(self._initial_state, strict=True)
        self.optimizer, self.scheduler = self._build_optimizer()

    def run(self, dataset: FinetuneDataset, epochs: int) -> list[float]:
        """Train ``epochs`` epochs on ``dataset``.

        The round is the ``round_index`` of the dataset, so that the augmentation of the items
        and the order of the epochs belong to the same round. The non-continual scheme resets
        the model, optimizer and schedule first, also when ``epochs`` is 0. Every epoch sets the
        dataset to the epoch counter of the round (from 0), visits the items in
        :func:`shuffle_order`, and steps the schedule once, after the batches of the epoch.
        The training flag of the model is restored at the end.

        Args:
            dataset: Training items of the round; its ``seed`` must equal the seed of this
                finetuner.
            epochs: Number of epochs, 0 or more.

        Returns:
            The mean loss of every epoch, weighted by batch size; empty for 0 epochs.

        Raises:
            ValueError: If ``epochs`` is negative, the seed of ``dataset`` differs from the
                seed of this finetuner, or ``dataset`` is empty while ``epochs`` is positive.
            FloatingPointError: If the loss of an epoch is not finite.
        """
        if epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {epochs}")
        if dataset.seed != self.seed:
            raise ValueError(
                f"dataset was built with seed {dataset.seed}, this finetuner has seed "
                f"{self.seed}; the augmentation and the order would use different seeds"
            )
        round_index = dataset.round_index
        if epochs > 0 and len(dataset) == 0:
            raise ValueError("dataset is empty; there is nothing to train on")
        if not self.cfg.continual:
            self._reset()
        if epochs == 0:
            return []

        n = len(dataset)
        sampler = FixedOrderSampler(n)
        workers = self.cfg.num_workers
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=init_loader_worker,
            # the loader draws its worker seeds from this generator, not from the global one
            generator=torch.Generator().manual_seed(
                derived_seed("loader", self.seed, round_index)
            ),
        )
        history: list[float] = []
        was_training = self.model.training
        self.model.train()
        try:
            for epoch in range(epochs):
                dataset.set_epoch(epoch)
                sampler.order = shuffle_order(n, self.seed, round_index, epoch).tolist()
                history.append(self._train_epoch(loader, n, round_index, epoch))
                self.scheduler.step()
                self.epochs_trained += 1
        finally:
            self.model.train(was_training)
        return history

    def _train_epoch(
        self, loader: DataLoader[Any], n: int, round_index: int, epoch: int
    ) -> float:
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        seen = 0
        for inputs, targets, weight in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            weight = weight.to(self.device, non_blocking=True)
            loss = heatmap_loss(self.model(inputs), targets, weight)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.detach().double() * inputs.shape[0]
            seen += inputs.shape[0]
        assert seen == n, (seen, n)
        mean = float(total) / n
        if not math.isfinite(mean):
            raise FloatingPointError(
                f"the loss is {mean} in epoch {epoch} of round {round_index}; "
                "training has diverged"
            )
        return mean

    def state_dict(self) -> dict[str, Any]:
        """A snapshot of everything that continues training: model, optimizer, schedule, counters.

        The snapshot is a copy. It holds tensors, numbers, strings and plain containers only, so
        ``torch.save`` and ``torch.load(weights_only=True)`` handle it.
        """
        return copy.deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "seed": self.seed,
                "epochs_trained": self.epochs_trained,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot of :meth:`state_dict`; the next round then continues identically.

        ``state`` is not modified and not shared with the finetuner.

        Raises:
            ValueError: If a key is missing or the snapshot was taken with another seed.
        """
        missing = [k for k in _STATE_KEYS if k not in state]
        if missing:
            raise ValueError(f"state is missing the keys {missing}")
        if state["seed"] != self.seed:
            raise ValueError(
                f"state was saved with seed {state['seed']}, this finetuner has seed {self.seed}"
            )
        state = copy.deepcopy(state)
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epochs_trained = int(state["epochs_trained"])
