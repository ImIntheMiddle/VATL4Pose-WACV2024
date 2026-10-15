"""Building blocks of an active-learning query strategy.

A query strategy answers one question per round: which ``k`` unlabeled people go to the
oracle? It is assembled from three kinds of component, each looked up by name in the registry
(:mod:`vatl4pose.strategies.registry`):

* a **scorer** gives every person one number (larger means query first), for example the
  summed peak probability (HP) or the Temporal Pose Continuity (TPC);
* a **combiner** turns the scores of all scorers into one total score per person;
* a **selector** selects the ``k`` people from the total score (and, for the filters, from the
  embeddings).

:class:`StrategyConfig` names the components and :class:`Strategy` runs them on a
:class:`RoundContext`, the snapshot of the pool at the start of a round.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from ..data.geometry import HEATMAP_SIZE
from ..keypoints import KeypointSpec


@dataclass
class RoundContext:
    """The pool of one active-learning round, as the strategies see it.

    All arrays cover every person of the video (``N`` samples), labeled ones included.
    Neighbors are linked by frame: ``prev_index[i]`` is the index of the person of the same track
    in the nearest earlier annotated frame and ``next_index[i]`` the index of the one in the
    nearest later frame; ``-1`` marks a missing neighbor.

    Attributes:
        round_index: Index of the round, 0 for the first query.
        spec: Keypoint spec of the dataset.
        heatmaps: ``(N, K, 64, 48)`` float32 heatmaps that the current pose estimator predicts
            from the crop of each person.
        coords: ``(N, K, 2)`` image coordinates decoded from ``heatmaps`` with ``bbox_crop``.
        maxvals: ``(N, K)`` peak values of ``heatmaps``.
        bbox_crop: ``(N, 4)`` crop boxes ``xmin, ymin, xmax, ymax``.
        prev_index: ``(N,)`` int64 index of the previous neighbor, or -1.
        next_index: ``(N,)`` int64 index of the next neighbor, or -1.
        labeled: ``(N,)`` bool, True for people who already have a label.
        embeddings: ``(N, D)`` backbone embeddings, ``None`` for estimators that have none.
        moks_previous: Mean OKS of the people queried in the previous round, 0.0 in round 0.
        rng: Source of randomness of the round. The strategies call ``rng.random(size)`` and
            ``rng.choice(array)`` only, so any object with these two methods works, for example a
            ``numpy.random.Generator``.
        extras: Objects the loop provides for strategies that need them, for example the
            whole-body autoencoder under ``"ae"``.
    """

    round_index: int
    spec: KeypointSpec
    heatmaps: torch.Tensor
    coords: np.ndarray
    maxvals: np.ndarray
    bbox_crop: np.ndarray
    prev_index: np.ndarray
    next_index: np.ndarray
    labeled: np.ndarray
    embeddings: np.ndarray | None
    moks_previous: float
    rng: np.random.Generator
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hm = self.heatmaps
        assert isinstance(hm, torch.Tensor), (
            f"heatmaps: expected a tensor, got {type(hm).__name__}"
        )
        assert hm.dtype == torch.float32, f"heatmaps: expected float32, got {hm.dtype}"
        k = self.spec.num_joints
        assert hm.ndim == 4 and tuple(hm.shape[1:]) == (k, *HEATMAP_SIZE), (
            f"heatmaps: expected shape (N, {k}, {HEATMAP_SIZE[0]}, {HEATMAP_SIZE[1]}) for "
            f"spec {self.spec.name!r}, got {tuple(hm.shape)}"
        )
        n = hm.shape[0]
        for name, array, shape in (
            ("coords", self.coords, (n, k, 2)),
            ("maxvals", self.maxvals, (n, k)),
            ("bbox_crop", self.bbox_crop, (n, 4)),
        ):
            assert isinstance(array, np.ndarray) and np.issubdtype(
                array.dtype, np.floating
            ), (
                f"{name}: expected a float array, got {getattr(array, 'dtype', type(array))}"
            )
            assert array.shape == shape, (
                f"{name}: expected shape {shape}, got {array.shape}"
            )
        for name, array in (
            ("prev_index", self.prev_index),
            ("next_index", self.next_index),
        ):
            assert isinstance(array, np.ndarray) and array.dtype == np.int64, (
                f"{name}: expected an int64 array, got {getattr(array, 'dtype', type(array))}"
            )
            assert array.shape == (n,), (
                f"{name}: expected shape ({n},), got {array.shape}"
            )
            assert array.min(initial=-1) >= -1 and array.max(initial=-1) < n, (
                f"{name}: entries must be -1 or a sample index below {n}"
            )
        assert (
            isinstance(self.labeled, np.ndarray)
            and self.labeled.dtype == np.bool_
            and self.labeled.shape == (n,)
        ), f"labeled: expected a bool array of shape ({n},)"
        if self.embeddings is not None:
            emb = self.embeddings
            assert (
                isinstance(emb, np.ndarray)
                and emb.ndim == 2
                and emb.shape[0] == n
                and np.issubdtype(emb.dtype, np.floating)
            ), f"embeddings: expected a float array of shape ({n}, D), got {emb.shape}"
        assert 0.0 <= float(self.moks_previous) <= 1.0, (
            f"moks_previous: expected a mean OKS in [0, 1], got {self.moks_previous}"
        )

    @property
    def num_samples(self) -> int:
        """Number of samples ``N`` of the pool."""
        return int(self.labeled.shape[0])

    @property
    def unlabeled_index(self) -> np.ndarray:
        """Ascending int64 indices of the unlabeled people."""
        return np.flatnonzero(~self.labeled).astype(np.int64)


class Scorer(Protocol):
    """Gives every person a number; larger means query first."""

    name: str

    def score(self, ctx: RoundContext) -> np.ndarray:
        """Return a float64 vector of shape ``(N,)``."""
        ...


class Combiner(Protocol):
    """Turns the scores of all scorers of a strategy into one total score."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        """Return a float64 vector of shape ``(N,)``; only the unlabeled entries are meaningful.

        ``scores`` maps the scorer names to their ``(N,)`` score vectors.
        """
        ...


class Selector(Protocol):
    """Selects the people to query.

    ``needs_embeddings`` tells the loop whether ``select`` reads ``ctx.embeddings``, so that the
    loop computes them. It is a class attribute, ``True`` for the selectors that cluster the
    embeddings or measure distances between them. A selector without the attribute counts as
    ``False``.
    """

    needs_embeddings: bool = False

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        """Return ``k`` distinct unlabeled sample indices as an int64 vector."""
        ...


@dataclass(frozen=True)
class StrategyConfig:
    """Names the components of a strategy.

    Attributes:
        name: Label of the strategy in outputs, e.g. ``"THC+WPU_Coresetfilter"``.
        scorers: Registered scorer names; empty for pure filters.
        combiner: Registered combiner name (``"single"``, ``"minmax_sum"``, ``"none"``).
        selector: Registered selector name (``"topk"``, ``"random"``, ``"kmeans"``, ...).
        params: Constructor arguments. A key that names one of ``scorers`` holds a mapping with
            the arguments of that scorer. Every other key is an argument of the selector. The
            combiner takes none.
    """

    name: str
    scorers: tuple[str, ...]
    combiner: str
    selector: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("StrategyConfig.name: must be a non-empty string")
        if isinstance(self.scorers, str):
            raise ValueError(
                f"StrategyConfig {self.name!r}: scorers must be a sequence of names, "
                f"got the string {self.scorers!r}"
            )
        scorers = tuple(self.scorers)
        for scorer in scorers:
            if not isinstance(scorer, str) or not scorer:
                raise ValueError(
                    f"StrategyConfig {self.name!r}: scorer names must be non-empty strings, "
                    f"got {scorer!r}"
                )
        for scorer in set(scorers):
            if scorers.count(scorer) > 1:
                raise ValueError(
                    f"StrategyConfig {self.name!r}: scorer {scorer!r} is listed twice"
                )
        for field_name in ("combiner", "selector"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"StrategyConfig {self.name!r}: {field_name} must be a non-empty string"
                )
        if not isinstance(self.params, Mapping):
            raise ValueError(
                f"StrategyConfig {self.name!r}: params must be a mapping, "
                f"got {type(self.params).__name__}"
            )
        # The dataclass is frozen: store the canonical forms with object.__setattr__.
        object.__setattr__(self, "scorers", scorers)
        object.__setattr__(self, "params", dict(self.params))


def unlabeled_for_query(ctx: RoundContext, k: int) -> np.ndarray:
    """Return the unlabeled indices after checking that ``k`` people can be queried.

    Raises:
        ValueError: If ``k`` is below 1 or above the number of unlabeled people.
    """
    unlabeled = ctx.unlabeled_index
    if not 1 <= k <= unlabeled.shape[0]:
        raise ValueError(
            f"cannot query k={k} people: k must be between 1 and the number of unlabeled "
            f"people, {unlabeled.shape[0]}"
        )
    return unlabeled


class Strategy:
    """A query strategy: scorers, a combiner and a selector run in that order."""

    def __init__(
        self,
        config: StrategyConfig,
        scorers: list[Scorer],
        combiner: Combiner,
        selector: Selector,
    ) -> None:
        self.config = config
        self.scorers = list(scorers)
        self.combiner = combiner
        self.selector = selector

    @property
    def needs_embeddings(self) -> bool:
        """Whether the selector reads ``ctx.embeddings`` (``Selector.needs_embeddings``)."""
        return bool(getattr(self.selector, "needs_embeddings", False))

    def query(
        self, ctx: RoundContext, k: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select ``k`` unlabeled people to query.

        When ``k`` is at least the number of unlabeled people, every choice is forced: the
        method returns all unlabeled people in index order, calls neither the scorers nor the
        selector, and the score dictionary is empty.

        Returns:
            ``(selected, raw)``: the selected sample indices (int64, shape ``(k,)``, in the
            selector's order) and the raw ``(N,)`` score of every scorer, by scorer name, for
            logging.

        Raises:
            ValueError: If ``k`` is below 1, or if a component returns an invalid selection.
        """
        if k < 1:
            raise ValueError(f"cannot query k={k} people: k must be at least 1")
        unlabeled = ctx.unlabeled_index
        if k >= unlabeled.shape[0]:
            return unlabeled, {}
        n = ctx.num_samples
        raw: dict[str, np.ndarray] = {}
        for scorer in self.scorers:
            score = np.asarray(scorer.score(ctx))
            assert score.dtype == np.float64 and score.shape == (n,), (
                f"scorer {scorer.name!r}: expected a float64 vector of shape ({n},), "
                f"got {score.dtype} {score.shape}"
            )
            raw[scorer.name] = score
        total = np.asarray(self.combiner.combine(raw, ctx))
        assert total.dtype == np.float64 and total.shape == (n,), (
            f"combiner {self.config.combiner!r}: expected a float64 vector of shape ({n},), "
            f"got {total.dtype} {total.shape}"
        )
        selected = np.asarray(self.selector.select(total, ctx, k))
        self._check_selection(selected, ctx, k)
        return selected.astype(np.int64, copy=False), raw

    def _check_selection(self, selected: np.ndarray, ctx: RoundContext, k: int) -> None:
        who = f"selector {self.config.selector!r}"
        if selected.shape != (k,) or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError(
                f"{who} returned an array of shape {selected.shape} and dtype "
                f"{selected.dtype}; expected {k} integer indices"
            )
        if selected.min() < 0 or selected.max() >= ctx.num_samples:
            raise ValueError(f"{who} returned an index outside [0, {ctx.num_samples})")
        if np.unique(selected).shape[0] != k:
            raise ValueError(f"{who} returned duplicate indices")
        if ctx.labeled[selected].any():
            raise ValueError(f"{who} returned an already labeled person")
