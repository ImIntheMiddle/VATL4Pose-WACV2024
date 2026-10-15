"""The numbers behind the explanation of every pick of a round.

For each person that a round selected, :func:`build_picks` gathers what the THC, WPU and DUW
figures show, from the arrays of ``details/round_XX.npz`` only: the heatmaps of the person and of
its neighbors in the previous and next frame with the joints whose heatmap changed most (THC),
the predicted and the reconstructed skeleton in crop pixels with the joints that hold the largest
parts of the WPU score (WPU), and the distance, uncertainty and value of the pick (DUW). :func:`tsne_map` writes the
embeddings of a round as the 2-D map of the DUW figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from ..data.geometry import HEATMAP_SIZE, INPUT_SIZE
from ..keypoints.spec import KeypointSpec
from .draw import shown_joints
from .load import RunInfo, frame_groups, neighbor_index

TOP_JOINTS = 3
"""Joints that the THC and the WPU figure single out."""

TSNE_SEED = 0
"""The seed of the t-SNE map, so that the same details give the same picture."""

_PERSON_BOX = 0.8
"""The crop box holds the person box times 1.25 (``box_to_center_scale``); this undoes it."""


@dataclass(frozen=True)
class Scene:
    """What a video holds that no round changes: the people, their frames and neighbors.

    Attributes:
        info: The run of the video.
        frames: ``(image path, samples)`` of every frame, in frame order.
        frame_of: ``(N,)`` position of each person's frame in ``frames``.
        track_id: ``(N,)`` track of each person.
        bbox_crop: ``(N, 4)`` float32 box of the crop of each person.
        prev: ``(N,)`` the same track in the previous frame, or ``-1``.
        next: ``(N,)`` the same track in the next frame, or ``-1``.
    """

    info: RunInfo
    frames: list[tuple[str, np.ndarray]]
    frame_of: np.ndarray
    track_id: np.ndarray
    bbox_crop: np.ndarray
    prev: np.ndarray
    next: np.ndarray


@dataclass(frozen=True)
class ThcFigure:
    """THC of one pick: the heatmaps of the joints that changed most over three frames.

    Attributes:
        people: Previous, this and next sample of the track; ``-1`` where the track has none.
        joints: ``(3,)`` joints with the largest difference, largest first.
        values: ``(K,)`` the difference of every joint (``thc_joint``).
        heat: ``(3, 3, 64, 48)`` uint8 heatmaps ``[frame][joint]``, the positive part in units of
            the largest value that the joint has in the three frames; 0 where a frame is missing.
            THC also counts the negative cells, which the picture leaves out.
    """

    people: tuple[int, int, int]
    joints: np.ndarray
    values: np.ndarray
    heat: np.ndarray


@dataclass(frozen=True)
class WpuFigure:
    """WPU of one pick: the predicted skeleton, the autoencoder's skeleton and their gap.

    Attributes:
        predicted: ``(K, 2)`` joints of the estimator in crop pixels (192 wide, 256 high).
        reconstructed: ``(K, 2)`` joints of the autoencoder in the same pixels.
        offsets: ``(K,)`` the part of the WPU score in the x and y offset of each joint.
        angles: ``(K,)`` the part of the WPU score in the angles at each joint; the skeleton does
            not show it. All parts of ``offsets`` and ``angles`` add up to the score.
        worst: ``(m,)`` the joints with the largest part, largest first (``m <= 3``).
    """

    predicted: np.ndarray
    reconstructed: np.ndarray
    offsets: np.ndarray
    angles: np.ndarray
    worst: np.ndarray


@dataclass(frozen=True)
class DuwFigure:
    """DUW of one pick: the numbers of its value ``(1 - m) * distance + lam * m * uncertainty``.

    ``distance`` and ``value`` are NaN for the pick that is made with nobody labeled or picked.
    """

    distance: float
    uncertainty: float
    value: float
    moks: float
    lam: float


@dataclass(frozen=True)
class Pick:
    """One selected person.

    Attributes:
        rank: 1 for the person that the strategy took first.
        person: The sample index.
        scores: ``{name: (value, percent of the candidates with a lower value)}`` of the scores
            that the selection computed (``thc``, ``wpu``, ..., ``total``); empty without them.
        thc: THC figure, or ``None`` if the strategy does not use THC.
        wpu: WPU figure, or ``None`` if the strategy does not use WPU.
        duw: DUW numbers, or ``None`` if the selector is not DUW.
    """

    rank: int
    person: int
    scores: dict[str, tuple[float, float]]
    thc: ThcFigure | None
    wpu: WpuFigure | None
    duw: DuwFigure | None


@dataclass(frozen=True)
class RoundData:
    """What one round shows.

    Attributes:
        index: The round.
        percentage: Share of the people labeled before the round selected, in percent.
        coords: ``(N, K, 2)`` joints that the model of the round predicts.
        shown: ``(N, K)`` bool, the joints confident enough to draw (:func:`.draw.shown_joints`).
        labeled: ``(N,)`` the people labeled before the selection.
        order: ``(k,)`` all people that the round selected, in the order of the selection; empty for
            a round without a selection.
        picks: The figures of the first picks (:func:`build_picks`).
        map_xy: ``(N, 2)`` t-SNE map of the embeddings, or ``None`` if the selection used none.
        moks: The mean OKS ``m`` that weighs the DUW value, or ``None`` without DUW.
        lam: The ``lam`` of the DUW value, or ``None`` without DUW.
    """

    index: int
    percentage: float
    coords: np.ndarray
    shown: np.ndarray
    labeled: np.ndarray
    order: np.ndarray
    picks: list[Pick]
    map_xy: np.ndarray | None
    moks: float | None
    lam: float | None

    @property
    def criteria(self) -> tuple[str, ...]:
        """The figures that the picks of this round have, from ``thc``, ``wpu`` and ``duw``."""
        if not self.picks:
            return ()
        first = self.picks[0]
        return tuple(
            name
            for name, figure in (
                ("thc", first.thc),
                ("wpu", first.wpu),
                ("duw", first.duw),
            )
            if figure is not None
        )


def build_round(
    scene: Scene, details: dict[str, np.ndarray], max_picks: int
) -> RoundData:
    """The :class:`RoundData` of a round; this computes its t-SNE map if the round has embeddings."""
    duw = "duw_value" in details
    return RoundData(
        index=int(details["round_index"]),
        percentage=float(details["percentage"]),
        coords=details["coords"],
        shown=shown_joints(details["maxvals"]),
        labeled=details["labeled"],
        order=details.get("pick_order", np.zeros(0, dtype=np.int64)),
        picks=build_picks(scene, details, max_picks),
        map_xy=tsne_map(details["embeddings"]) if "embeddings" in details else None,
        moks=float(details["duw_moks"]) if duw else None,
        lam=float(details["duw_lam"]) if duw else None,
    )


def build_scene(info: RunInfo, details: dict[str, np.ndarray]) -> Scene:
    """The :class:`Scene` of a video, from the details of any round."""
    frames = frame_groups(details["image_path"], details["frame_index"])
    n = details["frame_index"].shape[0]
    frame_of = np.full(n, -1, dtype=np.int64)
    for position, (_, rows) in enumerate(frames):
        frame_of[rows] = position
    assert (frame_of >= 0).all()
    track = np.asarray(details["track_id"])
    prev, nxt = neighbor_index(track, details["frame_index"])
    return Scene(
        info=info,
        frames=frames,
        frame_of=frame_of,
        track_id=track,
        bbox_crop=details["bbox_crop"].astype(np.float32),
        prev=prev,
        next=nxt,
    )


def person_boxes(bbox_crop: np.ndarray) -> np.ndarray:
    """``(N, 4)`` boxes of the people themselves, the crop boxes without their margin."""
    box = bbox_crop.astype(np.float64)
    center = (box[:, :2] + box[:, 2:]) / 2
    half = (box[:, 2:] - box[:, :2]) / 2 * _PERSON_BOX
    return np.concatenate([center - half, center + half], axis=1).astype(np.float32)


def build_picks(
    scene: Scene, details: dict[str, np.ndarray], max_picks: int
) -> list[Pick]:
    """The first ``max_picks`` picks of a round, in the order of the selection.

    Returns:
        An empty list for a round without a selection (the last round).
    """
    if "pick_order" not in details:
        return []
    order = details["pick_order"]
    pool = ~details["labeled"]
    rows = (
        {int(s): r for r, s in enumerate(details["heatmap_index"].tolist())}
        if ("heatmap_index" in details)
        else {}
    )
    picks = []
    for rank, person in enumerate(order[:max_picks].tolist(), start=1):
        scores = {}
        for key, values in details.items():
            if key == "total" or key.startswith("score/"):
                name = key.removeprefix("score/")
                scores[name] = (
                    float(values[person]),
                    float(100.0 * (values[pool] < values[person]).mean()),
                )
        picks.append(
            Pick(
                rank=rank,
                person=person,
                scores=scores,
                thc=_thc_figure(scene, details, rows, person)
                if "thc_joint" in details
                else None,
                wpu=_wpu_figure(scene, details, person)
                if "wpu_recon" in details
                else None,
                duw=_duw_figure(details, rank - 1) if "duw_value" in details else None,
            )
        )
    return picks


def _thc_figure(
    scene: Scene, details: dict[str, np.ndarray], rows: dict[int, int], person: int
) -> ThcFigure:
    values = details["thc_joint"][person].astype(np.float32)
    joints = np.argsort(values)[::-1][:TOP_JOINTS].astype(np.int64)
    people = (int(scene.prev[person]), person, int(scene.next[person]))
    for sample in people:
        assert sample < 0 or sample in rows, (
            f"the heatmaps of sample {sample}, a neighbor of pick {person}, were not recorded"
        )
    heat = np.zeros((3, joints.shape[0], *HEATMAP_SIZE), dtype=np.uint8)
    stored = details["heatmaps"]
    low, high = details["heatmap_low"], details["heatmap_high"]
    for t, joint in enumerate(joints.tolist()):
        peak = max(float(high[rows[s], joint]) for s in people if s >= 0)
        if peak <= 0:
            continue
        for c, sample in enumerate(people):
            if sample < 0:
                continue
            row = rows[sample]
            real = low[row, joint] + stored[row, joint].astype(np.float32) / 255.0 * (
                high[row, joint] - low[row, joint]
            )
            heat[c, t] = np.rint(np.clip(real / peak, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
    return ThcFigure(people=people, joints=joints, values=values, heat=heat)


def _wpu_figure(scene: Scene, details: dict[str, np.ndarray], person: int) -> WpuFigure:
    spec = scene.info.spec
    k = spec.num_joints
    box = scene.bbox_crop[person].astype(np.float64)
    height, width = INPUT_SIZE
    to_crop = np.array(
        [width / (box[2] - box[0]), height / (box[3] - box[1])], dtype=np.float64
    )

    def local(xy: np.ndarray) -> np.ndarray:
        return ((xy.astype(np.float64) - box[:2]) * to_crop).astype(np.float32)

    offsets, angles = wpu_contributions(details, person, spec)
    part = offsets + angles
    ranked = np.argsort(part, kind="stable")[::-1][:TOP_JOINTS]
    assert part.shape == (k,)
    return WpuFigure(
        predicted=local(details["coords"][person]),
        reconstructed=local(details["wpu_recon_coords"][person]),
        offsets=offsets.astype(np.float32),
        angles=angles.astype(np.float32),
        worst=ranked[part[ranked] > 0].astype(np.int64),
    )


def wpu_contributions(
    details: dict[str, np.ndarray], person: int, spec: KeypointSpec
) -> tuple[np.ndarray, np.ndarray]:
    """The WPU score of a person, split over its joints.

    The score is the mean squared gap between the standardised input and its reconstruction over
    the counted entries: those that WPU keeps (``spec.wpu_keep_index()``) and that hold a value
    (``wpu_valid``). Each counted entry adds its squared gap over the number of counted entries.

    Returns:
        ``(K,)`` float64 parts of the x and y offset of each joint and ``(K,)`` float64 parts of
        the angles at each joint, the middle joint ``b`` of a triplet ``(a, b, c)``. Together they
        add up to the score.
    """
    k = spec.num_joints
    gap = (
        details["wpu_recon"][person].astype(np.float64)
        - details["wpu_input"][person].astype(np.float64)
    ) ** 2
    counted = np.zeros(gap.shape[0], dtype=bool)
    keep = spec.wpu_keep_index()
    counted[keep] = details["wpu_valid"][person][keep]
    part = np.where(counted, gap, 0.0) / counted.sum()
    offsets = part[:k] + part[k : 2 * k]
    angles = np.zeros(k, dtype=np.float64)
    triplets = np.asarray(spec.angle_triplets, dtype=np.int64).reshape(-1, 3)
    np.add.at(angles, triplets[:, 1], part[2 * k :])
    return offsets, angles


def _duw_figure(details: dict[str, np.ndarray], position: int) -> DuwFigure:
    return DuwFigure(
        distance=float(details["duw_distance"][position]),
        uncertainty=float(details["duw_uncertainty"][position]),
        value=float(details["duw_value"][position]),
        moks=float(details["duw_moks"]),
        lam=float(details["duw_lam"]),
    )


def crop_samples(picks: list[Pick]) -> list[int]:
    """The people whose crops the figures of ``picks`` show, sorted."""
    samples: set[int] = set()
    for pick in picks:
        if pick.wpu is not None:
            samples.add(pick.person)
        if pick.thc is not None:
            samples.update(s for s in pick.thc.people if s >= 0)
    return sorted(samples)


def tsne_map(embeddings: np.ndarray) -> np.ndarray:
    """The embeddings of a round as a 2-D map, ``(N, 2)`` float32 in ``[0, 1]``.

    t-SNE with a fixed seed, started from the first two principal axes. The map is a projection:
    the distances in it only show which people lie close together in the embedding.

    Raises:
        ValueError: If there are fewer than 4 people, too few for a map.
    """
    n = embeddings.shape[0]
    if n < 4:
        raise ValueError(f"a map of the embeddings needs at least 4 people, got {n}")
    perplexity = float(min(30.0, (n - 1) / 3))
    # one OpenMP thread: with many threads on a busy machine the t-SNE of 100 people takes minutes,
    # and the thread count of the machine does not change the picture
    with threadpool_limits(limits=1, user_api="openmp"):
        xy = TSNE(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            random_state=TSNE_SEED,
        ).fit_transform(embeddings.astype(np.float32))
    low, high = xy.min(axis=0), xy.max(axis=0)
    return ((xy - low) / np.maximum(high - low, 1e-9)).astype(np.float32)
