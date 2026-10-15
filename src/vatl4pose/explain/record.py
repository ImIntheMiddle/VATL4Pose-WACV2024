"""Per-round details of the active-learning loop, written when ``loop.save_round_details`` is true.

Every round writes ``details/round_XX.npz`` into the run directory of the video. The file holds the
predictions of the round and, for a round that selects people, the scores, the pick order and the
inputs of the THC, WPU and DUW explanations. Building it reads the arrays of the round and draws no
random numbers, so the selection and the results are the same with and without it.

Keys (``N`` people, ``K`` joints, ``D`` autoencoder inputs, ``E`` embedding size, ``k`` picks,
``M`` stored heatmaps):

* always: ``schema``, ``round_index``, ``percentage``, ``image_path``, ``frame_index``,
  ``track_id``, ``ann_id``, ``bbox_crop`` ``(N, 4)``, ``coords`` ``(N, K, 2)``, ``maxvals``
  ``(N, K)``, ``labeled`` ``(N,)`` (before the selection of the round);
* a round that selects: ``pick_order`` ``(k,)`` in the order of the selection;
* a selection that is not a forced one: the stored heatmaps ``heatmap_index`` ``(M,)``,
  ``heatmaps`` ``(M, K, 64, 48)`` uint8, ``heatmap_low`` and ``heatmap_high`` ``(M, K)`` of the
  picks and their temporal neighbors (value = low + uint8 / 255 * (high - low), so negative values
  keep their sign), and ``embeddings`` ``(N, E)`` float32 if the selector uses them, stored
  exactly; a strategy with scorers adds ``score/<scorer>`` and ``total`` ``(N,)``;
* with THC: ``thc_joint`` ``(N, K)``, the per-joint heatmap distance summed like the THC score
  (its mean over the joints is the THC score);
* with WPU: ``wpu_input``, ``wpu_recon`` ``(N, D)`` standardised, ``wpu_valid`` ``(N, D)`` and
  ``wpu_recon_coords`` ``(N, K, 2)``, the reconstructed joints in image coordinates;
* with the DUW selector: ``duw_distance``, ``duw_uncertainty``, ``duw_value`` ``(k,)`` of every
  pick at the time of the pick, ``duw_moks`` and ``duw_lam``. A pick made with nothing labeled
  and nothing picked has no distance and takes the most uncertain person; its distance and
  value are NaN.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from ..data.cache import CacheArrays
from ..models.wholebody_ae import WholeBodyAE, compute_hybrid, hybrid_valid_mask
from ..strategies.base import RoundContext, Strategy
from ..strategies.vatl.wpu import crop_boxes_xywh
from ..train._common import write_atomic

SCHEMA = "vatl4pose.round_details/2"
DETAILS_DIR = "details"
_CHUNK = 128


def details_path(run_dir: pathlib.Path, round_index: int) -> pathlib.Path:
    """Where the details of round ``round_index`` of a video go."""
    return pathlib.Path(run_dir) / DETAILS_DIR / f"round_{round_index:02d}.npz"


def prediction_details(
    round_index: int,
    arrays: CacheArrays,
    coords: np.ndarray,
    maxvals: np.ndarray,
    labeled: np.ndarray,
) -> dict[str, np.ndarray]:
    """The keys that every round has: who is where and what the model of the round predicts."""
    n = labeled.shape[0]
    assert coords.ndim == 3 and coords.shape[0] == n and coords.shape[2] == 2, (
        coords.shape
    )
    assert maxvals.shape == coords.shape[:2], (maxvals.shape, coords.shape)
    return {
        "schema": np.array(SCHEMA),
        "round_index": np.array(round_index, dtype=np.int64),
        "percentage": np.array(100.0 * int(labeled.sum()) / n, dtype=np.float64),
        "image_path": np.asarray(arrays.image_path).astype(str),
        "frame_index": np.asarray(arrays.frame_index, dtype=np.int64),
        "track_id": np.asarray(arrays.track_id).astype(str),
        "ann_id": np.asarray(arrays.ann_id, dtype=np.int64),
        "bbox_crop": np.asarray(arrays.bbox_crop, dtype=np.float32),
        "coords": np.asarray(coords, dtype=np.float32),
        "maxvals": np.asarray(maxvals, dtype=np.float32),
        "labeled": labeled.astype(bool),
    }


def selection_details(
    strategy: Strategy,
    ctx: RoundContext,
    raw: Mapping[str, np.ndarray],
    order: np.ndarray,
    forced: bool,
) -> dict[str, np.ndarray]:
    """The keys of a round that selects: the pick order and what explains it.

    Args:
        strategy: The strategy that selected.
        ctx: The context the strategy received.
        raw: The raw scores it returned.
        order: The selected people in the order of the selection.
        forced: True if the query takes every unlabeled person, so that nothing was scored.
    """
    order = np.asarray(order, dtype=np.int64)
    out: dict[str, np.ndarray] = {"pick_order": order}
    if forced:
        return out
    scorers = strategy.config.scorers
    if raw:
        total = np.asarray(strategy.combiner.combine(raw, ctx), dtype=np.float64)
        out["total"] = total
        for name, score in raw.items():
            out[f"score/{name}"] = np.asarray(score, dtype=np.float64)
    if "thc" in scorers:
        out["thc_joint"] = thc_per_joint(ctx).astype(np.float32)
    if "wpu" in scorers:
        ae = ctx.extras.get("ae")
        assert isinstance(ae, WholeBodyAE), (
            "the WPU details need the autoencoder of the round"
        )
        out.update(wpu_details(ctx, ae))
    if strategy.config.selector == "coreset_duw":
        assert ctx.embeddings is not None and "total" in out
        out.update(
            duw_trace(ctx, out["total"], order, float(strategy.config.params["lam"]))
        )
    out.update(stored_heatmaps(ctx, order))
    if ctx.embeddings is not None:
        embeddings = ctx.embeddings.astype(np.float32)
        # the loop widens the float32 embeddings of the estimator, so float32 keeps them exactly
        assert np.array_equal(embeddings.astype(np.float64), ctx.embeddings)
        out["embeddings"] = embeddings
    return out


def write_round_details(
    run_dir: pathlib.Path, round_index: int, details: Mapping[str, np.ndarray]
) -> pathlib.Path:
    """Write ``details`` atomically to ``details/round_XX.npz`` of the run directory."""
    path = details_path(run_dir, round_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, lambda f: np.savez_compressed(f, **details))
    return path


def thc_per_joint(ctx: RoundContext) -> np.ndarray:
    """``(N, K)`` float64 per-joint L1 heatmap distance to the temporal neighbors.

    A joint's value is the sum of the absolute differences over its heatmap cells, added over the
    previous and the next neighbor; one neighbor counts twice and none gives 0, as in THC. The mean
    over the joints is the THC score.
    """
    n, k = ctx.num_samples, ctx.spec.num_joints
    total = np.zeros((n, k), dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for neighbor in (ctx.prev_index, ctx.next_index):
        samples = np.flatnonzero(neighbor >= 0)
        for start in range(0, samples.shape[0], _CHUNK):
            rows = samples[start : start + _CHUNK]
            other = neighbor[rows]
            diff = (ctx.heatmaps[rows] - ctx.heatmaps[other]).abs()
            total[rows] += diff.sum(dim=(2, 3), dtype=torch.float64).cpu().numpy()
        count[samples] += 1
    total[count == 1] *= 2.0
    return total


def wpu_details(ctx: RoundContext, ae: WholeBodyAE) -> dict[str, np.ndarray]:
    """Standardised input, reconstruction and reconstructed joints of every person.

    The hybrid feature starts with the joint offsets ``(x - c_x) / h`` and ``(y - c_y) / h``. The
    reconstructed joints undo the standardisation of these columns and map them back with the
    centre ``c`` and the box height ``h`` of the predicted pose.
    """
    spec = ctx.spec
    k = spec.num_joints
    keypoints = np.concatenate(
        [ctx.coords, ctx.maxvals[..., None]], axis=-1, dtype=np.float64
    )
    box = crop_boxes_xywh(ctx.bbox_crop)
    hybrid = compute_hybrid(box, keypoints, spec)
    valid = hybrid_valid_mask(ctx.maxvals, spec)
    assert hybrid.shape[1] == ae.input_dim, (hybrid.shape, ae.input_dim)
    device = next(ae.parameters()).device
    with torch.no_grad():
        x = ae.standardise(
            torch.tensor(hybrid.astype(np.float32), device=device),
            torch.tensor(valid, device=device),
        )
        recon = ae(x)
        raw = recon * ae.feature_std + ae.feature_mean
    raw_np = raw.cpu().numpy().astype(np.float64)
    weight = keypoints[..., 2]
    centre_x = (keypoints[..., 0] * weight).sum(axis=1) / weight.sum(axis=1)
    centre_y = (keypoints[..., 1] * weight).sum(axis=1) / weight.sum(axis=1)
    height = box[:, 3]
    recon_coords = np.stack(
        [
            centre_x[:, None] + raw_np[:, :k] * height[:, None],
            centre_y[:, None] + raw_np[:, k : 2 * k] * height[:, None],
        ],
        axis=-1,
    )
    return {
        "wpu_input": x.cpu().numpy().astype(np.float32),
        "wpu_recon": recon.cpu().numpy().astype(np.float32),
        "wpu_valid": valid,
        "wpu_recon_coords": recon_coords.astype(np.float32),
    }


def duw_trace(
    ctx: RoundContext, total: np.ndarray, order: np.ndarray, lam: float
) -> dict[str, np.ndarray]:
    """The distance, uncertainty and DUW value of every pick at the time of the pick.

    Replays the nearest-center distances of the greedy k-center selection in the given pick
    order: the distance of a person is the Euclidean distance of its embedding to the nearest
    labeled or already picked person, and the value is ``(1 - m) * distance + lam * m * total``.
    """
    assert ctx.embeddings is not None
    emb = ctx.embeddings
    m = float(ctx.moks_previous)
    min_distance: np.ndarray | None = None
    if ctx.labeled.any():
        min_distance = pairwise_distances(
            emb, emb[ctx.labeled], metric="euclidean"
        ).min(axis=1)
    k = order.shape[0]
    distance = np.full(k, np.nan, dtype=np.float64)
    value = np.full(k, np.nan, dtype=np.float64)
    for i, index in enumerate(order.tolist()):
        if min_distance is not None:
            distance[i] = min_distance[index]
            value[i] = (1 - m) * min_distance[index] + lam * m * total[index]
        step = pairwise_distances(emb, emb[[index]], metric="euclidean").reshape(-1)
        min_distance = step if min_distance is None else np.minimum(min_distance, step)
    return {
        "duw_distance": distance,
        "duw_uncertainty": total[order].astype(np.float64),
        "duw_value": value,
        "duw_moks": np.array(m, dtype=np.float64),
        "duw_lam": np.array(lam, dtype=np.float64),
    }


def stored_heatmaps(ctx: RoundContext, order: np.ndarray) -> dict[str, np.ndarray]:
    """8-bit heatmaps of the picks and their previous and next neighbors.

    Each joint map is stored between its own lowest and highest value, so the negative cells,
    which THC compares as well, keep their sign; a cell is off by at most half a step.
    """
    linked = np.concatenate([order, ctx.prev_index[order], ctx.next_index[order]])
    index = np.unique(linked[linked >= 0]).astype(np.int64)
    heatmaps = ctx.heatmaps[index].cpu().numpy().astype(np.float64)
    low = heatmaps.min(axis=(2, 3))
    high = heatmaps.max(axis=(2, 3))
    span = np.where(high > low, high - low, 1.0)[..., None, None]
    quantised = np.rint((heatmaps - low[..., None, None]) / span * 255.0).astype(np.uint8)
    return {
        "heatmap_index": index,
        "heatmaps": quantised,
        "heatmap_low": low.astype(np.float32),
        "heatmap_high": high.astype(np.float32),
    }
