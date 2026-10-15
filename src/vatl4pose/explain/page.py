"""``explain/index.html`` and ``explain/data.js``: the page for looking at one video.

The page is one HTML file that runs from ``file://`` and loads nothing from the network. Its data
is a script, ``data.js``, that sets ``window.VATL_EXPLAIN``: a script loads from a local folder where
``fetch`` is blocked. The frames of the video are JPEG files of ``frames/``, the person crops that the
figures show are JPEG files of ``crops/``; the browser draws the skeletons, boxes and figures
itself from the numbers of ``data.js``.
"""

from __future__ import annotations

import base64
import json
import pathlib
from importlib import resources

import numpy as np

from .draw import CENTER, GAP, LEFT, PICK, RECON, RIGHT, THC_JOINTS, limb_sides
from .figures import (
    DuwFigure,
    Pick,
    RoundData,
    Scene,
    ThcFigure,
    WpuFigure,
    person_boxes,
)

DATA_SCHEMA = "vatl4pose.explain/1"
INDEX_NAME = "index.html"
DATA_NAME = "data.js"
FRAMES_DIR = "frames"
CROPS_DIR = "crops"


def frame_name(position: int) -> str:
    """The path of frame ``position`` of the video below ``explain/``."""
    return f"{FRAMES_DIR}/frame_{position:04d}.jpg"


def crop_name(sample: int) -> str:
    """The path of the crop of person ``sample`` below ``explain/``."""
    return f"{CROPS_DIR}/person_{sample:06d}.jpg"


def _css(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _number(value: float, digits: int = 4) -> float | None:
    """A float for JSON; NaN, which JSON has no number for, becomes ``null``."""
    return None if np.isnan(value) else round(float(value), digits)


def _flat(xy: np.ndarray) -> list[float]:
    return np.round(xy.astype(np.float64).reshape(-1), 1).tolist()


def _thc(figure: ThcFigure) -> dict:
    return {
        "crops": [crop_name(s) if s >= 0 else None for s in figure.people],
        "joints": figure.joints.tolist(),
        "values": np.round(figure.values.astype(np.float64), 3).tolist(),
        "heat": [
            [base64.b64encode(h.tobytes()).decode("ascii") for h in frame]
            if sample >= 0
            else None
            for frame, sample in zip(figure.heat, figure.people, strict=True)
        ],
    }


def _wpu(figure: WpuFigure, person: int) -> dict:
    return {
        "crop": crop_name(person),
        "predicted": _flat(figure.predicted),
        "reconstructed": _flat(figure.reconstructed),
        "worst": figure.worst.tolist(),
        "offsets": np.round(figure.offsets.astype(np.float64), 5).tolist(),
        "angles": np.round(figure.angles.astype(np.float64), 5).tolist(),
    }


def _duw(figure: DuwFigure) -> dict:
    return {
        "distance": _number(figure.distance),
        "uncertainty": _number(figure.uncertainty),
        "value": _number(figure.value),
    }


def _pick(pick: Pick) -> dict:
    return {
        "rank": pick.rank,
        "person": pick.person,
        "scores": {
            name: [round(value, 5), round(percent, 2)]
            for name, (value, percent) in pick.scores.items()
        },
        "thc": None if pick.thc is None else _thc(pick.thc),
        "wpu": None if pick.wpu is None else _wpu(pick.wpu, pick.person),
        "duw": None if pick.duw is None else _duw(pick.duw),
    }


def _round(rd: RoundData) -> dict:
    n = rd.labeled.shape[0]
    rank = np.zeros(n, dtype=np.int64)
    rank[rd.order] = np.arange(1, rd.order.shape[0] + 1)
    return {
        "round": rd.index,
        "percentage": round(rd.percentage, 3),
        "labeled": rd.labeled.astype(np.int64).tolist(),
        "rank": rank.tolist(),
        "order": rd.order.tolist(),
        "coords": [_flat(c) for c in rd.coords],
        "shown": rd.shown.astype(np.int64).tolist(),
        "map": None
        if rd.map_xy is None
        else np.round(rd.map_xy.astype(np.float64), 4).tolist(),
        "duw": None
        if rd.moks is None
        else {"m": _number(rd.moks), "lam": _number(rd.lam)},
        "criteria": list(rd.criteria),
        "picks": [_pick(p) for p in rd.picks],
    }


def build_data(
    scene: Scene,
    rounds: list[RoundData],
    frame_sizes: list[tuple[int, int]],
    max_picks: int,
    videos: dict[str, dict[str, str]],
) -> dict:
    """The content of ``data.js``.

    Args:
        scene: The video.
        rounds: The rounds to show, in order.
        frame_sizes: ``(width, height)`` of every frame of ``scene.frames``.
        max_picks: The number of picks per round that carry a figure.
        videos: ``{"round": {round: file}, "compare": {round: file}}`` of the videos written.
    """
    info, spec = scene.info, scene.info.spec
    n = scene.frame_of.shape[0]
    boxes = person_boxes(scene.bbox_crop)
    return {
        "schema": DATA_SCHEMA,
        "video": {
            "id": info.video_id,
            "dataset": info.dataset,
            "strategy": info.strategy,
            "scorers": list(info.scorers),
            "estimator": info.estimator,
            "people": n,
            "panorama": info.panorama,
        },
        "palette": {
            "side": [_css(LEFT), _css(RIGHT), _css(CENTER)],
            "pick": _css(PICK),
            "labeled": "rgb(0,169,157)",
            "recon": _css(RECON),
            "gap": _css(GAP),
            "thc": [_css(c) for c in THC_JOINTS],
        },
        "skeleton": {
            "names": list(spec.joint_names),
            "limbs": [list(limb) for limb in limb_sides(spec)],
            "hidden": list(spec.redundant_joints),
        },
        "maxPicks": max_picks,
        "videos": videos,
        "people": {
            "frame": scene.frame_of.tolist(),
            "box": np.round(boxes.astype(np.float64), 1).tolist(),
            "track": [str(t) for t in scene.track_id],
        },
        "frames": [
            {
                "file": frame_name(position),
                "width": width,
                "height": height,
                "people": rows.tolist(),
            }
            for position, ((_, rows), (width, height)) in enumerate(
                zip(scene.frames, frame_sizes, strict=True)
            )
        ],
        "rounds": [_round(rd) for rd in rounds],
    }


def write_page(out_dir: pathlib.Path, data: dict) -> None:
    """Write ``index.html`` and ``data.js`` into ``out_dir``."""
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    (out_dir / DATA_NAME).write_text(
        f"window.VATL_EXPLAIN = {text};\n", encoding="utf-8"
    )
    html = resources.files(__package__).joinpath(INDEX_NAME).read_text(encoding="utf-8")
    (out_dir / INDEX_NAME).write_text(html, encoding="utf-8")
