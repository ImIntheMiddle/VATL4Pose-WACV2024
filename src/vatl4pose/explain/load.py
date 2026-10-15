"""Read the run directory of a video for the explanations.

The explanations need three files of the directory that ``vatl4pose-atl`` wrote for a video:
``settings.json`` (what the run was: keypoint layout, panorama, strategy), the ``details/round_XX.npz``
files of :mod:`vatl4pose.explain.record`, and the images of the video, whose absolute paths the
details hold.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass

import numpy as np

from ..cli._settings import SETTINGS_NAME
from ..keypoints import KeypointSpec, get_spec
from .record import DETAILS_DIR, SCHEMA

_ROUND_FILE = re.compile(r"round_(\d+)\.npz")

# keys of a round that every details file has (see record.prediction_details)
_ALWAYS = (
    "image_path",
    "frame_index",
    "track_id",
    "ann_id",
    "bbox_crop",
    "coords",
    "maxvals",
    "labeled",
)


@dataclass(frozen=True)
class RunInfo:
    """What the explanations need to know about the run of a video.

    Attributes:
        run_dir: The directory of the video's run.
        video_id: The video.
        dataset: The dataset name of the run.
        estimator: The estimator architecture.
        strategy: The name of the strategy.
        scorers: The scorers of the strategy (``thc``, ``wpu``, ...), in the order of its settings.
        selector: The selector of the strategy.
        spec: The keypoint layout that the skeleton is drawn with.
        panorama: Whether the frames are 360-degree strips.
    """

    run_dir: pathlib.Path
    video_id: str
    dataset: str
    estimator: str
    strategy: str
    scorers: tuple[str, ...]
    selector: str
    spec: KeypointSpec
    panorama: bool


def read_run_info(run_dir: pathlib.Path) -> RunInfo:
    """The :class:`RunInfo` of the run in ``run_dir``.

    Raises:
        FileNotFoundError: If ``run_dir`` has no ``settings.json``, so it is no run directory.
        ValueError: If the run used a keypoint layout of its own, which the settings do not hold
            (only its name), so that the skeleton cannot be drawn.
    """
    path = run_dir / SETTINGS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; give the run directory of one video, the one that holds "
            f"{SETTINGS_NAME} and {DETAILS_DIR}/"
        )
    settings = json.loads(path.read_text(encoding="utf-8"))
    name = settings["keypoints"]["name"]
    try:
        spec = get_spec(name)
    except ValueError as exc:
        raise ValueError(
            f"{path}: the run used the keypoint layout {name!r}, which is not a preset; the "
            "skeleton can be drawn for coco17, posetrack21 and jrdb_pose only"
        ) from exc
    strategy = settings["strategy"]
    return RunInfo(
        run_dir=run_dir,
        video_id=str(settings["video_id"]),
        dataset=str(settings["dataset"]),
        estimator=str(settings["estimator"]["arch"]),
        strategy=str(strategy["name"]),
        scorers=tuple(strategy["scorers"]),
        selector=str(strategy["selector"]),
        spec=spec,
        panorama=bool(settings["panorama"]),
    )


def recorded_rounds(run_dir: pathlib.Path) -> list[int]:
    """The rounds that have a details file, in ascending order.

    Raises:
        FileNotFoundError: If the directory has no ``details/`` folder or no file in it; the run
            was made without ``loop.save_round_details=true``.
    """
    folder = run_dir / DETAILS_DIR
    rounds = sorted(
        int(m.group(1))
        for p in (folder.glob("round_*.npz") if folder.is_dir() else [])
        if (m := _ROUND_FILE.fullmatch(p.name))
    )
    if not rounds:
        raise FileNotFoundError(
            f"{folder} holds no round_XX.npz; run vatl4pose-atl with loop.save_round_details=true"
        )
    return rounds


def load_round(run_dir: pathlib.Path, round_index: int) -> dict[str, np.ndarray]:
    """The arrays of ``details/round_XX.npz``.

    Raises:
        ValueError: If the file is not a round-details file of schema ``vatl4pose.round_details/2``
            or misses a key that every round has.
    """
    path = run_dir / DETAILS_DIR / f"round_{round_index:02d}.npz"
    with np.load(path, allow_pickle=False) as f:
        details = {key: f[key] for key in f.files}
    if "schema" not in details or str(details["schema"]) != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}")
    missing = [key for key in _ALWAYS if key not in details]
    if missing:
        raise ValueError(f"{path}: misses {', '.join(missing)}")
    n, k = details["coords"].shape[:2]
    assert details["coords"].shape == (n, k, 2), details["coords"].shape
    assert details["labeled"].shape == (n,) and details["labeled"].dtype == np.bool_
    assert int(details["round_index"]) == round_index, path
    return details


def neighbor_index(
    track_id: np.ndarray, frame_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The sample of the same track in the nearest earlier and later frame, or ``-1``.

    The same links as the ``prev_index`` and ``next_index`` of the video cache, which the details
    do not hold: a frame where the track has no sample is skipped.
    """
    n = track_id.shape[0]
    assert frame_index.shape == (n,)
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    for track in np.unique(track_id):
        rows = np.flatnonzero(track_id == track)
        rows = rows[np.argsort(frame_index[rows], kind="stable")]
        assert np.all(np.diff(frame_index[rows]) > 0), (
            f"track {track!r} has more than one person in a frame"
        )
        prev[rows[1:]] = rows[:-1]
        nxt[rows[:-1]] = rows[1:]
    return prev, nxt


def frame_groups(
    image_path: np.ndarray, frame_index: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    """The frames of the video in order: ``(image path, samples in the frame)``.

    Raises:
        ValueError: If two paths share a frame index, or one path has two frame indices.
    """
    order = np.argsort(frame_index, kind="stable")
    groups: dict[str, list[int]] = {}
    for i in order.tolist():
        groups.setdefault(str(image_path[i]), []).append(i)
    frames = [(path, np.array(rows, dtype=np.int64)) for path, rows in groups.items()]
    for path, rows in frames:
        if np.unique(frame_index[rows]).size != 1:
            raise ValueError(f"{path} has more than one frame index")
    indices = [int(frame_index[rows[0]]) for _, rows in frames]
    if len(set(indices)) != len(indices):
        raise ValueError("two image paths share a frame index")
    return frames


def locate_images(
    paths: list[str], image_root: pathlib.Path | None
) -> dict[str, pathlib.Path]:
    """Where each recorded image path is now.

    Without ``image_root`` an image is where the details say. With it, the folder that all the
    images share is replaced by ``image_root`` and the names below it stay.

    Raises:
        FileNotFoundError: If an image is missing; the message names ``--image-root``.
    """
    if image_root is None:
        found = {p: pathlib.Path(p) for p in paths}
    else:
        common = os.path.commonpath([os.path.dirname(p) for p in paths])
        found = {p: image_root / os.path.relpath(p, common) for p in paths}
    missing = [str(p) for p in found.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(found)} images are missing, for example {missing[0]}; "
            "give the folder that holds them now with --image-root"
        )
    return found
