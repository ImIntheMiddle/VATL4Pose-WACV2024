"""A video of the keypoints of everyone under the current estimator, for the annotation portal.

After the training of round ``r`` the loop predicts the joints of every person for round
``r + 1`` and keeps them in ``state/pending_{r+1}/predictions.npy``, in the order of the samples of
the video cache. The video draws them on every frame of the video. The limbs are blue on the
left, red on the right and green elsewhere, as in the editor; the joints are white dots. The
joints that the spec never annotates and those below the confidence threshold of the drawings
(:data:`vatl4pose.explain.draw.MIN_CONFIDENCE`) are left out, with their limbs. The video of a
round is written once, under ``estimates/`` of the session.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import cv2
import numpy as np

from annotation.labelstudio.queries_to_labelstudio import joint_color
from annotation.videos import write_video, written_once
from vatl4pose.data.cache import VideoCache
from vatl4pose.explain.draw import shown_joints
from vatl4pose.keypoints import KeypointSpec, get_spec


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def draw_people(
    image: np.ndarray, coords: np.ndarray, spec: KeypointSpec
) -> np.ndarray:
    """``image`` with the skeletons of ``coords`` drawn.

    ``coords`` is ``(n, K, 3)``: x and y in pixels and the confidence of the joint.
    """
    assert coords.ndim == 3 and coords.shape[1:] == (spec.num_joints, 3), coords.shape
    out = image.copy()
    t = max(2, round(2.2 * image.shape[0] / 480))
    for person in coords:
        skip = set(spec.redundant_joints) | set(
            np.flatnonzero(~shown_joints(person[:, 2])).tolist()
        )
        points = [tuple(int(v) for v in np.round(p)) for p in person[:, :2]]
        for a, b in spec.skeleton:
            if a in skip or b in skip:
                continue
            names = spec.joint_names[a] + spec.joint_names[b]
            side = "left" if "left" in names else "right" if "right" in names else ""
            color = _bgr(joint_color(side))
            cv2.line(out, points[a], points[b], color, t, cv2.LINE_AA)
        for j, point in enumerate(points):
            if j in skip:
                continue
            cv2.circle(out, point, t + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(out, point, max(1, t - 1), (255, 255, 255), -1, cv2.LINE_AA)
    return out


def render_estimates(
    frames: list[str],
    sample_frames: list[str],
    coords: np.ndarray,
    spec: KeypointSpec,
    fps: float,
    ffmpeg: str,
    target: Path,
) -> None:
    """Write ``target``: every frame of ``frames``, with the people whose frame (``sample_frames``,
    matched by file name) it is drawn at ``coords``."""
    assert len(sample_frames) == len(coords), (len(sample_frames), coords.shape)
    rows: dict[str, list[int]] = {}
    for i, path in enumerate(sample_frames):
        rows.setdefault(Path(path).name, []).append(i)
    first = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    assert first is not None, frames[0]
    height, width = first.shape[:2]

    def drawn() -> Iterator[np.ndarray]:
        for path in frames:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            assert image is not None, path
            people = rows.get(Path(path).name, [])
            yield draw_people(image, coords[people], spec) if people else image

    write_video(drawn(), width, height, fps, ffmpeg, target)


def estimate_video(
    run: Path,
    state_dir: Path,
    cache_path: Path,
    round_index: int,
    keypoints: str,
    fps: float,
    ffmpeg: str,
) -> Path:
    """The video of the predictions after the training of round ``round_index``.

    Raises:
        FileNotFoundError: If the loop keeps no predictions of round ``round_index + 1``, as after
            its last round.
    """
    predictions = state_dir / f"pending_{round_index + 1:02d}" / "predictions.npy"
    target = run / "estimates" / f"round_{round_index:02d}.mp4"
    if not target.is_file() and not predictions.is_file():
        raise FileNotFoundError(
            f"The predictions after round {round_index} are not kept"
        )

    def render() -> None:
        spec = get_spec(keypoints)
        sample_frames = [
            str(p) for p in VideoCache(cache_path, spec).arrays.image_path
        ]
        data = json.loads(
            (run / "dataset/annotations/clip.json").read_text(encoding="utf-8")
        )
        frames = [
            str(run / "dataset" / image["file_name"])
            for image in sorted(data["images"], key=lambda image: image["id"])
        ]
        render_estimates(
            frames, sample_frames, np.load(predictions), spec, fps, ffmpeg, target
        )

    return written_once(target, render)
