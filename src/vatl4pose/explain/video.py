"""The prediction videos of the rounds, written as H.264.

``round_XX.mp4`` shows the poses that the model of round ``XX`` predicts for everybody, drawn into
the frames at their original resolution. ``compare_00_XX.mp4`` shows round 0 and round ``XX`` of the
same frames next to each other, one above the other for a panorama. The writer is the ffmpeg of the
``imageio-ffmpeg`` package, an optional dependency (``pip install "vatl4pose[explain]"``): the pip
build of OpenCV has no H.264 encoder.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Callable, Iterable, Iterator

import cv2
import numpy as np

from .draw import draw_pose
from .figures import Scene

FPS = 8
"""Frames per second of the videos; the frames of a video are the annotated ones, so not 30."""

_GAP = 8
INSTALL_HINT = (
    "the videos are written with the ffmpeg of the imageio-ffmpeg package, which is not "
    "installed; install it with 'pip install \"vatl4pose[explain]\"' (or 'uv sync --extra "
    "explain'), or leave the videos out with --no-video"
)


def ffmpeg_exe() -> str:
    """The path of the ffmpeg binary of ``imageio-ffmpeg``.

    Raises:
        ModuleNotFoundError: If the package is not installed; the message says how to install it.
    """
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(INSTALL_HINT) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_h264(
    path: pathlib.Path, frames: Iterable[np.ndarray], fps: float = FPS
) -> None:
    """Write RGB frames to an H.264 file; a frame with an odd side gets one black row or column.

    Raises:
        RuntimeError: If ffmpeg fails, with its message.
    """
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    padded_h, padded_w = height + height % 2, width + width % 2
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{padded_w}x{padded_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in [first, *iterator]:
            assert frame.shape == (height, width, 3) and frame.dtype == np.uint8
            if (padded_h, padded_w) != (height, width):
                frame = np.pad(
                    frame, ((0, padded_h - height), (0, padded_w - width), (0, 0))
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
        message = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {path} (exit {code}): {message}")


def label_frame(image: np.ndarray, text: str) -> None:
    """Write ``text`` into the top left corner of an RGB frame, in place."""
    scale = max(0.5, image.shape[0] / 720)
    origin = (round(10 * scale), round(28 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, round(scale * 1.5))
    cv2.putText(
        image, text, origin, font, 0.8 * scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        image, text, origin, font, 0.8 * scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def draw_round_frame(
    image: np.ndarray, scene: Scene, rows: np.ndarray, coords: np.ndarray, text: str
) -> np.ndarray:
    """A copy of ``image`` with the skeletons of the people ``rows`` and a label."""
    out = image.copy()
    height, width = out.shape[:2]
    spec, panorama = scene.info.spec, scene.info.panorama
    for i in rows.tolist():
        xy = coords[i].astype(np.float64)
        if panorama:
            xy[:, 0] = np.mod(xy[:, 0], width)
        draw_pose(out, spec, xy, height / 480, width_x=width if panorama else None)
    label_frame(out, text)
    return out


def round_frames(
    scene: Scene,
    read_frame: Callable[[int], np.ndarray],
    coords: np.ndarray,
    text: str,
) -> Iterator[np.ndarray]:
    """The frames of ``round_XX.mp4``: the video's frames with the poses of the round."""
    for position, (_, rows) in enumerate(scene.frames):
        yield draw_round_frame(read_frame(position), scene, rows, coords, text)


def compare_frames(
    scene: Scene,
    read_frame: Callable[[int], np.ndarray],
    before: tuple[np.ndarray, str],
    after: tuple[np.ndarray, str],
) -> Iterator[np.ndarray]:
    """The frames of ``compare_00_XX.mp4``; ``before`` and ``after`` are ``(coords, label)``."""
    axis = 0 if scene.info.panorama else 1
    for position, (_, rows) in enumerate(scene.frames):
        image = read_frame(position)
        left = draw_round_frame(image, scene, rows, *before)
        right = draw_round_frame(image, scene, rows, *after)
        gap_shape = (
            (_GAP, image.shape[1], 3) if axis == 0 else (image.shape[0], _GAP, 3)
        )
        gap = np.full(gap_shape, 255, dtype=np.uint8)
        yield np.concatenate([left, gap, right], axis=axis)
