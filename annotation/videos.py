"""Write the browser videos of the annotation portal: H.264 files that are written once per path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import subprocess
import threading

import numpy as np

MAX_WIDTH = 1280

_guard = threading.Lock()
_locks: dict[Path, threading.Lock] = {}


def written_once(target: Path, render: Callable[[], None]) -> Path:
    """``target``, written by ``render`` on the first call; concurrent calls wait for it."""
    with _guard:
        lock = _locks.setdefault(target, threading.Lock())
    with lock:
        if not target.is_file():
            target.parent.mkdir(exist_ok=True)
            render()
    return target


def write_video(
    images: Iterable[np.ndarray],
    width: int,
    height: int,
    fps: float,
    ffmpeg: str,
    target: Path,
) -> None:
    """Encode BGR frames of ``width`` x ``height`` to ``target``, which appears only when complete.

    Frames wider than 1280 pixels are scaled down to that width.
    """
    partial = target.with_name(target.name + ".partial.mp4")
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            f"scale='min({MAX_WIDTH},iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    try:
        for image in images:
            assert image.shape == (height, width, 3), image.shape
            encoder.stdin.write(image.tobytes())
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited with code {code} for {target.name}")
    partial.replace(target)
