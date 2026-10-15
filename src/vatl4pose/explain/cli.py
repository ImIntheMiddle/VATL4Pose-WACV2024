"""``vatl4pose-explain``: figures, videos and a page that show which people a run selected and why.

    vatl4pose-explain outputs/mine/simplebaseline/THC+WPU_Coresetfilter/clip1

The argument is the run directory of one video that ``vatl4pose-atl`` wrote with
``loop.save_round_details=true``. The command reads ``details/round_XX.npz`` and writes ``explain/``:

* ``index.html`` with ``data.js`` and ``frames/``, ``crops/``: a page for one video that opens from
  ``file://``. It switches between the rounds, numbers the selected people in the order of the
  selection and, on a click on a person, shows the THC, WPU and DUW figures of the pick.
* ``round_XX.mp4``: the predictions of round ``XX`` in the frames of the video, and
  ``compare_00_XX.mp4``: round 0 next to round ``XX`` (H.264, needs ``vatl4pose[explain]``).
* ``picks/round_XX_rank_YY.png``: the explanation of the ``YY``-th selected person of round ``XX``.

A strategy without criteria (Random, for example) gets the numbers of the selection only.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import shutil
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from .draw import person_crop, shown_joints
from .figures import RoundData, Scene, build_round, build_scene, crop_samples
from .load import (
    load_round,
    locate_images,
    read_run_info,
    recorded_rounds,
)
from .page import (
    CROPS_DIR,
    FRAMES_DIR,
    build_data,
    crop_name,
    frame_name,
    write_page,
)
from .png import render_pick
from .video import compare_frames, ffmpeg_exe, round_frames, write_h264

OUT_DIR = "explain"
PICKS_DIR = "picks"
DEFAULT_MAX_PICKS = 40
_JPEG_QUALITY = 90


def parse_rounds(text: str) -> list[int]:
    """The rounds of ``--rounds``, e.g. ``0,2`` gives ``[0, 2]``."""
    try:
        rounds = sorted({int(part) for part in text.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--rounds takes round numbers separated by commas, got {text!r}"
        ) from exc
    if rounds[0] < 0:
        raise argparse.ArgumentTypeError("a round number is not negative")
    return rounds


def _read_rgb(path: pathlib.Path) -> np.ndarray:
    """An RGB frame as the loop reads it (``cv2.imread``, which applies the EXIF orientation)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot read the image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_jpeg(path: pathlib.Path, rgb: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    ):
        raise OSError(f"cannot write {path}")


def clean_output(out: pathlib.Path) -> None:
    """Remove what an earlier run of this command wrote into ``out``, and nothing else."""
    for name in (FRAMES_DIR, CROPS_DIR, PICKS_DIR):
        if (out / name).exists():
            shutil.rmtree(out / name)
    for pattern in ("index.html", "data.js", "round_*.mp4", "compare_*.mp4"):
        for path in out.glob(pattern):
            path.unlink()


def write_frames(
    out: pathlib.Path, scene: Scene, files: dict[str, pathlib.Path]
) -> list[tuple[int, int]]:
    """Write every frame of the video as ``frames/frame_XXXX.jpg``; returns ``(width, height)``."""
    sizes = []
    for position, (path, _) in enumerate(scene.frames):
        image = _read_rgb(files[path])
        sizes.append((image.shape[1], image.shape[0]))
        _write_jpeg(out / frame_name(position), image, _JPEG_QUALITY)
    return sizes


def write_crops(
    out: pathlib.Path,
    scene: Scene,
    files: dict[str, pathlib.Path],
    samples: list[int],
) -> dict[int, np.ndarray]:
    """Write the crop of each person in ``samples`` (sample indices) as ``crops/person_XXXXXX.jpg``.

    Returns:
        The crops by sample index.
    """
    wanted = set(samples)
    crops: dict[int, np.ndarray] = {}
    for path, rows in scene.frames:
        rows_here = [int(i) for i in rows.tolist() if int(i) in wanted]
        if not rows_here:
            continue
        image = _read_rgb(files[path])
        for i in rows_here:
            crops[i] = person_crop(image, scene.bbox_crop[i], scene.info.panorama)
            _write_jpeg(out / crop_name(i), crops[i], 95)
    assert set(crops) == wanted
    return crops


def frame_reader(out: pathlib.Path) -> Callable[[int], np.ndarray]:
    """A reader of the written frames; the last few frames stay in memory, so do not change them."""

    @functools.lru_cache(maxsize=6)
    def read(position: int) -> np.ndarray:
        return _read_rgb(out / frame_name(position))

    return read


def explain(
    run_dir: pathlib.Path,
    rounds: Sequence[int] | None = None,
    image_root: pathlib.Path | None = None,
    max_picks: int = DEFAULT_MAX_PICKS,
    video: bool = True,
) -> pathlib.Path:
    """Write ``explain/`` of a run directory (see the module docstring).

    Args:
        run_dir: The run directory of one video.
        rounds: The rounds to explain; ``None`` takes every round with details.
        image_root: The directory that holds the frames now, if they are not where the details say.
        max_picks: The number of picks of a round that get a figure and a PNG. All picks of a
            round are numbered in the page.
        video: Whether to write the videos, which need the ``explain`` extra.

    Returns:
        The ``explain/`` directory.

    Raises:
        FileNotFoundError: If the directory has no details or an image is missing.
        ModuleNotFoundError: If the videos are asked for and ``imageio-ffmpeg`` is not installed.
        ValueError: If a round has no details.
    """
    assert max_picks >= 1
    info = read_run_info(run_dir)
    recorded = recorded_rounds(run_dir)
    chosen = list(recorded) if rounds is None else sorted(set(rounds))
    absent = [r for r in chosen if r not in recorded]
    if absent:
        raise ValueError(
            f"no details for round {', '.join(map(str, absent))}; recorded rounds: "
            f"{', '.join(map(str, recorded))}"
        )
    if video:
        ffmpeg_exe()
    first = load_round(run_dir, chosen[0])
    scene = build_scene(info, first)
    files = locate_images([path for path, _ in scene.frames], image_root)
    out = run_dir / OUT_DIR
    out.mkdir(exist_ok=True)
    clean_output(out)

    sizes = write_frames(out, scene, files)
    data: list[RoundData] = []
    for r in chosen:
        details = first if r == chosen[0] else load_round(run_dir, r)
        assert np.array_equal(details["image_path"], first["image_path"]), (
            f"round {r} has other people than round {chosen[0]}"
        )
        data.append(build_round(scene, details, max_picks))
        print(f"round {r}: {len(data[-1].order)} selected", flush=True)

    samples = sorted({s for rd in data for s in crop_samples(rd.picks)})
    crops = write_crops(out, scene, files, samples)
    read = frame_reader(out)
    for rd in data:
        for pick in rd.picks:
            render_pick(
                out / PICKS_DIR / f"round_{rd.index:02d}_rank_{pick.rank:02d}.png",
                scene,
                rd,
                pick,
                crops,
                read,
            )
    videos: dict[str, dict[str, str]] = {"round": {}, "compare": {}}
    if video:
        base = data[0] if data[0].index == 0 else None
        if base is not None:
            zero: tuple[np.ndarray, np.ndarray, float] | None = (
                base.coords,
                base.shown,
                base.percentage,
            )
        elif 0 in recorded:
            details = load_round(run_dir, 0)
            zero = (
                details["coords"],
                shown_joints(details["maxvals"]),
                float(details["percentage"]),
            )
        else:
            zero = None
            print("no compare videos: round 0 has no details", flush=True)
        for rd in data:
            name = f"round_{rd.index:02d}.mp4"
            text = f"round {rd.index}: {rd.percentage:.1f} % labeled"
            write_h264(out / name, round_frames(scene, read, rd.coords, rd.shown, text))
            videos["round"][str(rd.index)] = name
            if rd.index != 0 and zero is not None:
                name = f"compare_00_{rd.index:02d}.mp4"
                before = (zero[0], zero[1], f"round 0: {zero[2]:.1f} % labeled")
                write_h264(
                    out / name,
                    compare_frames(scene, read, before, (rd.coords, rd.shown, text)),
                )
                videos["compare"][str(rd.index)] = name
            print(f"round {rd.index}: video written", flush=True)
    write_page(out, build_data(scene, data, sizes, max_picks, videos))
    print(f"wrote {out}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``vatl4pose-explain``; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="vatl4pose-explain",
        description="Show which people a run selected and why: a page, videos and one figure "
        "per selected person, written to explain/ of the run directory.",
    )
    parser.add_argument(
        "run_dir",
        type=pathlib.Path,
        help="run directory of one video, made with loop.save_round_details=true",
    )
    parser.add_argument(
        "--rounds",
        type=parse_rounds,
        default=None,
        metavar="0,2",
        help="rounds to explain (default: every round with details)",
    )
    parser.add_argument(
        "--image-root",
        type=pathlib.Path,
        default=None,
        help="directory that holds the frames of the video now, if they moved",
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=DEFAULT_MAX_PICKS,
        help="picks per round that get a figure and a PNG, in the order of the selection "
        f"(default: {DEFAULT_MAX_PICKS})",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="do not write the H.264 videos, which need 'pip install \"vatl4pose[explain]\"'",
    )
    args = parser.parse_args(argv)
    if args.max_picks < 1:
        parser.error("--max-picks must be at least 1")
    try:
        explain(
            args.run_dir,
            rounds=args.rounds,
            image_root=args.image_root,
            max_picks=args.max_picks,
            video=not args.no_video,
        )
    except ModuleNotFoundError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
