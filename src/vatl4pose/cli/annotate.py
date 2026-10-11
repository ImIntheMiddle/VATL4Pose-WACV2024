"""Launch the optional local video annotation application."""

from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path


def setup_ready(root: Path) -> bool:
    marker = root / ".deps/requirements.lock.sha256"
    lock = root / "requirements.lock"
    return (
        (root / ".venv/bin/python").is_file()
        and marker.is_file()
        and lock.is_file()
        and marker.read_text().strip() == hashlib.sha256(lock.read_bytes()).hexdigest()
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the prepared sample video through the same pipeline",
    )
    parser.add_argument(
        "--tools", type=Path, help="Path to the optional tracking tools folder"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--learning-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if (args.video is None) == (not args.sample):
        parser.error("Provide one video path or --sample")
    candidates = [
        Path.cwd() / "tracking",
        Path(__file__).resolve().parents[3].parent / "VATL4Pose-dev/tracking",
    ]
    if args.tools is not None:
        root = args.tools
    elif "VATL4POSE_TRACKING_ROOT" in os.environ:
        root = Path(os.environ["VATL4POSE_TRACKING_ROOT"])
    else:
        root = next((p for p in candidates if (p / "app.py").is_file()), candidates[0])
    root = root.resolve()
    if not (root / "app.py").is_file():
        parser.error(
            "Install the optional tracking tools and provide --tools /path/to/tracking"
        )
    video = (root / "samples/people.mp4") if args.sample else args.video.resolve()
    if not video.is_file():
        parser.error(f"Video does not exist: {video}")
    if not setup_ready(root):
        from vatl4pose.execution.process import run_child

        code = run_child(
            ["bash", str(root / "setup.sh")], root.parent, dict(os.environ), None
        )
        if code:
            raise SystemExit(code)
    command = [
        str(root / ".venv/bin/python"),
        str(root / "app.py"),
        str(video),
        "--port",
        str(args.port),
    ]
    for name in ("output", "gpu", "max_frames", "learning_device"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
