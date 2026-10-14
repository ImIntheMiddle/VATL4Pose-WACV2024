"""Launch the local video annotation application of the ``annotation/`` folder."""

from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path

TOOLS_ENV = "VATL4POSE_ANNOTATION_ROOT"


def default_tools_root() -> Path:
    """The ``annotation/`` folder of the repository that holds this package."""
    return Path(__file__).resolve().parents[3] / "annotation"


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
        help="Use the local file annotation/samples/people.mp4 instead of a video path",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        help="Path to the annotation folder; the default is annotation/ of the "
        f"repository ({TOOLS_ENV} sets it too)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--learning-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--target-oks",
        type=float,
        help="Accuracy θ of the stopping criterion of the paper, an OKS in (0, 1)",
    )
    args = parser.parse_args(argv)
    if (args.video is None) == (not args.sample):
        parser.error("Provide one video path or --sample")
    if args.tools is not None:
        root = args.tools
    elif TOOLS_ENV in os.environ:
        root = Path(os.environ[TOOLS_ENV])
    else:
        root = default_tools_root()
    root = root.resolve()
    if not (root / "app.py").is_file():
        parser.error(
            f"No annotation tool at {root}. Run vatl4pose from a clone of the repository, "
            "or provide --tools /path/to/annotation"
        )
    video = (root / "samples/people.mp4") if args.sample else args.video.resolve()
    if not video.is_file():
        hint = "; the repository has no sample clip, put your own there" if args.sample else ""
        parser.error(f"Video does not exist: {video}{hint}")
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
    for name in ("output", "gpu", "max_frames", "learning_device", "target_oks"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
