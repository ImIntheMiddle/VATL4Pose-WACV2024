"""The SHA256 manifest of the released pretrained weights.

``weights_manifest.json`` ships with the package. It maps each file of the release, as a
path relative to the weights root (the directory that holds ``jrdb/``, ``posetrack21/`` and
``wholebodyAE/``), to its SHA256. ``vatl4pose-download-weights`` and ``vatl4pose-doctor``
both check files against it.

To regenerate the JSON from a weights directory::

    python -m vatl4pose.cli.weights_manifest --weights-root /path/to/pretrained_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from collections.abc import Iterable, Sequence

MANIFEST_PATH = pathlib.Path(__file__).with_name("weights_manifest.json")
"""Location of the manifest that ships with the package."""

RELEASE_FILES: tuple[str, ...] = (
    "jrdb/fastpose_256x192.pth",
    "posetrack21/fastpose_256x192.pth",
    "posetrack21/hrnet_256x192.pth",
    "posetrack21/simplebaseline_256x192.pth",
    "wholebodyAE/JRDB2022/Hybrid/WholeBodyAE_zdim4.pth",
    "wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4.pth",
)
"""The files of the v1 release, relative to the weights root: the files v1 uses."""

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    """Return the lowercase hex SHA256 of the file at ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_relative_path(rel: str) -> str:
    """Return ``rel`` if it is a safe ``/``-separated path below a root, else raise.

    Raises:
        ValueError: If ``rel`` is empty, absolute, uses a backslash, or has an empty, ``.``
            or ``..`` component.
    """
    if not rel or rel.startswith("/") or "\\" in rel:
        raise ValueError(
            f"unsafe path {rel!r}: must be relative and use '/' separators"
        )
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise ValueError(f"unsafe path {rel!r}: empty, '.' or '..' component")
    return rel


def load_release_manifest(path: pathlib.Path | None = None) -> dict[str, str]:
    """Load a manifest (the shipped one by default) as ``{relative path: SHA256}``.

    Raises:
        ValueError: If the JSON is not an object of safe paths and 64-digit lowercase hex
            digests.
    """
    path = MANIFEST_PATH if path is None else pathlib.Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    manifest: dict[str, str] = {}
    for rel, digest in raw.items():
        try:
            check_relative_path(rel)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(
                f"{path}: entry {rel!r} is not a lowercase SHA256: {digest!r}"
            )
        manifest[rel] = digest
    return manifest


def compute_manifest(
    weights_root: pathlib.Path, files: Iterable[str] = RELEASE_FILES
) -> dict[str, str]:
    """Hash ``files`` below ``weights_root`` and return ``{relative path: SHA256}``.

    Raises:
        FileNotFoundError: If one of the files does not exist (the message holds its path).
    """
    return {rel: sha256_file(weights_root / check_relative_path(rel)) for rel in files}


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the shipped manifest from a weights directory that holds the release."""
    parser = argparse.ArgumentParser(
        prog="python -m vatl4pose.cli.weights_manifest", description=main.__doc__
    )
    parser.add_argument("--weights-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    manifest = compute_manifest(args.weights_root)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
