"""``vatl4pose-download-weights``: install the released pretrained weights.

The release is one zip. Its members are ``pretrained_models/<relative path>`` and
``MANIFEST.json``, which maps each member to its SHA256. :func:`install_from_zip` checks every
member against that file and against the manifest shipped with this package
(:mod:`vatl4pose.cli.weights_manifest`) before it moves anything into the weights root, so a
corrupted zip, a zip of another release and a zip with unsafe member names all stop with a
``ValueError`` and leave the weights root unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Mapping, Sequence

from .weights_manifest import check_relative_path, load_release_manifest

RELEASE_URL = (
    "https://github.com/ImIntheMiddle/VATL4Pose-WACV2024/releases/download/"
    "v1.0.0/vatl4pose-weights-v1.zip"
)
"""Location of the weights zip of release v1.0.0."""

_MEMBER_PREFIX = "pretrained_models/"
_MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


def _member_to_relative(name: str) -> str:
    """Return the path below the weights root for the zip member ``name``.

    Raises:
        ValueError: If ``name`` is not ``pretrained_models/<safe relative path>``.
    """
    if not name.startswith(_MEMBER_PREFIX):
        raise ValueError(f"member '{name}' is outside {_MEMBER_PREFIX}")
    try:
        return check_relative_path(name[len(_MEMBER_PREFIX) :])
    except ValueError as exc:
        raise ValueError(f"member '{name}' has an unsafe path: {exc}") from exc


def _read_zip_manifest(zf: zipfile.ZipFile, zip_path: pathlib.Path) -> dict[str, str]:
    """Return the ``MANIFEST.json`` of the zip as ``{member name: SHA256}``."""
    try:
        raw = json.loads(zf.read(_MANIFEST_NAME))
    except KeyError as exc:
        raise ValueError(f"{zip_path} has no {_MANIFEST_NAME}") from exc
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        raise ValueError(f"{zip_path}: cannot read {_MANIFEST_NAME}: {exc}") from exc
    if not isinstance(raw, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in raw.items()
    ):
        raise ValueError(
            f"{zip_path}: {_MANIFEST_NAME} must map member names to SHA256 strings"
        )
    return raw


def _plan_members(
    zf: zipfile.ZipFile, zip_path: pathlib.Path, release: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """Check the structure of the zip and return ``(member, relative path, SHA256)`` rows.

    Rows follow the order of the zip. Nothing is extracted and no member is read here.

    Raises:
        ValueError: On duplicate members, unsafe names, a member or manifest entry without
            counterpart, or a member set that differs from ``release``.
    """
    names = [info.filename for info in zf.infolist()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{zip_path}: duplicate member '{duplicates[0]}'")
    zip_manifest = _read_zip_manifest(zf, zip_path)
    files = [name for name in names if name != _MANIFEST_NAME]
    rows: list[tuple[str, str, str]] = []
    for name in files:
        if name.endswith("/"):
            if not name.startswith(_MEMBER_PREFIX) or ".." in name.split("/"):
                raise ValueError(f"member '{name}' is outside {_MEMBER_PREFIX}")
            continue
        rel = _member_to_relative(name)
        if name not in zip_manifest:
            raise ValueError(f"member '{name}' is not listed in {_MANIFEST_NAME}")
        rows.append((name, rel, zip_manifest[name]))
    in_zip = {name for name, _, _ in rows}
    for name in zip_manifest:
        _member_to_relative(name)
        if name not in in_zip:
            raise ValueError(
                f"{_MANIFEST_NAME} lists '{name}' but the zip has no such member"
            )
    release_rels = set(release)
    for name, rel, _ in rows:
        if rel not in release_rels:
            raise ValueError(f"member '{name}' is not part of this release")
    zip_rels = {rel for _, rel, _ in rows}
    missing = sorted(release_rels - zip_rels)
    if missing:
        raise ValueError(f"release file '{missing[0]}' is missing from the zip")
    return rows


def _stage_member(zf: zipfile.ZipFile, name: str, target: pathlib.Path) -> str:
    """Copy member ``name`` to ``target`` and return the SHA256 of the bytes written.

    Raises:
        ValueError: If the zip reports corrupted data for the member.
    """
    digest = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zf.open(name) as source, open(target, "wb") as sink:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                digest.update(chunk)
                sink.write(chunk)
    except (zipfile.BadZipFile, zlib.error) as exc:
        raise ValueError(f"member '{name}' is corrupted in the zip: {exc}") from exc
    return digest.hexdigest()


def install_from_zip(
    zip_path: pathlib.Path,
    weights_root: pathlib.Path,
    release_manifest: Mapping[str, str] | None = None,
) -> list[pathlib.Path]:
    """Install the weights of a release zip below ``weights_root``.

    Every member is verified against the SHA256 in the zip's ``MANIFEST.json`` and against
    ``release_manifest`` before any file reaches ``weights_root``. Members are staged in a
    temporary directory ``weights_root/.partial-<pid>-<random>/`` and moved into place with
    ``os.replace`` only after all of them passed, so a failure installs nothing.

    Args:
        zip_path: Release zip; members are ``pretrained_models/<relative path>`` plus
            ``MANIFEST.json``.
        weights_root: Directory that receives ``<relative path>``; created if missing.
        release_manifest: ``{relative path: SHA256}`` of the release the zip must match
            exactly. ``None`` selects the manifest shipped with this package.

    Returns:
        The installed paths, in the order of the zip.

    Raises:
        ValueError: For the first member whose SHA256 differs from the zip manifest or from
            the release manifest, for a member outside ``pretrained_models/`` or with a
            ``..`` component, and for any other structural problem of the zip.
    """
    zip_path, weights_root = pathlib.Path(zip_path), pathlib.Path(weights_root)
    release = (
        load_release_manifest() if release_manifest is None else dict(release_manifest)
    )
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{zip_path} is not a zip file: {exc}") from exc
    with archive:
        rows = _plan_members(archive, zip_path, release)
        weights_root.mkdir(parents=True, exist_ok=True)
        partial = pathlib.Path(
            tempfile.mkdtemp(prefix=f".partial-{os.getpid()}-", dir=weights_root)
        )
        try:
            staged: list[tuple[pathlib.Path, pathlib.Path]] = []
            for name, rel, expected in rows:
                if release[rel] != expected:
                    raise ValueError(
                        f"member '{name}': SHA256 {expected} in {_MANIFEST_NAME} differs from "
                        f"{release[rel]} in the release manifest of this vatl4pose version "
                        "(zip of another release?)"
                    )
                actual = _stage_member(archive, name, partial / rel)
                if actual != expected:
                    raise ValueError(
                        f"member '{name}': SHA256 {actual} does not match {expected} "
                        f"in {_MANIFEST_NAME}"
                    )
                staged.append((partial / rel, weights_root / rel))
            installed: list[pathlib.Path] = []
            for source, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                installed.append(destination)
        finally:
            shutil.rmtree(partial)
    return installed


def download_zip(url: str, destination: pathlib.Path) -> None:
    """Stream ``url`` into the file ``destination``."""
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        open(destination, "wb") as sink,
    ):
        shutil.copyfileobj(response, sink, _CHUNK)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``vatl4pose-download-weights``; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="vatl4pose-download-weights",
        description="Install the pretrained weights of the release, verified by SHA256.",
    )
    parser.add_argument(
        "--weights-root",
        type=pathlib.Path,
        required=True,
        help="directory that receives jrdb/, posetrack21/ and wholebodyAE/",
    )
    parser.add_argument(
        "--from-file",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="install from this zip instead of downloading it (offline install)",
    )
    args = parser.parse_args(argv)
    try:
        if args.from_file is not None:
            installed = install_from_zip(args.from_file, args.weights_root)
        else:
            args.weights_root.mkdir(parents=True, exist_ok=True)
            fd, download_name = tempfile.mkstemp(
                prefix=f".download-{os.getpid()}-", suffix=".zip", dir=args.weights_root
            )
            os.close(fd)
            download = pathlib.Path(download_name)
            try:
                print(f"downloading {RELEASE_URL}")
                download_zip(RELEASE_URL, download)
                installed = install_from_zip(download, args.weights_root)
            finally:
                download.unlink()
    except ValueError as exc:
        print(f"vatl4pose-download-weights: error: {exc}", file=sys.stderr)
        return 1
    print(f"installed {len(installed)} files into {args.weights_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
