"""Run bookkeeping for parallel execution: sharding, id lists, completion markers, locks.

Several processes, possibly on several hosts, share one filesystem and split the videos of a
dataset between them. Each process is one shard. :func:`select_shard` gives every shard a fixed,
disjoint set of videos, and every video has its own output directory that only the process of its
shard writes to. The other pieces keep those directories consistent when a process is killed or
started twice by mistake:

* :func:`read_id_list` reads the list of video ids to run (one id per line).
* :func:`video_seed` derives the random seed of a video from its id only, so a video gets the same
  seed whatever the number of shards is.
* :func:`mark_done` and :func:`is_done` record that a video finished, in a file that a reader
  never sees half-written.
* :class:`VideoLock` is a per-video lock file that refuses a second process.

The output directories usually live in a home directory on NFS that all hosts mount. The
primitives used here are the ones that work across hosts on NFS: ``open(O_CREAT | O_EXCL)`` is
atomic on NFSv3 and later with the Linux client, ``os.replace`` is an atomic rename, and every file
(and the directory after a rename) is ``fsync``-ed before another process can rely on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import socket
from collections import Counter
from datetime import datetime, timezone
from types import TracebackType

DONE_NAME = ".done.json"
DONE_TMP_NAME = ".done.json.tmp"
LOCK_NAME = ".lock"

_SHARD_RE = re.compile(r"(\d+)/(\d+)", re.ASCII)


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``"i/n"`` into ``(i, n)``: shard ``i`` of ``n`` shards, counted from 0.

    Raises ``ValueError`` unless ``spec`` is two non-negative integers around a slash with
    ``n >= 1`` and ``0 <= i < n``.
    """
    match = _SHARD_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f"shard must look like 'i/n' with integers i and n, got {spec!r}")
    shard, num_shards = int(match.group(1)), int(match.group(2))
    _check_shard(shard, num_shards)
    return shard, num_shards


def _check_shard(shard: int, num_shards: int) -> None:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard}/{num_shards} is invalid: need n >= 1 and 0 <= i < n")


def _duplicates(ids: list[str]) -> list[str]:
    return sorted(item for item, count in Counter(ids).items() if count > 1)


def read_id_list(path: pathlib.Path) -> list[str]:
    """Read video ids from a text file, one per line, in file order.

    Surrounding whitespace (including a ``\\r`` of Windows line ends) is stripped. Blank lines and
    lines that start with ``#`` are dropped. Raises ``ValueError`` that lists every id which occurs
    more than once.
    """
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    duplicates = _duplicates(ids)
    if duplicates:
        raise ValueError(f"{path} lists video ids more than once: {', '.join(duplicates)}")
    return ids


def select_shard(video_ids: list[str], shard: int, num_shards: int) -> list[str]:
    """Return the videos of one shard: the sorted ids taken round-robin, ``sorted_ids[shard::n]``.

    The assignment depends only on the set of ids and the shard numbers, so every process computes
    the same partition without talking to the others. The input list is not modified. Raises
    ``ValueError`` for an invalid shard or for a repeated id, because a repeated id would land in
    two shards and two processes would run the same video.
    """
    _check_shard(shard, num_shards)
    duplicates = _duplicates(video_ids)
    if duplicates:
        raise ValueError(f"video ids occur more than once: {', '.join(duplicates)}")
    return sorted(video_ids)[shard::num_shards]


def video_seed(base_seed: int, video_id: str) -> int:
    """Return the random seed of one video: ``base_seed`` plus 32 bits of the id's sha256.

    The seed depends on the id only, so a video gets the same seed whatever shard or number of
    shards runs it. It uses sha256 and not ``hash()``, whose value for a string changes between
    Python processes. The result can exceed ``2**32 - 1``; pass it to ``numpy.random.default_rng``
    or ``torch.manual_seed``, which take such values, and not to ``numpy.random.seed``, which does
    not.
    """
    digest = hashlib.sha256(video_id.encode()).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def is_done(video_dir: pathlib.Path) -> bool:
    """Return whether ``video_dir`` holds a completion marker written by :func:`mark_done`."""
    return (video_dir / DONE_NAME).is_file()


def mark_done(video_dir: pathlib.Path, payload: dict) -> None:
    """Record that the video of ``video_dir`` is finished, with ``payload`` as JSON in the marker.

    The payload is serialised first, so a payload that is not JSON-serialisable raises before any
    file exists. The marker is then written as ``.done.json.tmp``, ``fsync``-ed, and moved to
    ``.done.json`` with ``os.replace``, which is an atomic rename. The directory is ``fsync``-ed
    last so the rename survives a crash of the host. A reader therefore sees either no marker or
    the complete one, never a partial file.
    """
    text = json.dumps(payload, indent=2) + "\n"
    tmp = video_dir / DONE_TMP_NAME
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, video_dir / DONE_NAME)
    dir_fd = os.open(video_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class VideoLock:
    """Per-video mutual exclusion through the lock file ``<video_dir>/.lock``.

    The lock is used for everything that must have one writer per video: the build of the video's
    HDF5 cache (two builders of the same cache path would share one ``.partial`` file) and the
    active-learning run on the video.

    ``__enter__`` creates the file with ``os.open(O_CREAT | O_EXCL | O_WRONLY)``, which is atomic
    on NFSv3 and later with the Linux client, so exactly one of several processes on several hosts
    gets it. The process writes its payload ``{"host", "pid", "started"}`` through the descriptor
    it created and ``fsync``-s it, so a second process that is refused can read who holds the lock.
    If the file exists, ``__enter__`` raises ``RuntimeError`` that names the file and its content,
    and leaves the file alone.

    A lock left behind by a crashed process is never removed automatically, because the program
    cannot tell a crashed holder from a slow one on another host. The error message tells the user
    to check the host and the pid and to delete the file by hand.

    ``__exit__`` removes the lock only if the file still holds exactly the payload this process
    wrote. If the file is missing or holds another payload, it raises ``RuntimeError`` and does not
    delete anything, so it never removes the lock of another process. The check and the removal are
    two steps, so the check detects a lock that was replaced by hand, not a race with a deletion
    that happens in between.
    """

    def __init__(self, video_dir: pathlib.Path) -> None:
        self.path = video_dir / LOCK_NAME
        self._text: str | None = None

    def __enter__(self) -> VideoLock:
        payload = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": datetime.now(timezone.utc).isoformat(),
        }
        text = json.dumps(payload)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise RuntimeError(self._refusal_message()) from None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            # The file was created by this call, so it is this call's own lock and safe to remove.
            self.path.unlink()
            raise
        self._text = text
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._text is not None, "VideoLock.__exit__ called without a successful __enter__"
        own, self._text = self._text, None
        try:
            found = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RuntimeError(
                f"Lock file {self.path} is missing at release; this process wrote {own}. "
                "Someone deleted it while the video was being processed."
            ) from None
        if found != own:
            raise RuntimeError(
                f"Lock file {self.path} no longer holds this process's payload {own}; "
                f"it holds {found!r}. It was not removed."
            )
        self.path.unlink()

    def _refusal_message(self) -> str:
        try:
            held = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            held = "<released while this process was reading it; try again>"
        return (
            f"Video is locked: {self.path} already exists and holds {held!r}. "
            "Another process is working on this video. If no process with that host and pid is "
            "running any more, the lock is stale after a crash: check the host and the pid, then "
            "delete the file by hand. It is never removed automatically."
        )
