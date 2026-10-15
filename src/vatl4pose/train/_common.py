"""Helpers that the training code and the active-learning loop share.

* :func:`derived_seed` and :func:`shuffle_order` give seeds and epoch orders that depend on their
  arguments only, never on the global random state.
* :class:`FixedOrderSampler` lets a loader visit the items in an order that is set before every
  epoch.
* :func:`write_atomic` writes a file so that a reader sees no file or the complete file;
  :func:`write_file` and :func:`fsync_directory` are the steps it is made of.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from collections.abc import Callable, Iterator
from typing import BinaryIO

import torch
from torch.utils.data import Sampler


def derived_seed(*parts: int | str) -> int:
    """A 63-bit seed from ``parts`` through SHA-256; equal on every run and every machine.

    The hashed text starts with the fixed prefix ``vatl4pose.finetune``; changing it changes every
    derived seed and epoch order.
    """
    text = "/".join(["vatl4pose.finetune", *map(str, parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def shuffle_order(n: int, seed: int, round_index: int, epoch: int) -> torch.Tensor:
    """The order in which the ``n`` items are visited in one epoch of one round.

    The permutation is a function of ``(seed, round_index, epoch)`` only; it does not read or
    change the global random state.

    Returns:
        ``(n,)`` int64 permutation of ``0 .. n - 1``.
    """
    generator = torch.Generator().manual_seed(
        derived_seed("shuffle", seed, round_index, epoch)
    )
    order = torch.randperm(n, generator=generator)
    assert order.dtype == torch.int64 and tuple(order.shape) == (n,)
    return order


class FixedOrderSampler(Sampler[int]):
    """Yields ``order``; the training loop replaces it before every epoch."""

    def __init__(self, n: int) -> None:
        self.order: list[int] = list(range(n))

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def write_atomic(path: pathlib.Path, write: Callable[[BinaryIO], None]) -> None:
    """Write a file so that a reader sees no file or the complete file.

    ``write`` receives a binary file object for ``<path>.tmp``; the file is ``fsync``-ed, moved to
    ``path`` with ``os.replace`` and the directory is ``fsync``-ed. A failure removes the temporary
    file and leaves ``path`` as it was.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        write_file(tmp, write)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def write_file(path: pathlib.Path, write: Callable[[BinaryIO], None]) -> None:
    """Write ``path`` through ``write`` and ``fsync`` the file; the file may be incomplete after a crash."""
    with open(path, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())


def fsync_directory(directory: pathlib.Path) -> None:
    """``fsync`` a directory, so that the files created or moved in it are recorded."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
