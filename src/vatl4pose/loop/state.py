"""The round state of the active-learning loop and its files.

A run that is killed, or that waits for human annotations, must continue with the round after the
last one it finished and give the result an uninterrupted run would have given. Everything that the
next round reads is therefore written when a round is finished, to the directory ``state/`` of the
run directory:

``state.json``
    The record of every finished round (:class:`RoundRecord`: what was queried, the answers of the
    oracle, the metrics, the timings), the sorted list of labeled samples, the state of the numpy
    random generator and the settings of the run (``config``), which a resume compares with its
    own settings.
``torch_rng.pt``
    The state of the global torch random generator.
``finetuner.pt``
    The state of the :class:`~vatl4pose.train.finetune.Finetuner`: model, optimizer, schedule.

Both torch files carry the number of finished rounds, and :func:`load_state` checks that all three
files agree. :func:`save_state` writes the three temporary files first (the slow part), then moves
them into place one after the other and ``state.json`` last. ``state.json`` is the commit marker: a
run killed while the files are moved leaves either the old state complete, or a torch file that is
one round ahead of ``state.json``, which :func:`load_state` reports as an error.

Everything is JSON or ``torch.save`` data of tensors, numbers, strings and plain containers, so
``torch.load(weights_only=True)`` reads it and no pickled code is involved.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, BinaryIO

import torch

STATE_SCHEMA = "vatl4pose.state/1"
STATE_NAME = "state.json"
TORCH_RNG_NAME = "torch_rng.pt"
FINETUNER_NAME = "finetuner.pt"


def write_atomic(path: pathlib.Path, write: Callable[[BinaryIO], None]) -> None:
    """Write a file so that a reader sees no file or the complete file.

    ``write`` receives a binary file object for ``<path>.tmp``; the file is ``fsync``-ed, moved to
    ``path`` with ``os.replace`` and the directory is ``fsync``-ed. A failure removes the temporary
    file and leaves ``path`` as it was.
    """
    tmp = _write_temporary(path, write)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _write_temporary(
    path: pathlib.Path, write: Callable[[BinaryIO], None]
) -> pathlib.Path:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _fsync_directory(directory: pathlib.Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass
class RoundRecord:
    """Everything one finished round produced. Lists hold plain Python numbers.

    Attributes:
        percentage: Share of labeled samples in percent when the round evaluated, before its query.
        performance: The ten AP and AR values of the predictions of all samples.
        performance_ann: The same after the labeled samples' predictions were replaced by their
            labels.
        ospa: OSPA of the predictions.
        ospa_ann: OSPA after the replacement.
        combine_weight: Mean local peak of the heatmaps of the unlabeled samples; ``None`` when
            nobody is unlabeled.
        score_means: Mean raw score of every scorer over the unlabeled samples; ``None`` when the
            round scored nothing (no query, an empty query, or a forced query that takes everybody).
        queried: Sorted sample indices queried in the round; ``[]`` for a query of size 0; ``None``
            in the final round, which has no query.
        answers: Labels of the oracle for ``queried``, row for row, as ``(k, K, 3)`` nested lists
            of x, y and visibility; ``None`` where ``queried`` is ``None`` or empty.
        moks_queried: Mean OKS of the predictions for the queried people against their labels, from
            the model before it learned from them; ``None`` without a query.
        timings: Seconds of each step (``infer``, ``eval``, ``select``, ``oracle``, ``retrain``,
            ``ae``) and ``total``, the whole round without the write of the state.
    """

    percentage: float
    performance: dict[str, float]
    performance_ann: dict[str, float]
    ospa: float
    ospa_ann: float
    combine_weight: float | None
    score_means: dict[str, float | None]
    queried: list[int] | None
    answers: list[list[list[float]]] | None
    moks_queried: float | None
    timings: dict[str, float]


@dataclass
class LoopState:
    """The finished rounds of a run and the generator state after them.

    Attributes:
        config: The settings of the run as JSON data; a resume must have the same.
        rounds: One record per finished round.
        rng: ``bit_generator.state`` of the numpy generator after the last finished round.
    """

    config: dict[str, Any]
    rng: dict[str, Any]
    rounds: list[RoundRecord] = field(default_factory=list)

    @property
    def rounds_done(self) -> int:
        """Number of finished rounds."""
        return len(self.rounds)

    @property
    def labeled(self) -> list[int]:
        """Sorted sample indices queried in the finished rounds."""
        return sorted(i for record in self.rounds for i in (record.queried or []))


@dataclass
class LoadedState:
    """What :func:`load_state` reads: the round state and the two torch files."""

    state: LoopState
    torch_rng: torch.Tensor
    finetuner: dict[str, Any]


def save_state(
    state_dir: pathlib.Path,
    state: LoopState,
    torch_rng: torch.Tensor,
    finetuner_state: dict[str, Any],
) -> None:
    """Write the three state files of the finished rounds of ``state`` (see the module docstring).

    Raises:
        ValueError: If ``state`` holds a value that is not valid JSON (NaN, infinity).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    rounds_done = state.rounds_done
    document = {
        "schema": STATE_SCHEMA,
        "rounds_done": rounds_done,
        "config": state.config,
        "labeled": state.labeled,
        "rng": state.rng,
        "rounds": [asdict(record) for record in state.rounds],
    }
    text = json.dumps(document, indent=1, allow_nan=False) + "\n"
    files: list[tuple[pathlib.Path, Callable[[BinaryIO], None]]] = [
        (
            state_dir / FINETUNER_NAME,
            lambda f: torch.save(
                {"rounds_done": rounds_done, "finetuner": finetuner_state}, f
            ),
        ),
        (
            state_dir / TORCH_RNG_NAME,
            lambda f: torch.save({"rounds_done": rounds_done, "cpu": torch_rng}, f),
        ),
        (state_dir / STATE_NAME, lambda f: f.write(text.encode("utf-8"))),
    ]
    temporaries: list[pathlib.Path] = []
    try:
        for path, write in files:
            temporaries.append(_write_temporary(path, write))
    except BaseException:
        for tmp in temporaries:
            tmp.unlink(missing_ok=True)
        raise
    for (path, _), tmp in zip(files, temporaries, strict=True):
        os.replace(tmp, path)
    _fsync_directory(state_dir)


def _load_torch(path: pathlib.Path, rounds_done: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing although {STATE_NAME} exists; the state is incomplete. "
            f"Remove {path.parent} to start the run again."
        )
    content = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(content, dict) or content.get("rounds_done") != rounds_done:
        found = content.get("rounds_done") if isinstance(content, dict) else None
        raise ValueError(
            f"{path} holds the state after {found} rounds, but {STATE_NAME} says {rounds_done}: "
            "the run was killed while it wrote its state. "
            f"Remove {path.parent} to start the run again."
        )
    return content


def load_state(state_dir: pathlib.Path) -> LoadedState | None:
    """Read the state of the finished rounds, or return ``None`` if the run has none yet.

    ``state.json`` is the commit marker: without it the run starts from the beginning, whatever
    torch files a killed first save left behind.

    Raises:
        ValueError: If the schema is not :data:`STATE_SCHEMA`, a record does not fit
            :class:`RoundRecord`, ``labeled`` disagrees with the queries of the rounds, or the
            torch files belong to another number of rounds.
        FileNotFoundError: If a torch file is missing.
    """
    path = state_dir / STATE_NAME
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != STATE_SCHEMA:
        raise ValueError(
            f"{path}: the schema is {document.get('schema')!r}, this code reads {STATE_SCHEMA!r}"
        )
    rounds: list[RoundRecord] = []
    for position, entry in enumerate(document["rounds"]):
        try:
            rounds.append(RoundRecord(**entry))
        except TypeError as error:
            raise ValueError(
                f"{path}: rounds[{position}] does not fit RoundRecord: {error}"
            ) from error
    state = LoopState(config=document["config"], rng=document["rng"], rounds=rounds)
    if document["rounds_done"] != state.rounds_done:
        raise ValueError(
            f"{path}: rounds_done is {document['rounds_done']} but the file holds "
            f"{state.rounds_done} rounds"
        )
    if document["labeled"] != state.labeled:
        raise ValueError(
            f"{path}: the list 'labeled' differs from the samples queried in the rounds"
        )
    rng = _load_torch(state_dir / TORCH_RNG_NAME, state.rounds_done)["cpu"]
    finetuner = _load_torch(state_dir / FINETUNER_NAME, state.rounds_done)["finetuner"]
    return LoadedState(state=state, torch_rng=rng, finetuner=finetuner)
