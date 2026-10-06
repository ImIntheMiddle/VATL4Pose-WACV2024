"""The round state of the active-learning loop and its files.

A run that is killed, or that waits for human annotations, must continue with the round after the
last one it finished and give the result an uninterrupted run would have given. Everything that the
next round reads is therefore written when a round is finished, to a directory of ``state/``::

    state/current.json            points at the directory of the last finished round
    state/round_02/state.json     the records of rounds 0 to 2, the labeled samples, the settings
    state/round_02/torch_rng.pt   the state of the global torch random generator
    state/round_02/finetuner.pt   model, optimizer and schedule of the Finetuner
    state/pending_03/...          the query of round 3 while it waits for annotations

``state.json`` holds a record per finished round (:class:`RoundRecord`: what was queried, the
answers of the oracle, the metrics, the timings), the sorted list of labeled samples, the state of
the numpy random generator and the settings of the run (``config``), which a resume compares with
its own settings.

Atomic commit. :func:`save_state` writes the three files into a new directory ``round_NN`` (``NN``
is the index of the finished round) and only then replaces ``current.json`` with a pointer to it,
through a temporary file and ``os.replace``. :func:`load_state` reads the directory that the
pointer names and nothing else. A run killed at any moment therefore leaves either the old round or
the new round as the current one, never a mix; a directory written before the pointer moved is
ignored and replaced by the next save. The directories of older rounds are removed after the
pointer has moved.

Pending query. When the oracle has no answers yet (a human has to annotate), the round has already
chosen its people. :func:`save_pending` stores that choice, the predictions that were given to the
oracle and what the round has computed so far, so that the next ``run`` asks the oracle for the same
people without inference or selection (:func:`load_pending`). The directory ``pending_NN`` belongs
to round ``NN`` and is removed when the state of round ``NN`` is committed.

Everything is JSON, ``.npy`` or ``torch.save`` data of tensors, numbers, strings and plain
containers, so ``torch.load(weights_only=True)`` and ``np.load(allow_pickle=False)`` read it and no
pickled code is involved.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from ..train._common import fsync_directory, write_atomic, write_file

STATE_SCHEMA = "vatl4pose.state/1"
PENDING_SCHEMA = "vatl4pose.pending/1"
POINTER_NAME = "current.json"
STATE_NAME = "state.json"
TORCH_RNG_NAME = "torch_rng.pt"
FINETUNER_NAME = "finetuner.pt"
PENDING_NAME = "pending.json"
PREDICTIONS_NAME = "predictions.npy"

_ROUND_DIR = re.compile(r"round_(\d+)")
_PENDING_DIR = re.compile(r"pending_(\d+)")


@dataclass
class RoundRecord:
    """Everything one finished round produced. Lists hold plain Python numbers.

    Attributes:
        percentage: Share of labeled samples in percent when the round evaluated, before its query.
        performance: The ten AP and AR values of the predictions of all samples; ``None`` in a video
            in which some person has no ground truth.
        performance_ann: The same after the labeled samples' predictions were replaced by their
            labels; ``None`` as ``performance``.
        ospa: OSPA of the predictions; ``None`` as ``performance``.
        ospa_ann: OSPA after the replacement; ``None`` as ``performance``.
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
            ``ae``) and ``total``, the work of the whole round without the write of the state.
    """

    percentage: float
    performance: dict[str, float] | None
    performance_ann: dict[str, float] | None
    ospa: float | None
    ospa_ann: float | None
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
        rng: ``bit_generator.state`` of the numpy generator after the last finished round.
        rounds: One record per finished round.
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


@dataclass
class PendingQuery:
    """The query of a round that waits for annotations, and what the round has computed so far.

    Attributes:
        round_index: The round that waits.
        config: The settings of the run, as in :attr:`LoopState.config`.
        queried: Sorted sample indices the oracle was asked for.
        predictions: ``(N, K, 3)`` float64 predictions (x, y, peak value) that were given to the
            oracle, from the model before it learns from the answers.
        evaluation: The metrics of the round (percentage, AP and OSPA with and without labels,
            ``combine_weight``) as JSON data.
        score_means: Mean raw score of every scorer that scored, as in :class:`RoundRecord`.
        rng: ``bit_generator.state`` of the numpy generator after the selection.
        timings: Seconds of the steps that ran before the oracle was asked, and ``total``.
    """

    round_index: int
    config: dict[str, Any]
    queried: list[int]
    predictions: np.ndarray
    evaluation: dict[str, Any]
    score_means: dict[str, float]
    rng: dict[str, Any]
    timings: dict[str, float]


def round_dir_name(round_index: int) -> str:
    """Name of the directory that holds the state after round ``round_index``."""
    return f"round_{round_index:02d}"


def pending_dir_name(round_index: int) -> str:
    """Name of the directory that holds the pending query of round ``round_index``."""
    return f"pending_{round_index:02d}"


def _move_pointer(state_dir: pathlib.Path, pointer: dict[str, Any]) -> None:
    """Make ``current.json`` point at a finished round; this is the commit of a round."""
    text = json.dumps(pointer, indent=1) + "\n"
    write_atomic(state_dir / POINTER_NAME, lambda f: f.write(text.encode("utf-8")))


def _prune(state_dir: pathlib.Path, keep: str, rounds_done: int) -> None:
    """Remove the directories of older rounds and of the queries that are no longer pending."""
    for entry in sorted(state_dir.iterdir()):
        if not entry.is_dir():
            continue
        finished = _ROUND_DIR.fullmatch(entry.name)
        waiting = _PENDING_DIR.fullmatch(entry.name)
        if (finished and entry.name != keep) or (
            waiting and int(waiting.group(1)) < rounds_done
        ):
            shutil.rmtree(entry)


def save_state(
    state_dir: pathlib.Path,
    state: LoopState,
    torch_rng: torch.Tensor,
    finetuner_state: dict[str, Any],
) -> None:
    """Commit the state after the last round of ``state`` (see the module docstring).

    Raises:
        ValueError: If no round is finished, or ``state`` holds a value that is not valid JSON
            (NaN, infinity).
    """
    rounds_done = state.rounds_done
    if rounds_done < 1:
        raise ValueError("there is no finished round to save")
    name = round_dir_name(rounds_done - 1)
    document = {
        "schema": STATE_SCHEMA,
        "rounds_done": rounds_done,
        "config": state.config,
        "labeled": state.labeled,
        "rng": state.rng,
        "rounds": [asdict(record) for record in state.rounds],
    }
    text = json.dumps(document, indent=1, allow_nan=False) + "\n"
    state_dir.mkdir(parents=True, exist_ok=True)
    directory = state_dir / name
    if directory.exists():
        shutil.rmtree(directory)  # left by a run killed before the pointer moved
    directory.mkdir()
    write_file(
        directory / FINETUNER_NAME,
        lambda f: torch.save(
            {"rounds_done": rounds_done, "finetuner": finetuner_state}, f
        ),
    )
    write_file(
        directory / TORCH_RNG_NAME,
        lambda f: torch.save({"rounds_done": rounds_done, "cpu": torch_rng}, f),
    )
    write_file(directory / STATE_NAME, lambda f: f.write(text.encode("utf-8")))
    fsync_directory(directory)
    fsync_directory(state_dir)
    _move_pointer(
        state_dir,
        {"schema": STATE_SCHEMA, "directory": name, "rounds_done": rounds_done},
    )
    _prune(state_dir, name, rounds_done)


def _load_torch(path: pathlib.Path, rounds_done: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing although {POINTER_NAME} points at its directory; "
            f"the state is damaged. Remove {path.parent.parent} to start the run again."
        )
    content = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(content, dict) or content.get("rounds_done") != rounds_done:
        found = content.get("rounds_done") if isinstance(content, dict) else None
        raise ValueError(
            f"{path} holds the state after {found} rounds, but {STATE_NAME} says {rounds_done}: "
            f"the state is damaged. Remove {path.parent.parent} to start the run again."
        )
    return content


def load_state(state_dir: pathlib.Path) -> LoadedState | None:
    """Read the state of the last committed round, or return ``None`` if the run has none yet.

    ``current.json`` is the commit marker: without it the run starts from the beginning, whatever
    round directories a killed first save left behind.

    Raises:
        ValueError: If a schema is not the one of this code, the pointer or a record does not fit,
            ``labeled`` disagrees with the queries of the rounds, or the torch files belong to
            another number of rounds.
        FileNotFoundError: If a file that the pointer promises is missing.
    """
    pointer_path = state_dir / POINTER_NAME
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != STATE_SCHEMA:
        raise ValueError(
            f"{pointer_path}: the schema is {pointer.get('schema')!r}, this code reads "
            f"{STATE_SCHEMA!r}"
        )
    name = pointer["directory"]
    if not _ROUND_DIR.fullmatch(name):
        raise ValueError(f"{pointer_path}: {name!r} is not a round directory")
    directory = state_dir / name
    path = directory / STATE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing although {POINTER_NAME} points at {name}; the state is damaged. "
            f"Remove {state_dir} to start the run again."
        )
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
    if not (pointer["rounds_done"] == document["rounds_done"] == state.rounds_done):
        raise ValueError(
            f"{pointer_path} says {pointer['rounds_done']} finished rounds, {path} says "
            f"{document['rounds_done']} and holds {state.rounds_done}"
        )
    if document["labeled"] != state.labeled:
        raise ValueError(
            f"{path}: the list 'labeled' differs from the samples queried in the rounds"
        )
    rng = _load_torch(directory / TORCH_RNG_NAME, state.rounds_done)["cpu"]
    finetuner = _load_torch(directory / FINETUNER_NAME, state.rounds_done)["finetuner"]
    return LoadedState(state=state, torch_rng=rng, finetuner=finetuner)


def save_pending(state_dir: pathlib.Path, pending: PendingQuery) -> None:
    """Store the query of a round that waits for annotations.

    The predictions are written first and ``pending.json`` last, so ``pending.json`` is the commit
    marker of the query.

    Raises:
        ValueError: If ``pending`` holds a value that is not valid JSON, or the predictions are not
            a float64 array of shape ``(N, K, 3)``.
    """
    predictions = pending.predictions
    if (
        predictions.ndim != 3
        or predictions.shape[2] != 3
        or predictions.dtype != np.float64
    ):
        raise ValueError(
            f"predictions: expected a float64 array of shape (N, K, 3), got {predictions.dtype} "
            f"{predictions.shape}"
        )
    document = {
        "schema": PENDING_SCHEMA,
        "round_index": pending.round_index,
        "config": pending.config,
        "queried": pending.queried,
        "evaluation": pending.evaluation,
        "score_means": pending.score_means,
        "rng": pending.rng,
        "timings": pending.timings,
    }
    text = json.dumps(document, indent=1, allow_nan=False) + "\n"
    directory = state_dir / pending_dir_name(pending.round_index)
    directory.mkdir(parents=True, exist_ok=True)
    fsync_directory(state_dir)
    write_atomic(
        directory / PREDICTIONS_NAME,
        lambda f: np.save(f, predictions, allow_pickle=False),
    )
    write_atomic(directory / PENDING_NAME, lambda f: f.write(text.encode("utf-8")))


def load_pending(state_dir: pathlib.Path, round_index: int) -> PendingQuery | None:
    """The stored query of round ``round_index``, or ``None`` if it has none.

    Raises:
        ValueError: If the file is of another schema or belongs to another round.
        FileNotFoundError: If ``pending.json`` exists without its predictions.
    """
    directory = state_dir / pending_dir_name(round_index)
    path = directory / PENDING_NAME
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != PENDING_SCHEMA:
        raise ValueError(
            f"{path}: the schema is {document.get('schema')!r}, this code reads {PENDING_SCHEMA!r}"
        )
    if document["round_index"] != round_index:
        raise ValueError(
            f"{path} belongs to round {document['round_index']}, not to round {round_index}"
        )
    predictions_path = directory / PREDICTIONS_NAME
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"{predictions_path} is missing although {path} exists; remove {directory} to ask "
            "for the query again."
        )
    predictions = np.load(predictions_path, allow_pickle=False)
    if (
        predictions.ndim != 3
        or predictions.shape[2] != 3
        or predictions.dtype != np.float64
    ):
        raise ValueError(
            f"{predictions_path}: expected a float64 array of shape (N, K, 3), got "
            f"{predictions.dtype} {predictions.shape}"
        )
    return PendingQuery(
        round_index=round_index,
        config=document["config"],
        queried=document["queried"],
        predictions=predictions,
        evaluation=document["evaluation"],
        score_means=document["score_means"],
        rng=document["rng"],
        timings=document["timings"],
    )
