"""Logging of the runs: ``log.jsonl`` always, W&B on request.

The loop calls the round logger once for every finished round (``ActiveLearningLoop(on_round=...)``).
The JSON log is one line per round in ``<run_dir>/log.jsonl``; a resumed run appends the rounds it
finishes. ``logger=wandb`` writes the same numbers to a W&B run named
``<MMDD>_<dataset>_<strategy>_<video_id>`` as well.

``vatl4pose-train`` writes its own ``log.jsonl`` and calls the epoch logger once for every logged
epoch. With ``logger=wandb`` that logger writes the numbers of the epoch to a W&B run named
``<MMDD>_<dataset>_<run.name>``.

The ``wandb`` package is imported only for ``logger=wandb``.
"""

from __future__ import annotations

import datetime
import json
import pathlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

from vatl4pose.config import WandbSection
from vatl4pose.loop.state import RoundRecord

LOG_NAME = "log.jsonl"
RoundLogger = Callable[[int, RoundRecord], None]
EpochLogger = Callable[[dict[str, Any]], None]

_WANDB_MISSING = (
    "logger=wandb needs the 'wandb' package, which is an optional extra. Install it with "
    "`uv sync --extra wandb` (a source checkout) or `pip install 'vatl4pose[wandb]'`, or use "
    "logger=json."
)


def wandb_run_name(
    day: datetime.date,
    dataset: str,
    strategy: str,
    video_id: str,
    tag: Optional[str] = None,
) -> str:
    """``<MMDD>_<dataset>_<strategy>_<video_id>``, with ``_<tag>`` appended if there is a tag."""
    parts = [day.strftime("%m%d"), dataset, strategy, video_id]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def train_wandb_run_name(day: datetime.date, dataset: str, run_name: str) -> str:
    """``<MMDD>_<dataset>_<run.name>``, the name of the W&B run of a training."""
    return "_".join([day.strftime("%m%d"), dataset, run_name])


def round_line(video_id: str, round_index: int, record: RoundRecord) -> dict[str, Any]:
    """The numbers of one round as one flat-enough JSON object (no label arrays)."""
    return {
        "video_id": video_id,
        "round": round_index,
        "percentage": record.percentage,
        "performance": record.performance,
        "performance_ann": record.performance_ann,
        "ospa": record.ospa,
        "ospa_ann": record.ospa_ann,
        "combine_weight": record.combine_weight,
        "moks_queried": record.moks_queried,
        "stopping": record.stopping,
        "n_queried": None if record.queried is None else len(record.queried),
        "timings": record.timings,
    }


def _wandb_metrics(
    line: dict[str, Any], skip: tuple[str, ...] = ("video_id", "round")
) -> dict[str, float]:
    """The scalars of a log line, with the nested dictionaries flattened to ``a/b/c`` keys.

    The keys in ``skip`` and the values that are ``None`` are left out.
    """
    metrics: dict[str, float] = {}
    for key, value in line.items():
        if key in skip or value is None:
            continue
        if isinstance(value, dict):
            for inner, number in _wandb_metrics(value, skip=()).items():
                metrics[f"{key}/{inner}"] = number
        else:
            metrics[key] = value
    return metrics


def _import_wandb() -> Any:
    try:
        import wandb
    except ImportError as error:
        raise ImportError(_WANDB_MISSING) from error
    return wandb


@contextmanager
def _wandb_run(
    logger: str,
    wandb_cfg: WandbSection,
    name: str,
    group: str,
    config: dict[str, Any],
    output_root: pathlib.Path,
) -> Iterator[Any]:
    """Yield the W&B run of ``logger=wandb`` (``None`` for ``json``) and close it on exit.

    A run that leaves the block with an error is finished with exit code 1.

    Raises:
        ImportError: If ``logger`` is ``wandb`` and the package is not installed.
        ValueError: If ``logger`` is neither ``json`` nor ``wandb``.
    """
    if logger not in ("json", "wandb"):
        raise ValueError(f"logger must be 'json' or 'wandb', got {logger!r}")
    run = None
    if logger == "wandb":
        wandb = _import_wandb()
        output_root.mkdir(parents=True, exist_ok=True)
        # W&B puts its files in <dir>/wandb, that is <output_root>/wandb
        run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=name,
            group=group,
            config=config,
            mode=wandb_cfg.mode,
            dir=str(output_root),
        )
    try:
        yield run
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()


@contextmanager
def open_round_log(
    logger: str,
    wandb_cfg: WandbSection,
    run_dir: pathlib.Path,
    name: str,
    group: str,
    config: dict[str, Any],
    output_root: pathlib.Path,
    video_id: str,
) -> Iterator[RoundLogger]:
    """Yield the round logger of one video and close its W&B run on exit.

    Args:
        logger: ``json`` or ``wandb``.
        wandb_cfg: Project, entity and mode of the W&B run.
        run_dir: Directory of the run; ``log.jsonl`` goes here.
        name: Name of the W&B run.
        group: Group of the W&B run (the experiment name).
        config: Settings stored with the W&B run, including the git commit of the package; it is
            used for ``logger=wandb`` only.
        output_root: ``paths.output_root``; W&B keeps its files in ``<output_root>/wandb``
            (it makes the ``wandb`` directory below the directory it is given).
        video_id: Written into every line of the log.

    Raises:
        ImportError: If ``logger`` is ``wandb`` and the package is not installed.
        ValueError: If ``logger`` is neither ``json`` nor ``wandb``.
    """
    with _wandb_run(logger, wandb_cfg, name, group, config, output_root) as run:
        path = run_dir / LOG_NAME

        def log_round(round_index: int, record: RoundRecord) -> None:
            line = round_line(video_id, round_index, record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, allow_nan=False) + "\n")
            if run is not None:
                run.log(_wandb_metrics(line), step=round_index)

        yield log_round


@contextmanager
def open_epoch_log(
    logger: str,
    wandb_cfg: WandbSection,
    name: str,
    group: str,
    config: dict[str, Any],
    output_root: pathlib.Path,
) -> Iterator[Optional[EpochLogger]]:
    """Yield the epoch logger of a training, or ``None`` for ``logger=json``, and close the W&B run.

    The training writes ``log.jsonl`` itself. The epoch logger takes the log line of an epoch (a
    dictionary with ``epoch`` and numbers) and sends the numbers to W&B at the step ``epoch``.

    Args:
        logger: ``json`` or ``wandb``.
        wandb_cfg: Project, entity and mode of the W&B run.
        name: Name of the W&B run (:func:`train_wandb_run_name`).
        group: Group of the W&B run.
        config: Settings stored with the W&B run; used for ``logger=wandb`` only.
        output_root: ``paths.output_root``; W&B keeps its files in ``<output_root>/wandb``.

    Raises:
        ImportError: If ``logger`` is ``wandb`` and the package is not installed.
        ValueError: If ``logger`` is neither ``json`` nor ``wandb``.
    """
    with _wandb_run(logger, wandb_cfg, name, group, config, output_root) as run:
        if run is None:
            yield None
            return

        def log_epoch(line: dict[str, Any]) -> None:
            run.log(_wandb_metrics(line, skip=("epoch",)), step=line["epoch"])

        yield log_epoch
