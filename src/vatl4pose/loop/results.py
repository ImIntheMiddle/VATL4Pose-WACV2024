"""The result file of a run: ``result.json``, schema ``vatl4pose.result/1``.

One file per video holds what the summaries need: for every evaluation (the 0 % model and the model
after each query) the labeled share, the AP and AR values with and without the replacement of the
labeled samples by their labels, the OSPA, and for every query the people chosen. Every per-round
list has one entry per evaluation round (``len(query_ratio) + 1`` entries, the last for the final
model) and has ``null`` where the round computed nothing:

``percentages``
    Share of labeled samples in percent when the round evaluated.
``performances``, ``performances_ann``
    The ten values of :data:`vatl4pose.eval.METRIC_KEYS` of the predictions of all samples, and of
    the same predictions after the labeled samples were replaced by their labels.
``ospa``, ``ospa_ann``
    The OSPA of both.
``query_list``
    ``{"Round0": [...], ...}``: the sorted sample indices queried in each round that has a query
    (every round but the last); ``[]`` for a query of size 0.
``score_means``
    ``{scorer: [...]}``: the mean raw score of the scorer over the unlabeled samples in the
    round. ``null`` in the final round, for an empty query, and for a forced query that takes all
    unlabeled samples (the strategy then scores nothing). Strategies without scorers give ``{}``.
``combine_weight``
    The mean local peak of the heatmaps of the unlabeled samples (the weight of the legacy code
    for the uncertainty term); ``null`` when nobody is unlabeled.
``moks_queried``
    The mean OKS of the people queried in the round, measured on the model before it learned
    from them; ``null`` without a query or for an empty one.
``timings``
    Seconds per step: ``infer``, ``eval``, ``select`` (the extras and the strategy), ``oracle``,
    ``retrain``, ``ae`` (the fine-tuning of the autoencoder) and ``total`` (the round without the
    write of its state). A step that did not run is 0.
``git_commit``
    The commit that the package was run from, or ``null`` when it was not run from a git checkout
    (an installed wheel carries no commit).
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import vatl4pose

from .state import RoundRecord, write_atomic

RESULT_SCHEMA = "vatl4pose.result/1"
RESULT_NAME = "result.json"

_COMMIT = re.compile(r"[0-9a-f]{40,64}")


def git_commit(package_dir: pathlib.Path | None = None) -> str | None:
    """The ``HEAD`` commit of the git checkout that ``package_dir`` lies in, or ``None``.

    ``None`` means that no directory from ``package_dir`` up to the root has a ``.git`` entry: the
    package is installed, not run from a checkout. Where a ``.git`` entry exists, git must answer;
    a failure raises instead of reporting ``None``.

    Args:
        package_dir: The directory to start from; the directory of the ``vatl4pose`` package if
            omitted.

    Raises:
        RuntimeError: If a ``.git`` entry exists and ``git rev-parse HEAD`` fails or does not
            print a commit hash.
    """
    start = (
        pathlib.Path(vatl4pose.__file__).resolve().parent
        if package_dir is None
        else package_dir
    )
    start = pathlib.Path(start).resolve()
    for folder in (start, *start.parents):
        if not (folder / ".git").exists():
            continue
        try:
            done = subprocess.run(
                ["git", "-C", str(folder), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"{folder} is a git checkout but the program 'git' cannot be started: {error}"
            ) from error
        commit = done.stdout.strip()
        if done.returncode != 0 or not _COMMIT.fullmatch(commit):
            raise RuntimeError(
                f"`git -C {folder} rev-parse HEAD` failed (exit code {done.returncode}): "
                f"{done.stderr.strip() or commit!r}"
            )
        return commit
    return None


def build_result(
    identity: Mapping[str, Any],
    scorers: Sequence[str],
    rounds: Sequence[RoundRecord],
) -> dict[str, Any]:
    """Assemble the result document of a finished run from its round records.

    Args:
        identity: ``dataset``, ``video_id``, ``estimator``, ``strategy``, ``params`` and
            ``n_samples`` of the run.
        scorers: The scorer names of the strategy; they key ``score_means``.
        rounds: The records of all rounds, in order.
    """
    return {
        "schema": RESULT_SCHEMA,
        "dataset": identity["dataset"],
        "video_id": identity["video_id"],
        "estimator": identity["estimator"],
        "strategy": identity["strategy"],
        "params": identity["params"],
        "percentages": [r.percentage for r in rounds],
        "performances": [r.performance for r in rounds],
        "performances_ann": [r.performance_ann for r in rounds],
        "ospa": [r.ospa for r in rounds],
        "ospa_ann": [r.ospa_ann for r in rounds],
        "query_list": {
            f"Round{i}": r.queried
            for i, r in enumerate(rounds)
            if r.queried is not None
        },
        "score_means": {
            name: [r.score_means[name] for r in rounds] for name in scorers
        },
        "combine_weight": [r.combine_weight for r in rounds],
        "moks_queried": [r.moks_queried for r in rounds],
        "timings": [r.timings for r in rounds],
        "n_samples": identity["n_samples"],
        "vatl4pose_version": vatl4pose.__version__,
        "git_commit": git_commit(),
    }


def write_result(run_dir: pathlib.Path, result: Mapping[str, Any]) -> pathlib.Path:
    """Write ``run_dir/result.json`` so that a reader never sees a half-written file.

    Raises:
        ValueError: If ``result`` holds NaN or infinity, which JSON cannot represent.
    """
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = run_dir / RESULT_NAME
    write_atomic(path, lambda f: f.write(text.encode("utf-8")))
    return path
