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
``ap_protocol``
    The name of the AP protocol (:data:`vatl4pose.eval.AP_PROTOCOL`): the 2023 protocol of the
    paper, whose detection area differs from the standard COCO keypoint AP. The AP values of
    ``performances`` and ``performances_ann`` are comparable with the paper, not with another COCO
    evaluation.
``git_commit``
    The commit that the package was run from, or ``null`` when it was not run from its own git
    checkout (an installed wheel carries no commit) or git could not name the commit (one warning
    says why).

A video in which some person has no ground truth (boxes without keypoints, see
:mod:`vatl4pose.data.annotations`) has no AP and no OSPA. ``performances``, ``performances_ann``,
``ospa`` and ``ospa_ann`` are ``null`` in every round of such a video, and ``vatl4pose-summarize``
lists it as "no ground truth". All other lists are computed as usual.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import vatl4pose

from ..eval import AP_PROTOCOL
from ..train._common import write_atomic
from .state import RoundRecord

RESULT_SCHEMA = "vatl4pose.result/1"
RESULT_NAME = "result.json"

_COMMIT = re.compile(r"[0-9a-f]{40,64}")
_WARNED: set[pathlib.Path] = set()
logger = logging.getLogger(__name__)


def _warn_once(root: pathlib.Path, reason: str) -> None:
    """Log that the commit of the checkout ``root`` is unknown, once per process and checkout."""
    if root in _WARNED:
        return
    _WARNED.add(root)
    logger.warning(
        "the git commit of the checkout %s is not recorded (null in result.json): %s",
        root,
        reason,
    )


def git_commit(package_dir: pathlib.Path | None = None) -> str | None:
    """The ``HEAD`` commit of the checkout that this package runs from, or ``None``.

    The package runs from its own checkout when ``package_dir`` is ``<root>/src/vatl4pose`` and
    ``<root>`` has a ``.git`` entry. Any other place gives ``None`` without a message: an installed
    package (``site-packages`` has no ``src`` parent), also one inside the ``.venv`` of another
    project, whose repository is never asked. When the checkout exists but git cannot answer for
    it (the program is missing, git refuses the directory, ``<root>/.git`` is no repository so
    that git would find another one above it), the result is ``None`` and one warning per process
    and checkout says why. The commit is information about the run, and a failure to read it must
    not end an hours-long run.

    Args:
        package_dir: The directory of the ``vatl4pose`` package; the running one if omitted.
    """
    package = (
        pathlib.Path(vatl4pose.__file__).parent
        if package_dir is None
        else pathlib.Path(package_dir)
    ).resolve()
    root = package.parent.parent
    if package.parent.name != "src" or not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        _warn_once(root, f"the program 'git' cannot be started ({error})")
        return None
    lines = done.stdout.splitlines()
    if done.returncode != 0 or len(lines) != 2 or not _COMMIT.fullmatch(lines[1]):
        reason = done.stderr.strip() or done.stdout.strip()
        _warn_once(root, f"`git rev-parse HEAD` failed (exit code {done.returncode}): {reason}")
        return None
    if pathlib.Path(lines[0]).resolve() != root:
        _warn_once(
            root,
            f"git answers for another repository, {lines[0]}, because {root}/.git is not "
            "a repository",
        )
        return None
    return lines[1]


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
        "ap_protocol": AP_PROTOCOL,
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
