"""The result file of a run: ``result.json``, schema ``vatl4pose.result/1``.

One file per video holds what the summaries need: for every evaluation (the 0 % estimator and the
estimator after each query) the share of labeled people, the AP and AR values with and without the
replacement of the labeled people by their labels, the OSPA, and for every query the selected
people. Every per-round list has one entry per evaluation round (``len(query_ratio) + 1`` entries,
the last for the final estimator) and has ``null`` where the round computed nothing:

``percentages``
    Share of labeled people in percent when the round evaluated.
``performances``, ``performances_ann``
    The ten values of :data:`vatl4pose.eval.METRIC_KEYS` of the predictions of all people, and of
    the same predictions after the labeled people were replaced by their labels.
``ospa``, ``ospa_ann``
    The OSPA of both.
``query_list``
    ``{"Round0": [...], ...}``: the sorted sample indices selected in each round that has a query
    (every round but the last); ``[]`` for a query of size 0.
``score_means``
    ``{scorer: [...]}``: the mean raw score of the scorer over the unlabeled people in the
    round. ``null`` in the final round, for an empty query, and for a forced query that takes all
    unlabeled people (the strategy then scores nothing). Strategies without scorers give ``{}``.
``combine_weight``
    The mean local peak of the heatmaps of the unlabeled people in the round; ``null`` when nobody
    is unlabeled.
``moks_queried``, ``stopping``, ``timings``
    As in :class:`~vatl4pose.loop.state.RoundRecord`, one entry per round.
``ap_protocol``
    The name of the AP protocol (:data:`vatl4pose.eval.AP_PROTOCOL`): the AP protocol of the
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

The poses of everybody after the last round
-------------------------------------------
The last round evaluates the estimator after its last retraining on every person of the video,
with or without ground truth. The loop writes that inference next to ``result.json`` as two COCO
keypoint results lists (the format of ``COCO.loadRes``), so that a run on your own video
(``oracle=coco_file``) ends with a pose for every person:

``predictions.json``
    The prediction for every person.
``predictions_ann.json``
    The same list with the labels in place of the predictions for the labeled people. These are
    the poses that ``performances_ann`` evaluates: a labeled person whose label shows no joint
    keeps the prediction.

Each list has one entry per person in the sample order of the video, so entry ``i`` belongs to
the sample index ``i`` of ``query_list``. An entry holds ``image_id`` and ``id`` (the image id and
the annotation id of the annotation file, to join the entry with it), ``track_id`` (the track of
the person as a string, as the cache holds it; the PoseTrack21 ids carry the video id as a
prefix), ``category_id`` (always 1), ``keypoints`` and ``score``. ``keypoints`` is the flat list
``x, y, c`` of every joint in the order of the keypoint spec, ``3 * K`` numbers. ``c`` is the peak
value of the heatmap for an estimator joint and the visibility (0 or 1) for a label joint.
``score`` is the pose score of the ``K`` peak values (:func:`vatl4pose.eval.pose_score`); the
label of a person does not change it. ``x`` and ``y`` are the pixel coordinates that
``performances`` evaluates.

For a ``panorama`` dataset (JRDB-Pose) the coordinates keep the convention of the evaluation: the
unwrapped coordinates of the 360-degree strip next to the box of the person (see
:mod:`vatl4pose.data.annotations`). A joint of a person on the seam can have ``x`` below 0 or above
``W - 1``, and adding or subtracting the frame width ``W`` gives its position in the frame. The
labels of ``predictions_ann.json`` follow the same convention.

The two files are written one after the other, each through a temporary file and a rename, before
the state of the last round is saved, so a run that stops in between writes both again on resume.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import vatl4pose

from ..eval import AP_PROTOCOL
from ..train._common import write_atomic
from .state import RoundRecord

RESULT_SCHEMA = "vatl4pose.result/1"
RESULT_NAME = "result.json"
PREDICTIONS_NAME = "predictions.json"
PREDICTIONS_ANN_NAME = "predictions_ann.json"
_PERSON_CATEGORY = 1  # the one category of the COCO keypoint results

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


def git_commit() -> str | None:
    """The ``HEAD`` commit of the checkout that this package runs from, or ``None``.

    A package that is not at ``<root>/src/vatl4pose`` with a ``.git`` entry in ``<root>`` (an
    installed package, for example) gives ``None`` without a message. If the checkout exists but
    git cannot name its commit, the result is ``None`` and one warning per process and checkout
    says why; a failure to read the commit must not end a long run.
    """
    package = pathlib.Path(vatl4pose.__file__).parent.resolve()
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
        "stopping": [r.stopping for r in rounds],
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


def build_predictions(
    keypoints: np.ndarray,
    score: np.ndarray,
    ann_id: np.ndarray,
    image_id: np.ndarray,
    track_id: Sequence[str],
) -> list[dict[str, Any]]:
    """The COCO keypoint results list of the people of a video, one entry per person in order.

    Args:
        keypoints: ``(N, K, 3)`` float x, y and the third value of every joint (the peak value of
            an estimator joint, the visibility of a label joint), in image pixels.
        score: ``(N,)`` float pose score of every person.
        ann_id: ``(N,)`` integer annotation id of every person.
        image_id: ``(N,)`` integer image id of every person.
        track_id: ``N`` track ids as strings.

    Returns:
        The entries described in the module docstring.
    """
    n = len(keypoints)
    assert (
        keypoints.ndim == 3
        and keypoints.shape[2] == 3
        and np.issubdtype(keypoints.dtype, np.floating)
    ), (
        f"keypoints: expected a float array of shape (N, K, 3), got {keypoints.dtype} "
        f"{keypoints.shape}"
    )
    assert score.shape == (n,) and np.issubdtype(score.dtype, np.floating), (
        f"score: expected a float array of shape ({n},), got {score.dtype} {score.shape}"
    )
    for name, ids in (("ann_id", ann_id), ("image_id", image_id)):
        assert ids.shape == (n,) and np.issubdtype(ids.dtype, np.integer), (
            f"{name}: expected an integer array of shape ({n},), got {ids.dtype} {ids.shape}"
        )
    assert len(track_id) == n, f"track_id: expected {n} entries, got {len(track_id)}"
    return [
        {
            "image_id": int(image_id[i]),
            "category_id": _PERSON_CATEGORY,
            "keypoints": keypoints[i].reshape(-1).tolist(),
            "score": float(score[i]),
            "id": int(ann_id[i]),
            "track_id": str(track_id[i]),
        }
        for i in range(n)
    ]


def write_predictions(
    run_dir: pathlib.Path,
    predictions: Sequence[Mapping[str, Any]],
    predictions_ann: Sequence[Mapping[str, Any]],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``predictions.json`` and ``predictions_ann.json`` into ``run_dir``.

    Both lists are serialized before either file is written, and every file goes through a
    temporary file and a rename, so a reader never sees a half-written file.

    Raises:
        ValueError: If a list holds NaN or infinity, which JSON cannot represent.
    """
    documents = [
        (PREDICTIONS_NAME, json.dumps(predictions, allow_nan=False) + "\n"),
        (PREDICTIONS_ANN_NAME, json.dumps(predictions_ann, allow_nan=False) + "\n"),
    ]
    paths = []
    for name, text in documents:
        path = run_dir / name
        write_atomic(path, lambda f, text=text: f.write(text.encode("utf-8")))
        paths.append(path)
    return paths[0], paths[1]
