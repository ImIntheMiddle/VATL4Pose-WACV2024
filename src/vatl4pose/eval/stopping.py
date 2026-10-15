"""The stopping points of Table 5 of the paper, read from the result of one video.

After every query three criteria are judged with the target OKS θ (``retrain.finish_acc``), all
on the OKS that the pose estimator reaches before its retraining:

* ``oracle`` (the actual stop): every person of the video, labeled or not, has OKS >= θ.
* ``min_error`` (SC_Min of the paper): the mean OKS of the people queried in the round is
  >= θ.
* ``sc_all`` (SC_All of the paper): every labeled person, the people queried in the round
  included, has OKS >= θ.

A criterion stops the annotation at the share of labeled people after the query of the first round
that meets it, ``100 * labeled / n`` (``labeled`` counts the people queried so far, the round
included). A criterion that is never met stops at 100 %. :func:`stop_points` computes these numbers
for one video from a ``result.json`` of ``vatl4pose-atl`` run with ``loop.stopping``. It ignores the
rounds after the round of the actual stop and reads the AP@θ (``performances_ann``) of the round
whose share of labeled people is nearest to the stop (the first one on a tie).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coco_ap import METRIC_KEYS

CRITERIA = ("oracle", "min_error", "sc_all")
"""The criteria of Table 5: the actual stop (key ``oracle``), SC_Min and SC_All."""
_WITH_AP = ("min_error", "sc_all")  # the criteria whose AP at the stop is read


@dataclass(frozen=True)
class StopPoints:
    """Where the three criteria stop the annotation of one video.

    Attributes:
        theta: The target OKS θ of the judgments.
        labeled: Share of labeled people, in percent, at the stop of every criterion; 100.0 when the
            criterion is not met in any round up to the round of the actual stop.
        ap: AP@θ in percent at the stop, for ``min_error`` and ``sc_all``; ``None`` when
            ``performances_ann`` holds no AP at that threshold (see :func:`ap_key`).
    """

    theta: float
    labeled: dict[str, float]
    ap: dict[str, float | None]


def ap_key(theta: float) -> str | None:
    """The key of ``performances_ann`` that holds AP@θ (``"AP .6"`` for 0.6).

    Returns:
        The key, or ``None`` when the result holds no AP at that threshold (it has 0.5, 0.6, 0.7,
        0.75, 0.8 and 0.95).
    """
    key = f"AP {theta:g}".replace("AP 0.", "AP .")
    return key if key in METRIC_KEYS else None


def stop_points(result: dict[str, Any]) -> StopPoints | None:
    """The stops of the three criteria of a video, or ``None`` when its rounds were not judged.

    Args:
        result: The ``result.json`` of a video with ground truth, run with ``loop.stopping``. It
            needs ``percentages``, ``performances_ann``, ``query_list``, ``moks_queried``,
            ``stopping`` and ``n_samples``.

    Returns:
        The stop points, or ``None`` when the result has no ``stopping`` record or all of it is
        ``null`` (the video ran without ``loop.stopping``).

    Raises:
        ValueError: If a round that queried people has no judgment, a judgment has no ``oracle``
            record (the video has no ground truth), the judgments use different values of
            the target OKS θ, or ``performances_ann`` has an entry without AP (a video without
            ground truth).
    """
    judgments = result.get("stopping")
    if judgments is None or all(j is None for j in judgments):
        return None
    queries = result["query_list"]
    sizes = [len(queries[f"Round{r}"]) for r in range(len(queries))]
    n = int(result["n_samples"])
    thetas = {float(j["theta"]) for j in judgments if j is not None}
    if len(thetas) != 1:
        raise ValueError(
            f"the stopping judgments use several values of the target OKS θ: {sorted(thetas)}"
        )
    (theta,) = thetas

    met: dict[str, list[bool]] = {c: [False] * len(sizes) for c in CRITERIA}
    for r, size in enumerate(sizes):
        if size == 0:  # nobody queried, nothing to judge
            continue
        status = judgments[r]
        if status is None:
            raise ValueError(
                f"round {r} queried {size} people but has no stopping judgment"
            )
        oracle = status.get("oracle")
        if oracle is None:
            raise ValueError(
                f"the stopping judgment of round {r} has no oracle record: run the video again "
                "with loop.stopping on a video whose people all have ground truth"
            )
        moks = result["moks_queried"][r]
        met["oracle"][r] = bool(oracle["met"])
        met["min_error"][r] = moks is not None and moks >= theta
        met["sc_all"][r] = bool(status["met"])

    oracle_round = next((r for r, hit in enumerate(met["oracle"]) if hit), None)
    last = len(sizes) - 1 if oracle_round is None else oracle_round
    labeled: dict[str, float] = {}
    cumulative = np.cumsum(sizes)
    for criterion in CRITERIA:
        stop = 100.0
        for r in range(last + 1):
            share = (int(cumulative[r]) / n) * 100
            if met[criterion][r] and share < stop:
                stop = share
        labeled[criterion] = stop

    performances = result["performances_ann"]
    if any(p is None for p in performances):
        raise ValueError(
            "the result has no AP (a video without ground truth has no Table 5 row)"
        )
    if oracle_round is not None:
        # the annotation ends in the round of the actual stop, so later rounds repeat that round
        performances = performances[: oracle_round + 1] + [
            performances[oracle_round]
        ] * (len(performances) - oracle_round - 1)
    key = ap_key(theta)
    percentages = np.asarray(result["percentages"], dtype=np.float64)
    ap: dict[str, float | None] = {}
    for criterion in _WITH_AP:
        index = int(np.abs(percentages - labeled[criterion]).argmin())
        ap[criterion] = None if key is None else 100.0 * float(performances[index][key])
    return StopPoints(theta=theta, labeled=labeled, ap=ap)
