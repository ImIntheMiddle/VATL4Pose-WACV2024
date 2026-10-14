"""``vatl4pose-summarize``: the ALC table and the learning curves of finished runs.

    vatl4pose-summarize experiment=wacv2024_posetrack21 [summarize.ids_file=my_videos.txt]

Reads ``<output_root>/<run.name>/<estimator>/<strategy>/<video>/result.json`` of every strategy
directory and every video of the id list (the dataset's, or ``summarize.ids_file``), and writes to
``<output_root>/<run.name>/summary/<estimator>[_<ids file name>]/``:

* ``summary.json`` and ``summary.md``: for each strategy the mean ALC over its videos of ``AP .6``
  and ``AP .75`` (``performances_ann``) and of the OSPA (``ospa_ann``), the number of videos, and
  ``percent_range``, the range of labeled people in percent that the curves of all its videos
  cover. The ALC is 100 times the area under the curve over the share of labeled people, so it is
  at most 100 for a run that labels all people. A run that stops at 20 % labeled people (a budget
  run, ``loop.query_ratio`` ending below 1) covers only that range, and the ALC of a perfect curve
  is 20; compare ALC values of runs with the same range only. A video without ``result.json`` is counted as missing; a video whose
  curve of a metric holds ``-1`` (no ground truth in the area range) is left out of that metric and
  counted as skipped, as in the 2023 summaries. A video whose metrics are ``null`` (some person has
  no ground-truth keypoints, so the loop computed no AP and no OSPA) is listed as "no ground truth"
  and left out of every mean and curve; it still counts in ``n_videos``.
* ``learning_curves_<metric>.png``: the mean over the videos on the 0 to 100 % label grid, one line
  per strategy.
* ``stopping`` in ``summary.json`` and the stopping table of ``summary.md`` (Table 5 of the paper)
  for a strategy whose videos ran with ``loop.stopping=true``: for each of the three criteria of
  :mod:`vatl4pose.eval.stopping` (SC_Min, SC_All and the oracle stop) the mean over the videos of
  the share of labeled people at the stop and, for SC_Min and SC_All, of the AP at OKS θ
  (``retrain.finish_acc``) at the stop, as ``scripts/detailed_result.py`` of the 2023 code
  computes them. A video that never meets a criterion stops at 100 % and counts in the means.
  The oracle stop and the AP come from the ground truth, so a video without it is left out.

The plots are made here, from the saved results, and nowhere else.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig

from vatl4pose.config import (
    PathsSection,
    RunSection,
    SummarizeSection,
    section,
)
from vatl4pose.eval import (
    CRITERIA,
    PERCENT_GRID,
    ap_key,
    interpolate_curve,
    stop_points,
)
from vatl4pose.loop.results import RESULT_NAME
from vatl4pose.train._common import write_atomic

from ._common import (
    alc_percent,
    has_ground_truth,
    hydra_command,
    load_result,
    metric_curve,
    output_root,
    read_ids,
    require_unsharded,
    run_name,
)

METRICS = ("AP .6", "AP .75", "OSPA")
"""The summarized metrics: ALC of two AP thresholds (``performances_ann``) and of the OSPA."""


def collect(base: pathlib.Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Summarize every strategy directory of ``base``.

    Returns:
        ``{strategy: {"n_videos", "n_missing", "no_ground_truth", "percent_range", metric:
        {"alc_mean", "n", "n_skipped"}, "curves": {metric: (percent grid, mean curve)}}}``.
        ``no_ground_truth`` lists the ids of the videos whose result has no metrics; the means
        cover the other videos. ``percent_range`` is ``[low, high]``, the range of labeled people
        in percent that the results of all found videos cover (``None`` without a result).
        ``curves`` is for the plots and is dropped from the JSON. ``stopping`` is ``None`` unless
        the videos with ground truth ran with ``loop.stopping``; otherwise it is the dict of
        :func:`_stopping_summary`.
    """
    if not base.is_dir():
        raise FileNotFoundError(f"no results below {base}; run vatl4pose-atl first")
    strategies = sorted(p.name for p in base.iterdir() if p.is_dir())
    if not strategies:
        raise FileNotFoundError(f"{base} holds no strategy directory")
    summary: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        found: dict[str, dict[str, Any]] = {}
        for video_id in ids:
            run_dir = base / strategy / video_id
            if (run_dir / RESULT_NAME).is_file():
                found[video_id] = load_result(run_dir)
        no_ground_truth = [v for v, r in found.items() if not has_ground_truth(r)]
        results = [r for v, r in found.items() if v not in no_ground_truth]
        entry: dict[str, Any] = {
            "n_videos": len(found),
            "n_missing": len(ids) - len(found),
            "no_ground_truth": no_ground_truth,
            "percent_range": _covered_range(list(found.values())),
            "stopping": _stopping_summary(
                [(v, r) for v, r in found.items() if v not in no_ground_truth]
            ),
            "curves": {},
        }
        for metric in METRICS:
            alcs = [alc_percent(result, metric) for result in results]
            valid = [a for a in alcs if a is not None]
            entry[metric] = {
                "alc_mean": float(np.mean(valid)) if valid else None,
                "n": len(valid),
                "n_skipped": len(alcs) - len(valid),
            }
            usable = [r for r, a in zip(results, alcs, strict=True) if a is not None]
            if usable:
                entry["curves"][metric] = _mean_curve(usable, metric)
        summary[strategy] = entry
    return summary


def _stopping_summary(
    videos: list[tuple[str, dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """Table 5 of the paper for the videos of one strategy, or ``None`` without judged videos.

    Returns:
        ``{"theta", "ap_key", "n", criterion: {"labeled_mean", "n_never", "ap_mean"}}`` for
        ``oracle``, ``min_error`` and ``sc_all``. ``labeled_mean`` is the mean share of labeled
        people at the stop in percent and ``n_never`` the number of videos that stop at 100 %.
        ``ap_mean`` is the mean AP at OKS ``theta`` in percent at the stop (``min_error`` and
        ``sc_all`` only), ``None`` when the result holds no AP at ``theta`` (``ap_key`` is then
        ``None`` too).

    Raises:
        ValueError: If only some of the videos were judged, or they use different values of theta.
    """
    points = [(video, stop_points(result)) for video, result in videos]
    unjudged = [video for video, point in points if point is None]
    stops = [point for _, point in points if point is not None]
    if not stops:
        return None
    if unjudged:
        raise ValueError(
            f"{len(unjudged)} video(s) have no stopping judgment ({', '.join(unjudged)}) and "
            f"{len(stops)} have; run all of them with loop.stopping=true, or leave some out with "
            "summarize.ids_file"
        )
    thetas = sorted({point.theta for point in stops})
    if len(thetas) != 1:
        raise ValueError(
            f"the videos were judged with several values of theta: {thetas}; summarize one "
            "value (one run.name per theta)"
        )
    theta = thetas[0]
    summary: dict[str, Any] = {"theta": theta, "ap_key": ap_key(theta), "n": len(stops)}
    for criterion in CRITERIA:
        labeled = [point.labeled[criterion] for point in stops]
        row: dict[str, Any] = {
            "labeled_mean": float(np.mean(labeled)),
            "n_never": sum(share == 100.0 for share in labeled),
        }
        if criterion in stops[0].ap:
            aps = [point.ap[criterion] for point in stops]
            row["ap_mean"] = None if None in aps else float(np.mean(aps))
        summary[criterion] = row
    return summary


def _covered_range(results: list[dict[str, Any]]) -> Optional[list[float]]:
    """``[low, high]``: the largest first and the smallest last label percentage of ``results``."""
    if not results:
        return None
    return [
        float(max(r["percentages"][0] for r in results)),
        float(min(r["percentages"][-1] for r in results)),
    ]


def _mean_curve(
    results: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over ``results`` on the label grid they all cover (0.1 % steps)."""
    low = max(r["percentages"][0] for r in results)
    high = min(r["percentages"][-1] for r in results)
    grid = PERCENT_GRID[(PERCENT_GRID >= low) & (PERCENT_GRID <= high)]
    curves = [
        interpolate_curve(
            r["percentages"], 100.0 * np.asarray(metric_curve(r, metric)), grid
        )
        for r in results
    ]
    return grid, np.mean(curves, axis=0)


def _markdown(title: str, summary: dict[str, dict[str, Any]], n_ids: int) -> str:
    def cell(entry: dict[str, Any], metric: str) -> str:
        value = entry[metric]["alc_mean"]
        return "-" if value is None else f"{value:.2f}"

    def covered(entry: dict[str, Any]) -> str:
        if entry["percent_range"] is None:
            return "-"
        low, high = entry["percent_range"]
        return f"{low:.4g}-{high:.4g}"

    lines = [
        f"# {title}",
        "",
        "Mean ALC over the videos: 100 times the area under the curve of the metric over the share "
        "of labeled people (`performances_ann` for AP, `ospa_ann` for the OSPA, so a lower OSPA is "
        "better). `labeled %` is the range of labeled people that the curves cover. The ALC is at "
        "most the range, which is 100 for a run that labels all people and less for a budget run "
        "that stops earlier, so compare ALC values of runs with the same range only.",
        "",
        "| strategy | videos | labeled % | " + " | ".join(f"ALC {m}" for m in METRICS) + " |",
        "|---|---|---|" + "---|" * len(METRICS),
    ]
    for strategy, entry in summary.items():
        # a video without ground truth has a result but no metrics, so it is not evaluated
        without = len(entry["no_ground_truth"])
        evaluated = f"{entry['n_videos'] - without} / {n_ids}"
        if without:
            evaluated += f" ({without} no GT)"
        row = [strategy, evaluated, covered(entry)] + [cell(entry, m) for m in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    notes = []
    for strategy, entry in summary.items():
        if entry["n_missing"]:
            notes.append(
                f"- {strategy}: {entry['n_missing']} video(s) have no result.json"
            )
        if entry["no_ground_truth"]:
            notes.append(
                f"- {strategy}: {len(entry['no_ground_truth'])} video(s) have no ground truth "
                f"({', '.join(entry['no_ground_truth'])}); they are left out of the means"
            )
        for metric in METRICS:
            if entry[metric]["n_skipped"]:
                notes.append(
                    f"- {strategy}: {entry[metric]['n_skipped']} video(s) left out of {metric} "
                    "(a value of -1 in the curve)"
                )
    if notes:
        lines += ["", *notes]
    lines += _stopping_markdown(summary)
    return "\n".join(lines) + "\n"


_CRITERION_LABELS = {
    "min_error": "SC_Min (mean OKS of the queried people)",
    "sc_all": "SC_All (every annotated person)",
    "oracle": "actual (every person of the video)",
}


def _stopping_markdown(summary: dict[str, dict[str, Any]]) -> list[str]:
    """The stopping table (Table 5 of the paper) of the strategies that have one; ``[]`` if none."""
    judged = {s: e["stopping"] for s, e in summary.items() if e["stopping"] is not None}
    if not judged:
        return []
    lines = [
        "",
        "## Stopping criteria",
        "",
        "Table 5 of the paper, with `loop.stopping=true` and θ = `retrain.finish_acc`. A criterion "
        "stops the annotation at the share of labeled people after the first round that meets "
        "it; a video that never meets it stops at 100 % and counts in the means. SC_Min: the "
        "mean OKS of the people queried in the round reaches θ. SC_All: every person annotated "
        "so far reaches θ. Actual: every person of the video reaches θ, and the rounds after it "
        "are ignored, as the 2023 runs ended there. AP@θ is the AP at OKS θ (`performances_ann`) "
        "in the round nearest to the stop, in percent; the means run over the videos.",
        "",
        "| strategy | θ | videos | criterion | AP@θ (%) | stopped (%) | stopped at 100 % |",
        "|---|---|---|---|---|---|---|",
    ]
    notes = []
    for strategy, entry in judged.items():
        for criterion in ("min_error", "sc_all", "oracle"):
            row = entry[criterion]
            ap = row.get("ap_mean")
            lines.append(
                f"| {strategy} | {entry['theta']:g} | {entry['n']} | "
                f"{_CRITERION_LABELS[criterion]} | {'-' if ap is None else f'{ap:.2f}'} | "
                f"{row['labeled_mean']:.2f} | {row['n_never']} |"
            )
        if entry["ap_key"] is None:
            notes.append(
                f"- {strategy}: the results hold no AP at OKS {entry['theta']:g} (only 0.5, 0.6, "
                "0.7, 0.75, 0.8 and 0.95), so AP@θ is not reported"
            )
    if notes:
        lines += ["", *notes]
    return lines


def _plot(
    path: pathlib.Path,
    title: str,
    metric: str,
    summary: dict[str, dict[str, Any]],
) -> None:
    """Draw the mean learning curves of ``metric``; no file is written if no strategy has a curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 4.5))
    colors = plt.get_cmap("tab20")
    drawn = False
    for position, (strategy, entry) in enumerate(summary.items()):
        if metric not in entry["curves"]:
            continue
        grid, curve = entry["curves"][metric]
        axes.plot(
            grid, curve, label=strategy, color=colors(position % 20), linewidth=1.5
        )
        drawn = True
    axes.set_xlabel("labeled people (%)")
    axes.set_ylabel(f"{metric} (x 100)")
    axes.set_title(f"{title}: {metric}")
    axes.grid(alpha=0.3)
    if drawn:
        axes.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    plt.close(figure)


def run(cfg: DictConfig) -> None:
    """Write the summary tables and plots of the run ``run.name``."""
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    require_unsharded(run_, "vatl4pose-summarize")
    # only the architecture names the result directory; the weights are not needed here
    arch: str = cfg.estimator.arch
    summarize = section(cfg, "summarize", SummarizeSection)
    if summarize.ids_file is not None:
        ids = read_ids(None, summarize.ids_file, "summarize.ids_file")
        suffix = "_" + pathlib.Path(summarize.ids_file).stem
    else:
        # the ids only: the dataset root is not needed to read results
        listed = cfg.dataset.ids
        ids = read_ids(
            None if listed is None else [str(i) for i in listed],
            cfg.dataset.ids_file,
            "dataset.ids / dataset.ids_file",
        )
        suffix = ""
    name = run_name(run_)
    base = output_root(paths) / name / arch
    summary = collect(base, ids)
    out_dir = output_root(paths) / name / "summary" / f"{arch}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{name} / {arch}"
    document = {
        "run": name,
        "estimator": arch,
        "n_ids": len(ids),
        "metrics": list(METRICS),
        "strategies": {
            strategy: {k: v for k, v in entry.items() if k != "curves"}
            for strategy, entry in summary.items()
        },
    }
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    write_atomic(out_dir / "summary.json", lambda f: f.write(text.encode("utf-8")))
    markdown = _markdown(title, summary, len(ids))
    write_atomic(out_dir / "summary.md", lambda f: f.write(markdown.encode("utf-8")))
    for metric in METRICS:
        file = out_dir / f"learning_curves_{metric.replace(' ', '_')}.png"
        _plot(file, title, metric, summary)
    print(markdown, flush=True)
    print(f"written to {out_dir}", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-summarize``."""
