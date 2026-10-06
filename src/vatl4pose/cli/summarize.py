"""``vatl4pose-summarize``: the ALC table and the learning curves of finished runs.

    vatl4pose-summarize experiment=wacv2024_posetrack21 [summarize.ids_file=my_videos.txt]

Reads ``<output_root>/<run.name>/<estimator>/<strategy>/<video>/result.json`` of every strategy
directory and every video of the id list (the dataset's, or ``summarize.ids_file``), and writes to
``<output_root>/<run.name>/summary/<estimator>[_<ids file name>]/``:

* ``summary.json`` and ``summary.md``: for each strategy the mean ALC over its videos of ``AP .6``
  and ``AP .75`` (``performances_ann``) and of the OSPA (``ospa_ann``), all on the 0 to 100 scale,
  and the number of videos. A video without ``result.json`` is counted as missing; a video whose
  curve of a metric holds ``-1`` (no ground truth in the area range) is left out of that metric and
  counted as skipped, as in the 2023 summaries. A video whose metrics are ``null`` (some person has
  no ground-truth keypoints, so the loop computed no AP and no OSPA) is listed as "no ground truth"
  and left out of every mean and curve; it still counts in ``n_videos``.
* ``learning_curves_<metric>.png``: the mean over the videos on the 0 to 100 % label grid, one line
  per strategy.

The plots are made here, from the saved results, and nowhere else.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from omegaconf import DictConfig

from vatl4pose.config import (
    PathsSection,
    RunSection,
    SummarizeSection,
    section,
)
from vatl4pose.eval import PERCENT_GRID, interpolate_curve
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
        ``{strategy: {"n_videos", "n_missing", "no_ground_truth", metric: {"alc_mean", "n",
        "n_skipped"}, "curves": {metric: (percent grid, mean curve)}}}``. ``no_ground_truth`` lists
        the ids of the videos whose result has no metrics; the means cover the other videos.
        ``curves`` is for the plots and is dropped from the JSON.
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

    lines = [
        f"# {title}",
        "",
        "Mean ALC over the videos on the 0 to 100 scale (`performances_ann` for AP, `ospa_ann` for "
        "the OSPA, so a lower OSPA is better).",
        "",
        "| strategy | videos | " + " | ".join(f"ALC {m}" for m in METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for strategy, entry in summary.items():
        row = [strategy, f"{entry['n_videos']} / {n_ids}"] + [
            cell(entry, m) for m in METRICS
        ]
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
    return "\n".join(lines) + "\n"


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
