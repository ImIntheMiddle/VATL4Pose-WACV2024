"""``vatl4pose-tune``: choose the weight lambda of the DUW selector on the validation videos.

    vatl4pose-tune experiment=wacv2024_posetrack21 strategy=thc_wpu_duw 'tune.lams=[0.001,0.01,0.1]'

For every lambda the loop runs on the validation videos of the config (``tune.split`` and
``tune.ids_file``; the experiments set the 14 PoseTrack21 and 2 JRDB-Pose validation videos) with
the schedule ``tune.query_ratio`` and the simulated oracle. The objective is the one of the 2023
tuning: the ALC of ``tune.metric`` (default ``AP .95``) on ``performances_ann``, averaged over the
videos, on the 0 to 100 scale. The runs go to
``<output_root>/<run.name>/tune/<estimator>/<strategy>/lam_<lam>/<video>/``. A finished run is not
repeated.

The table (``tune.json`` and ``tune.md`` next to the ``lam_*`` directories) is rebuilt after every
call from all ``lam_*`` directories that exist, so that successive or concurrent calls with
different ``tune.lams`` give one table. A directory made with other settings than this call (see
:mod:`._settings`, for example another schedule) raises instead of entering the table. A lambda
whose videos are not all finished is listed but cannot be the best.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from typing import Any

import numpy as np
from omegaconf import DictConfig

from vatl4pose.config import TuneSection, section
from vatl4pose.eval import METRIC_KEYS
from vatl4pose.loop.results import RESULT_NAME
from vatl4pose.train._common import write_atomic

from ._common import (
    Job,
    alc_percent,
    hydra_command,
    load_job,
    load_result,
    output_root,
    read_ids,
    require_unsharded,
    run_name,
    run_video,
)
from ._settings import check_settings, fingerprint


def lam_dirs(base: pathlib.Path) -> list[tuple[float, pathlib.Path]]:
    """The ``lam_<number>`` directories of ``base`` as ``(lam, directory)``, sorted by lam."""
    found = []
    for path in base.glob("lam_*"):
        if not path.is_dir():
            continue
        text = re.fullmatch(r"lam_(.+)", path.name)
        assert text is not None
        try:
            lam = float(text.group(1))
        except ValueError as error:
            raise ValueError(
                f"{path} is in the tuning directory but its name is not lam_<number>"
            ) from error
        found.append((lam, path))
    return sorted(found)


def collect_rows(
    job: Job, tune: TuneSection, base: pathlib.Path, ids: list[str]
) -> list[dict[str, Any]]:
    """One row per ``lam_*`` directory: the mean ALC of the metric over the finished videos.

    Raises:
        ValueError: If a run directory was made with other settings than ``job`` has for its lambda.
    """
    rows = []
    for lam, directory in lam_dirs(base):
        strategy = dataclasses.replace(
            job.strategy, params={**job.strategy.params, "lam": lam}
        )
        trial = dataclasses.replace(job, strategy=strategy)
        per_video: dict[str, float | None] = {}
        missing = 0
        for video_id in ids:
            run_dir = directory / video_id
            if not (run_dir / RESULT_NAME).is_file():
                missing += 1
                continue
            check_settings(run_dir, fingerprint(trial, video_id, tune.split))
            per_video[video_id] = alc_percent(load_result(run_dir), tune.metric)
        valid = [a for a in per_video.values() if a is not None]
        rows.append(
            {
                "lam": lam,
                "alc_mean": float(np.mean(valid)) if valid else None,
                "n_videos": len(valid),
                "n_skipped": len(per_video) - len(valid),
                "n_missing": missing,
                "alc_per_video": per_video,
            }
        )
    return rows


def run(cfg: DictConfig) -> None:
    """Run the loop for every lambda on the validation videos and write the ALC table."""
    job = load_job(cfg)
    require_unsharded(job.run, "vatl4pose-tune")
    tune = section(cfg, "tune", TuneSection)
    if not tune.lams:
        raise ValueError(
            "tune.lams is empty: give the weights to try, e.g. 'tune.lams=[0.01,1]'"
        )
    if tune.metric not in METRIC_KEYS:
        raise ValueError(
            f"tune.metric must be one of {list(METRIC_KEYS)}, got {tune.metric!r}"
        )
    if job.strategy.selector != "coreset_duw":
        raise ValueError(
            f"strategy {job.strategy.name!r} has the selector {job.strategy.selector!r}, which has "
            "no weight lam; tune a strategy with the selector coreset_duw (thc_coreset, "
            "wpu_coreset, thc_wpu_duw)"
        )
    if job.oracle.name != "simulated":
        raise ValueError(
            "vatl4pose-tune needs oracle=simulated: the ground truth answers"
        )
    ids = read_ids(None, tune.ids_file, "tune.ids_file")
    loop = dataclasses.replace(job.loop, query_ratio=tuple(tune.query_ratio))
    base = (
        output_root(job.paths)
        / run_name(job.run)
        / "tune"
        / job.estimator.arch
        / job.strategy.name
    )
    for lam in tune.lams:
        strategy = dataclasses.replace(
            job.strategy, params={**job.strategy.params, "lam": float(lam)}
        )
        trial = dataclasses.replace(job, strategy=strategy, loop=loop)
        for video_id in ids:
            run_dir = base / f"lam_{float(lam)!r}" / video_id
            outcome = run_video(
                trial, video_id, tune.split, run_dir, tag=f"lam{float(lam)!r}"
            )
            print(f"lam {lam}: {outcome.message}", flush=True)
    rows = collect_rows(dataclasses.replace(job, loop=loop), tune, base, ids)
    complete = [r for r in rows if r["n_missing"] == 0 and r["alc_mean"] is not None]
    best = max(complete, key=lambda r: r["alc_mean"])["lam"] if complete else None
    document = {
        "run": job.run.name,
        "estimator": job.estimator.arch,
        "strategy": job.strategy.name,
        "metric": tune.metric,
        "query_ratio": list(loop.query_ratio),
        "videos": ids,
        "best_lam": best,
        "rows": rows,
    }
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    write_atomic(base / "tune.json", lambda f: f.write(text.encode("utf-8")))
    table = [
        f"# lambda tuning: {job.run.name} / {job.estimator.arch} / {job.strategy.name}",
        "",
        f"ALC of `{tune.metric}` (`performances_ann`, 0 to 100), mean over {len(ids)} video(s); "
        f"schedule {list(loop.query_ratio)}. All `lam_*` directories of this run are listed.",
        "",
        "| lam | mean ALC | videos |",
        "|---|---|---|",
    ]
    for row in rows:
        value = "-" if row["alc_mean"] is None else f"{row['alc_mean']:.2f}"
        table.append(f"| {row['lam']:g} | {value} | {row['n_videos']} / {len(ids)} |")
    unfinished = [f"{r['lam']:g}" for r in rows if r["n_missing"]]
    if unfinished:
        table += [
            "",
            f"Not all videos are finished for lam {', '.join(unfinished)}; "
            "these cannot be the best.",
        ]
    if best is not None:
        table += ["", f"Best lam: {best:g}"]
    markdown = "\n".join(table) + "\n"
    write_atomic(base / "tune.md", lambda f: f.write(markdown.encode("utf-8")))
    print(markdown, flush=True)
    print(f"written to {base}", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-tune``."""
