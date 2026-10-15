"""``vatl4pose-atl``: run the active-learning loop on the videos of a dataset.

    vatl4pose-atl experiment=wacv2024_posetrack21 strategy=thc_wpu_duw run.device=cuda run.shard=0/4

For every video of the shard the command takes the lock of the run directory of the video, skips the
video if it is done, runs (or resumes) the loop, and marks the video done. The output is
``<output_root>/<run.name>/<estimator>/<strategy.name>/<video>/result.json``. Next to it the last
round writes ``predictions.json`` and ``predictions_ann.json``, the poses of every person of the
video under the final estimator, without and with the labels of the labeled people (see
:mod:`vatl4pose.loop.results`). With ``oracle=coco_file`` the run stops at a round whose
annotations are not in place: the command prints a ``Pending`` line for that video, goes on with
the next video, and exits with status 0; run it again after annotating.

The videos need their cache from ``vatl4pose-prepare``. Plots are made by ``vatl4pose-summarize``
from the saved results, never here.
"""

from __future__ import annotations

from collections import Counter

from omegaconf import DictConfig

from ._common import (
    dataset_ids,
    hydra_command,
    load_job,
    run_dir_of,
    run_video,
    shard_ids,
)


def run(cfg: DictConfig) -> None:
    """Run the loop on every video of the shard ``run.shard``."""
    job = load_job(cfg)
    ids = shard_ids(dataset_ids(job.dataset), job.run)
    print(
        f"shard {job.run.shard}: {len(ids)} video(s), strategy {job.strategy.name}",
        flush=True,
    )
    counts: Counter[str] = Counter()
    for video_id in ids:
        run_dir = run_dir_of(
            job.paths, job.run, job.estimator, job.strategy.name, video_id
        )
        outcome = run_video(job, video_id, job.dataset.split, run_dir)
        counts[outcome.status] += 1
        print(outcome.message, flush=True)
    print(
        f"{counts['done']} done, {counts['skipped']} skipped, {counts['pending']} pending",
        flush=True,
    )


main = hydra_command(run)
"""Console entry point of ``vatl4pose-atl``."""
