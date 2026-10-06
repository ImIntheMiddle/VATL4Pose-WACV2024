"""``vatl4pose-prepare``: build the per-video HDF5 caches of the crops and annotations.

    vatl4pose-prepare dataset=posetrack21 paths.data_root=/data paths.output_root=/out
    vatl4pose-prepare dataset=coco_custom dataset.root=/data/mine 'dataset.ids=[a,b]' run.shard=0/2

Writes ``<output_root>/cache/<dataset>[/<split>]/<video>.h5`` for the videos of the shard. A cache
that matches its video (same keypoint spec, same annotation file, same number of people) is kept;
one that does not match raises, and ``prepare.overwrite=true`` rebuilds it. The build of a video
holds the lock ``<cache dir>/.locks/<video>/.lock``.
"""

from __future__ import annotations

import pathlib

from omegaconf import DictConfig

from vatl4pose.config import (
    DatasetSection,
    PathsSection,
    PrepareSection,
    RunSection,
    keypoint_spec,
    section,
)
from vatl4pose.data.cache import VideoCache, annotation_sha256, build_video_cache
from vatl4pose.loop import VideoLock

from ._common import (
    LOCKS_DIR,
    cache_dir,
    cache_file,
    dataset_ids,
    hydra_command,
    read_video,
    shard_ids,
)


def prepare_video(
    paths: PathsSection,
    dataset: DatasetSection,
    prepare: PrepareSection,
    video_id: str,
) -> tuple[str, pathlib.Path]:
    """Make sure the cache of one video exists and matches it; returns ``(status, path)``.

    The status is ``built``, ``rebuilt`` or ``up to date``.

    Raises:
        ValueError: If a cache exists that does not match the video and ``prepare.overwrite`` is
            false.
    """
    split = dataset.split
    path = cache_file(paths, dataset, split, video_id)
    lock_dir = cache_dir(paths, dataset, split) / LOCKS_DIR / video_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    with VideoLock(lock_dir):
        video = read_video(paths, dataset, video_id, split)
        if not path.exists():
            build_video_cache(video, path, prepare.num_workers)
            return "built", path
        spec, digest = keypoint_spec(dataset), annotation_sha256(video)
        try:
            cache = VideoCache(path, spec, digest)
            if len(cache) != len(video.people):
                raise ValueError(
                    f"{path} holds {len(cache)} people, but the annotation file now gives "
                    f"{len(video.people)}"
                )
        except ValueError as error:
            if not prepare.overwrite:
                raise ValueError(
                    f"the cache {path} does not match video {video_id!r}: {error}. "
                    "Run with prepare.overwrite=true to rebuild it."
                ) from error
            build_video_cache(video, path, prepare.num_workers)
            return "rebuilt", path
        return "up to date", path


def run(cfg: DictConfig) -> None:
    """Build the caches of the videos of the shard ``run.shard``."""
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    dataset = section(cfg, "dataset", DatasetSection)
    prepare = section(cfg, "prepare", PrepareSection)
    ids = shard_ids(dataset_ids(dataset), run_)
    print(f"shard {run_.shard}: {len(ids)} video(s) of {dataset.name}", flush=True)
    for video_id in ids:
        status, path = prepare_video(paths, dataset, prepare, video_id)
        print(f"{status}: {video_id} -> {path}", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-prepare``."""
