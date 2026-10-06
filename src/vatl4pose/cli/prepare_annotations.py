"""``vatl4pose-prepare-annotations``: make the per-video annotation files from the official releases.

    vatl4pose-prepare-annotations dataset=posetrack21 paths.data_root=/data
    vatl4pose-prepare-annotations dataset=posetrack21 dataset.split=train
    vatl4pose-prepare-annotations dataset=jrdb_pose dataset.split=test 'dataset.ids=[00,01]'

Run it once per split before ``vatl4pose-prepare``. It reads the release in ``dataset.root`` (below
``paths.data_root``) and writes ``<dataset.root>/activelearning/<dataset.split>/<video>_*.json``,
the files that the readers of :mod:`vatl4pose.data.annotations` take. The layout of the release and
the rules of the conversion are in :mod:`vatl4pose.data.prepare_annotations`.

``dataset.split`` is ``train``, ``train_val`` or ``val`` for PoseTrack21 and ``train``, ``val`` or
``test`` for JRDB-Pose; the dataset config's own split is the default. The videos of a split are
fixed by the id lists packaged with vatl4pose (``vatl4pose/data/splits``), and ``dataset.ids=[...]``
selects some of them. ``dataset.ids_file`` is not read: the split decides the list.

A file that already exists is kept if it holds the same data. One with other data raises, and
``prepare.overwrite=true`` replaces it. ``prepare.num_workers`` is the number of threads that read
the image headers.
"""

from __future__ import annotations

import pathlib
from collections import Counter
from collections.abc import Sequence
from typing import Optional

from omegaconf import DictConfig

from vatl4pose.config import (
    SPLITS_DIR,
    DatasetSection,
    PathsSection,
    PrepareSection,
    RunSection,
    section,
    split_file,
)
from vatl4pose.data.prepare_annotations import (
    JRDB_POSE_SPLITS,
    POSETRACK21_SPLITS,
    PreparedVideo,
    prepare_jrdb_pose,
    prepare_posetrack21,
)
from vatl4pose.loop import read_id_list

from ._common import dataset_root, hydra_command, require_unsharded

COMMAND = "vatl4pose-prepare-annotations"
SPLITS = {"posetrack21": POSETRACK21_SPLITS, "jrdb_pose": JRDB_POSE_SPLITS}


def select_videos(
    requested: Optional[list[str]], available: Sequence[str], split: str, source: str
) -> list[str]:
    """The videos of ``dataset.ids`` if given, else all ``available``; ``source`` names the list.

    Raises:
        ValueError: If ``requested`` is empty, repeats an id or holds an id that is not in
            ``available``.
    """
    if requested is None:
        return list(available)
    ids = [str(video_id) for video_id in requested]
    if not ids:
        raise ValueError("dataset.ids is empty")
    repeated = sorted(i for i, count in Counter(ids).items() if count > 1)
    if repeated:
        raise ValueError(
            f"dataset.ids: video ids occur more than once: {', '.join(repeated)}"
        )
    unknown = [video_id for video_id in ids if video_id not in available]
    if unknown:
        raise ValueError(
            f"dataset.ids {unknown} are not videos of the split {split!r}; the videos of this "
            f"split are listed in {source}"
        )
    return ids


def prepare_split(
    dataset: DatasetSection,
    root: pathlib.Path,
    overwrite: bool,
    num_workers: int,
) -> list[PreparedVideo]:
    """Write the annotation files of the split ``dataset.split`` of ``dataset`` below ``root``."""
    name, split = dataset.name, dataset.split
    if name not in SPLITS:
        raise ValueError(
            f"{COMMAND} prepares posetrack21 and jrdb_pose from their official releases, got "
            f"dataset {name!r}; the files of coco_custom are written by you "
            "(see vatl4pose.data.annotations)"
        )
    if split not in SPLITS[name]:
        raise ValueError(
            f"dataset.split must be one of {', '.join(SPLITS[name])} for {name}, got {split!r}"
        )
    if dataset.ids_file is not None and pathlib.Path(
        dataset.ids_file
    ).resolve().parent != (SPLITS_DIR):
        raise ValueError(
            f"dataset.ids_file={dataset.ids_file} is not read by {COMMAND}: the split fixes the "
            "list of videos. Select some of them with dataset.ids=[...]."
        )
    if name == "posetrack21":
        source = f"posetrack21_{split}.txt"
        available = read_id_list(pathlib.Path(split_file(source)))
        ids = select_videos(dataset.ids, available, split, source)
        return prepare_posetrack21(
            root, split, ids, overwrite=overwrite, num_workers=num_workers
        )
    source = f"jrdb_pose_{split}_scenes.txt"
    scenes = read_id_list(pathlib.Path(split_file(source)))
    available = [f"{index:02d}" for index in range(len(scenes))]
    ids = select_videos(dataset.ids, available, split, source)
    return prepare_jrdb_pose(
        root, split, scenes, ids, overwrite=overwrite, num_workers=num_workers
    )


def run(cfg: DictConfig) -> None:
    """Write the annotation files of the split of the dataset config and print one line per video."""
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    require_unsharded(run_, COMMAND)
    dataset = section(cfg, "dataset", DatasetSection)
    prepare = section(cfg, "prepare", PrepareSection)
    root = dataset_root(paths, dataset)
    results = prepare_split(dataset, root, prepare.overwrite, prepare.num_workers)
    for result in results:
        dropped = (
            f", {result.num_dropped} annotation(s) without a box dropped"
            if result.num_dropped
            else ""
        )
        print(
            f"{result.status}: {result.video_id} -> {result.path} "
            f"({result.num_images} images, {result.num_annotations} annotations{dropped})",
            flush=True,
        )
    print(
        f"{dataset.name} {dataset.split}: {len(results)} video(s) in "
        f"{root / 'activelearning' / str(dataset.split)}",
        flush=True,
    )


main = hydra_command(run)
"""Console entry point of ``vatl4pose-prepare-annotations``."""
