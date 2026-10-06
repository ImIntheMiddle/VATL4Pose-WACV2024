"""``vatl4pose-train``: the initial training of the pose estimator or the whole-body autoencoder.

    vatl4pose-train train=simplebaseline_posetrack21 paths.data_root=/data run.device=cuda
    vatl4pose-train train=fastpose_jrdb run.device=cuda          # after fastpose_posetrack21
    vatl4pose-train train=ae_posetrack21

The ``train`` configs hold the recipe of spec 13.3 (the 2023 learning rates, milestones and batch
sizes, with the validation AP of the loop selecting the best epoch). For your own data use
``train.target=estimator`` or ``train.target=ae`` with ``dataset=coco_custom``,
``train.train_ids_file`` and ``train.val_ids_file``. Outputs go to
``<output_root>/train/<run.name>/``: ``best.pth`` (the weights to give ``estimator.weights`` or
``ae.checkpoint``), ``log.jsonl``, ``last.pth`` (estimator) and ``summary.json``. A directory that
already holds a run is refused.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from omegaconf import DictConfig

from vatl4pose.config import (
    DatasetSection,
    EstimatorSection,
    PathsSection,
    RunSection,
    TrainSection,
    ae_train_config,
    estimator_config,
    initial_train_config,
    keypoint_spec,
    section,
)
from vatl4pose.data import VideoAnnotation
from vatl4pose.train import train_estimator, train_wholebody_ae
from vatl4pose.train._common import write_atomic

from ._common import (
    hydra_command,
    output_root,
    read_ids,
    read_video,
    require_unsharded,
    run_name,
    torch_device,
)

TARGETS = ("estimator", "ae")


def _videos(
    paths: PathsSection,
    dataset: DatasetSection,
    ids_file: Optional[str],
    split: Optional[str],
    what: str,
) -> list[VideoAnnotation]:
    ids = read_ids(None, ids_file, what)
    return [read_video(paths, dataset, i, split) for i in ids]


def run(cfg: DictConfig) -> None:
    """Train the target of ``train.target`` and write its files."""
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    require_unsharded(run_, "vatl4pose-train")
    train = section(cfg, "train", TrainSection)
    if train.target not in TARGETS:
        raise ValueError(
            f"train.target must be one of {TARGETS} (or choose a config: train=<name>), "
            f"got {train.target!r}"
        )
    dataset = section(cfg, "dataset", DatasetSection)
    spec = keypoint_spec(dataset)
    device = torch_device(run_)
    out_dir = output_root(paths) / "train" / run_name(run_)
    train_videos = _videos(
        paths, dataset, train.train_ids_file, train.train_split, "train.train_ids_file"
    )
    val_videos = _videos(
        paths, dataset, train.val_ids_file, train.val_split, "train.val_ids_file"
    )
    print(
        f"{train.target}: {len(train_videos)} training and {len(val_videos)} validation video(s)",
        flush=True,
    )
    summary: dict[str, Any]
    if train.target == "estimator":
        estimator = section(cfg, "estimator", EstimatorSection)
        summary = train_estimator(
            train_videos,
            val_videos,
            estimator_config(estimator, spec, paths.weights_root),
            initial_train_config(train, paths.weights_root),
            out_dir,
            device,
            run_.seed,
        )
    else:
        summary = train_wholebody_ae(
            train_videos,
            val_videos,
            spec,
            ae_train_config(train.ae),
            out_dir,
            device,
            run_.seed,
        )
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write_atomic(out_dir / "summary.json", lambda f: f.write(text.encode("utf-8")))
    print(text, end="", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-train``."""
