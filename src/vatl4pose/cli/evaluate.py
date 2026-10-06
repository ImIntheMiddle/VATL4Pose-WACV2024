"""``vatl4pose-eval``: AP and OSPA of a pose estimator on the videos of a dataset.

    vatl4pose-eval dataset=posetrack21 estimator=simplebaseline \
        estimator.weights=posetrack21/simplebaseline_256x192.pth

Every person is cropped from its ground-truth box, as in the loop, and the metrics are computed
over all people of all videos of the id list together (micro average, like the 2023
``poseestimator_eval``), with the OKS sigmas of the dataset's keypoint spec. The result goes to
``eval.output`` or ``<output_root>/eval/<dataset>_<split>_<estimator>_<weights file>.json``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch
from omegaconf import DictConfig

from vatl4pose.config import (
    DatasetSection,
    EstimatorSection,
    EvalSection,
    PathsSection,
    RunSection,
    estimator_config,
    keypoint_spec,
    resolve_path,
    section,
)
from vatl4pose.eval import METRIC_KEYS
from vatl4pose.models.estimator import build_estimator
from vatl4pose.train import evaluate_estimator
from vatl4pose.train._common import write_atomic

from ._common import (
    dataset_ids,
    hydra_command,
    output_root,
    read_video,
    require_unsharded,
    torch_device,
)


def output_path(
    paths: PathsSection,
    dataset: DatasetSection,
    estimator: EstimatorSection,
    eval_: EvalSection,
) -> pathlib.Path:
    """The JSON file of the evaluation: ``eval.output``, or a name made from the settings."""
    if eval_.output is not None:
        return resolve_path(".", eval_.output)
    stem = (
        "random-init"
        if estimator.weights is None
        else pathlib.Path(estimator.weights).stem
    )
    name = "_".join(p for p in (dataset.name, dataset.split, estimator.arch, stem) if p)
    return output_root(paths) / "eval" / f"{name}.json"


def run(cfg: DictConfig) -> None:
    """Evaluate the estimator on the videos of the dataset config and write the JSON file."""
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    require_unsharded(run_, "vatl4pose-eval")
    dataset = section(cfg, "dataset", DatasetSection)
    estimator = section(cfg, "estimator", EstimatorSection)
    eval_ = section(cfg, "eval", EvalSection)
    device = torch_device(run_)
    ids = dataset_ids(dataset)
    videos = [read_video(paths, dataset, i, dataset.split) for i in ids]
    est_cfg = estimator_config(estimator, keypoint_spec(dataset), paths.weights_root)
    # the initial weights of a network without a checkpoint follow run.seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(run_.seed)
        model = build_estimator(est_cfg, device)
    metrics = evaluate_estimator(
        model,
        videos,
        batch_size=eval_.batch_size,
        num_workers=eval_.num_workers,
        device=device,
    )
    document: dict[str, Any] = {
        "dataset": dataset.name,
        "split": dataset.split,
        "n_videos": len(videos),
        "n_people": sum(len(v.people) for v in videos),
        "estimator": estimator.arch,
        "weights": None if est_cfg.weights is None else str(est_cfg.weights),
        "metrics": {
            **{k: float(metrics.ap[k]) for k in METRIC_KEYS},
            "OSPA": float(metrics.ospa),
        },
    }
    out = output_path(paths, dataset, estimator, eval_)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    write_atomic(out, lambda f: f.write(text.encode("utf-8")))
    print(text, end="", flush=True)
    print(f"written to {out}", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-eval``."""
