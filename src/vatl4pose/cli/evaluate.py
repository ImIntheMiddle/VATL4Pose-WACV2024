"""``vatl4pose-eval``: AP and OSPA of a pose estimator on the videos of a dataset.

    vatl4pose-eval dataset=posetrack21 estimator=simplebaseline \
        estimator.weights=posetrack21/simplebaseline_256x192.pth

Every person is cropped from its ground-truth box, as in the loop, and the metrics are computed
over all people of all videos of the id list together (micro average, like the 2023
``poseestimator_eval``), with the OKS sigmas of the dataset's keypoint spec. The result goes to
``eval.output`` or to ``<output_root>/eval/<name>.json``, where the name holds the dataset, split,
estimator, the directory and the stem of the weights file, and the first 8 hex digits of the
SHA-256 of the sorted video ids, so that evaluations of other weights or other videos never share
a file. An existing file is not overwritten unless ``eval.overwrite=true``.

AP protocol. The AP is the AP of the paper, not the standard COCO keypoint AP. The 2023 evaluation
gives a detection that carries a box the area ``(w - x) * (h - y)`` of its box ``[x, y, w, h]``,
and this package keeps that rule so that its numbers stay comparable with the paper. A detection
that no label matches and whose area lies outside the area range is then ignored instead of
counted as a false positive. On the 2023 predictions of JRDB-Pose video 00 in round 0, the AP@0.6
is 0.592 with this rule and 0.458 with the standard area ``w * h``. The file names the protocol in
``ap_protocol``, and ``vatl4pose.eval.coco_ap`` explains it. The file also records ``panorama``, the
setting of ``dataset.panorama`` that the labels and the crops of the evaluation followed. The ten values are computed in this
package from the precision and recall arrays of ``pycocotools``. Issue #40 came from the summary
of the stock ``pycocotools``, which lists AP75 at the position of the key ``AP .6``.
The JSON also records the video ids, device, seed, weight digest and keypoint-spec digest to
identify the inputs of an evaluation.
"""

from __future__ import annotations

import hashlib
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
from vatl4pose.data.cache import spec_sha256
from vatl4pose.eval import AP_PROTOCOL, METRIC_KEYS
from vatl4pose.models.estimator import build_estimator
from vatl4pose.train import evaluate_estimator
from vatl4pose.train._common import write_atomic

from ._common import (
    dataset_ids,
    hydra_command,
    logger_name,
    output_root,
    read_video,
    require_unsharded,
    torch_device,
)
from ._settings import file_sha256


def output_path(
    paths: PathsSection,
    dataset: DatasetSection,
    estimator: EstimatorSection,
    eval_: EvalSection,
    ids: list[str],
) -> pathlib.Path:
    """The JSON file of the evaluation: ``eval.output``, or a name made from the settings."""
    if eval_.output is not None:
        return resolve_path(".", eval_.output)
    if estimator.weights is None:
        weights = ["random-init"]
    else:
        file = pathlib.Path(estimator.weights)
        weights = [file.parent.name, file.stem]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:8]
    parts = [dataset.name, dataset.split, estimator.arch, *weights, digest]
    return output_root(paths) / "eval" / f"{'_'.join(p for p in parts if p)}.json"


def run(cfg: DictConfig) -> None:
    """Evaluate the estimator on the videos of the dataset config and write the JSON file.

    Raises:
        ValueError: If ``logger=wandb``; the command writes one JSON file and logs nothing else.
    """
    if logger_name(cfg) != "json":
        raise ValueError(
            "vatl4pose-eval writes its metrics to the JSON file and does not log to W&B; "
            "use logger=json (the default). logger=wandb is read by vatl4pose-atl, "
            "vatl4pose-tune and vatl4pose-train"
        )
    paths = section(cfg, "paths", PathsSection)
    run_ = section(cfg, "run", RunSection)
    require_unsharded(run_, "vatl4pose-eval")
    dataset = section(cfg, "dataset", DatasetSection)
    estimator = section(cfg, "estimator", EstimatorSection)
    eval_ = section(cfg, "eval", EvalSection)
    device = torch_device(run_)
    ids = dataset_ids(dataset)
    out = output_path(paths, dataset, estimator, eval_, ids)
    if out.exists() and not eval_.overwrite:
        raise FileExistsError(
            f"the evaluation file {out} exists; set eval.overwrite=true to replace it, or "
            "give another eval.output"
        )
    videos = [read_video(paths, dataset, i, dataset.split) for i in ids]
    spec = keypoint_spec(dataset)
    est_cfg = estimator_config(estimator, spec, paths.weights_root)
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
        "video_ids": sorted(ids),
        "device": str(device),
        "seed": run_.seed,
        "keypoints_sha256": spec_sha256(spec),
        "n_people": sum(len(v.people) for v in videos),
        "estimator": estimator.arch,
        "weights": None if est_cfg.weights is None else str(est_cfg.weights),
        "weights_sha256": None if est_cfg.weights is None else file_sha256(
            est_cfg.weights, "estimator.weights"
        ),
        "ap_protocol": AP_PROTOCOL,
        "panorama": dataset.panorama,
        "metrics": {
            **{k: float(metrics.ap[k]) for k in METRIC_KEYS},
            "OSPA": float(metrics.ospa),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    write_atomic(out, lambda f: f.write(text.encode("utf-8")))
    print(text, end="", flush=True)
    print(f"written to {out}", flush=True)


main = hydra_command(run)
"""Console entry point of ``vatl4pose-eval``."""
