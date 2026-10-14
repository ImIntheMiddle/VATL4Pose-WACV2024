"""Detect people and track them with SAMIDARE, then prepare an unlabeled VATL4Pose dataset."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import importlib.util
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from annotation.mot_to_coco import convert
from annotation.person_detector import person_detections
from annotation.memory import configure_memory
from vatl4pose.execution.gpu import GpuLease
from vatl4pose.execution.locks import OwnedLock
from vatl4pose.execution.status import Status, atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    upstream = ROOT / ".deps/SAMIDARE/samidare_for_vis.py"
    checkpoint = ROOT / ".deps/checkpoints/sam2.1_hiera_large.pt"
    if not upstream.is_file() or not checkpoint.is_file():
        raise RuntimeError("Run bash annotation/setup.sh first")

    def digest(path: pathlib.Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    expected = read_json(ROOT / "models.json")[checkpoint.name]
    if digest(checkpoint) != expected:
        raise ValueError(
            f"{checkpoint}: checksum differs; rerun setup with a clean checkpoint"
        )
    source = pathlib.Path(__file__).resolve().parent
    identity = {
        "video": str(args.video.resolve()),
        "video_sha256": digest(args.video),
        "max_frames": args.max_frames,
        "models": read_json(ROOT / "models.json"),
        "upstream": read_json(ROOT / "upstream.json"),
        "adapter": {
            name: digest(source / name)
            for name in [
                "prepare_video.py",
                "person_detector.py",
                "memory.py",
                "mot_to_coco.py",
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with OwnedLock(
        args.output.parent / (args.output.name + ".tracking.lock"), command="tracking"
    ):
        if args.output.exists():
            if (args.output / ".done.json").is_file() and (
                args.output / "tracking_settings.json"
            ).is_file():
                if read_json(args.output / "tracking_settings.json") == identity:
                    print(f"Already completed: {args.output}")
                    return
            raise FileExistsError(
                f"{args.output}: input/settings cannot be verified; choose a new output directory"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        attempt = pathlib.Path(
            tempfile.mkdtemp(
                prefix=args.output.name + ".attempt_", dir=args.output.parent
            )
        )
        print(f"Tracking attempt: {attempt}", flush=True)
        with Status(attempt, "tracking", video=str(args.video)):
            with GpuLease(args.gpu, recheck=False) as lease:
                previous = {key: os.environ.get(key) for key in lease.env}
                try:
                    os.environ.update(lease.env)
                    prepare(args.video, attempt, args.max_frames)
                    atomic_json(attempt / "tracking_settings.json", identity)
                    atomic_json(attempt / ".done.json", {"command": "tracking"})
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
        if args.output.exists():
            raise FileExistsError(
                f"{args.output}: appeared during tracking; attempt preserved at {attempt}"
            )
        attempt.rename(args.output)
        print(
            f"Dataset committed: {args.output}; next: bash annotation/cycle.sh {args.output} atl"
        )


def prepare(video: pathlib.Path, output: pathlib.Path, max_frames: int | None) -> None:
    """Execute tracking after the caller acquires and binds a GPU lease."""
    upstream = ROOT / ".deps/SAMIDARE/samidare_for_vis.py"
    checkpoint = ROOT / ".deps/checkpoints/sam2.1_hiera_large.pt"
    os.environ["TORCH_HOME"] = str(ROOT / ".deps/torch-models")
    import cv2
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tracking needs exactly one visible CUDA GPU")
    work = output.resolve()
    images = work / "dataset/images/clip"
    images.mkdir(parents=True)
    sequence = work / "sequence"
    (sequence / "det").mkdir(parents=True)
    (sequence / "img1").symlink_to(
        pathlib.Path("../dataset/images/clip"), target_is_directory=True
    )
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    person = weights.meta["categories"].index("person")
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).cuda().eval()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video {video}")
    rows = []
    frame_id = 0
    with torch.inference_mode():
        while max_frames is None or frame_id < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            path = images / f"{frame_id:06d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Cannot write {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255
            pred = detector([tensor])[0]
            for box, score in person_detections(pred, person):
                x1, y1, x2, y2 = box
                rows.append(
                    f"{frame_id},-1,{x1},{y1},{x2 - x1},{y2 - y1},{score},-1,-1,-1\n"
                )
    cap.release()
    del detector
    torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("No people were detected in the video")
    detections = sequence / "det/det.txt"
    detections.write_text("".join(rows))
    module_spec = importlib.util.spec_from_file_location("samidare_upstream", upstream)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(checkpoint),
        device="cuda",
        apply_postprocessing=False,
    )
    tracker = module.SAM2MOT(predictor, module.GTLoader(str(detections)), device="cuda")
    configure_memory(predictor, tracker)
    with torch.inference_mode():
        results = tracker.track_sequence(
            str(sequence),
            video_output_path=str(work / "tracking.mp4"),
            output_dir=str(work),
        )
    if not results:
        raise RuntimeError(
            "SAMIDARE returned no tracks; inspect detector and tracker logs"
        )
    tracks = work / "sam2mot_results.txt"
    with tracks.open("w") as handle:
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            handle.write(
                f"{result['frame_id']},{result['track_id']},{x1},{y1},{x2 - x1},{y2 - y1},{result['confidence']},-1,-1,-1\n"
            )
    annotation = convert(tracks, images, work / "dataset", "clip")
    (work / "input.json").write_text(
        json.dumps(
            {
                "video": str(video.resolve()),
                "frames": frame_id,
                "detector": "fasterrcnn_resnet50_fpn_v2_coco_person",
                "detector_score_threshold": 0.7,
                "sam2_state_location": "cpu",
                "upstream": json.loads(
                    (ROOT / ".deps/upstream_commits.json").read_text()
                ),
            },
            indent=2,
        )
    )
    print(f"Dataset ready: {annotation}. Next: bash annotation/cycle.sh {work} atl")


if __name__ == "__main__":
    main()
