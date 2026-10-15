"""Run one step (atl, annotate, answers or stop) of the manual annotation cycle on a run directory."""

from __future__ import annotations
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import signal

from vatl4pose.execution.gpu import GpuLease
from vatl4pose.execution.process import _stop, run_child

ROOT = Path(__file__).resolve().parent
LABELSTUDIO = ROOT / "labelstudio"
# the directory that vatl4pose-download-weights --weights-root pretrained_models fills
WEIGHTS_ROOT = ROOT.parent / "pretrained_models"
KEYPOINTS = "posetrack21"  # the keypoint spec of the pretrained estimator
CACHE_FILE = "atl/cache/coco_custom/clip.h5"  # written by vatl4pose-prepare under atl/


def atl_commands(
    run_dir: Path, device: str, bin_dir: Path, target_oks: float | None = None
) -> list[list[str]]:
    """The ``vatl4pose-prepare`` and ``vatl4pose-atl`` commands for a run directory.

    They use the PoseTrack21 keypoint spec, the pretrained SimpleBaseline estimator and the
    whole-body autoencoder.

    With ``target_oks`` (the target OKS θ of the paper) the loop judges its stopping criterion in
    every round, and the retraining keeps the labeled people below ``θ + 0.05`` (Eq. 3). The
    commands of a run whose cache exists leave out ``vatl4pose-prepare``.
    """
    judged = (
        []
        if target_oks is None
        else [f"retrain.finish_acc={target_oks}", "loop.stopping=true"]
    )
    common = [
        "dataset=coco_custom",
        f"dataset.root={run_dir}/dataset",
        "dataset.ids=[clip]",
        f"dataset.keypoints={KEYPOINTS}",
        f"paths.output_root={run_dir}/atl",
        f"paths.weights_root={WEIGHTS_ROOT}",
        "run.name=tracked",
        f"run.device={device}",
    ]
    commands = [
        [str(bin_dir / "vatl4pose-prepare"), *common],
        [
            str(bin_dir / "vatl4pose-atl"),
            *common,
            "strategy=thc_wpu_duw",
            "oracle=coco_file",
            "estimator=simplebaseline",
            "estimator.weights=posetrack21/simplebaseline_256x192.pth",
            "ae.checkpoint=wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth",
            "retrain.num_workers=0",
            "loop.save_final_model=true",
            "loop.save_estimator=true",
            *judged,
        ],
    ]
    # vatl4pose-atl refuses a cache that does not match the annotation file (SHA-256), so the
    # prepare command is needed only until the cache exists
    return commands[1:] if (run_dir / CACHE_FILE).is_file() else commands


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("action", choices=["atl", "annotate", "answers", "stop"])
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    if not run.is_dir():
        parser.error(f"Run directory does not exist: {run}")
    if (
        (args.action in {"annotate", "stop"} and args.extra)
        or (args.action == "answers" and len(args.extra) != 1)
        or (args.action == "atl" and len(args.extra) > 1)
    ):
        parser.error(
            "atl accepts one optional GPU id; answers needs one JSON export; annotate/stop need no extra arguments"
        )
    index = None
    if args.action == "atl" and args.extra:
        try:
            index = int(args.extra[0])
        except ValueError:
            parser.error("GPU id must be a nonnegative integer")
        if index < 0:
            parser.error("GPU id must be a nonnegative integer")
    env = {
        **os.environ,
        "DATA_ROOT": str(run / "dataset"),
        "OUTPUT_ROOT": str(run / "atl"),
        "RUN_NAME": "tracked",
        "VIDEO_ID": "clip",
        "LS_DATA_DIR": str(run / "label-studio"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        if args.action == "atl":
            with (
                GpuLease(index, recheck=False)
                if index is not None
                else nullcontext(None)
            ) as lease:
                if lease is not None:
                    env.update(lease.env)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    env.pop("VATL4POSE_GPU_LEASE", None)
                    env.pop("VATL4POSE_GPU_TOKEN", None)
                for command in atl_commands(
                    run, "cuda" if lease is not None else "cpu", ROOT / ".venv/bin"
                ):
                    code = run_child(command, Path.cwd(), env, None)
                    if code != 0:
                        raise SystemExit(code if code > 0 else 128 - code)
        elif args.action in {"annotate", "stop"}:
            subprocess.run(
                [
                    "bash",
                    str(
                        LABELSTUDIO
                        / ("launch.sh" if args.action == "annotate" else "stop.sh")
                    ),
                ],
                env=env,
                check=True,
            )
        else:
            oracle = (
                run / "atl/tracked/simplebaseline/THC+WPU_Coresetfilter/clip/oracle"
            )
            queries = sorted(oracle.glob("round_*_queries.json"))
            if not queries:
                raise FileNotFoundError(f"No queries in {oracle}; run atl first")
            query = queries[-1]
            answer = query.with_name(query.name.replace("_queries", "_answers"))
            if answer.exists():
                raise FileExistsError(answer)
            subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    str(LABELSTUDIO / "ls_export_to_answers.py"),
                    "--export",
                    args.extra[0],
                    "--queries",
                    str(query),
                    "--out",
                    str(answer),
                ],
                env=env,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    saved = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(signum)
        if previous != signal.SIG_IGN:
            saved[signum] = previous
            signal.signal(signum, _stop)
    try:
        _main(argv)
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
