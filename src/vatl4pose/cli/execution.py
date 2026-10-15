"""Annotate videos, run, inspect and resume workflows, and hold a GPU lease for a command."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from vatl4pose.execution.status import Status, atomic_json, read_json, status_report


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["annotate"]:
        from vatl4pose.cli.annotate import main as annotate
        annotate(arguments[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("annotate", help="Annotate a video in one local browser session")
    status = commands.add_parser(
        "status", help="Read execution states without changing a run"
    )
    status.add_argument("run_dir", type=Path)
    status.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="Run an immutable YAML workflow")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--gpu", type=int)
    resume = commands.add_parser(
        "resume", help="Resume failed, stopped and annotation-waiting jobs"
    )
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--only-failed", action="store_true")
    resume.add_argument("--gpu", type=int)
    gpu = commands.add_parser("gpu-run", help="Hold a GPU lease for one child command")
    gpu.add_argument("--gpu", type=int, required=True)
    gpu.add_argument("--lease-root", type=Path)
    gpu.add_argument("--ready-file", type=Path)
    gpu.add_argument("--status-dir", type=Path, required=True)
    gpu.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            report = status_report(args.run_dir)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                for record in report["runs"]:
                    print(f"{record['status']:20} {record['path']}")
                print(
                    ", ".join(
                        f"{count} {state}" for state, count in report["counts"].items()
                    )
                    or "No execution records"
                )
            if any(
                "error" in record for record in report["runs"]
            ) or "error" in report.get("workflow", {}):
                raise SystemExit(2)
        elif args.command in {"run", "resume"}:
            from vatl4pose.execution.workflow import execute, load_plan

            if args.command == "run":
                result = execute(load_plan(args.config, args.gpu))
            else:
                manifest = read_json(args.run_dir / "manifest.json")
                index = args.gpu if args.gpu is not None else manifest.get("gpu_index")
                plan = load_plan(Path(manifest["config_path"]), index)
                if Path(plan["run_dir"]).resolve() != args.run_dir.resolve():
                    raise ValueError(
                        "Resume run_dir does not match the workflow manifest"
                    )
                result = execute(plan, resume=True, only_failed=args.only_failed)
            print(f"Workflow: {result}")
            if result in {"failed", "blocked", "stopped"}:
                raise SystemExit(1)
        else:
            from vatl4pose.execution.gpu import GpuLease
            from vatl4pose.execution.process import _stop, run_child

            command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            if not command:
                parser.error("gpu-run needs a child command after --")
            saved = {}
            for signum in (signal.SIGTERM, signal.SIGHUP):
                old = signal.getsignal(signum)
                if old != signal.SIG_IGN:
                    saved[signum] = old
                    signal.signal(signum, _stop)
            try:
                with Status(args.status_dir, "gpu-run") as state:
                    with GpuLease(args.gpu, args.lease_root) as lease:
                        state.update("running", gpu_index=args.gpu, gpu_uuid=lease.uid)
                        if args.ready_file is not None:
                            atomic_json(
                                args.ready_file,
                                {"pid": os.getpid(), "gpu_index": args.gpu},
                            )
                        code = run_child(
                            command, Path.cwd(), {**os.environ, **lease.env}, None
                        )
                        if code != 0:
                            state.update(
                                "stopped" if code < 0 or code >= 128 else "failed",
                                exit_code=code,
                            )
                            raise SystemExit(code if code > 0 else 128 - code)
            finally:
                for signum, handler in saved.items():
                    signal.signal(signum, handler)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(f"Execution refused: {error}", file=sys.stderr)
        raise SystemExit(3) from error


if __name__ == "__main__":
    main()
