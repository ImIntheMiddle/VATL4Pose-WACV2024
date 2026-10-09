"""Immutable workflow manifests and sequential, resumable subprocess execution."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any

from omegaconf import OmegaConf
from .gpu import GpuLease
from .process import _stop, run_child
from .locks import OwnedLock
from .status import Status, atomic_json, read_json, read_status

MANIFEST = "manifest.json"
REPORT_ENV = "VATL4POSE_COMMAND_REPORT"
PLAN_SCHEMA = "vatl4pose.workflow/1"


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def code_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    roots = [("vatl4pose", root)]
    vendor = find_spec("third_party")
    if vendor is None or vendor.submodule_search_locations is None:
        raise RuntimeError("Bundled third_party model sources are unavailable")
    roots.extend(
        (f"third_party_{i}", Path(location))
        for i, location in enumerate(vendor.submodule_search_locations)
    )
    for label, directory in roots:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".txt"}:
                digest.update((label + "/" + str(path.relative_to(directory))).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Invalid job/video id: {value!r}")
    return value


def environment_info(programs: list[str]) -> dict[str, Any]:
    """Inspect Python distribution metadata without importing model packages."""
    records = {}
    probe = 'import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"prefix":sys.prefix,"packages":{d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"]}}))'
    for program in programs:
        interpreter = program if Path(program).name.startswith("python") else None
        if interpreter is None:
            with Path(program).open("rb") as handle:
                first = handle.readline(1024)
            if first.startswith(b"#!"):
                parts = first[2:].decode().strip().split()
                if len(parts) == 1 and Path(parts[0]).name.startswith("python"):
                    interpreter = parts[0]
        if interpreter is not None and interpreter not in records:
            done = subprocess.run(
                [interpreter, "-c", probe], capture_output=True, text=True, check=True
            )
            records[interpreter] = json.loads(done.stdout)
    return records


def load_plan(path: Path, gpu: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) - {
        "schema",
        "run_dir",
        "jobs",
        "inputs",
    }:
        raise ValueError("Workflow requires run_dir and jobs; unknown top-level field")
    if raw.get("schema", PLAN_SCHEMA) != PLAN_SCHEMA:
        raise ValueError("Unsupported workflow schema")
    if not isinstance(raw.get("run_dir"), str) or not raw["run_dir"]:
        raise ValueError("Workflow run_dir must be a path")
    root = Path(raw["run_dir"])
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    source_jobs = raw.get("jobs")
    if not isinstance(source_jobs, list) or not source_jobs:
        raise ValueError("Workflow jobs must be a nonempty list")
    jobs = []
    seen: set[str] = set()
    for source in source_jobs:
        if not isinstance(source, dict) or set(source) - {
            "id",
            "argv",
            "videos",
            "depends_on",
            "gpu",
        }:
            raise ValueError("Invalid workflow job fields")
        name = _name(source.get("id"))
        argv = source.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(a, str) for a in argv)
            or not argv[0]
        ):
            raise ValueError(f"{name}: argv must be a nonempty argument list")
        requires_gpu = source.get("gpu", False)
        if type(requires_gpu) is not bool:
            raise ValueError(f"{name}: gpu must be true or false")
        if requires_gpu and (type(gpu) is not int or gpu < 0):
            raise ValueError(
                f"{name}: GPU work requires --gpu with a nonnegative index"
            )
        videos = source.get("videos", [None])
        if not isinstance(videos, list) or not videos:
            raise ValueError(f"{name}: videos must be a nonempty list")
        deps = source.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            raise ValueError(f"{name}: depends_on must be a list of ids")
        for video in videos:
            if video is not None:
                _name(video)

            def expand(text: str) -> str:
                text = text.replace("{run_dir}", str(root))
                if "{video_id}" in text and video is None:
                    raise ValueError(f"{name}: video_id placeholder requires videos")
                return (
                    text.replace("{video_id}", str(video))
                    if video is not None
                    else text
                )

            ident = name if video is None else f"{name}.{video}"
            if ident in seen:
                raise ValueError(f"Duplicate job id {ident}")
            resolved = [expand(a) for a in argv]
            executable = resolved[0]
            if "/" in executable and not Path(executable).is_absolute():
                executable = str((path.parent / executable).resolve())
            executable = shutil.which(executable)
            if executable is None:
                raise FileNotFoundError(resolved[0])
            resolved[0] = executable
            if any("{gpu_id}" in a for a in resolved) and not requires_gpu:
                raise ValueError(f"{ident}: gpu_id placeholder requires gpu: true")
            parents = [expand(d) for d in deps]
            if any(parent not in seen for parent in parents):
                raise ValueError(
                    f"{ident}: dependencies must refer to earlier jobs: {parents}"
                )
            jobs.append(
                {
                    "id": ident,
                    "argv": resolved,
                    "depends_on": parents,
                    "gpu": requires_gpu,
                }
            )
            seen.add(ident)
    inputs = raw.get("inputs", [])
    if not isinstance(inputs, list) or any(not isinstance(p, str) for p in inputs):
        raise ValueError("inputs must be a list of immutable file paths")
    hashes = {}
    for value in inputs:
        source = (path.parent / value).resolve()
        hashes[str(source)] = file_digest(source)
    # Record executable/script contents without importing model libraries.
    for job in jobs:
        for value in job["argv"]:
            if value != job["argv"][0] and Path(value).suffix not in {".py", ".sh"}:
                continue
            candidate = (path.parent / value).resolve()
            if candidate.is_file():
                hashes[str(candidate)] = file_digest(candidate)
    return {
        "schema": PLAN_SCHEMA,
        "config_path": str(path),
        "run_dir": str(root),
        "cwd": str(path.parent),
        "jobs": jobs,
        "inputs": hashes,
        "source_sha256": code_identity(),
        "environment": environment_info([j["argv"][0] for j in jobs]),
        "gpu_index": gpu,
    }


def ensure_manifest(plan: dict[str, Any], resume: bool) -> None:
    path = Path(plan["run_dir"]) / MANIFEST
    if resume:
        stored = read_json(path)
        expected = {k: v for k, v in plan.items() if k != "gpu_index"}
        actual = {k: v for k, v in stored.items() if k != "gpu_index"}
        if actual != expected:
            raise ValueError(
                f"{path}: workflow manifest differs; commands, code or immutable inputs changed"
            )
    else:
        if path.exists():
            raise FileExistsError(f"{path}: use resume or choose a new run_dir")
        atomic_json(path, plan)


def publish_outcome(video_id: str, state: str, run_dir: Path) -> None:
    """Let native loop commands report annotation waits independently of their exit code."""
    value = os.environ.get(REPORT_ENV)
    if not value:
        return
    path = Path(value)
    report = (
        read_json(path)
        if path.exists()
        else {"schema": "vatl4pose.command-outcomes/1", "outcomes": []}
    )
    if report.get("schema") != "vatl4pose.command-outcomes/1":
        raise ValueError(f"{path}: invalid command outcome schema")
    report["outcomes"].append(
        {"video_id": video_id, "status": state, "run_dir": str(run_dir)}
    )
    atomic_json(path, report)


def execute(
    plan: dict[str, Any], resume: bool = False, only_failed: bool = False
) -> str:
    root = Path(plan["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    saved_signals = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(signum)
        if previous != signal.SIG_IGN:
            saved_signals[signum] = previous
            signal.signal(signum, _stop)
    try:
        with OwnedLock(root / ".workflow.lock"), ExitStack() as resources:
            ensure_manifest(plan, resume)
            with Status(root, "workflow", kind="workflow") as overall:
                states: dict[str, str] = {}
                lease = None
                for job in plan["jobs"]:
                    folder = root / "jobs" / job["id"]
                    prior = (
                        read_status(folder)
                        if (folder / ".status.json").exists()
                        else {"status": "planned"}
                    )
                    if prior["status"] == "completed" or (
                        only_failed and prior["status"] == "waiting_annotation"
                    ):
                        states[job["id"]] = prior["status"]
                        continue
                    if any(states[p] != "completed" for p in job["depends_on"]):
                        status = Status(folder, "job", job_id=job["id"], kind="job")
                        status.update("blocked", depends_on=job["depends_on"])
                        states[job["id"]] = "blocked"
                        continue
                    attempt = int(prior.get("attempt", 0)) + 1
                    attempt_dir = folder / f"attempt_{attempt:02d}"
                    attempt_dir.mkdir(parents=True, exist_ok=False)
                    env = {**os.environ, REPORT_ENV: str(attempt_dir / "outcomes.json")}
                    argv = list(job["argv"])
                    with Status(
                        folder,
                        "job",
                        kind="job",
                        job_id=job["id"],
                        attempt=attempt,
                        argv=argv,
                    ) as status:
                        if job["gpu"]:
                            if lease is None:
                                lease = resources.enter_context(
                                    GpuLease(plan["gpu_index"])
                                )
                            env.update(lease.env)
                            argv = [
                                a.replace("{gpu_id}", str(plan["gpu_index"]))
                                for a in argv
                            ]
                        else:
                            env["CUDA_VISIBLE_DEVICES"] = ""
                            env.pop("VATL4POSE_GPU_LEASE", None)
                            env.pop("VATL4POSE_GPU_TOKEN", None)
                        status.update(
                            "running",
                            argv=argv,
                            gpu_index=plan["gpu_index"] if job["gpu"] else None,
                            gpu_uuid=lease.uid if job["gpu"] else None,
                        )
                        print(f"Running {job['id']} (attempt {attempt})", flush=True)
                        code = run_child(
                            argv, Path(plan["cwd"]), env, attempt_dir / "command.log"
                        )
                        if code != 0:
                            status.update(
                                "stopped" if code < 0 or code >= 128 else "failed",
                                exit_code=code,
                                log=str(attempt_dir / "command.log"),
                            )
                        elif (attempt_dir / "outcomes.json").is_file():
                            report = read_json(attempt_dir / "outcomes.json")
                            if report.get("schema") != "vatl4pose.command-outcomes/1":
                                raise ValueError("Invalid command outcome schema")
                            state = (
                                "waiting_annotation"
                                if any(
                                    o["status"] == "pending" for o in report["outcomes"]
                                )
                                else "completed"
                            )
                            status.update(
                                state,
                                outcomes=report["outcomes"],
                                log=str(attempt_dir / "command.log"),
                            )
                        states[job["id"]] = (
                            status.data["status"]
                            if status.data["status"] != "running"
                            else "completed"
                        )
                final = (
                    "failed"
                    if any(s in {"failed", "stopped"} for s in states.values())
                    else "waiting_annotation"
                    if "waiting_annotation" in states.values()
                    else "blocked"
                    if "blocked" in states.values()
                    else "completed"
                )
                overall.update(final, jobs=states)
                return final
    finally:
        for signum, handler in saved_signals.items():
            signal.signal(signum, handler)
