"""Atomic execution records and read-only inspection without model dependencies."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Any
import uuid

SCHEMA = "vatl4pose.status/1"
STATUS_FILE = ".status.json"
STATES = {
    "planned",
    "running",
    "waiting_annotation",
    "completed",
    "failed",
    "stopped",
    "blocked",
    "unknown",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Commit complete JSON with a unique temporary file and a directory fsync."""
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _start_time(pid: int) -> str:
    text = Path(f"/proc/{pid}/stat").read_text()
    return text[text.rfind(")") + 2 :].split()[19]


def process_identity() -> dict[str, Any]:
    owner: dict[str, Any] = {"host": socket.gethostname(), "pid": os.getpid()}
    # Non-Linux readers cannot prove process identity; they report unknown owners.
    try:
        owner.update(
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            pid_namespace=os.readlink("/proc/self/ns/pid"),
            start_time=_start_time(os.getpid()),
        )
    except OSError:
        owner.update(boot_id=None, pid_namespace=None, start_time=None)
    return owner


def owner_alive(owner: dict[str, Any]) -> bool | None:
    """Verify host, boot, PID namespace and start identity before trusting a PID."""
    current = process_identity()
    if owner.get("host") != current["host"]:
        return None
    if not owner.get("boot_id") or not current.get("boot_id"):
        return None
    if owner["boot_id"] != current["boot_id"]:
        return False
    if owner.get("pid_namespace") != current.get("pid_namespace") or not owner.get(
        "start_time"
    ):
        return None
    try:
        pid = owner["pid"]
        if not isinstance(pid, int) or pid <= 0:
            return None
        return _start_time(pid) == owner["start_time"]
    except FileNotFoundError:
        return False
    except (OSError, KeyError):
        return None


class Status:
    """Record a single writer's lifecycle; the caller holds its existing run lock."""

    def __init__(self, root: Path, command: str, **details: Any) -> None:
        self.root = Path(root)
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "command": command,
            "owner": process_identity(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }

    def update(self, state: str, **details: Any) -> None:
        if state not in STATES:
            raise ValueError(f"Unknown execution status {state!r}")
        self.data.update(
            details, status=state, updated_utc=datetime.now(timezone.utc).isoformat()
        )
        atomic_json(self.root / STATUS_FILE, self.data)

    def __enter__(self) -> Status:
        self.update("running")
        return self

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> None:
        if error is not None:
            stopped = isinstance(error, KeyboardInterrupt) or (
                isinstance(error, SystemExit)
                and isinstance(error.code, int)
                and error.code >= 128
            )
            details = {"type": type(error).__name__, "message": str(error)}
            try:
                self.update("stopped" if stopped else "failed", error=details)
            except Exception as write_error:
                error.add_note(f"Execution status could not be saved: {write_error}")
        elif self.data["status"] == "running":
            self.update("completed")


def read_status(root: Path) -> dict[str, Any]:
    """Status of one run directory, read from ``.status.json``, ``.done.json``,
    ``.lock`` and the oracle query files."""
    root = Path(root)
    path = root / STATUS_FILE
    record: dict[str, Any] = {"schema": SCHEMA, "path": str(root), "status": "unknown"}
    if path.is_file():
        data = read_json(path)
        if data.get("schema") != SCHEMA or data.get("status") not in STATES:
            raise ValueError(f"{path}: invalid status schema or state")
        if data["status"] == "running" and not isinstance(data.get("owner"), dict):
            raise ValueError(f"{path}: running status requires an owner object")
        record.update(data)
    if (root / ".done.json").is_file():
        read_json(root / ".done.json")
        record["status"] = "completed"
        return record
    if path.is_file():
        if record["status"] == "running":
            alive = owner_alive(record.get("owner", {}))
            if alive is not True:
                record["recorded_status"] = "running"
                record["status"] = "stopped" if alive is False else "unknown"
                record["reason"] = (
                    "owner exited or was replaced"
                    if alive is False
                    else "owner cannot be verified here"
                )
        return record
    if (root / ".lock").exists():
        record["reason"] = (
            "a lock file exists but there is no status file; "
            "verify its owner before recovery"
        )
        return record
    queries = sorted((root / "oracle").glob("round_*_queries.json"))
    if queries:
        query = queries[-1]
        if not query.with_name(query.name.replace("_queries", "_answers")).exists():
            record.update(status="waiting_annotation", queries=str(query))
        else:
            record["reason"] = "answers exist; run the command again to resume"
    return record


def status_report(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root / "manifest.json").is_file():
        manifest = read_json(root / "manifest.json")
        if manifest.get("schema") != "vatl4pose.workflow/1" or not isinstance(
            manifest.get("jobs"), list
        ):
            raise ValueError(f"{root}/manifest.json: invalid workflow manifest")
        records = []
        for job in manifest["jobs"]:
            ident = job.get("id") if isinstance(job, dict) else None
            if (
                not isinstance(ident, str)
                or not ident
                or ident in {".", ".."}
                or "/" in ident
                or "\\" in ident
            ):
                raise ValueError("Invalid job id in workflow manifest")
            folder = root / "jobs" / ident
            try:
                record = (
                    read_status(folder)
                    if (folder / STATUS_FILE).exists()
                    else {"path": str(folder), "status": "planned"}
                )
            except ValueError as error:
                record = {"path": str(folder), "status": "unknown", "error": str(error)}
            records.append(record)
        try:
            overall = read_status(root)
        except ValueError as error:
            overall = {"path": str(root), "status": "unknown", "error": str(error)}
        return {
            "schema": "vatl4pose.status-report/1",
            "root": str(root),
            "workflow": overall,
            "runs": records,
            "counts": dict(sorted(Counter(r["status"] for r in records).items())),
        }
    paths: set[Path] = set()
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in {
                "oracle",
                "state",
                "cache",
                "dataset",
                "images",
                "data",
                "weights",
                "label-studio",
            }
        ]
        if (
            STATUS_FILE in files
            or ".done.json" in files
            or "settings.json" in files
        ):
            paths.add(Path(current))
        # "oracle" is pruned from the walk above, so look for it here.
        if (Path(current) / "oracle").is_dir():
            paths.add(Path(current))
    records = []
    for path in sorted(paths):
        try:
            records.append(read_status(path))
        except ValueError as error:
            records.append(
                {"path": str(path), "status": "unknown", "error": str(error)}
            )
    return {
        "schema": "vatl4pose.status-report/1",
        "root": str(root),
        "runs": records,
        "counts": dict(sorted(Counter(r["status"] for r in records).items())),
    }
