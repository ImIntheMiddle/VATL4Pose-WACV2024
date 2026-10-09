"""Signal propagation and owned process-group cleanup without model imports."""

from __future__ import annotations
from contextlib import nullcontext
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

GRACE_SECONDS = 30


class ChildCleanupError(RuntimeError):
    """Retain GPU ownership when live owned processes cannot be stopped safely."""

    keep_gpu_lease = True


def _stop(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def _stat(pid: int) -> list[str]:
    text = Path(f"/proc/{pid}/stat").read_text()
    return text[text.rfind(")") + 2 :].split()


def _members(group: int, leader_start: str) -> list[int]:
    leader = Path(f"/proc/{group}/stat")
    if leader.exists():
        try:
            if _stat(group)[19] != leader_start:
                raise ChildCleanupError(
                    f"Process group leader identity changed: {group}"
                )
        except FileNotFoundError:
            pass
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            data = _stat(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(data[2]) == group and int(data[3]) == group and data[0] != "Z":
            members.append(int(path.name))
    return members


def _send(group: int, signum: int) -> None:
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        raise ChildCleanupError(f"Cannot stop owned process group {group}") from error


def _cleanup(group: int, leader_start: str) -> None:
    if not _members(group, leader_start):
        return
    _send(group, signal.SIGTERM)
    deadline = time.monotonic() + GRACE_SECONDS
    while _members(group, leader_start) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _members(group, leader_start):
        _send(group, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _members(group, leader_start) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _members(group, leader_start):
        raise ChildCleanupError(
            f"Live owned process group {group} remains; inspect it before reclaiming the GPU"
        )


def run_child(argv: list[str], cwd: Path, env: dict[str, str], log: Path | None) -> int:
    """Wait for the child and stop remaining members of its owned process group.

    GPU commands must run foreground work, not launch a detached new session.
    Linux /proc is required to distinguish live members from unreaped zombies.
    """
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("Managed subprocess execution requires Linux /proc")
    with log.open("w") if log is not None else nullcontext(None) as handle:
        child = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT if handle is not None else None,
            start_new_session=True,
        )
        leader_start = _stat(child.pid)[19]
        try:
            code = child.wait()
        except BaseException:
            _cleanup(child.pid, leader_start)
            child.wait()
            raise
        _cleanup(child.pid, leader_start)
        return code
