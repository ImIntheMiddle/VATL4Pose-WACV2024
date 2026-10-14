"""Cooperative GPU ownership for commands and their sequential subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
from typing import Any

from .status import owner_alive, read_json
from .locks import OwnedLock

LEASE_PATH = "VATL4POSE_GPU_LEASE"
LEASE_TOKEN = "VATL4POSE_GPU_TOKEN"
DEFAULT_LEASE_ROOT = Path.home() / ".cache/vatl4pose/gpu-leases"


def _query(arguments: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def gpu_uuid(index: int) -> str:
    if type(index) is not int or index < 0:
        raise ValueError("GPU index must be a nonnegative integer")
    for line in _query(
        ["--query-gpu=index,uuid", "--format=csv,noheader"]
    ).splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"Invalid nvidia-smi GPU row: {line!r}")
        if int(fields[0]) == index:
            if not fields[1].startswith("GPU-"):
                raise ValueError(f"Invalid GPU UUID: {fields[1]!r}")
            return fields[1]
    raise ValueError(f"GPU {index} does not exist")


def check_free(uid: str) -> None:
    rows = _query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    occupied = []
    for line in rows.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if (
            len(fields) != 2
            or not fields[0].startswith("GPU-")
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
        ):
            raise ValueError(
                f"Invalid nvidia-smi process row: {line!r}; occupancy cannot be verified"
            )
        if fields[0] == uid:
            occupied.append(line)
    if occupied:
        raise RuntimeError(f"GPU {uid} has compute processes: {'; '.join(occupied)}")


def wait_recheck() -> None:
    """Use the same POSIX sleep command as the existing research launcher."""
    subprocess.run(["sleep", "180"], check=True)


def _ancestor(pid: int) -> bool:
    current = os.getpid()
    while current > 0:
        if current == pid:
            return True
        try:
            text = Path(f"/proc/{current}/stat").read_text()
            parent = int(text[text.rfind(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if parent == current:
            return False
        current = parent
    return False


class GpuLease:
    """Hold an exclusive UUID lock until all owned child work has exited.

    New owners check for compute processes twice, 180 seconds apart; with
    ``recheck=False`` they check once and do not wait. The environment
    permits only descendants of a live owner to borrow the same lease. Unrelated
    launchers can still race; this is not a cluster scheduler.
    """

    def __init__(
        self, index: int, root: Path | None = None, *, recheck: bool = True
    ) -> None:
        if type(index) is not int or index < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        self.index = index
        self.recheck = recheck
        self.root = (
            Path(root)
            if root is not None
            else Path(os.environ.get("VATL4POSE_GPU_LEASE_ROOT", DEFAULT_LEASE_ROOT))
        )
        self.borrowed = False
        self.env: dict[str, str] = {}
        self.path = self.root / "unacquired"

    def __enter__(self) -> GpuLease:
        inherited = os.environ.get(LEASE_PATH)
        if inherited:
            self.path = Path(inherited)
            data = read_json(self.path)
            if data.get("schema") != "vatl4pose.gpu-lease/1" or data.get(
                "gpu_uuid"
            ) != gpu_uuid(self.index):
                raise RuntimeError(
                    "Inherited GPU lease has an invalid schema or device identity"
                )
            if data.get("gpu_index") != self.index:
                raise RuntimeError("Inherited lease belongs to a different GPU")
            owner = data.get("owner", {})
            if (
                data.get("token") != os.environ.get(LEASE_TOKEN)
                or owner_alive(owner) is not True
                or not _ancestor(owner["pid"])
            ):
                raise RuntimeError(
                    "Inherited GPU lease must have a live ancestor owner and matching token"
                )
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES") != str(self.index)
                or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                raise RuntimeError(
                    "Inherited GPU lease has an inconsistent CUDA environment"
                )
            self.uid = data["gpu_uuid"]
            self.borrowed = True
            self.env = self._environment(data["token"])
            return self
        uid = gpu_uuid(self.index)
        self.uid = uid
        self.root.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname().replace("/", "_")
        self.path = self.root / f"{host}_{uid}.lock"
        self.lock = OwnedLock(
            self.path,
            schema="vatl4pose.gpu-lease/1",
            gpu_index=self.index,
            gpu_uuid=uid,
        )
        self.lock.__enter__()
        try:
            check_free(uid)
            if self.recheck:
                print(
                    f"GPU {self.index} ({uid}) is free; checking again in 180 seconds",
                    flush=True,
                )
                wait_recheck()
                check_free(uid)
            else:
                print(f"GPU {self.index} ({uid}) is free", flush=True)
            self.env = self._environment(self.lock.data["token"])
            return self
        except BaseException as error:
            self.lock.__exit__(type(error), error, error.__traceback__)
            raise

    def _environment(self, token: str) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(self.index),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            LEASE_PATH: str(self.path.resolve()),
            LEASE_TOKEN: token,
        }

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if self.borrowed:
            return
        if getattr(error, "keep_gpu_lease", False):
            error.add_note(f"GPU lease preserved for manual recovery: {self.path}")
            return
        self.lock.__exit__(kind, error, traceback)
