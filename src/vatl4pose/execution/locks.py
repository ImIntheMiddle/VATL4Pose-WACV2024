"""Owner-checked exclusive files for execution coordination on shared storage."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import uuid
from .status import process_identity


class OwnedLock:
    """Never remove a replacement lock and never automatically reclaim an existing one."""

    def __init__(self, path: Path, **details: Any) -> None:
        self.path = Path(path)
        self.data = {"owner": process_identity(), "token": uuid.uuid4().hex, **details}
        self.text = json.dumps(self.data)

    def __enter__(self) -> OwnedLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(
                f"Execution resource is locked: {self.path}; verify its owner before recovery"
            ) from None
        created = os.fstat(fd)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(self.text)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            if self.path.exists():
                found = self.path.stat()
                if (found.st_dev, found.st_ino) == (created.st_dev, created.st_ino):
                    self.path.unlink()
            raise
        return self

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> None:
        try:
            if not self.path.is_file() or self.path.read_text() != self.text:
                raise RuntimeError(
                    f"Execution resource ownership changed: {self.path}; it was not removed"
                )
            self.path.unlink()
        except Exception as release_error:
            if error is None:
                raise
            error.add_note(str(release_error))
