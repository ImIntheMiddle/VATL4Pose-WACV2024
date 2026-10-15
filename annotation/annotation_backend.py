"""Run tracking and the active learning loop for a session and manage its Label Studio editor."""

from __future__ import annotations

import http.cookiejar
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any

from annotation.labelstudio.ls_setup import call
from annotation.cycle import atl_commands
from vatl4pose.execution import gpu
from vatl4pose.execution.process import run_child, _stat, _cleanup
from vatl4pose.execution.status import atomic_json, read_json, process_identity


def preview_command(source: Path, target: Path, encoder: str) -> list[str]:
    return [
        encoder,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]


class NoIdleGpu(RuntimeError):
    """All available devices are occupied or have retained leases."""


def idle_gpu() -> int:
    """Choose a candidate; GpuLease checks it again under ownership."""
    for row in gpu._query(
        ["--query-gpu=index,uuid", "--format=csv,noheader"]
    ).splitlines():
        fields = [p.strip() for p in row.split(",")]
        if (
            len(fields) != 2
            or not fields[0].isdigit()
            or not fields[1].startswith("GPU-")
        ):
            raise ValueError(f"Invalid GPU inventory: {row}")
        index, uid = int(fields[0]), fields[1]
        try:
            gpu.check_free(uid)
        except RuntimeError:
            continue
        root = Path(os.environ.get("VATL4POSE_GPU_LEASE_ROOT", gpu.DEFAULT_LEASE_ROOT))
        if not (root / f"{socket.gethostname()}_{uid}.lock").exists():
            return index
    raise NoIdleGpu("No idle GPU is available. Waiting for a GPU to become free.")


class Backend:
    def __init__(
        self,
        root: Path,
        index: int | None,
        max_frames: int | None,
        learning_device: str,
    ) -> None:
        self.root, self.index, self.max_frames = root, index, max_frames
        self.learning_device = learning_device
        self.portal: Any = None
        self.child: subprocess.Popen | None = None
        self.start_time: str | None = None
        self.child_record: Path | None = None
        self.handle: Any = None
        self.base = ""
        self.token = secrets.token_hex(24)
        self.password = secrets.token_urlsafe(24)
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )

    def load_credentials(self, run: Path) -> None:
        path = run / "editor-credentials.json"
        if path.exists():
            data = read_json(path)
            self.token, self.password = data["token"], data["password"]
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({"token": self.token, "password": self.password}, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def _execute(
        self, session: Any, commands: list[list[str]], env: dict[str, str]
    ) -> None:
        for i, command in enumerate(commands):
            log = session.run / f"{time.time_ns()}_{i}.log"
            code = run_child(command, self.root.parent, env, log)
            if code != 0:
                lines = log.read_text(errors="replace").splitlines()[-12:]
                raise RuntimeError(
                    f"Processing exited with code {code}:\n" + "\n".join(lines)
                )

    def _process(self, session: Any, initial: bool) -> None:
        from annotation.app import sha256

        receipt = session.run / "prepared-receipt.json"
        artifacts = [
            "prepared/tracking_settings.json",
            "prepared/dataset/annotations/clip.json",
            "preview.mp4",
        ]
        if initial and receipt.exists():
            recorded = read_json(receipt)
            if (
                set(recorded) != set(artifacts)
                or not (session.run / "prepared/.done.json").is_file()
                or any(
                    not (session.run / name).is_file()
                    or sha256(session.run / name) != recorded[name]
                    for name in artifacts
                )
            ):
                raise ValueError(
                    "Prepared tracking artifacts changed; choose a new output directory"
                )
            self._process(session, False)
            return
        if not initial and self.learning_device == "cpu":
            self._cpu_learning(session)
            return
        session.update("checking_gpu")
        if self.index is not None:
            index = self.index
        else:
            while True:
                try:
                    index = idle_gpu()
                    session.update("checking_gpu")
                    break
                except NoIdleGpu:
                    session.update("waiting_gpu")
                    time.sleep(10)
        with gpu.GpuLease(index, recheck=False) as lease:
            env = {
                **os.environ,
                **lease.env,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            if initial:
                session.update("tracking")
                command = [
                    str(self.root / ".venv/bin/python"),
                    str(self.root / "prepare_video.py"),
                    "--video",
                    str(session.video),
                    "--output",
                    str(session.run / "prepared"),
                    "--gpu",
                    str(index),
                ]
                if self.max_frames is not None:
                    command.extend(["--max-frames", str(self.max_frames)])
                self._execute(session, [command], env)
                dataset = session.run / "dataset"
                if not dataset.exists():
                    dataset.symlink_to(
                        Path("prepared/dataset"), target_is_directory=True
                    )
                import imageio_ffmpeg

                self._execute(
                    session,
                    [
                        preview_command(
                            session.run / "prepared/tracking.mp4",
                            session.run / "preview.mp4",
                            imageio_ffmpeg.get_ffmpeg_exe(),
                        )
                    ],
                    env,
                )
                atomic_json(
                    receipt, {name: sha256(session.run / name) for name in artifacts}
                )
            if self.learning_device == "cuda":
                session.update("learning")
                self._execute(
                    session,
                    atl_commands(
                        session.run, "cuda", self.root / ".venv/bin", session.target_oks
                    ),
                    env,
                )
        if self.learning_device == "cpu":
            self._cpu_learning(session)

    def _cpu_learning(self, session: Any) -> None:
        session.update("learning")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        env.pop(gpu.LEASE_PATH, None)
        env.pop(gpu.LEASE_TOKEN, None)
        self._execute(
            session,
            atl_commands(session.run, "cpu", self.root / ".venv/bin", session.target_oks),
            env,
        )

    def render_estimate(self) -> None:
        """Write the estimate video of the round just trained."""
        self.portal.estimate()

    def initial(self, session: Any) -> None:
        self._process(session, True)

    def learn(self, session: Any) -> None:
        self._process(session, False)

    def _start_studio(self, session: Any) -> None:
        if self.child is not None:
            if self.child.poll() is not None:
                raise RuntimeError(
                    "The annotation editor exited; restart the same command"
                )
            return
        env_dir = self.root / ".venv-labelstudio"
        self.load_credentials(session.run)
        if not (env_dir / "bin/label-studio").is_file():
            self._execute(
                session,
                [
                    ["uv", "venv", "--python", "3.12", str(env_dir)],
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(env_dir / "bin/python"),
                        "label-studio==1.23.2",
                    ],
                ],
                dict(os.environ),
            )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        data = session.run / "label-studio"
        data.mkdir(exist_ok=True)
        self.handle = (data / "server.log").open("a")
        env = {
            **os.environ,
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(
                (session.run / "dataset").resolve()
            ),
            "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN": "true",
            "COLLECT_ANALYTICS": "false",
            "LATEST_VERSION_CHECK": "false",
            "LABEL_STUDIO_CSRF_TRUSTED_ORIGINS": self.portal.url,
        }
        self.child = subprocess.Popen(
            [
                str(env_dir / "bin/label-studio"),
                "start",
                "--no-browser",
                "--internal-host",
                "127.0.0.1",
                "-p",
                str(port),
                "--data-dir",
                str(data),
                "--username",
                "annotator@localhost.test",
                "--password",
                self.password,
                "--user-token",
                self.token,
            ],
            env=env,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.start_time = _stat(self.child.pid)[19]
        self.child_record = session.run / "editor-process.json"
        atomic_json(
            self.child_record,
            {
                "schema": "vatl4pose.annotation-editor/1",
                "owner": {
                    **process_identity(),
                    "pid": self.child.pid,
                    "start_time": self.start_time,
                },
            },
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError(
                    f"Annotation editor exited; see {data / 'server.log'}"
                )
            try:
                with urllib.request.urlopen(
                    self.base + "/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise RuntimeError("Annotation editor did not start within five minutes")
        with self.opener.open(self.base + "/user/login/") as response:
            html = response.read().decode()
        match = re.search(
            r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)', html
        )
        if match is None:
            raise RuntimeError("Editor login form has no CSRF token")
        body = urllib.parse.urlencode(
            {
                "email": "annotator@localhost.test",
                "password": self.password,
                "csrfmiddlewaretoken": match.group(1),
                "persist_session": "on",
            }
        ).encode()
        request = urllib.request.Request(
            self.base + "/user/login/",
            data=body,
            headers={"Referer": self.base + "/user/login/"},
        )
        with self.opener.open(request) as response:
            response.read()
        if not any(c.name == "sessionid" for c in self.cookies):
            raise RuntimeError("Automatic editor login failed")

    def publish(self, session: Any, path: Path) -> int:
        from annotation.app import imported_tasks
        from annotation.labelstudio.queries_to_labelstudio import label_config

        self._start_studio(session)
        round_index = int(path.stem.split("_")[1])
        project_file = session.run / f"project_{round_index:02d}.json"
        if project_file.exists():
            project = read_json(project_file)
            # Existing editor DB, tasks and human answers must survive restart.
            call(self.base, self.token, "GET", f"/api/projects/{project['id']}")
            return project["id"]
        queries = read_json(path)
        category = queries["categories"][0]
        project = call(
            self.base,
            self.token,
            "POST",
            "/api/projects",
            {
                "title": f"{session.video.name} · round {round_index + 1}",
                "label_config": label_config(category["keypoints"]),
                # a skipped task is not saved, and the round needs every task saved
                "show_skip_button": False,
            },
        )
        pid = project["id"]
        call(
            self.base,
            self.token,
            "POST",
            "/api/storages/localfiles",
            {
                "project": pid,
                "path": str((session.run / "dataset/images").resolve()),
                "use_blob_urls": True,
                "regex_filter": ".*jpg",
            },
        )
        call(
            self.base,
            self.token,
            "POST",
            f"/api/projects/{pid}/import?commit_to_project=true",
            imported_tasks(queries, round_index),
        )
        atomic_json(project_file, {"id": pid})
        return pid

    def export_tasks(self, project: int) -> list[dict[str, Any]]:
        result = call(
            self.base,
            self.token,
            "GET",
            f"/api/projects/{project}/export?exportType=JSON&download_all_tasks=true",
        )
        if not isinstance(result, list):
            raise ValueError("Editor returned an invalid task export")
        return result

    def task_counts(self, project: int) -> dict[str, int]:
        """The saved tasks (with an answer that is not cancelled) and all tasks of a project."""
        data = call(self.base, self.token, "GET", f"/api/projects/{project}")
        return {"saved": data["finished_task_number"], "total": data["task_number"]}

    def cookie_header(self) -> str:
        return "; ".join(f"{c.name}={c.value}" for c in self.cookies)

    def close(self) -> None:
        if self.child is not None:
            try:
                if self.start_time is not None:
                    _cleanup(self.child.pid, self.start_time)
                elif self.child.poll() is None:
                    raise RuntimeError(
                        "The editor's process identity cannot be verified"
                    )
                self.child.wait()
                if self.child_record is not None:
                    self.child_record.unlink(missing_ok=True)
            finally:
                if self.handle is not None:
                    self.handle.close()
