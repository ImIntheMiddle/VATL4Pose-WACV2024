"""Serve one localhost URL for progress, editor, and advancing a saved round."""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


class Portal:
    def __init__(self, session: Any, event: threading.Event, port: int) -> None:
        self.session, self.event = session, event
        self.csrf = secrets.token_urlsafe(32)
        self.queued = False
        self.choice = "advance"  # the last request: advance, finish or continue
        self.request_lock = threading.Lock()
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def respond(self, code: int, data: bytes, kind: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def local_origin(self) -> bool:
                host = self.headers.get("Host", "")
                expected = {f"localhost:{portal.port}", f"127.0.0.1:{portal.port}"}
                origin = self.headers.get("Origin")
                return host in expected and (
                    origin is None or origin == f"http://{host}"
                )

            def do_GET(self) -> None:
                if not self.local_origin():
                    self.respond(403, b"Invalid origin", "text/plain")
                    return
                if self.path == "/":
                    self.respond(200, portal.page(), "text/html; charset=utf-8")
                elif self.path == "/__vatl__/state":
                    state = portal.session.snapshot()
                    state["queued"] = portal.queued
                    state["progress"] = portal.progress()
                    state.update(portal.details(state))
                    self.respond(200, json.dumps(state).encode(), "application/json")
                elif self.path == "/__vatl__/video":
                    self.video()
                elif urllib.parse.urlsplit(self.path).path == "/__vatl__/tracklet":
                    self.tracklet()
                elif urllib.parse.urlsplit(self.path).path == "/__vatl__/estimate":
                    try:
                        self.send_video(portal.estimate())
                    except FileNotFoundError as error:
                        self.respond(404, str(error).encode(), "text/plain")
                else:
                    self.proxy()

            def do_POST(self) -> None:
                if not self.local_origin():
                    self.respond(403, b"Invalid origin", "text/plain")
                    return
                phases = {
                    "/__vatl__/advance": "waiting_annotation",
                    "/__vatl__/finish": "round_done",
                    "/__vatl__/continue": "round_done",
                }
                if self.path in phases:
                    if self.headers.get("X-VATL4POSE-Session") != portal.csrf:
                        self.respond(403, b"Invalid session", "text/plain")
                        return
                    with portal.request_lock:
                        if (
                            portal.queued
                            or portal.session.snapshot()["phase"] != phases[self.path]
                        ):
                            self.respond(409, b"Session is busy", "text/plain")
                            return
                        portal.queued = True
                        portal.choice = self.path.rsplit("/", 1)[1]
                        portal.event.set()
                    self.respond(202, b"{}", "application/json")
                else:
                    self.proxy()

            def do_PATCH(self) -> None:
                if self.local_origin():
                    self.proxy()
                else:
                    self.respond(403, b"Invalid origin", "text/plain")

            do_DELETE = do_PATCH
            do_PUT = do_PATCH

            def video(self) -> None:
                path = portal.session.run / "preview.mp4"
                if not path.is_file():
                    self.respond(404, b"Tracking preview is not ready", "text/plain")
                    return
                self.send_video(path)

            def tracklet(self) -> None:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                ann = query.get("ann", [""])[0]
                if not ann.isdigit():
                    self.respond(400, b"ann must be an annotation id", "text/plain")
                    return
                try:
                    path = portal.tracklet(int(ann))
                except KeyError as error:
                    self.respond(404, error.args[0].encode(), "text/plain")
                    return
                self.send_video(path)

            def send_video(self, path: Path) -> None:
                size = path.stat().st_size
                start, end = 0, size - 1
                part = self.headers.get("Range")
                if part:
                    import re

                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", part)
                    if match is None:
                        self.respond(416, b"Unsupported byte range", "text/plain")
                        return
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else end, end)
                    if start > end:
                        self.respond(416, b"Invalid byte range", "text/plain")
                        return
                self.send_response(206 if part else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if part:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = handle.read(min(remaining, 1 << 20))
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

            def proxy(self) -> None:
                backend = portal.session.backend
                if not backend.base:
                    self.respond(503, b"Editor is starting", "text/plain")
                    return
                # Keep the target on the owned loopback server; never accept an arbitrary URL.
                if not self.path.startswith("/") or self.path.startswith("//"):
                    self.respond(400, b"Invalid path", "text/plain")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else None
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() in {"content-type", "x-csrftoken", "accept", "range"}
                }
                headers.update(
                    {
                        "Authorization": f"Token {backend.token}",
                        "Cookie": backend.cookie_header(),
                        "Referer": backend.base + "/",
                        "Origin": backend.base,
                    }
                )
                request = urllib.request.Request(
                    backend.base + self.path,
                    data=body,
                    method=self.command,
                    headers=headers,
                )
                opener = urllib.request.build_opener(NoRedirect)
                try:
                    response = opener.open(request, timeout=60)
                except urllib.error.HTTPError as error:
                    response = error
                except urllib.error.URLError:
                    self.respond(
                        502,
                        b"The editor stopped. Restart the same command.",
                        "text/plain",
                    )
                    return
                with response:
                    data = response.read()
                    self.send_response(response.code)
                    for name, value in response.headers.items():
                        if name.lower() in {
                            "content-type",
                            "content-range",
                            "accept-ranges",
                            "x-frame-options",
                        }:
                            self.send_header(name, value)
                        elif name.lower() == "location":
                            self.send_header(name, value.replace(backend.base, ""))
                    for cookie in backend.cookies:
                        if cookie.name == "csrftoken":
                            self.send_header(
                                "Set-Cookie",
                                f"csrftoken={cookie.value}; Path=/; SameSite=Lax",
                            )
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://localhost:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def progress(self) -> dict[str, Any]:
        """The start of the current phase, and the step of the loop and the tracked frame if they
        were written during this phase (older files belong to an earlier phase)."""
        run = self.session.run
        state = run / "app-state.json"
        out: dict[str, Any] = {"now": time.time(), "since": None, "loop": None, "frames": None}
        if not state.is_file():
            return out
        since = out["since"] = state.stat().st_mtime
        loop = self.session.oracle.parent / "progress.json"
        if loop.is_file() and loop.stat().st_mtime >= since:
            out["loop"] = json.loads(loop.read_text(encoding="utf-8"))
        logs = sorted(run.glob("*.log"), key=lambda path: path.stat().st_mtime)
        if logs and logs[-1].stat().st_mtime >= since:
            with logs[-1].open("rb") as handle:
                handle.seek(max(0, handle.seek(0, 2) - (1 << 16)))
                frames = re.findall(rb"=== Frame (\d+)/(\d+) ===", handle.read())
            if frames:
                out["frames"] = [int(frames[-1][0]), int(frames[-1][1])]
        return out

    def details(self, state: dict[str, Any]) -> dict[str, Any]:
        """What the page shows besides the phase (see ``AnnotationSession.details``)."""
        return self.session.details(state)

    def fps(self) -> float:
        import cv2

        capture = cv2.VideoCapture(str(self.session.video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        assert fps > 0, f"no frame rate in {self.session.video}"
        return fps

    def estimate(self) -> Path:
        """The video of the keypoints of everyone after the last trained round (see
        :mod:`annotation.estimates`)."""
        import imageio_ffmpeg

        from annotation.cycle import KEYPOINTS
        from annotation.estimates import estimate_video

        round_index = self.session.snapshot()["round"]
        if round_index is None:
            raise FileNotFoundError("No round is trained yet")
        session = self.session
        return estimate_video(
            session.run,
            session.oracle.parent / "state",
            session.run / "atl/cache/coco_custom/clip.h5",
            round_index,
            KEYPOINTS,
            self.fps(),
            imageio_ffmpeg.get_ffmpeg_exe(),
        )

    def tracklet(self, ann_id: int) -> Path:
        """The video of the track of annotation ``ann_id`` (see :mod:`annotation.tracklet`)."""
        import imageio_ffmpeg

        from annotation.tracklet import tracklet_video

        return tracklet_video(
            self.session.run, ann_id, self.fps(), imageio_ffmpeg.get_ffmpeg_exe()
        )

    def __enter__(self) -> Portal:
        self.thread.start()
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.session.backend.close()

    def page(self) -> bytes:
        template = (Path(__file__).parent / "portal.html").read_text(encoding="utf-8")
        return (
            template.replace("__VIDEO__", html.escape(self.session.video.name))
            .replace("__TOKEN__", json.dumps(self.csrf))
            .encode()
        )
