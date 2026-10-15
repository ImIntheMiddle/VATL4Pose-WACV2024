"""One local video annotation session using the existing tracking and ATL recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from annotation.labelstudio.queries_to_labelstudio import convert as make_tasks
from annotation.labelstudio.ls_export_to_answers import convert as make_answers
from annotation.cycle import WEIGHTS_ROOT
from vatl4pose.execution.status import atomic_json, read_json, owner_alive
from vatl4pose.execution.locks import OwnedLock
from vatl4pose.execution.process import _cleanup as cleanup_editor


def recover_annotation(run: Path) -> None:
    """Recover only a provably dead same-host owner; never reclaim a GPU lease."""
    path = run / ".annotation.lock"
    if not path.exists():
        return
    with OwnedLock(run / ".annotation-recovery.lock", command="annotation recovery"):
        text = path.read_text()
        record = read_json(path)
        if owner_alive(record.get("owner", {})) is not False:
            return
        child = run / "editor-process.json"
        if child.exists():
            data = read_json(child)
            owner = data.get("owner", {})
            if (
                data.get("schema") != "vatl4pose.annotation-editor/1"
                or owner_alive(owner) is None
            ):
                raise RuntimeError(
                    "The old editor's ownership cannot be verified; inspect it before recovery"
                )
            cleanup_editor(owner["pid"], owner["start_time"])
            child.unlink()
        if path.read_text() != text:
            raise RuntimeError("Annotation ownership changed during recovery")
        path.unlink()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def session_inputs() -> dict[str, str]:
    """SHA-256 of the files that fix what a session computes, for the identity of the session.

    The keys name the files relative to the repository, so a session resumes after the clone moves.
    A missing file raises, so the identity never rests on a file that is not there.
    """
    tool = [
        "app.py",
        "annotation_backend.py",
        "cycle.py",
        "prepare_video.py",
        "upstream.json",
        "models.json",
        "labelstudio/queries_to_labelstudio.py",
        "labelstudio/ls_export_to_answers.py",
    ]
    weights = [
        "posetrack21/simplebaseline_256x192.pth",
        "wholebodyAE/Posetrack21/Hybrid/WholeBodyAE_zdim4_v1.pth",
    ]
    inputs = {f"annotation/{name}": sha256(ROOT / name) for name in tool}
    inputs.update(
        {f"pretrained_models/{name}": sha256(WEIGHTS_ROOT / name) for name in weights}
    )
    return inputs


def imported_tasks(queries: dict[str, Any], round_index: int) -> list[dict[str, Any]]:
    tasks = make_tasks(
        queries,
        {"left_ear", "right_ear"} & set(queries["categories"][0]["keypoints"]),
        round_index,
    )
    for task in tasks:
        result = task.pop("annotations")[0]["result"]
        task["predictions"] = [{"result": result, "model_version": "vatl4pose"}]
    return tasks


def validate_export(
    tasks: list[dict[str, Any]], queries: dict[str, Any]
) -> dict[str, Any]:
    if any(
        not any(not a.get("was_cancelled") for a in t.get("annotations", []))
        for t in tasks
    ):
        raise ValueError(
            "Every task must be saved with Submit or Update before continuing"
        )
    return make_answers(tasks, queries)


class AnnotationSession:
    """Persist the same session for sample and user-supplied videos.

    After the training of every round the session stops at ``round_done`` and shows the results;
    the annotator then opens the next round (``continue``) or ends the session (``finish``). The
    session marks the rounds it opened in ``opened.json`` and the end in ``finished.json``, so that
    a restart comes back to the same point.
    """

    def __init__(
        self,
        video: Path,
        run: Path,
        backend: Any,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.video, self.run, self.backend = video.resolve(), run.resolve(), backend
        self.target_oks: float | None = (settings or {}).get("target_oks")
        self.oracle = (
            self.run / "atl/tracked/simplebaseline/THC+WPU_Coresetfilter/clip/oracle"
        )
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {
            "phase": "starting",
            "video": video.name,
            "round": None,
            "project": None,
            "error": None,
        }
        identity = {
            "schema": "vatl4pose.annotation-session/1",
            "video": str(self.video),
            "video_sha256": sha256(self.video),
            "settings": settings or {},
        }
        self.run.mkdir(parents=True, exist_ok=True)
        manifest = self.run / "session.json"
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("Video or settings changed; choose a new output directory")
        if not manifest.exists():
            atomic_json(manifest, identity)

    def update(self, phase: str, **fields: Any) -> None:
        with self.lock:
            self.state.update(phase=phase, **fields)
            atomic_json(self.run / "app-state.json", self.state)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def pending(self) -> Path | None:
        for path in sorted(self.oracle.glob("round_*_queries.json")):
            if not path.with_name(path.name.replace("_queries", "_answers")).exists():
                return path
        return None

    def publish(self) -> None:
        path = self.pending()
        if path is None:
            if (self.oracle.parent / ".done.json").is_file():
                self.update("completed", project=None)
                return
            raise RuntimeError(
                "Learning stopped without an annotation query or completion record"
            )
        self.update("opening_annotation")
        round_index = int(path.stem.split("_")[1])
        project = self.backend.publish(self, path)
        atomic_json(self.run / "opened.json", {"round": round_index})
        self.update(
            "waiting_annotation",
            round=round_index,
            project=project,
            error=None,
        )

    def opened(self) -> int:
        """The last round whose editor was opened, -1 before the first."""
        path = self.run / "opened.json"
        return read_json(path)["round"] if path.is_file() else -1

    def start(self) -> None:
        finished = self.run / "finished.json"
        if finished.is_file():
            self.update("completed", round=read_json(finished)["round"], project=None)
            return
        path = self.pending()
        if path is not None and 0 < int(path.stem.split("_")[1]) > self.opened():
            self.update("round_done", round=int(path.stem.split("_")[1]) - 1)
            return
        if path is not None or (self.oracle.parent / ".done.json").is_file():
            self.publish()
            return
        self.backend.initial(self)
        self.publish()

    def advance(self, choice: str = "advance") -> None:
        """Act on a request of the page: ``advance`` (train on the saved round), ``continue``
        or ``finish`` (the choice after a trained round)."""
        state = self.snapshot()
        if choice in ("continue", "finish"):
            if state["phase"] != "round_done":
                raise ValueError("No trained round waits for a choice")
            if choice == "continue":
                self.publish()
            else:
                atomic_json(self.run / "finished.json", {"round": state["round"]})
                self.update("completed", project=None, error=None)
            return
        if state["phase"] != "waiting_annotation":
            raise ValueError("The session is not waiting for annotations")
        path = self.pending()
        if path is None:
            raise ValueError("No pending annotation round")
        tasks = self.backend.export_tasks(state["project"])
        answers = validate_export(tasks, read_json(path))
        atomic_json(path.with_name(path.name.replace("_queries", "_answers")), answers)
        self.update("learning", project=None, error=None)
        self.backend.learn(self)
        if self.pending() is None:
            self.publish()  # the loop labeled everybody: the session is complete
        else:
            self.backend.render_estimate()  # the results page then shows the video at once
            self.update("round_done", error=None)

    def details(self, state: dict[str, Any]) -> dict[str, Any]:
        """What the page shows besides the phase: the saved tasks of the open round, the
        judgment of the stopping criterion in every trained round, and the estimator file."""
        tasks = None
        if state["phase"] == "waiting_annotation" and state["project"] is not None:
            tasks = self.backend.task_counts(state["project"])
        stopping = None
        if self.target_oks is not None:
            stopping = {"theta": self.target_oks, "rounds": self.judged_rounds()}
        estimator = self.oracle.parent / "estimator.pth"
        return {
            "tasks": tasks,
            "stopping": stopping,
            "estimator": str(estimator) if estimator.is_file() else None,
        }

    def judged_rounds(self) -> list[dict[str, Any]]:
        """One row per judged round of ``log.jsonl``, with the labeled share after the round."""
        run = self.oracle.parent
        if not (run / "log.jsonl").is_file():
            return []
        current = read_json(run / "state/current.json")["directory"]
        n = read_json(run / "state" / current / "state.json")["config"]["n_samples"]
        rows, labeled = [], 0
        for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            labeled += entry["n_queried"] or 0
            judgment = entry["stopping"]
            if judgment is not None:
                rows.append(
                    {
                        "round": entry["round"],
                        "labeled": 100 * labeled / n,
                        **{k: judgment[k] for k in ("people", "above", "lowest", "met")},
                    }
                )
        return rows


def main(argv: list[str] | None = None) -> None:
    from annotation.annotation_backend import Backend
    from annotation.annotation_web import Portal
    from vatl4pose.execution.process import _stop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gpu",
        type=int,
        help="Optional physical GPU index; otherwise choose an idle GPU",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--learning-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--target-oks",
        type=float,
        help="Accuracy θ of the stopping criterion of the paper, an OKS in (0, 1)",
    )
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if args.target_oks is not None and not 0 < args.target_oks < 1:
        parser.error("target-oks must be in (0, 1)")
    run = args.output or Path.cwd() / "outputs/annotation" / (
        args.video.stem + "-" + sha256(args.video)[:12]
    )
    settings = {
        "max_frames": args.max_frames,
        "device": args.learning_device,
        "target_oks": args.target_oks,
        "recipe": "existing tracking ATL defaults",
        "inputs": session_inputs(),
    }
    event = threading.Event()
    saved = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        saved[signum] = signal.getsignal(signum)
        signal.signal(signum, _stop)
    try:
        recover_annotation(run)
        with OwnedLock(run / ".annotation.lock", command="annotate"):
            backend = Backend(ROOT, args.gpu, args.max_frames, args.learning_device)
            session = AnnotationSession(args.video, run, backend, settings)
            with Portal(session, event, args.port) as portal:
                backend.portal = portal
                print(f"Open {portal.url} to annotate {args.video.name}", flush=True)
                print(
                    f"Remote access: ssh -L {portal.port}:localhost:{portal.port} USER@{os.uname().nodename}",
                    flush=True,
                )
                try:
                    session.start()
                except Exception as error:
                    session.update("failed", error=str(error))
                while True:
                    event.wait()
                    event.clear()
                    try:
                        session.advance(portal.choice)
                    except ValueError as error:
                        session.update(session.snapshot()["phase"], error=str(error))
                    except Exception as error:
                        session.update("failed", error=str(error))
                    finally:
                        portal.queued = False
    except (KeyboardInterrupt, SystemExit):
        print("Annotation session stopped; run the same command to resume", flush=True)
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
