"""A video of the track of one selected person, for the annotation portal.

The video runs from the first to the last frame of the track at the frame rate of the input video.
Only the box of the track is drawn and the frame outside it is dimmed; the frame of the query has
a thicker box and a label. The videos are written once per annotation id under ``tracklets/`` of
the session, because the tracking of a session does not change.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import cv2
import numpy as np

from annotation.videos import write_video, written_once

BOX = (0, 165, 255)  # BGR of #ffa500, the person box of the editor
DIM = 0.45  # brightness outside the box


def _draw(
    image: np.ndarray, box: list[float] | None, queried: bool, scale: float
) -> np.ndarray:
    if box is None:
        return (image * DIM).astype(np.uint8)
    h, w = image.shape[:2]
    x, y, bw, bh = box
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    out = (image * DIM).astype(np.uint8)
    out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    thickness = max(2, round((5 if queried else 2) * scale))
    cv2.rectangle(out, (x1, y1), (x2, y2), BOX, thickness, cv2.LINE_AA)
    if queried:
        size = 0.8 * scale
        (tw, th), base = cv2.getTextSize(
            "annotating", cv2.FONT_HERSHEY_SIMPLEX, size, 2
        )
        top = max(0, y1 - th - base - 8)
        cv2.rectangle(out, (x1, top), (x1 + tw + 12, top + th + base + 8), BOX, -1)
        cv2.putText(
            out,
            "annotating",
            (x1 + 6, top + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _render(dataset: Path, ann_id: int, fps: float, ffmpeg: str, target: Path) -> None:
    data = json.loads((dataset / "annotations/clip.json").read_text(encoding="utf-8"))
    query = next((a for a in data["annotations"] if a["id"] == ann_id), None)
    if query is None:
        raise KeyError(f"annotation {ann_id} is not in the tracking result")
    boxes = {
        a["image_id"]: a["bbox"]
        for a in data["annotations"]
        if a["track_id"] == query["track_id"]
    }
    frames = sorted(
        (i for i in data["images"] if min(boxes) <= i["id"] <= max(boxes)),
        key=lambda i: i["id"],
    )
    width, height = frames[0]["width"], frames[0]["height"]
    scale = height / 720

    def drawn() -> Iterator[np.ndarray]:
        for image in frames:
            frame = cv2.imread(str(dataset / image["file_name"]), cv2.IMREAD_COLOR)
            assert frame is not None, image
            yield _draw(
                frame, boxes.get(image["id"]), image["id"] == query["image_id"], scale
            )

    write_video(drawn(), width, height, fps, ffmpeg, target)


def tracklet_video(run: Path, ann_id: int, fps: float, ffmpeg: str) -> Path:
    """The video of the track of annotation ``ann_id``; written on the first request.

    Raises:
        KeyError: If the tracking result has no annotation ``ann_id``.
    """
    target = run / "tracklets" / f"{ann_id}.mp4"
    return written_once(
        target, lambda: _render(run / "dataset", ann_id, fps, ffmpeg, target)
    )
