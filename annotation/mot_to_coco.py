"""Convert SAMIDARE MOT results into an unlabeled VATL4Pose dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import shutil

from PIL import Image


def convert(
    tracks: pathlib.Path, images: pathlib.Path, output: pathlib.Path, video_id: str
) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError(
            "video_id must contain only letters, digits, underscores or hyphens"
        )
    target = output / "annotations" / f"{video_id}.json"
    if target.exists():
        raise FileExistsError(target)
    files = sorted(images.glob("*.jpg"))
    image_rows = []
    sizes = {}
    for file in files:
        if not file.stem.isdigit():
            raise ValueError(f"frame filename must be numeric: {file}")
        frame = int(file.stem)
        if file.name != f"{frame:06d}.jpg":
            raise ValueError(f"frame filename must have six-digit zero padding: {file}")
        if frame < 1 or frame in sizes:
            raise ValueError(f"duplicate or invalid frame: {file}")
        with Image.open(file) as image:
            width, height = image.size
        sizes[frame] = (width, height)
        image_rows.append(
            {
                "id": frame,
                "file_name": f"images/{video_id}/{file.name}",
                "width": width,
                "height": height,
            }
        )
    if not sizes:
        raise ValueError(f"no JPG frames in {images}")
    if sorted(sizes) != list(range(1, len(sizes) + 1)):
        raise ValueError("frame images must be contiguous and start at 000001.jpg")
    annotations = []
    seen = set()
    with tracks.open() as handle:
        for number, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != 10:
                raise ValueError(f"{tracks}:{number}: expected 10 MOT columns")
            frame, track = int(row[0]), int(row[1])
            bbox = [float(x) for x in row[2:6]]
            if frame not in sizes:
                raise ValueError(f"{tracks}:{number}: frame {frame} has no image")
            if track < 1:
                raise ValueError(
                    f"{tracks}:{number}: expected a positive track id, got {track}"
                )
            if (frame, track) in seen:
                raise ValueError(
                    f"{tracks}:{number}: duplicate frame/track {frame}/{track}"
                )
            x, y, w, h = bbox
            width, height = sizes[frame]
            if (
                not all(math.isfinite(v) for v in bbox)
                or w <= 0
                or h <= 0
                or x >= width
                or y >= height
                or x + w <= 0
                or y + h <= 0
            ):
                raise ValueError(f"{tracks}:{number}: invalid bbox {bbox}")
            seen.add((frame, track))
            annotations.append(
                {"image_id": frame, "track_id": track, "bbox": bbox, "category_id": 1}
            )
    if not annotations:
        raise ValueError(f"{tracks}: no tracks")
    annotations.sort(key=lambda a: (a["image_id"], a["track_id"]))
    for index, annotation in enumerate(annotations, 1):
        annotation["id"] = index
    destination = output / "images" / video_id
    destination.mkdir(parents=True, exist_ok=True)
    if images.resolve() != destination.resolve():
        for file in files:
            if (destination / file.name).exists():
                raise FileExistsError(destination / file.name)
        for file in files:
            shutil.copy2(file, destination / file.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "images": image_rows,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=pathlib.Path, required=True)
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--video-id", default="clip")
    args = parser.parse_args()
    print(convert(args.tracks, args.images, args.output, args.video_id))


if __name__ == "__main__":
    main()
