"""Turn a Label Studio JSON export into the ``round_XX_answers.json`` that ``oracle=coco_file`` reads.

    ls_export_to_answers.py --export export.json --queries round_00_queries.json --out round_00_answers.json

Use the JSON export of Label Studio ("JSON", not "COCO"). Label Studio's own COCO export numbers the
annotations 0, 1, 2, ... in export order and cannot give the ids of the queries back, so the answer
of a task is matched through ``task.data.ann_id``, which ``queries_to_labelstudio.py`` wrote.

For each task the newest submitted annotation is used. A joint with a point becomes ``x, y, 2`` in
the frame pixels; a joint without a point becomes the position of the queries file with visibility 0
(not labeled). The script stops with a message if a task has no annotation, a joint has two points,
or the ids differ from the queries.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def convert(export: list[dict], queries: dict) -> dict:
    category = queries["categories"][0]
    joints: list[str] = category["keypoints"]
    images = {image["id"]: image for image in queries["images"]}
    asked = {ann["id"]: ann for ann in queries["annotations"]}

    answers: dict[int, dict] = {}
    for task in export:
        ann_id = task["data"]["ann_id"]
        if ann_id not in asked:
            raise ValueError(
                f"task {task['id']}: ann_id {ann_id} is not in the queries file"
            )
        if ann_id in answers:
            raise ValueError(
                f"ann_id {ann_id} appears in two tasks; export one project only"
            )
        ask = asked[ann_id]
        if task["data"].get("file_name") != images[ask["image_id"]]["file_name"]:
            raise ValueError(
                f"task {task['id']}: its frame is not the frame of annotation {ann_id}"
            )
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            raise ValueError(
                f"task {task['id']} (ann_id {ann_id}) has no submitted annotation"
            )
        annotation = max(done, key=lambda a: a.get("updated_at") or "")
        width, height = (
            images[ask["image_id"]]["width"],
            images[ask["image_id"]]["height"],
        )
        points: dict[str, tuple[float, float]] = {}
        for region in annotation["result"]:
            if region.get("type") != "keypointlabels":
                continue
            name = region["value"]["keypointlabels"][0]
            if name not in joints:
                raise ValueError(f"ann_id {ann_id}: unknown joint {name!r}")
            if name in points:
                raise ValueError(f"ann_id {ann_id}: joint {name!r} has two points")
            points[name] = (
                region["value"]["x"] / 100 * region["original_width"],
                region["value"]["y"] / 100 * region["original_height"],
            )
            assert (region["original_width"], region["original_height"]) == (
                width,
                height,
            )
        flat: list[float] = []
        for j, name in enumerate(joints):
            if name in points:
                flat += [points[name][0], points[name][1], 2]
            else:
                flat += [ask["keypoints"][3 * j], ask["keypoints"][3 * j + 1], 0]
        answers[ann_id] = {
            "id": ann_id,
            "image_id": ask["image_id"],
            "category_id": category["id"],
            "bbox": ask["bbox"],
            "keypoints": flat,
            "num_keypoints": len(points),
        }

    missing = sorted(set(asked) - set(answers))
    if missing:
        raise ValueError(f"no task for the annotation ids {missing}")
    return {
        "images": queries["images"],
        "annotations": [answers[i] for i in sorted(answers)],
        "categories": queries["categories"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", type=pathlib.Path, required=True)
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    export = json.loads(args.export.read_text(encoding="utf-8"))
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    answers = convert(export, queries)
    args.out.write_text(json.dumps(answers) + "\n", encoding="utf-8")
    print(f"{len(answers['annotations'])} answers -> {args.out}")


if __name__ == "__main__":
    main()
