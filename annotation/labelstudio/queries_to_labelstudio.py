"""Convert a vatl4pose ``round_XX_queries.json`` (COCO keypoints) to Label Studio tasks and a labeling config.

One task per queried person. The task shows the whole frame, the person box (read only) and the
prediction of the model as keypoint regions, named by the joints of the keypoint spec. The task data
keeps the annotation id of the queries file in ``ann_id`` (and ``vatl_sample_index``), because Label
Studio numbers its own tasks and annotations and the answers must keep the ids of the queries.

The joints in ``--skip-joints`` are not pre-filled (the PoseTrack21 layout never annotates the ears).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from xml.sax.saxutils import quoteattr

POINT_WIDTH = 0.5  # size of a keypoint in percent of the image width
IMAGE_URL = "/data/local-files/?d="  # Label Studio local file serving, relative to its document root


def joint_color(joint: str) -> str:
    """The colour of a joint in the editor: left joints blue, right red, the others green."""
    if joint.startswith("left"):
        return "#1f77b4"
    if joint.startswith("right"):
        return "#d62728"
    return "#2ca02c"


def label_config(joints: list[str]) -> str:
    """The labeling config: image, read-only person box, and one keypoint label per joint."""
    labels = "\n".join(
        f"    <Label value={quoteattr(j)} background={quoteattr(joint_color(j))} model_index={quoteattr(str(i))}/>"
        for i, j in enumerate(joints)
    )
    return f"""<View>
  <Header value="Person $ann_id in frame $image_id: move the points onto the joints, add missing ones, delete the ones you cannot place (select the point and press Backspace)"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="box" toName="image" strokeWidth="2">
    <Label value="person" background="#ffa500"/>
  </RectangleLabels>
  <KeyPointLabels name="kp" toName="image" strokeWidth="2">
{labels}
  </KeyPointLabels>
</View>
"""


def convert(queries: dict, skip_joints: set[str], round_index: int) -> list[dict]:
    """Label Studio tasks of a queries file; the prediction of the model is imported as an annotation."""
    joints = queries["categories"][0]["keypoints"]
    unknown = skip_joints - set(joints)
    if unknown:
        raise ValueError(f"--skip-joints names unknown joints: {sorted(unknown)}")
    images = {image["id"]: image for image in queries["images"]}
    tasks = []
    for ann in queries["annotations"]:
        image = images[ann["image_id"]]
        width, height = image["width"], image["height"]
        x, y, w, h = ann["bbox"]
        common = {
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
        }
        result = [
            {
                **common,
                "id": "box",
                "from_name": "box",
                "type": "rectanglelabels",
                "readonly": True,
                "value": {
                    "x": x / width * 100,
                    "y": y / height * 100,
                    "width": w / width * 100,
                    "height": h / height * 100,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
            }
        ]
        flat = ann["keypoints"]
        for j, name in enumerate(joints):
            px, py, visibility = flat[3 * j : 3 * j + 3]
            if name in skip_joints or visibility == 0:
                continue
            result.append(
                {
                    **common,
                    "id": f"kp_{name}",
                    "from_name": "kp",
                    "type": "keypointlabels",
                    "value": {
                        "x": px / width * 100,
                        "y": py / height * 100,
                        "width": POINT_WIDTH,
                        "keypointlabels": [name],
                    },
                }
            )
        tasks.append(
            {
                "data": {
                    "image": IMAGE_URL + image["file_name"],
                    "ann_id": ann["id"],
                    "vatl_sample_index": ann["attributes"]["vatl_sample_index"],
                    "image_id": ann["image_id"],
                    "file_name": image["file_name"],
                    "round": round_index,
                },
                "annotations": [{"result": result, "was_cancelled": False}],
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--out-tasks", type=pathlib.Path, required=True)
    parser.add_argument("--out-config", type=pathlib.Path, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_index")
    parser.add_argument(
        "--skip-joints", default="", help="comma separated joint names not to pre-fill"
    )
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    skip = {j for j in args.skip_joints.split(",") if j}
    tasks = convert(queries, skip, args.round_index)
    args.out_tasks.write_text(json.dumps(tasks), encoding="utf-8")
    args.out_config.write_text(
        label_config(queries["categories"][0]["keypoints"]), encoding="utf-8"
    )
    print(
        f"{len(tasks)} tasks -> {args.out_tasks}; labeling config -> {args.out_config}"
    )


if __name__ == "__main__":
    main()
