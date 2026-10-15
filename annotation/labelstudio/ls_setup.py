"""Create the Label Studio project of one round through the API and import its tasks.

Runs against a started Label Studio (``launch.sh`` starts it). It is safe to run again: a project
with the same title is kept, and tasks are imported only into an empty project.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request


def call(
    base: str, token: str, method: str, path: str, body: object | None = None
) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--tasks", type=pathlib.Path, required=True)
    parser.add_argument(
        "--images-dir", type=pathlib.Path, required=True, help="directory with the frames"
    )
    args = parser.parse_args()

    projects = call(args.url, args.token, "GET", "/api/projects?page_size=1000")[
        "results"
    ]
    project = next((p for p in projects if p["title"] == args.title), None)
    if project is None:
        project = call(
            args.url,
            args.token,
            "POST",
            "/api/projects",
            {
                "title": args.title,
                "label_config": args.config.read_text(encoding="utf-8"),
            },
        )
        print(f"created project {project['id']}: {args.title}")
    else:
        print(f"project {project['id']} exists: {args.title}")
    pid = project["id"]

    # local file serving is allowed only for the directories of a project's local storage
    storages = call(
        args.url, args.token, "GET", f"/api/storages/localfiles?project={pid}"
    )
    if not storages:
        call(
            args.url,
            args.token,
            "POST",
            "/api/storages/localfiles",
            {
                "project": pid,
                "title": "demo frames",
                "path": str(args.images_dir),
                "use_blob_urls": True,
                "regex_filter": ".*jpg",
            },
        )
        print(
            f"added local storage {args.images_dir} (no sync: the tasks come from the import)"
        )

    if call(args.url, args.token, "GET", f"/api/projects/{pid}")["task_number"] == 0:
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        info = call(
            args.url,
            args.token,
            "POST",
            f"/api/projects/{pid}/import?commit_to_project=true",
            tasks,
        )
        print(f"imported {info.get('task_count')} tasks")
    else:
        print("project already has tasks, nothing imported")
    print(f"project url: {args.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
