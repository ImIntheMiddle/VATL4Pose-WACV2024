"""The settings fingerprint of an AL run directory: ``settings.json``.

A run directory holds the state and result of one video under one set of settings. The first start
writes ``settings.json``; every later start (a resume, or a check before skipping a finished video)
compares its settings with that file and raises an error that names the differing keys and asks for
a new ``run.name`` (see :func:`check_settings`).

The fingerprint covers the package version, dataset, split, video id, keypoint spec, panorama flag,
seed, estimator, autoencoder checkpoint, oracle, strategy, loop and retraining settings; weights
files enter by path and SHA-256. Settings that do not change what a round computes are left out
(``_LOOP_NEUTRAL``, ``_RETRAIN_NEUTRAL`` and the device), the same ones the loop leaves out when it
resumes.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Any, Optional

import vatl4pose
from vatl4pose.config import keypoint_spec, resolve_path
from vatl4pose.data.cache import spec_sha256
from vatl4pose.loop import is_done
from vatl4pose.loop.results import RESULT_NAME
from vatl4pose.train._common import write_atomic

if TYPE_CHECKING:
    from ._common import Job

SETTINGS_NAME = "settings.json"
_LOOP_NEUTRAL = ("infer_batch_size", "save_final_model", "save_estimator", "stopping", "save_round_details")
_RETRAIN_NEUTRAL = ("num_workers",)
_CHUNK = 1 << 20


@functools.lru_cache(maxsize=32)
def _sha256_of(path: str, mtime_ns: int, size: int) -> str:
    """SHA-256 of a file; the arguments after the path key the cache on the file's identity."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: pathlib.Path, what: str) -> str:
    """SHA-256 of ``path``, computed once per process for an unchanged file.

    Raises:
        FileNotFoundError: If ``path`` is not a file; ``what`` names the setting.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{what}: the file {path} does not exist")
    stat = path.stat()
    return _sha256_of(str(path), stat.st_mtime_ns, stat.st_size)


def _file(
    root: str, name: Optional[str], what: str
) -> tuple[Optional[str], Optional[str]]:
    if name is None:
        return None, None
    path = resolve_path(root, name)
    return str(path), file_sha256(path, what)


def fingerprint(job: "Job", video_id: str, split: Optional[str]) -> dict[str, Any]:
    """The settings of the run of one video as plain JSON data."""
    weights, weights_sha = _file(
        job.paths.weights_root, job.estimator.weights, "estimator.weights"
    )
    ae, ae_sha = _file(job.paths.weights_root, job.ae.checkpoint, "ae.checkpoint")
    loop = {
        k: v for k, v in dataclasses.asdict(job.loop).items() if k not in _LOOP_NEUTRAL
    }
    retrain = {
        k: v
        for k, v in dataclasses.asdict(job.retrain).items()
        if k not in _RETRAIN_NEUTRAL
    }
    spec = keypoint_spec(job.dataset)
    data = {
        "package_version": vatl4pose.__version__,
        "dataset": job.dataset.name,
        "split": split,
        "video_id": video_id,
        "keypoints": {"name": spec.name, "sha256": spec_sha256(spec)},
        "panorama": job.dataset.panorama,
        "seed": job.run.seed,
        "estimator": {
            "arch": job.estimator.arch,
            "weights": weights,
            "weights_sha256": weights_sha,
            "imagenet_init": job.estimator.imagenet_init,
        },
        "ae": {"checkpoint": ae, "checkpoint_sha256": ae_sha},
        "oracle": job.oracle.name,
        "strategy": dataclasses.asdict(job.strategy),
        "loop": loop,
        "retrain": retrain,
    }
    return json.loads(json.dumps(data, allow_nan=False))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def differences(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """``key: stored X, now Y`` for every dotted key whose value differs, sorted by key."""
    old, new = _flatten(stored), _flatten(current)
    lines = []
    for key in sorted(set(old) | set(new)):
        if key in old and key in new and old[key] == new[key]:
            continue
        before = repr(old[key]) if key in old else "<absent>"
        after = repr(new[key]) if key in new else "<absent>"
        lines.append(f"{key}: stored {before}, now {after}")
    return lines


def check_settings(run_dir: pathlib.Path, current: dict[str, Any]) -> bool:
    """Compare ``current`` with the ``settings.json`` of ``run_dir``.

    A run directory holds a run once it has a ``state`` directory, ``oracle`` files, a result or a
    done marker. A start that failed before that (a missing autoencoder checkpoint, weights that do
    not fit the estimator) leaves a ``settings.json`` that no run depends on, so other settings
    replace it and the corrected start needs no new ``run.name``.

    Returns:
        ``True`` if the file exists and matches, ``False`` if the directory holds no run yet and
        the caller has to write the settings.

    Raises:
        ValueError: If the settings differ from those of a run in the directory, or the directory
            holds a run that has no ``settings.json``.
    """
    leftovers = [
        name for name in ("state", "oracle", RESULT_NAME) if (run_dir / name).exists()
    ] + (["done marker"] if is_done(run_dir) else [])
    path = run_dir / SETTINGS_NAME
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        found = differences(stored, current)
        if not found:
            return True
        if leftovers:
            raise ValueError(
                f"{run_dir} was made with other settings ({'; '.join(found)}). Use another "
                f"run.name for the new settings, or delete {run_dir} to start this video again."
            )
        return False
    if leftovers:
        raise ValueError(
            f"{run_dir} holds a run ({', '.join(leftovers)}) but no {SETTINGS_NAME}, so its "
            "settings are unknown. Use another run.name, or delete the directory to start again."
        )
    return False


def write_settings(run_dir: pathlib.Path, current: dict[str, Any]) -> None:
    """Write ``settings.json`` atomically; called when a run starts for the first time."""
    text = json.dumps(current, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_atomic(run_dir / SETTINGS_NAME, lambda f: f.write(text.encode("utf-8")))
