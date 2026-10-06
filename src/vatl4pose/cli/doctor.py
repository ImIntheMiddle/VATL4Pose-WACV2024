"""``vatl4pose-doctor``: check the environment, the dataset roots and the weights.

Run it before a long job::

    vatl4pose-doctor paths.weights_root=/path/to/pretrained_models paths.data_root=/path/to/data

Options are ``key=value`` overrides of :class:`DoctorConfig`. The command prints one row per
check and exits with status 1 if any check failed. CUDA is informational: a machine without
a GPU passes.

Layout of ``paths.data_root``: ``PoseTrack21/`` and ``jrdb-pose/``, each with the
sub-directories the readers in :mod:`vatl4pose.data.annotations` open. Without
``paths.data_root`` the dataset checks are listed as skipped.
"""

from __future__ import annotations

import importlib
import pathlib
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from omegaconf import MISSING, OmegaConf

from vatl4pose.keypoints import get_spec

from .weights_manifest import load_release_manifest, sha256_file

_USAGE = """usage: vatl4pose-doctor paths.weights_root=DIR [paths.data_root=DIR]

Checks torch and CUDA, the dataset layout below paths.data_root, the files of the release
below paths.weights_root (SHA256) and the input dimension of every whole-body autoencoder
checkpoint. Exits with status 1 if a check failed."""

_DATASETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # (preset name, directory below data_root, sub-paths the readers open)
    ("posetrack21", "PoseTrack21", ("activelearning/val", "images/val")),
    ("jrdb_pose", "jrdb-pose", ("activelearning/test", "images/image_stitched")),
)

_AE_DATASETS: Mapping[str, str] = {
    "Posetrack21": "posetrack21",
    "JRDB2022": "jrdb_pose",
}
"""Directory below ``wholebodyAE/`` -> keypoint preset of the checkpoints inside."""

_RAW_VALUES_PER_JOINT = 3
"""The ``Raw`` autoencoder input is x, y and score of every joint (legacy ``kp_direct``: 51 for 17 joints)."""

_AE_ERRORS = (ValueError, RuntimeError, EOFError, OSError, pickle.UnpicklingError)


@dataclass
class PathsConfig:
    """Locations the doctor inspects."""

    weights_root: pathlib.Path = MISSING
    data_root: Optional[pathlib.Path] = None


@dataclass
class DoctorConfig:
    """Options of ``vatl4pose-doctor``."""

    paths: PathsConfig = field(default_factory=PathsConfig)


class CheckResult(NamedTuple):
    """Outcome of one check: its name, whether it passed, and a one-line explanation."""

    name: str
    ok: bool
    detail: str


def _check_torch() -> CheckResult:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return CheckResult("torch", False, f"import failed: {exc}")
    return CheckResult("torch", True, f"version {torch.__version__}")


def _check_cuda() -> CheckResult:
    """Report CUDA devices. This check never fails because of a missing GPU."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return CheckResult(
            "cuda", True, "unknown (torch is not importable, see the torch check)"
        )
    if not torch.cuda.is_available():
        return CheckResult("cuda", True, "not available (CPU only)")
    names = ", ".join(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )
    return CheckResult(
        "cuda",
        True,
        f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s): {names}",
    )


def _check_datasets(data_root: pathlib.Path | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for key, directory, subpaths in _DATASETS:
        if data_root is None:
            results.append(
                CheckResult(
                    f"data:{key}",
                    True,
                    "not configured (paths.data_root unset), skipped",
                )
            )
            continue
        for sub in subpaths:
            path = data_root / directory / sub
            detail = str(path) if path.is_dir() else f"missing directory {path}"
            results.append(CheckResult(f"data:{key}:{sub}", path.is_dir(), detail))
    return results


def _check_weight_files(
    weights_root: pathlib.Path, release_manifest: Mapping[str, str]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for rel, expected in release_manifest.items():
        path = weights_root / rel
        name = f"weight:{rel}"
        if not path.is_file():
            results.append(
                CheckResult(name, False, f"{path}: missing, expected SHA256 {expected}")
            )
            continue
        actual = sha256_file(path)
        if actual == expected:
            results.append(CheckResult(name, True, f"{path}: SHA256 ok"))
        else:
            results.append(
                CheckResult(
                    name,
                    False,
                    f"{path}: SHA256 mismatch, expected {expected}, actual {actual}",
                )
            )
    return results


def _check_autoencoders(weights_root: pathlib.Path) -> list[CheckResult]:
    """Check the input dimension of every ``Hybrid`` and ``Raw`` checkpoint of each dataset."""
    from vatl4pose.models.wholebody_ae import read_input_dim

    results: list[CheckResult] = []
    for directory, preset in _AE_DATASETS.items():
        spec = get_spec(preset)
        expected_dim = {
            "Hybrid": spec.ae_input_dim,
            "Raw": _RAW_VALUES_PER_JOINT * spec.num_joints,
        }
        for kind, expected in expected_dim.items():
            folder = weights_root / "wholebodyAE" / directory / kind
            for ckpt in sorted(folder.glob("*.pth")) if folder.is_dir() else []:
                name = f"ae:{ckpt.relative_to(weights_root)}"
                try:
                    actual = read_input_dim(ckpt)
                except _AE_ERRORS as exc:
                    results.append(
                        CheckResult(name, False, f"{ckpt}: cannot read: {exc}")
                    )
                    continue
                detail = f"{ckpt}: input dim {actual}"
                if actual == expected:
                    results.append(CheckResult(name, True, detail))
                else:
                    results.append(
                        CheckResult(
                            name,
                            False,
                            f"{detail}, expected {expected} ({spec.name} {kind} feature)",
                        )
                    )
    return results


def run_checks(
    cfg: DoctorConfig, release_manifest: Mapping[str, str] | None = None
) -> list[CheckResult]:
    """Run every check and return one result per row.

    Args:
        cfg: Weights and dataset locations.
        release_manifest: ``{relative path: SHA256}`` of the files that must exist below
            ``cfg.paths.weights_root``. ``None`` selects the manifest shipped with the package.
    """
    manifest = load_release_manifest() if release_manifest is None else release_manifest
    weights_root = pathlib.Path(cfg.paths.weights_root)
    data_root = (
        None if cfg.paths.data_root is None else pathlib.Path(cfg.paths.data_root)
    )
    torch_result = _check_torch()
    results = [torch_result, _check_cuda()]
    results.append(
        CheckResult("weights_root", weights_root.is_dir(), str(weights_root))
    )
    results += _check_datasets(data_root)
    results += _check_weight_files(weights_root, manifest)
    if torch_result.ok:
        results += _check_autoencoders(weights_root)
    else:
        results.append(
            CheckResult(
                "autoencoders", False, "torch is not importable, checkpoints not read"
            )
        )
    return results


def format_table(results: Sequence[CheckResult]) -> str:
    """Render ``results`` as a plain-text table followed by a summary line."""
    width = max(len(r.name) for r in results)
    lines = [f"{'status':<6}  {'check':<{width}}  detail"]
    for r in results:
        lines.append(f"{'ok' if r.ok else 'FAIL':<6}  {r.name:<{width}}  {r.detail}")
    failed = sum(not r.ok for r in results)
    lines.append(f"{len(results)} checks, {failed} failed")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``vatl4pose-doctor``; returns 1 if any check failed, else 0."""
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in ("-h", "--help") for arg in args):
        print(_USAGE)
        return 0
    merged = OmegaConf.merge(
        OmegaConf.structured(DoctorConfig), OmegaConf.from_dotlist(args)
    )
    cfg = OmegaConf.to_object(merged)
    assert isinstance(cfg, DoctorConfig), type(cfg)
    results = run_checks(cfg)
    print(format_table(results))
    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
