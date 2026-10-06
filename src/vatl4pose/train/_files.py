"""Output directory of the training runs of this package."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence


def prepare_output_dir(
    out_dir: pathlib.Path, names: Sequence[str]
) -> tuple[pathlib.Path, ...]:
    """Create ``out_dir`` and return the paths of the files ``names`` in it, in order.

    Raises:
        FileExistsError: If ``out_dir`` already holds one of the files. A run neither overwrites
            nor continues another run; the log of two runs in one file would be wrong.
    """
    out_dir = pathlib.Path(out_dir)
    present = [name for name in names if (out_dir / name).exists()]
    if present:
        raise FileExistsError(
            f"{out_dir} already holds {present}; training neither overwrites nor resumes a run"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return tuple(out_dir / name for name in names)
