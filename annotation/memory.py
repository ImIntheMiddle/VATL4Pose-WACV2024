"""Limit retained tracking state without changing upstream source files."""

from __future__ import annotations

from functools import partial
from typing import Any


def configure_memory(predictor: Any, tracker: Any) -> None:
    """Offload SAM 2 state and retain boxes instead of historical output masks."""
    predictor.init_state = partial(
        predictor.init_state, offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    original = tracker.track_frame_sequential

    def track_frame(*args: Any, **kwargs: Any) -> Any:
        results, reset, debug = original(*args, **kwargs)
        for result in results:
            result.pop("mask", None)
        return results, reset, debug

    tracker.track_frame_sequential = track_frame
