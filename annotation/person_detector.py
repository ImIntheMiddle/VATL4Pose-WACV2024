"""Select COCO person detections for SAMIDARE's detector input."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def person_detections(
    prediction: dict[str, torch.Tensor],
    person: int,
    threshold: float = 0.7,
) -> list[tuple[list[float], float]]:
    keep = (prediction["labels"] == person) & (prediction["scores"] >= threshold)
    return list(
        zip(
            prediction["boxes"][keep].cpu().tolist(),
            prediction["scores"][keep].cpu().tolist(),
            strict=True,
        )
    )
