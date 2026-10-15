"""The explanation of one selected person as a single PNG, ``picks/round_XX_rank_YY.png``.

The picture has the frame with the person boxed and numbered, and for each criterion that the
strategy used its figure: THC (the heatmaps of the joints that changed most in the previous, this and
the next frame), WPU (the predicted and the reconstructed skeleton) and DUW (the map of the
embeddings with the numbers of the pick). A strategy without criteria gets the frame only.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from ..data.geometry import HEATMAP_SIZE, INPUT_SIZE
from .draw import (
    GAP,
    PICK,
    RECON,
    SIDE_COLOURS,
    THC_JOINTS,
    draw_pose,
    limb_sides,
    window,
)
from .figures import (
    DuwFigure,
    Pick,
    RoundData,
    Scene,
    ThcFigure,
    WpuFigure,
    person_boxes,
)

_WINDOW = 1600
"""Width in pixels of the part of a panorama that the picture shows around the person."""


def _rgb(colour: tuple[int, int, int]) -> tuple[float, float, float]:
    return colour[0] / 255, colour[1] / 255, colour[2] / 255


def _name(scene: Scene, joint: int) -> str:
    return scene.info.spec.joint_names[joint].replace("_", " ")


def render_pick(
    path: pathlib.Path,
    scene: Scene,
    rd: RoundData,
    pick: Pick,
    crops: dict[int, np.ndarray],
    read_frame: Callable[[int], np.ndarray],
) -> None:
    """Write the picture of one pick of a round to ``path``.

    Args:
        path: The PNG file.
        scene: The video.
        rd: The round of the pick.
        pick: The pick.
        crops: ``(256, 192, 3)`` uint8 crops of the people that the figures show, by sample.
        read_frame: The RGB frame at a position of ``scene.frames``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    has_right = pick.thc is not None or pick.wpu is not None or rd.map_xy is not None
    figure = Figure(figsize=(15.0 if has_right else 7.5, 6.4), dpi=110)
    outer = figure.add_gridspec(
        1,
        2 if has_right else 1,
        width_ratios=[1.3, 1.7] if has_right else [1.0],
        left=0.02,
        right=0.99,
        top=0.9,
        bottom=0.2 if has_right else 0.1,
    )
    frame_ax = figure.add_subplot(outer[0, 0])
    frame_ax.set_anchor("N")
    _frame_panel(frame_ax, scene, rd, pick, read_frame)
    if has_right:
        rows = int(pick.thc is not None) + int(
            pick.wpu is not None or rd.map_xy is not None
        )
        right = outer[0, 1].subgridspec(rows, 1, hspace=0.35)
        row = 0
        if pick.thc is not None:
            _thc_panel(figure, right[row], scene, pick.thc, crops)
            row += 1
        if pick.wpu is not None or rd.map_xy is not None:
            columns = int(pick.wpu is not None) + int(rd.map_xy is not None)
            lower = right[row].subgridspec(
                1, columns, width_ratios=[1.0, 1.6][-columns:], wspace=0.2
            )
            column = 0
            if pick.wpu is not None:
                _wpu_panel(
                    figure.add_subplot(lower[0, column]),
                    scene,
                    pick.wpu,
                    crops[pick.person],
                )
                column += 1
            if rd.map_xy is not None:
                _duw_panel(figure.add_subplot(lower[0, column]), scene, rd, pick)
    if pick.duw is not None:
        figure.text(0.02, 0.02, _duw_text(pick.duw, pick.rank), ha="left", va="bottom", fontsize=9)
    figure.suptitle(_title(scene, rd, pick), fontsize=12, x=0.02, ha="left")
    figure.savefig(path, bbox_inches="tight", facecolor="white")


def _title(scene: Scene, rd: RoundData, pick: Pick) -> str:
    text = (
        f"{scene.info.video_id} · {scene.info.strategy} · round {rd.index} "
        f"({rd.percentage:.1f} % labeled) · pick {pick.rank} of {rd.order.shape[0]}"
    )
    if not pick.scores:
        text += " · no criterion, the order of the selection"
    return text


def _frame_panel(
    ax: Axes,
    scene: Scene,
    rd: RoundData,
    pick: Pick,
    read_frame: Callable[[int], np.ndarray],
) -> None:
    spec, panorama = scene.info.spec, scene.info.panorama
    position = int(scene.frame_of[pick.person])
    image = read_frame(position)
    height, width = image.shape[:2]
    boxes = person_boxes(scene.bbox_crop)
    x0 = 0
    if panorama:
        centre = (boxes[pick.person, 0] + boxes[pick.person, 2]) / 2
        x0 = int(round(centre - _WINDOW / 2))
        view = np.ascontiguousarray(window(image, x0, x0 + min(_WINDOW, width), True))
    else:
        view = image.copy()

    def shift(x: np.ndarray | float) -> np.ndarray | float:
        return np.mod(x - x0, width) if panorama else x

    rank_of = {int(p): r for r, p in enumerate(rd.order.tolist(), start=1)}
    _, rows = scene.frames[position]
    for i in rows.tolist():
        xy = rd.coords[i].astype(np.float64)
        xy[:, 0] = shift(xy[:, 0])
        draw_pose(
            view,
            spec,
            xy,
            height / 480,
            width_x=width if panorama else None,
            shown=rd.shown[i],
        )
    ax.imshow(view)
    for i in rows.tolist():
        if i not in rank_of:
            continue
        x1, y1, x2, y2 = boxes[i].astype(np.float64)
        left = float(shift(x1))
        this = i == pick.person
        ax.add_patch(
            Rectangle(
                (left, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=_rgb(PICK),
                linewidth=3.0 if this else 1.2,
                alpha=1.0 if this else 0.8,
            )
        )
        # a box can reach past the frame (a crop margin), so the number stays inside the frame
        ax.text(
            min(max(left, 0.0), view.shape[1] - 1.0),
            min(max(y1, 0.0), view.shape[0] - 1.0),
            str(rank_of[i]),
            color="black",
            fontsize=11 if this else 8,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": _rgb(PICK), "edgecolor": "none", "pad": 1.5},
        )
    ax.set_axis_off()
    ax.set_title(
        f"frame {position + 1} of {len(scene.frames)}, track {scene.track_id[pick.person]}",
        fontsize=9,
    )


def _crop_axes(ax: Axes, crop: np.ndarray | None) -> None:
    """Show a ``(256, 192, 3)`` crop in crop pixels, or a note that the track has none."""
    height, width = INPUT_SIZE
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if crop is None:
        ax.text(
            width / 2, height / 2, "no frame", ha="center", va="center", color="gray"
        )
        return
    ax.imshow(crop, extent=(0, width, height, 0))


def _heat_overlay(heat: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """RGBA picture of a ``(64, 48)`` uint8 heatmap in one colour."""
    assert heat.shape == HEATMAP_SIZE and heat.dtype == np.uint8
    level = heat.astype(np.float32) / 255.0
    rgba = np.zeros((*heat.shape, 4), dtype=np.float32)
    rgba[..., :3] = _rgb(colour)
    rgba[..., 3] = level**0.7 * 0.85
    return rgba


def _thc_panel(
    figure: Figure,
    cell: SubplotSpec,
    scene: Scene,
    thc: ThcFigure,
    crops: dict[int, np.ndarray],
) -> None:
    height, width = INPUT_SIZE
    inner = cell.subgridspec(1, 3, wspace=0.05)
    titles = ("previous frame", "this frame", "next frame")
    axes = [figure.add_subplot(inner[0, c]) for c in range(3)]
    for c, (ax, sample) in enumerate(zip(axes, thc.people, strict=True)):
        _crop_axes(ax, crops[sample] if sample >= 0 else None)
        ax.set_title(titles[c], fontsize=9)
        if sample < 0:
            continue
        for t in range(thc.joints.shape[0]):
            ax.imshow(
                _heat_overlay(thc.heat[c, t], THC_JOINTS[t]),
                extent=(0, width, height, 0),
                interpolation="bicubic",
            )
    handles = [
        Patch(
            facecolor=_rgb(THC_JOINTS[t]),
            label=f"{_name(scene, int(j))}: {thc.values[j]:.1f}",
        )
        for t, j in enumerate(thc.joints.tolist())
    ]
    axes[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        title="THC: joints whose heatmap changed most (L1 difference to the neighbors;"
        " the overlay shows the positive part)",
        title_fontsize=8,
        columnspacing=1.2,
        handlelength=1.2,
    )


def _wpu_panel(ax: Axes, scene: Scene, wpu: WpuFigure, crop: np.ndarray) -> None:
    height, width = INPUT_SIZE
    _crop_axes(ax, crop)
    ax.imshow(np.ones((height, width, 3)), extent=(0, width, height, 0), alpha=0.6)
    for xy, colour, line_width in (
        (wpu.reconstructed, RECON, 3.5),
        (wpu.predicted, None, 1.8),
    ):
        for a, b, side in limb_sides(scene.info.spec):
            ax.plot(
                xy[[a, b], 0],
                xy[[a, b], 1],
                color=_rgb(colour or SIDE_COLOURS[side]),
                linewidth=line_width,
                solid_capstyle="round",
            )
    for joint in wpu.worst.tolist():
        ax.add_patch(
            Circle(
                tuple(wpu.predicted[joint]),
                14,
                fill=False,
                edgecolor=_rgb(GAP),
                linewidth=2.5,
            )
        )
    offsets, angles = float(wpu.offsets.sum()), float(wpu.angles.sum())
    ax.set_title(
        "WPU: predicted and reconstructed skeleton\n"
        f"offsets {offsets / (offsets + angles):.0%}, angles {angles / (offsets + angles):.0%}"
        " of the score",
        fontsize=9,
    )
    names = ", ".join(
        f"{_name(scene, j)} ({'angle' if wpu.angles[j] > wpu.offsets[j] else 'offset'})"
        for j in wpu.worst.tolist()
    )
    handles = [
        Line2D([], [], color=_rgb(RECON), linewidth=3.5, label="autoencoder"),
        Line2D([], [], color="gray", linewidth=1.8, label="estimator"),
        Line2D(
            [],
            [],
            color=_rgb(GAP),
            marker="o",
            markerfacecolor="none",
            linestyle="",
            label=f"largest parts: {names}",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,
    )


def _number(value: float, digits: int = 3) -> str:
    return "n/a" if np.isnan(value) else f"{value:.{digits}g}"


def _duw_panel(ax: Axes, scene: Scene, rd: RoundData, pick: Pick) -> None:
    assert rd.map_xy is not None
    xy = rd.map_xy
    picked = np.zeros(xy.shape[0], dtype=bool)
    picked[rd.order] = True
    ax.scatter(
        *xy[~rd.labeled & ~picked].T, s=7, color="#c8c8c8", label="not annotated"
    )
    ax.scatter(*xy[rd.labeled].T, s=9, color="#4a4a4a", label="annotated")
    ax.scatter(*xy[rd.order].T, s=46, color=_rgb(PICK), label="selected")
    shown = {p.person: p.rank for p in rd.picks}
    for person, rank in shown.items():
        ax.annotate(
            str(rank),
            xy[person],
            fontsize=6.5,
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.scatter(
        *xy[pick.person],
        s=190,
        facecolor=_rgb(PICK),
        edgecolor="black",
        linewidth=1.6,
        zorder=5,
    )
    ax.annotate(
        str(pick.rank),
        xy[pick.person],
        fontsize=8.5,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("embedding map (t-SNE; a projection)", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        markerscale=1.0,
    )


def _duw_text(duw: DuwFigure, rank: int) -> str:
    return (
        f"DUW of pick {rank}: distance {_number(duw.distance)} to the nearest annotated or earlier "
        f"pick, uncertainty {_number(duw.uncertainty)}\n"
        f"m = {_number(duw.moks)}, lam = {_number(duw.lam)}, "
        f"value = (1 - m) * distance + lam * m * uncertainty = {_number(duw.value)}"
    )
