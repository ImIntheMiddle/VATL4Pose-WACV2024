"""VL4Pose: the pose whose link lengths are least likely is queried first (Shukla et al., BMVC 2022).

An auxiliary network (:class:`~vatl4pose.models.auxnet.AuxNet`) predicts, for every skeleton link,
a Gaussian over the link length from the backbone features of the person. The score of a person is
the negative log-likelihood of the link lengths of its predicted pose under those Gaussians.

The predicted pose has one peak per joint: the arg-max cell of the heatmap. A link length is the
distance between the arg-max cells of its two joints, in heatmap cells (not in image pixels, which
depend on the crop box). With one peak per joint the likelihood of the pose is the product of the
likelihoods of its links, so the score is the sum over the links of ``gaussian_nll``.

The links are the skeleton links of the keypoint spec without those that touch a redundant joint
(:func:`vl4pose_links`): a joint that is never annotated gives a link no training target.
The loop builds the auxiliary network with ``len(vl4pose_links(spec))`` links, trains it on these
links, predicts the Gaussians of all people with
:func:`~vatl4pose.models.auxnet.predict_link_gaussians` and hands them to the scorer as
:class:`LinkGaussians` in ``ctx.extras["vl4pose"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...keypoints import KeypointSpec
from ...models.auxnet import gaussian_nll, link_lengths
from ..base import RoundContext
from ..registry import register_scorer

EXTRAS_KEY = "vl4pose"
"""Key of ``RoundContext.extras`` that holds the :class:`LinkGaussians` of the pool."""


@dataclass(frozen=True, eq=False)
class LinkGaussians:
    """The Gaussian over the length of every link, for every person of the pool.

    Attributes:
        mean: ``(N, L)`` float array, the predicted mean length of each link in heatmap cells.
        log_var: ``(N, L)`` float array, the predicted log-variance of the length.
    """

    mean: np.ndarray
    log_var: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "log_var"):
            value = getattr(self, name)
            assert (
                isinstance(value, np.ndarray)
                and value.ndim == 2
                and np.issubdtype(value.dtype, np.floating)
            ), (
                f"{name}: expected a float array of shape (N, L), got "
                f"{getattr(value, 'dtype', type(value))} {getattr(value, 'shape', None)}"
            )
        assert self.log_var.shape == self.mean.shape, (
            f"log_var: expected shape {self.mean.shape} of mean, got {self.log_var.shape}"
        )


def vl4pose_links(spec: KeypointSpec) -> tuple[tuple[int, int], ...]:
    """The links of the baseline: ``spec.skeleton`` without the links that touch a redundant joint.

    The order of the skeleton is kept. PoseTrack21 has 19 skeleton links, 4 of which end in an
    ear (a redundant joint), so it keeps 15; JRDB-Pose keeps its 15.

    Raises:
        ValueError: If no link is left.
    """
    redundant = set(spec.redundant_joints)
    links = tuple(
        (int(u), int(v))
        for u, v in spec.skeleton
        if u not in redundant and v not in redundant
    )
    if not links:
        raise ValueError(
            f"spec {spec.name!r} has no link left for VL4Pose: its skeleton has "
            f"{len(spec.skeleton)} links and the redundant joints are {sorted(redundant)}"
        )
    return links


def heatmap_peaks(heatmaps: torch.Tensor) -> torch.Tensor:
    """Arg-max cell of every heatmap (the first one on ties).

    The 2023 code took ``peak_local_max(min_distance=8)``, which skipped the cells near the border;
    this is the plain arg-max of the rest of the package.

    Args:
        heatmaps: ``(N, K, H, W)`` float32.

    Returns:
        ``(N, K, 2)`` float32 positions ``(x, y)`` = ``(column, row)`` in heatmap cells.
    """
    assert heatmaps.ndim == 4 and heatmaps.dtype == torch.float32, (
        f"heatmaps: expected a float32 tensor of shape (N, K, H, W), got {heatmaps.dtype} "
        f"{tuple(heatmaps.shape)}"
    )
    n, k, _, width = heatmaps.shape
    cell = heatmaps.reshape(n, k, -1).argmax(dim=-1)
    peaks = torch.stack([cell % width, cell // width], dim=-1).to(torch.float32)
    assert peaks.shape == (n, k, 2)
    return peaks


@register_scorer("vl4pose")
class VL4PoseScorer:
    """Negative log-likelihood of the predicted link lengths; larger means query first.

    Raises:
        ValueError: From :meth:`score`, if ``ctx.extras["vl4pose"]`` is missing, is not a
            :class:`LinkGaussians`, or does not have one Gaussian per person and link.
    """

    @torch.no_grad()
    def score(self, ctx: RoundContext) -> np.ndarray:
        if EXTRAS_KEY not in ctx.extras:
            raise ValueError(
                f"scorer 'vl4pose' needs ctx.extras[{EXTRAS_KEY!r}], the LinkGaussians of the "
                "pool; it is missing (the loop predicts them with the trained auxiliary network)"
            )
        gaussians = ctx.extras[EXTRAS_KEY]
        if not isinstance(gaussians, LinkGaussians):
            raise ValueError(
                f"scorer 'vl4pose' needs ctx.extras[{EXTRAS_KEY!r}] to be a LinkGaussians, "
                f"got {type(gaussians).__name__}"
            )
        links = vl4pose_links(ctx.spec)
        expected = (ctx.num_samples, len(links))
        if gaussians.mean.shape != expected:
            raise ValueError(
                f"ctx.extras[{EXTRAS_KEY!r}] has shape {gaussians.mean.shape}, expected "
                f"{expected}: one Gaussian per person and per link of vl4pose_links(spec) of spec "
                f"{ctx.spec.name!r}"
            )
        lengths = link_lengths(heatmap_peaks(ctx.heatmaps).cpu().double(), links)
        nll = gaussian_nll(
            lengths,
            torch.from_numpy(np.asarray(gaussians.mean, dtype=np.float64)),
            torch.from_numpy(np.asarray(gaussians.log_var, dtype=np.float64)),
        )
        return nll.numpy()
