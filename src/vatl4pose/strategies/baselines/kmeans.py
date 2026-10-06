"""K-Means filter: query the sample nearest to each of ``k`` cluster centers."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..base import RoundContext, unlabeled_for_query
from ..registry import register_selector

KMEANS_SEED = 318


def nearest_member_per_cluster(
    x: np.ndarray, labels: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Return, for each cluster, the position of its member closest to the cluster center.

    Distances are squared Euclidean. Among equally close members the lowest position wins.

    Args:
        x: ``(M, D)`` points.
        labels: ``(M,)`` cluster of every point, in ``[0, C)``.
        centers: ``(C, D)`` cluster centers.

    Returns:
        ``(C,)`` int64 positions into ``x``, in cluster order.

    Raises:
        ValueError: If a cluster has no member.
    """
    c = centers.shape[0]
    assert x.ndim == 2 and centers.ndim == 2 and x.shape[1] == centers.shape[1], (
        f"x {x.shape} and centers {centers.shape} must have the same dimension"
    )
    assert labels.shape == (x.shape[0],), f"labels: expected shape ({x.shape[0]},)"
    counts = np.bincount(labels, minlength=c)
    if counts.shape[0] != c or (counts == 0).any():
        empty = np.flatnonzero(counts[:c] == 0).tolist()
        raise ValueError(
            f"cluster(s) {empty} of {c} have no member; the embeddings have fewer than {c} "
            "distinct points"
        )
    distance = ((x - centers[labels]) ** 2).sum(axis=1)
    # Sort by cluster, then distance, then position: the first row of each cluster is its nearest
    # member (lowest position on a tie).
    order = np.lexsort((np.arange(x.shape[0]), distance, labels))
    first = np.cumsum(counts) - counts
    return order[first].astype(np.int64)


@register_selector("kmeans")
class KMeansSelector:
    """Cluster the unlabeled embeddings into ``k`` groups and query one sample per group.

    ``KMeans(n_clusters=k, random_state=318, n_init=10, algorithm="lloyd")`` clusters the
    embeddings of the unlabeled samples (in index order). The queried sample of a cluster is the
    member nearest to the cluster center. The result is in cluster order.
    """

    needs_embeddings = True

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        if ctx.embeddings is None:
            raise ValueError(
                "selector 'kmeans' needs ctx.embeddings; the pose model provides none"
            )
        x = ctx.embeddings[unlabeled]
        learner = KMeans(
            n_clusters=k, random_state=KMEANS_SEED, n_init=10, algorithm="lloyd"
        )
        labels = learner.fit_predict(x)
        return unlabeled[
            nearest_member_per_cluster(x, labels, learner.cluster_centers_)
        ]
