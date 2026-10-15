"""The inputs that some query strategies need besides the heatmaps, and how the loop builds them.

* The WPU scorer reads a whole-body autoencoder. Every round the loop starts from the pretrained
  checkpoint and tunes it on the hybrid features of the labels of the labeled people, with the
  invisible joints masked out (:func:`load_pretrained_ae`, :func:`finetune_wholebody_ae`).
* The VL4Pose scorer reads the Gaussian over every link length that an auxiliary network predicts
  for every person. Every round the loop trains a new network on the labeled people and predicts
  all people (:func:`fit_link_gaussians`).

Every function here depends on its arguments and on the seeds it is given only. The global random
state of numpy and torch is left as it was.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from ..data.geometry import HEATMAP_SIZE, box_to_center_scale, get_affine_transform
from ..keypoints import KeypointSpec
from ..models.auxnet import AuxNet, predict_link_gaussians, train_auxnet
from ..models.wholebody_ae import WholeBodyAE, load_wholebody_ae, masked_mse
from ..strategies.baselines.vl4pose import LinkGaussians, vl4pose_links


def load_pretrained_ae(
    ckpt: pathlib.Path, spec: KeypointSpec, z_dim: int, device: torch.device
) -> WholeBodyAE:
    """Load the autoencoder checkpoint; the random initial weights it overwrites cost no random state."""
    with torch.random.fork_rng(devices=[]):
        return load_wholebody_ae(ckpt, spec, z_dim, device)


def finetune_wholebody_ae(
    ae: WholeBodyAE,
    features: np.ndarray,
    valid: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> list[float]:
    """Tune ``ae`` in place on hybrid features: Adam, masked mean squared reconstruction error.

    The features are standardised with the statistics stored in ``ae``, which stay as they are
    (the pretrained normalisation is fixed; the buffers are not parameters). An entry that is not
    valid is 0 in the input and does not enter the loss. Every epoch visits all rows once in a
    random order that depends on ``seed`` only (the last batch may be smaller). The autoencoder is
    left in evaluation mode.

    Args:
        ae: The autoencoder, usually just loaded from its pretrained checkpoint.
        features: ``(n, ae.input_dim)`` raw hybrid features, one row per labeled person, of the
            labels that the oracle gave.
        valid: ``(n, ae.input_dim)`` bool mask of the entries of ``features`` that hold a measured
            value, :func:`~vatl4pose.models.wholebody_ae.hybrid_valid_mask` of the label
            visibilities.
        epochs: Number of passes; 0 changes nothing.
        lr: Learning rate of Adam.
        batch_size: Rows per step.
        seed: Seed of the order of the rows.

    Returns:
        The mean loss of every epoch (``epochs`` floats), each weighted by the number of valid
        entries of its batches.

    Raises:
        ValueError: If ``features`` has no row or the wrong width, ``valid`` does not fit
            ``features``, or ``epochs`` is negative or ``batch_size`` is below 1.
    """
    if epochs < 0 or batch_size < 1:
        raise ValueError(
            f"epochs must be >= 0 and batch_size >= 1, got {epochs} and {batch_size}"
        )
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] != ae.input_dim
    ):
        raise ValueError(
            f"features: expected shape (n, {ae.input_dim}) with n >= 1, got {features.shape}"
        )
    if valid.shape != features.shape or valid.dtype != np.bool_:
        raise ValueError(
            f"valid: expected a bool array of shape {features.shape}, got {valid.dtype} "
            f"of shape {valid.shape}"
        )
    device = next(ae.parameters()).device
    mask = torch.from_numpy(valid).to(device)
    x = ae.standardize(
        torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device), mask
    )
    n = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    history: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        total, count = 0.0, 0
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            batch = x[rows]
            loss = masked_mse(ae(batch), batch, mask[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            entries = int(mask[rows].sum())
            total += float(loss.detach()) * entries
            count += entries
        history.append(total / count)
    ae.eval()
    return history


def image_to_heatmap_coords(bbox_xyxy: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Map joints from image pixels to the cells of the heatmaps the model predicts for the same crop.

    The crop of a person comes from its box with an affine map; this applies the map whose output
    is the ``64 x 48`` heatmap instead of the ``256 x 192`` crop (the heatmap has a quarter of the
    crop's resolution). It is the inverse of what :func:`vatl4pose.data.geometry.heatmaps_to_coords`
    does to a heatmap cell.

    Args:
        bbox_xyxy: ``(n, 4)`` float boxes the crops were made from (the ground-truth boxes).
        keypoints: ``(n, K, 3)`` or ``(n, K, 2)`` joints, x and y in image pixels first.

    Returns:
        ``(n, K, 2)`` float32 positions ``(x, y)`` = ``(column, row)`` in heatmap cells.
    """
    box = np.asarray(bbox_xyxy)
    joints = np.asarray(keypoints)
    assert box.ndim == 2 and box.shape[1] == 4, (
        f"bbox_xyxy: expected (n, 4), got {box.shape}"
    )
    assert (
        joints.ndim == 3
        and joints.shape[0] == box.shape[0]
        and joints.shape[2] in (2, 3)
    ), f"keypoints: expected (n, K, 2 or 3) with n={box.shape[0]}, got {joints.shape}"
    out = np.empty((box.shape[0], joints.shape[1], 2), dtype=np.float32)
    heatmap_size = (HEATMAP_SIZE[1], HEATMAP_SIZE[0])  # (width, height)
    for i in range(box.shape[0]):
        center, scale = box_to_center_scale(box[i])
        matrix = get_affine_transform(center, scale, 0.0, heatmap_size)
        out[i] = joints[i, :, :2].astype(np.float64) @ matrix[:, :2].T + matrix[:, 2]
    return out


def fit_link_gaussians(
    embeddings: np.ndarray,
    labeled: np.ndarray,
    labels: np.ndarray,
    bbox_gt: np.ndarray,
    spec: KeypointSpec,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    init_seed: int,
    train_seed: int,
) -> LinkGaussians:
    """Train a new auxiliary network on the labeled people and predict the link Gaussians of everybody.

    The auxiliary network of the VL4Pose baseline starts with the global average of the backbone
    output and so reads the ``(2048,)`` embedding that the inference pass returns with the
    heatmaps. The embeddings are given to it as ``(2048, 1, 1)`` maps, which it averages to the same
    values, so no backbone output of the video is kept. Its training targets are the link lengths
    of the labels, in heatmap cells (:func:`image_to_heatmap_coords`), for the links of
    :func:`~vatl4pose.strategies.baselines.vl4pose.vl4pose_links`; a link counts only when both of
    its joints are labeled visible. With nobody labeled the network keeps its random initial
    weights.

    Args:
        embeddings: ``(N, C)`` float32 embeddings of all people from this round's pass.
        labeled: ``(n,)`` int64 indices of the labeled people, ``n >= 0``.
        labels: ``(n, K, 3)`` float64 labels (x, y, visibility) of those people.
        bbox_gt: ``(N, 4)`` ground-truth boxes of all people.
        spec: Keypoint spec of the dataset.
        epochs: Training epochs.
        lr: Learning rate of Adam.
        batch_size: Samples per training step and per prediction batch.
        init_seed: Seed of the initial weights.
        train_seed: Seed of the order of the training batches.

    Returns:
        The predicted mean and log-variance of every link of every person, ``(N, L)`` each.
    """
    n_all, channels = embeddings.shape
    links = vl4pose_links(spec)
    assert labels.shape == (labeled.shape[0], spec.num_joints, 3), (
        f"labels: expected {(labeled.shape[0], spec.num_joints, 3)}, got {labels.shape}"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        aux = AuxNet(channels, len(links))
    features = torch.from_numpy(embeddings).reshape(n_all, channels, 1, 1)
    if labeled.shape[0] > 0:
        train_auxnet(
            aux=aux,
            features=features[torch.from_numpy(labeled)],
            gt_coords_hm=torch.from_numpy(
                image_to_heatmap_coords(bbox_gt[labeled], labels)
            ),
            links=links,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=train_seed,
            joint_exist=torch.from_numpy(labels[:, :, 2] > 0),
        )
    means: list[torch.Tensor] = []
    log_vars: list[torch.Tensor] = []
    for start in range(0, n_all, batch_size):
        mean, log_var = predict_link_gaussians(
            aux, features[start : start + batch_size]
        )
        means.append(mean)
        log_vars.append(log_var)
    return LinkGaussians(
        mean=torch.cat(means).numpy(), log_var=torch.cat(log_vars).numpy()
    )
