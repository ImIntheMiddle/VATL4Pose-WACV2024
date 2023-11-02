import os
import sys
import copy
import math
from pathlib import Path

import torch
import scipy.io
import numpy as np
from tqdm import tqdm
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

import umap
from sklearn.decomposition import PCA

plt.style.use('ggplot')
plt.switch_backend('agg')


def visualize_image(image_info, save_dir, bbox=False, adjusttext=False):
    '''
    :param image_info: (dict)
    '''
    uv_to_xy = lambda uv: (uv[1], uv[0])

    mpii_skelton = [['head', 'neck', (0, 0, 1)], ['neck', 'thorax', (0, 0, 1)],
                    ['thorax', 'lsho', (0, 1, 0)], ['lsho', 'lelb', (0, 1, 1)], ['lelb', 'lwri', (0, 1, 1)],
                    ['thorax', 'rsho', (0, 1, 0)], ['rsho', 'relb', (1, 0, 0)], ['relb', 'rwri', (1, 0, 0)],
                    ['lsho', 'lhip', (0, 1, 0)], ['rsho', 'rhip', (0, 1, 0)],
                    ['pelvis', 'lhip', (0, 1, 0)], ['lhip', 'lknee', (1, 0, 1)], ['lknee', 'lankl', (1, 0, 1)],
                    ['pelvis', 'rhip', (0, 1, 0)], ['rhip', 'rknee', (1, 1, 0)], ['rknee', 'rankl', (1, 1, 0)]]

    lsp_skeleton = [['head', 'neck', (0, 0, 1)], ['lhip', 'rhip', (0, 1, 0)],
                    ['neck', 'lsho', (0, 1, 0)], ['lsho', 'lelb', (0, 1, 1)], ['lelb', 'lwri', (0, 1, 1)],
                    ['neck', 'rsho', (0, 1, 0)], ['rsho', 'relb', (1, 0, 0)], ['relb', 'rwri', (1, 0, 0)],
                    ['lsho', 'lhip', (0, 1, 0)], ['lhip', 'lknee', (1, 0, 1)], ['lknee', 'lankl', (1, 0, 1)],
                    ['rsho', 'rhip', (0, 1, 0)], ['rhip', 'rknee', (1, 1, 0)], ['rknee', 'rankl', (1, 1, 0)]]



    colour = {'rankl': (1, 1, 0), 'rknee': (1, 1, 0), 'rhip': (1, 1, 0),
              'lankl': (1, 0, 1), 'lknee': (1, 0, 1), 'lhip': (1, 0, 1), 'pelvis': (0, 1, 0),
              'rwri': (1, 0, 0), 'relb': (1, 0, 0), 'rsho': (1, 0, 0),
              'lwri': (0, 1, 1), 'lelb': (0, 1, 1), 'lsho': (0, 1, 1),
              'head': (0, 1, 1), 'neck': (0, 1, 1), 'thorax': (0, 1, 0)}

    os.makedirs(os.path.join(save_dir, 'skeleton_visualizations'), exist_ok=True)
    img_dump = os.path.join(save_dir, 'skeleton_visualizations')

    # Currently will iterate over MPII and LSPET and LSP
    for dataset_name_ in image_info.keys():
        # Iterate over all images
        for i in tqdm(range(len(image_info[dataset_name_]['img']))):

            fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False)
            ax.set_axis_off()

            img = image_info[dataset_name_]['img'][i]
            img_name = image_info[dataset_name_]['img_name'][i]
            img_pred = image_info[dataset_name_]['img_pred'][i]
            img_gt = image_info[dataset_name_]['img_gt'][i]
            img_dataset = image_info[dataset_name_]['dataset'][i]
            img_string = image_info[dataset_name_]['display_string'][i]

            # One list for each, ground truth and predictions
            text_overlay = []
            ax.set_title('Name: {}, Shape: {}, Dataset: {}'.format(img_string, str(img.shape), img_dataset),
                         color='orange')
            ax.imshow(img)


            if dataset_name_ == 'mpii':
                skeleton = mpii_skelton
            else:
                assert dataset_name_ == 'lsp' or dataset_name_ == 'lspet'
                skeleton = lsp_skeleton

            for link in skeleton:
                joint_1_name = link[0]
                joint_2_name = link[1]
                color = link[2]

                joint_1 = img_pred[joint_1_name][0]
                joint_2 = img_pred[joint_2_name][0]

                if joint_1[2] == 1 and joint_2[2] == 1:
                    joint_1 = uv_to_xy(joint_1)
                    joint_2 = uv_to_xy(joint_2)

                    ax.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], color=color)


            joint_names = list(colour.keys())
            for jnt in joint_names:
                if (dataset_name_ in ['lsp', 'lspet']) and (jnt in ['pelvis', 'thorax']):
                    continue
                for jnt_gt in img_gt[jnt]:
                    if jnt_gt[2] >= 0 and jnt_gt[1] >= 0 and jnt_gt[0] >= 0:
                        jnt_gt = uv_to_xy(jnt_gt)
                        text_overlay.append(ax.text(x=jnt_gt[0], y=jnt_gt[1], s=jnt, color=colour[jnt], fontsize=6))
                        ax.add_patch(Circle(jnt_gt[:2], radius=1.5, color=colour[jnt], fill=False))

            if bbox:
                for person_patch in range(image_info[dataset_name_]['bbox_coords'].shape[1]):
                    coords = image_info[dataset_name_]['bbox_coords'][i, person_patch]
                    ax.add_patch(Rectangle(xy=(coords[0], coords[1]), height=(coords[3] - coords[1]), width=(coords[2]-coords[0]),
                                           linewidth=1, edgecolor='r', fill=False))

            if adjusttext:
                adjust_text(text_overlay)

            plt.savefig(fname=os.path.join(img_dump, '{}'.format(img_string)),
                        facecolor='black', edgecolor='black', bbox_inches='tight', dpi=500)

            del fig, ax
            plt.close()


def heatmap_loss(combined_hm_preds, heatmaps, egl=False):
    '''

    :param combined_hm_preds:
    :param heatmaps:
    :param nstack:
    :return:
    '''

    calc_loss = lambda pred, gt: ((pred - gt) ** 2).mean(dim=[1, 2, 3])

    combined_loss = []
    nstack = combined_hm_preds.shape[1]

    for i in range(nstack):
        if egl:
            combined_loss.append(calc_loss(combined_hm_preds[:, i], heatmaps[:, i].to(combined_hm_preds[:, i].device)))
        else:
            combined_loss.append(calc_loss(combined_hm_preds[:, i], heatmaps.to(combined_hm_preds[:, i].device)))

    combined_loss = torch.stack(combined_loss, dim=1)
    return combined_loss


def heatmap_generator(joints, occlusion, hm_shape=(0, 0), img_shape=(0, 0)):
    '''

    :param joints:
    :return:
    '''

    def draw_heatmap(pt_uv, use_occlusion, hm_shape, sigma=1.75):
        '''
        2D gaussian (exponential term only) centred at given point.
        No constraints on point to be integer only.
        :param im: (Numpy array of size=64x64) Heatmap
        :param pt: (Numpy array of size=2) Float values denoting point on the heatmap
        :param sigma: (Float) self.joint_size which determines the standard deviation of the gaussian
        :return: (Numpy array of size=64x64) Heatmap with gaussian centred around point.
        '''

        im = np.zeros(hm_shape, dtype=np.float32)

        # If joint is absent
        if pt_uv[2] == -1:
            return im, 0

        elif pt_uv[2] == 0:
            if not use_occlusion:
                return im, 0

        else:
            assert pt_uv[2] == 1, "joint[2] should be (-1, 0, 1), but got {}".format(pt_uv[2])

        # Point around which Gaussian will be centred.
        pt_uv = pt_uv[:2]
        pt_uv_rint = np.rint(pt_uv).astype(int)

        # Size of 2D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Check whether gaussian intersects with im:
        if (pt_uv_rint[0] - (size//2) >= hm_shape[0]) or (pt_uv_rint[0] + (size//2) <= 0) \
                or (pt_uv_rint[1] - (size//2) > hm_shape[1]) or (pt_uv_rint[1] + (size//2) < 0):

            return im, 0

        else:
            # Generate gaussian, with window=size and variance=sigma
            u = np.arange(pt_uv_rint[0] - (size // 2), pt_uv_rint[0] + (size // 2) + 1)
            v = np.arange(pt_uv_rint[1] - (size // 2), pt_uv_rint[1] + (size // 2) + 1)
            uu, vv = np.meshgrid(u, v, sparse=True)
            z = np.exp(-((uu - pt_uv[0]) ** 2 + (vv - pt_uv[1]) ** 2) / (2 * (sigma ** 2)))
            z = z.T

            # Identify indices in im that will define the crop area
            top = max(0, pt_uv_rint[0] - (size//2))
            bottom = min(hm_shape[0], pt_uv_rint[0] + (size//2) + 1)
            left = max(0, pt_uv_rint[1] - (size//2))
            right = min(hm_shape[1], pt_uv_rint[1] + (size//2) + 1)

            im[top:bottom, left:right] = \
                z[top - (pt_uv_rint[0] - (size//2)): top - (pt_uv_rint[0] - (size//2)) + (bottom - top),
                  left - (pt_uv_rint[1] - (size//2)): left - (pt_uv_rint[1] - (size//2)) + (right - left)]

            return im, 1   # heatmap, joint_exist


    assert len(joints.shape) == 3, 'Joints should be rank 3:' \
                                   '(num_person, num_joints, [u,v,vis]), but is instead {}'.format(joints.shape)

    heatmaps = np.zeros([joints.shape[1], hm_shape[0], hm_shape[1]], dtype=np.float32)
    joints_exist = np.zeros([joints.shape[1]], dtype=np.uint8)

    # Downscale
    downscale = [(img_shape[0] - 1)/(hm_shape[0] - 1), ((img_shape[1] - 1)/(hm_shape[1] - 1))]
    joints /= np.array([downscale[0], downscale[1], 1]).reshape(1, 1, 3)

    # Iterate over number of heatmaps
    for i in range(joints.shape[1]):

        # Create new heatmap for joint
        hm_i = np.zeros(hm_shape, dtype=np.float32)

        # Iterate over persons
        for p in range(joints.shape[0]):
            hm_, joint_present = draw_heatmap(pt_uv=joints[p, i, :], use_occlusion=occlusion, hm_shape=hm_shape)
            joints_exist[i] = max(joints_exist[i], joint_present)
            hm_i = np.maximum(hm_i, hm_)

        heatmaps[i] = hm_i

    return heatmaps, joints_exist


def uv_from_heatmap(hm=None, threshold=None, img_shape=(256, 256)):
    '''

    :param hm:
    :param threshold:
    :param img_shape:
    :return:
    '''
    max_uv = arg_max(hm)
    corrected_uv = weight_avg_centre(hm=hm, max_uv=max_uv)

    if hm[int(corrected_uv[0]), int(corrected_uv[1])] < threshold:
        return np.array([-1, -1, -1])

    else:
        joints = np.array([corrected_uv[0], corrected_uv[1], 1])
        hm_shape = hm.shape
        upscale = [(img_shape[0] - 1) / (hm_shape[0] - 1), ((img_shape[1] - 1) / (hm_shape[1] - 1))]
        joints *= np.array([upscale[0], upscale[1], 1])

        return joints


def arg_max(img):
    '''
    Find the indices corresponding to maximum values in the heatmap
    :param img: (Numpy array of size=64x64) Heatmap
    :return: (Torch tensor of size=2) argmax of the image
    '''
    img = torch.tensor(img)
    assert img.dim() == 2, 'Expected img.dim() == 2, got {}'.format(img.dim())

    h = img.shape[0]
    w = img.shape[1]

    rawmaxidx = img.flatten().argmax()

    max_u = int(rawmaxidx) // int(w)
    max_v = int(rawmaxidx) % int(w)

    return torch.FloatTensor([max_u, max_v])


def fast_argmax(_heatmaps):
    """
    Direct argmax from the heatmap, does not perform smoothing of heatmaps
    :param _heatmaps:
    :return:
    """
    batch_size = _heatmaps.shape[0]
    num_jnts = _heatmaps.shape[1]
    spatial_dim = _heatmaps.shape[3]
    assert _heatmaps.shape[2] == _heatmaps.shape[3]

    assert len(_heatmaps.shape) == 4, "Heatmaps should be of shape: BatchSize x num_joints x 64 x64"
    _heatmaps = _heatmaps.reshape(batch_size, num_jnts, -1)
    indices = torch.argmax(_heatmaps, dim=2)
    indices = torch.cat(((indices // spatial_dim).view(batch_size, num_jnts, 1),
                         (indices % spatial_dim).view(batch_size, num_jnts, 1)),
                        dim=2)
    return indices.type(torch.float32)


def weight_avg_centre(hm, max_uv=None, jnt_size=1.75):
    '''
    Weighted average of points around the maxima. Weighted average avoids solitary spikes being identified.
    :param hm: (Numpy array of size 64x64)
    :param jnt_size: (Float) Windows size around the maxima to compute weighted average.
    :return: (Numpy array of size=2)
    '''

    hm = torch.clamp(torch.from_numpy(hm), min=0.0)
    mx = max_uv

    # Dimension of the heatmap
    siz = torch.Tensor([hm.shape[0], hm.shape[1]]).float()

    # Clip indices if needed so that start and end indices are valid points.
    st_idx = torch.max(torch.zeros(2), mx - np.ceil(jnt_size))
    end_idx = torch.min(siz - 1, mx + np.ceil(jnt_size))

    # Crop around the maxima.
    img_crop = hm[int(st_idx[0]):int(end_idx[0] + 1), int(st_idx[1]):int(end_idx[1] + 1)].clone()
    img_crop = img_crop.type(torch.FloatTensor)
    img_sum = img_crop.sum()
    if img_sum == 0:
        img_sum = img_sum + 0.000001

    # Weighted average along column/row
    u = img_crop.sum(1).mul(torch.arange(st_idx[0], end_idx[0] + 1)).div(img_sum).sum()
    v = img_crop.sum(0).mul(torch.arange(st_idx[1], end_idx[1] + 1)).div(img_sum).sum()

    return np.array([u, v])


def principal_component_analysis(encodings, n_components=2):
    '''
    Compute the principal component transform of the encodings
    :param encodings: Encodings generated by LLAL network
    :param n_components: Number of components to retain
    :return: Principal Component Transform of encodings
    '''
    pca = PCA(n_components=n_components)
    pca.fit(encodings)
    pca_encodings = pca.transform(encodings)

    return pca_encodings


def umap_fn(encodings, n_components=2):
    '''
    NUMPY
    https://umap-learn.readthedocs.io/en/latest/how_umap_works.html
    Compute the UMAP transform of the encodings
    :param encodings: Encodings generated by LLAL network
    :param n_components: Number of components to retain
    :return: UMAP transform of the encodings
    '''
    # Number of neighbours balances the local versus the global structure of the data
    umap_transform = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=n_components).fit(encodings)
    umap_encodings = umap_transform.transform(encodings)

    return umap_encodings


def shannon_entropy(probs):
    '''
    Computes the Shannon Entropy for a distribution
    :param probs_array: 2D-Tensor; Probability distribution along axis=1
    :return: Scalar; H(p)
    '''
    return torch.sum(-probs * torch.log(probs), dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pairwise_joint_distances(heatmaps):
    '''
        Computes pairwise distances between joints, given a batch of heatmaps
        - param heatmaps: Tensor of size (batch_size, num_joints, hm_size, hm_size)
        - return pairwise_dists: List[2D lists]
                                    - length = batch_size
                                    - Each 2D List (pairwise distances) of dim [num_joints,num_joints]
    '''

    # Batch Size, num_jnts, 64, 64
    assert heatmaps.dim() == 4, 'Dimension of input heatmaps must be 4 but was {}'.format(heatmaps.shape)

    # Batch Size, num_jnts, 2
    joint_uv = fast_argmax(heatmaps)
    pairwise_dists = torch.cdist(joint_uv, joint_uv)

    return pairwise_dists