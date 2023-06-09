# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from .transforms import get_max_pred_batch, _integral_tensor

class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def calc_iou(pred, target):
    """Calculate mask iou"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().data.numpy()

    pred = pred >= 0.5
    target = target >= 0.5

    intersect = (pred == target) * pred * target
    union = np.maximum(pred, target)

    if pred.ndim == 2:
        iou = np.sum(intersect) / np.sum(union)
    elif pred.ndim == 3 or pred.ndim == 4:
        n_samples = pred.shape[0]
        intersect = intersect.reshape(n_samples, -1)
        union = union.reshape(n_samples, -1)

        iou = np.mean(np.sum(intersect, axis=1) / np.sum(union, axis=1))

    return iou


def mask_cross_entropy(pred, target):
    return F.binary_cross_entropy_with_logits(
        pred, target, reduction='mean')[None]


def evaluate_mAP(res_file, ann_type='bbox', ann_file='./data/coco/annotations/person_keypoints_val2017.json', silence=False):
    """Evaluate mAP result for coco dataset.

    Parameters
    ----------
    res_file: str
        Path to result json file.
    ann_type: str
        annotation type, including: `bbox`, `segm`, `keypoints`.
    ann_file: str
        Path to groundtruth file.
    silence: bool
        True: disable running log.

    """
    class NullWriter(object):
        def write(self, arg):
            pass

    # ann_file = os.path.join('./data/coco/annotations/', ann_file)

    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output
    else:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)

    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if isinstance(cocoEval.stats[0], dict):
        stats_names = ['AP', 'AP .5', 'AP .6', 'AP .7', 'AP .75', 'AP .8', 'AP .95', 'AP (M)', 'AP (L)','AR']
        parts = ['body', 'foot', 'face', 'hand', 'fullbody']

        info = {}
        for i, part in enumerate(parts):
            info[part] = cocoEval.stats[i][part][0]
        return info
    else:
        stats_names = ['AP', 'AP .5', 'AP .6', 'AP .7', 'AP .75', 'AP .8', 'AP .95', 'AP (M)', 'AP (L)','AR']
        info_str = {}
        for ind, name in enumerate(stats_names):
            info_str[name] = cocoEval.stats[ind]
        return info_str


def calc_accuracy(preds, labels, thr=0.5):
    """Calculate heatmap accuracy."""
    preds = preds.cpu().data.numpy() # preds.shape = (batch_size, 17, 64, 48)
    labels = labels.cpu().data.numpy() # labels.shape = (batch_size, 17, 64, 48)

    num_joints = preds.shape[1] # preds.shape[1] = 17

    norm = 1.0
    hm_h = preds.shape[2]
    hm_w = preds.shape[3]

    preds, _ = get_max_pred_batch(preds) # preds.shape = (batch_size, 17, 2) (x, y)
    labels, _ = get_max_pred_batch(labels)
    norm = np.ones((preds.shape[0], 2)) * np.array([hm_w, hm_h]) / 10 # norm.shape = (batch_size, 2), norm = [6.4, 4.8]

    dists = calc_dist(preds, labels, norm) # calc_dist: calculate normalized distances

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i], thr=thr) # dist_acc: calculate accuracy with given input distance
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_integral_accuracy(preds, labels, label_masks, output_3d=False, norm_type='softmax'):
    """Calculate integral coordinates accuracy."""
    def integral_op(hm_1d):
        hm_1d = hm_1d * torch.cuda.comm.broadcast(torch.arange(hm_1d.shape[-1]).type(
            torch.cuda.FloatTensor), devices=[hm_1d.device.index])[0]
        return hm_1d

    preds = preds.detach()
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]

    if output_3d:
        hm_depth = hm_height
        num_joints = preds.shape[1] // hm_depth
    else:
        hm_depth = 1
        num_joints = preds.shape[1]

    with torch.no_grad():
        pred_jts, _ = _integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth, integral_op, norm_type=norm_type)

    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    if output_3d:
        coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    else:
        coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 2), 2))
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    if output_3d:
        labels = labels.cpu().data.numpy().reshape(preds.shape[0], num_joints, 3)
        label_masks = label_masks.cpu().data.numpy().reshape(preds.shape[0], num_joints, 3)

        labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
        labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height
        labels[:, :, 2] = (labels[:, :, 2] + 0.5) * hm_depth

        coords[:, :, 2] = (coords[:, :, 2] + 0.5) * hm_depth
    else:
        labels = labels.cpu().data.numpy().reshape(preds.shape[0], num_joints, 2)
        label_masks = label_masks.cpu().data.numpy().reshape(preds.shape[0], num_joints, 2)

        labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
        labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height

    coords = coords * label_masks
    labels = labels * label_masks

    if output_3d:
        norm = np.ones((preds.shape[0], 3)) * np.array([hm_width, hm_height, hm_depth]) / 10
    else:
        norm = np.ones((preds.shape[0], 2)) * np.array([hm_width, hm_height]) / 10

    dists = calc_dist(coords, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i])
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_dist(preds, target, normalize):
    """Calculate normalized distances of coordinates between preds and target."""
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]): # each image
        for c in range(preds.shape[1]): # each joint
            if target[n, c, 0] > 1 and target[n, c, 1] > 1: # if the joint is visible
                normed_preds = preds[n, c, :] / normalize[n] # normalize the distance
                normed_targets = target[n, c, :] / normalize[n] # normalize the distance
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets) # calculate the distance(L2 norm). dists[c, n] has 0~1 value
            else:
                dists[c, n] = 0 # if the joint is not visible, set the distance to -1
    return dists # dists.shape = (17, batch_size)


def dist_acc(dists, thr=0.5):
    """Calculate accuracy with given input distance."""
    dist_cal = np.not_equal(dists, 0) # not_equal: return True if two arrays are element-wise not equal
    num_dist_cal = dist_cal.sum() # the number of joints that are visible
    if num_dist_cal > 0: # if at least one joint is visible: valid
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal # return the accuracy. np.less: return True if x1 < x2 element-wise
    else: # if all joints are not visible, return -1 as the accuracy: invalid
        return -1
