import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pdb
import torch
from scipy.stats import spearmanr

def plot_learning_curves(savedir, video_id, strategy, percentages, performances, ann=False):
    """Plot the learning curves"""
    fig, ax = plt.subplots()
    c = ["blue","orange","green","red","black"]      # 各プロットの色
    ax.set_xlabel('Label Percentage (%)')  # x軸ラベル
    ax.set_ylabel('AP Performance (%)')  # y軸ラベル
    ax.set_title(f'Active Learning Result on {video_id}') # グラフタイトル
    ax.grid()            # 罫線
    ax.set_xlim(0, 100)  # x軸の範囲
    ax.set_ylim(0, 100)  # y軸の範囲
    ax.plot(percentages, performances, label=strategy, color=c[0])

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    if ann:
        savepath = os.path.join(savedir, f'learning_curve_{strategy}_{video_id}_ann.png')
    else:
        savepath = os.path.join(savedir, f'learning_curve_{strategy}_{video_id}.png')
    plt.savefig(savepath)
    print(f'Experiment result saved to... {savepath}!\n')
    return savepath

def compute_alc(percentages, performances):
    """Compute the area under the learning curve of active learning"""
    # pdb.set_trace()
    alc = metrics.auc(0.01*np.array(percentages), 0.01*np.array(performances)) # ALC: 0 ~ 1
    # print(f'[Evaluation] ALC: {alc:.3f}!')
    return alc

OKS_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
OKS_vars = (OKS_sigmas * 2)**2
OKS_k = len(OKS_sigmas)

def compute_OKS(bb, predkpts, GTkpts):
    """Calculate OKS between predicted keypoints and GT keypoints.
    bb: gt bbox, [x, y, w, h] format
    """
    # print(f'bbox_ann: {bb}')
    d, g = np.array(predkpts), np.array(GTkpts)
    # if np.allclose(d, g): # predicted keypoints are the same as the GT keypoints
    #     oks = 1
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3] # check the coordinates and visibility of the keypoints
    k1 = np.count_nonzero(vg > 0) # count the number of keypoints that are visible in the gt
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2 # x0 and x1 are the x coordinates of the left and right boundaries of the ignore region
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    body_area = bb[2] * bb[3] # use bbox size instead
    xd = d[0::3]; yd = d[1::3]
    if k1>0: # check if there is at least one keypoint visible
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((OKS_k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    e_vec = (dx**2 + dy**2) / OKS_vars / (body_area+np.spacing(1)) * 0.5 # oks_top is the numerator of oks
    if k1 > 0: # check if there is at least one keypoint visible
        e_vec=e_vec[vg > 0] # only consider the visible keypoints in the gt
    oks = np.sum(np.exp(-e_vec)) / e_vec.shape[0] # oks is the final oks score. oks = 0 ~ 1
    return oks

def compute_Spearmanr(unc_dict, oks_dict):
    """compute Spearmanr correlation coefficient between uncertainty and OKS"""
    unc_list = []
    oks_list = []
    for key in unc_dict.keys(): # unc_dict[key] = uncertainty
        unc_list.append(unc_dict[key])
        oks_list.append(oks_dict[key]) # oks_dict[key] = OKS
    unc_list = np.array(unc_list)
    oks_list = np.array(oks_list)

    # compute Spearmanr correlation coefficient
    corr, pval = spearmanr(unc_list, oks_list)
    return corr

def compute_corr(unc_dict, oks_dict):
    """compute correlation coefficient between uncertainty and OKS"""
    # sort the uncertainty and OKS in ascending order based on key
    unc_list = []
    oks_list = []
    for key in unc_dict.keys(): # unc_dict[key] = uncertainty
        unc_list.append(unc_dict[key])
        oks_list.append(oks_dict[key]) # oks_dict[key] = OKS
    unc_list = np.array(unc_list)
    oks_list = np.array(oks_list)
    # print(f'unc_list: {unc_list}')
    # print(f'oks_list: {oks_list}')
    # pdb.set_trace()
    # compute correlation coefficient
    corr = np.corrcoef(unc_list, oks_list)[0,1]
    return corr