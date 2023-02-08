import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pdb

def plot_learning_curves(savedir, video_id, strategy, percentages, performances):
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
    savepath = os.path.join(savedir, f'learning_curve_{strategy}_{video_id}.png')
    plt.savefig(savepath)
    print(f'Experiment result saved to... {savepath}!\n')
    return savepath

def compute_alc(percentages, performances):
    """Compute the area under the learning curve of active learning"""
    # pdb.set_trace()
    alc = metrics.auc(0.01*np.array(percentages), 0.01*np.array(performances)) # ALC: 0 ~ 1
    print(f'[Evaluation] ALC: {alc:.3f}!')
    return alc