"""能動学習を実行する本体部分のスクリプト。全てはここをベースにして開発してゆく。
1. 準備. PoseTrack21の該当するディレクトリから動画の各フレーム読み込み, 初期のデータローダーを作る
2. 準備. 初期姿勢推定器を用意する（MS COCO で事前学習済み） -> これを使って能動学習を行う
3. 準備. Unlabeled listとLabeled listを作る
4. メインの能動学習イテレーションを実行する
    4-1. Unlabeledデータに対し予測, 予測結果とヒートマップ用いてcriteriaとval_error計算
    4-2. 予測結果からmetricを計算. ここで能動学習の終了判定を行う．
    4-3. criteriaの上位から順にannotation対象を選択
    4-4. 選択結果に基づきUnlabeled list, Labeled list, Unlabeled Dataloader, Labeled Dataloaerを更新
    4-5. 更新されたLabeled Dataloaderでmodelを再学習し, modelを更新. AIFTのやり方に従う.
5. 能動学習を終了．能動学習の実行結果をグラフ，動画などに出力する．
"""

"""---------------------------------- Import Libraries ----------------------------------"""
# general libraries
import argparse
import os
import platform
import sys
import time
import random

# python general libraries
import numpy as np
import torch
from tqdm import tqdm
from skimage.feature import peak_local_max

# 3rd party libraries
from alipy.experiment import StoppingCriteria

# additional functions
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord)


"""---------------------------------- Functions Set ----------------------------------"""
def parse_args():
    """
    parse given arguments before active learning execution
    return: args parsed by parser
    """
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('--cfg', type=str, required=True, default="configs/active_learning/al_settings.yaml",
                        help='experiment configure file name')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    args = parser.parse_args()
    return args

def setup_opt(opt):
    """Setup opt for active learning
    opt: parsed arguments
    return: opt  updated
    """
    opt.gpus = [int(i) for i in opt.gpus.split(',')]
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus[0] >= 0 else torch.device('cpu')
    # default settings
    opt.format = "coco"
    opt.min_box_area = 0 # min box area to filter out
    opt.qsize = 1024 # the length of result buffer, where reducing it will lower requirement of cpu memory

    if opt.debug: # for efficient debug
        import pdb;pdb.set_trace() # import python debugger
        opt.vis = True # visualize option
        opt.profile = True
        opt.save_video = True
        opt.vis_fast = True
    return opt

"""---------------------------------- Main Class ----------------------------------"""
class ActiveLearning:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.model = self.initialize_estimator()
        self.stopping_criterion = StoppingCriteria(stopping_criteria=None)


        self.norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg) # function to get final prediction from heatmap

    def is_stop(self):
        return self.stopping_criterion.is_stop()

    def initialize_estimator(self): # Load an initial pose estimator
        """construct a initial pose estimator
        Args:
            cfg: configuration file
            opt: experiment option
        Returns:
            model: Initial pose estimator
        """
        model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        print(f'Loading model from {self.cfg.MODEL.PRETRAINED}...') # pretrained by MSCOCO2017
        model.load_state_dict(torch.load(self.cfg.MODEL.PRETRAINED))
        model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).cuda()
        return model

    def main(): # rough flow of active learning

        # Initializaton of active learning
        eval_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
        print(len(unlabel_dataset))
        self.eval_joints = self.unlabel_dataset.EVAL_JOINTS
        eval_loader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), 
                                                    drop_last=False, pin_memory=True)


"""---------------------------------- Execution ----------------------------------"""
if __name__ == '__main__': # Do active learning
    """
    1. get experiment settings.
    2. construct ActiveLearning class instance
    3. execute active learning.
    """
    opt = parse_args() # get exp settings
    opt = setup_opt(opt) # setup option
    cfg = update_config(opt.cfg) # update config
    if not os.path.exists(cfg.RESULT.OUTDIR):
        os.makedirs(cfg.RESULT.OUTDIR)

    # CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    al = ActiveLearning(cfg, opt) # initialize active learning state

    ##　初期性能の評価

    # active learning iteration continue until termination conditions have been met
    while not al.is_stop():
        ## モデルによるUnlabeledデータの予測。アノテーション対象のサンプルを出す

        ## インデックスの更新
        ## データセットの更新（教師付けに対応）

        ## モデルのFineTuningを行い、モデルを更新
        ## 予測対象動画への予測の評価
        ## 能動学習の状態を更新、保存（終了判定含む）

        print(f'al state: {al.is_stop()} -> continue!')

    # The condition is met and break the loop.

    ## 実験の評価、結果の保存

    print("Successfully finished!!")


"""---------------------------------- Memo ----------------------------------"""
    # for sample in unlabeled_list:
        # 姿勢推定器によるUnlabeled dataの予測。index必ず取り出す

        # 予測結果のヒートマップから局所ピークを拾う 局所ピークの座標が返ってくる
        # local_peaks = peak_local_max(hp, min_distance=7) # min_distance: filter size
            # そのサンプルのindexをlabeledに追加。unlabeledから抜く。