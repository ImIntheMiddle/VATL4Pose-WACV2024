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
import pdb # import python debugger
os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
import platform
import sys
import time
import random

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print detail information')
    parser.add_argument('--speedup', default=True, action='store_true',
                        help='apply speeding up technique like fix random seed, fast viz, etc.')
    parser.add_argument('--seedfix', default=False, action='store_true',
                        help='fix random seed')
    args = parser.parse_args()
    return args

def setup_opt(opt):
    """Setup opt for active learning
    opt: parsed arguments
    return: opt  updated
    """
    opt.gpus = [int(i) for i in opt.gpus.split(',')]
    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using GPU : {}'.format(opt.gpus))

    opt.format = "coco" # default settings
    opt.min_box_area = 0 # min box area to filter out
    opt.qsize = 1024 # the length of result buffer, where reducing it will lower requirement of cpu memory

    if opt.verbose: # for visualization and explanation
        opt.vis = True # visualize option
        opt.profile = True
        opt.save_video = True
    if opt.speedup: # for speed up
        opt.vis_fast = True
        opt.vis_thre = 0.3
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if opt.seedfix: # fix seed for reproducibility
        SEED = 318
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return opt

"""---------------------------------- Main Class ----------------------------------"""
class ActiveLearning:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.model = self.initialize_estimator()
        self.stopping_criterion = StoppingCriteria(stopping_criteria=None)
        pdb.set_trace()
        self.eval_dataset = builder.build_dataset(self.cfg.DATASET.EVAL, preset_cfg=self.cfg.DATA_PRESET, train=False, get_prenext=False)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.cfg.VAL.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(),
                                                    drop_last=False, pin_memory=True)
        if self.opt.verbose: # test dataset and dataloader
            test_dataset(self.eval_dataset)
            test_dataloader(self.eval_loader)

        self.eval_joints = self.eval_dataset.EVAL_JOINTS
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
        if self.opt.device == torch.device('cuda'):
            model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).cuda()
        return model

"""---------------------------------- Execution ----------------------------------"""

if __name__ == '__main__': # Do active learning
    """Execution of active learning
    1. Setting up experiment
    2. Initialize active learning
    3. Evaluation of the initial pose estimator
    4. start active learning
    """
    # 1. Setting up experiment
    opt = parse_args() # get exp settings
    opt = setup_opt(opt) # setup option
    cfg = update_config(opt.cfg) # update config
    if not os.path.exists(cfg.RESULT.OUTDIR):
        os.makedirs(cfg.RESULT.OUTDIR)

    # 2. Initialize active learning
    # pdb.set_trace()
    al = ActiveLearning(cfg, opt) # initialize active learning state

    # 3. Evaluation of the initial pose estimator

    # active learning iteration continue until termination conditions have been met
    while not al.is_stop():
        ## モデルによるUnlabeledデータの予測。アノテーション対象のサンプルを出す

        ## インデックスの更新
        ## データセットの更新（教師付けに対応）

        ## モデルのFineTuningを行い、モデルを更新
        ## 予測対象動画への予測の評価
        ## 能動学習の状態を更新、保存

        ## Judge whether to stop active learning
        print(f'al state: {al.is_stop()}')
        if al.is_stop(): # The condition is met and break the loop.
            print("Active Learning is finished!")
            break
        else:
            print(f'Active Learning is not finished yet.\nContinue!')

    ## 最終性能の評価、実験結果のまとめ、保存



"""---------------------------------- Memo ----------------------------------"""
    # for sample in unlabeled_list:
        # 姿勢推定器によるUnlabeled dataの予測。index必ず取り出す

        # 予測結果のヒートマップから局所ピークを拾う 局所ピークの座標が返ってくる
        # local_peaks = peak_local_max(hp, min_distance=7) # min_distance: filter size
            # そのサンプルのindexをlabeledに追加。unlabeledから抜く。

"""---------------------------------- Test ----------------------------------"""
def test_dataset(dataset):
    print(len(dataset))
    image, label, category = dataset[0]
    print(image.shape, type(image), label, category, len(image_dataset))
    plt.imshow(image.transpose(2, 1, 0).astype(np.uint8))

def test_dataloader(dataloader):
    print(len(dataloader))