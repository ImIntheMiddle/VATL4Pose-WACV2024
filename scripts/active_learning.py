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

# general libraries
import argparse
import os
import platform
import sys
import time
# python general libraries
import numpy as np
import torch
from tqdm import tqdm
from skimage.feature import peak_local_max

# additional functions
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
"""---------------------------------- Functions Set ----------------------------------"""
def parse_args():
    """
    parse given arguments before active learning execution
    return: args parsed by parser
    """
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('--cfg', type=str, required=True, default="configs/active_learning/al_settings.yaml",
                        help='experiment configure file name')
    parser.add_argument('--outdir', dest='outdir',
                        help='output-directory', default="examples/res/")
    parser.add_argument('--save_img', default=False, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str, default="coco", 
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    # video options:
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=True)
    args = parser.parse_args()
    if args.debug:
        import pdb # import python debugger
    return args

def setup_opt(opt):
    """Setup opt for active learning
    opt: parsed arguments
    return: opt  updated
    """
    opt.gpus = [int(i) for i in opt.gpus.split(',')]
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus[0] >= 0 else torch.device('cpu')
    if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
    cfg = update_config(opt.cfg)
    return opt, cfg

def initial_estimator(cfg):
    """Setup initial pose estimator.
    cfg: config file
    return: model
    """
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f'Loading model from {cfg.MODEL.PRETRAINED}...') # pretrained by MSCOCO2017
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED)) 
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    return model

"""---------------------------------- Main Process ----------------------------------"""
def main(): # rough flow of active learning
    """
    Setup for active learning.
    1. get experiment settings.
    2. prepare models, data loaders, lists for labeled & unlabeled data, respectively.
    3. set the stopping criteria of  active learning.
    """
    opt = parse_args() # get exp settings
    opt, cfg = setup_opt(opt) # update opt, cfg
    model = initial_estimator(cfg) # Load an initial pose estimator
    # Information about the dataset
    eval_joints = val_dataset.EVAL_JOINTS
    # Information about heatmap
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    heatmap_to_coord = get_func_heatmap_to_coord(cfg) # way to get final prediction from heatmap

    # Create the DataLoader for Prediction
    val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, num_workers=32, drop_last=False)
    print(val_loader.__len__())

    ###能動学習イテレーション###
    FINISH_AL = False
    while not FINISH_AL: # 終了条件が満たされない限り続ける
        break

        for sample in unlabeled_list:
            # 姿勢推定器によるUnlabeled dataの予測。index必ず取り出す   

            # 予測結果のヒートマップから局所ピークを拾う 局所ピークの座標が返ってくる
            local_peaks = peak_local_max(hp, min_distance=7) # min_distance: filter size
                # そのサンプルのindexをlabeledに追加。unlabeledから抜く。


        print('##### gt box: {} mAP #####'.format(gt_AP))

"""---------------------------------- Execution ----------------------------------"""
if __name__ == '__main__': # Do active learning
    """
    execute active learning for the input directory.
    after whole active learning process, print "finish".
    """
    main()
    print("Successfully finished!!")