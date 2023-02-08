"""能動学習を実行する本体部分のスクリプト。全てはここをベースにして開発してゆく。
1. 準備. PoseTrack21の該当するディレクトリから動画の各フレーム読み込み, 初期のデータローダーを作る
2. 準備. 初期姿勢推定器を用意する(PoseTrack21で事前学習済み) -> これを使って能動学習を行う
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
import platform
import sys
import time
import random
import json
from datetime import datetime

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from skimage.feature import peak_local_max
from cachetools import cached
from torch.multiprocessing import Pool, Process, set_start_method

# 3rd party libraries
from alipy.experiment import StoppingCriteria
from alipy.index import IndexCollection

# additional functions
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP, calc_accuracy, DataLogger
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord)
from alphapose.utils.vis import vis_frame_fast, vis_frame

# import custom libraries
from active_learning import ActiveLearning
from active_learning.al_metric import plot_learning_curves, compute_alc

"""---------------------------------- Functions Set ----------------------------------"""
def parse_args():
    """
    parse given arguments before active learning execution
    return: args parsed by parser
    """
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('--cfg', type=str, default="configs/al_simple.yaml",
                        help='experiment configure file name')
    parser.add_argument('--uncertainty', type=str, default="None", help='uncertainty type') # HP, TPC, THC_L1, WPU_hybrid
    parser.add_argument('--representativeness', type=str, default="None", help='representativeness type') # Influence, Random
    parser.add_argument('--filter', type=str, default="None", help='filter type') # None, Random, Diversity
    parser.add_argument('--video_id', type=str, required=True, help = 'id of the video for test')
    parser.add_argument('--verbose', action='store_true',
                        help='print detail information')
    parser.add_argument('--speedup', action='store_true',
                        help='apply speeding up technique like fast viz, etc. But lose reproducibility')
    parser.add_argument('--seedfix', action='store_true',
                        help='fix random seed')
    parser.add_argument('--vis', action='store_true',
                        help='visualize & save')
    parser.add_argument('--memo', type=str, default="test",
                        help='memo for this experiment')
    parser.add_argument("--onebyone", action="store_true", help="one by one annotation")
    args = parser.parse_args()
    return args

def setup_opt(opt):
    """Setup opt for active learning
    opt: parsed arguments
    return: opt  updated
    """
    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    opt.num_gpu = torch.cuda.device_count()
    print('Available GPU : {}'.format(opt.num_gpu))
    opt.gpus = [int(i) for i in range(opt.num_gpu)]
    print('Using GPU : {}'.format(opt.gpus))
    opt.format = "coco" # default settings
    opt.min_box_area = 0 # min box area to filter out
    opt.qsize = 1024 # the length of result buffer, where reducing it will lower requirement of cpu memory

    if opt.verbose: # for visualization and explanation
        opt.profile = True
        opt.save_video = True
        print('Verbose mode applied.')
    if opt.vis: # for visualization
        pass
    if opt.speedup: # for speed up
        opt.vis_fast = True
        opt.vis_thre = 0.3
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print('Speed up technique applied. (Not reproducible)')
    if opt.seedfix: # fix seed for reproducibility
        SEED = 318
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed fixed for reproducibility')
    return opt

def main(cfg, opt):
    # Set up strategy
    if opt.uncertainty == "None" and opt.representativeness == "None": # raise error
        raise ValueError('Uncertainty and representativeness cannot be None at the same time! \n --> Please specify one of them.')
    elif opt.uncertainty == "None": # representativeness only
        opt.strategy = opt.representativeness
    elif opt.representativeness == "None": # uncertainty only
        opt.strategy = opt.uncertainty
    else: # both uncertainty and representativeness
        opt.strategy = opt.uncertainty + '+' + opt.representativeness
    if opt.filter != "None": # add filter
        opt.strategy = opt.strategy + '_' + opt.filter + 'filter' # add filter

    # Set up experiment directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # get current date and time
    opt.work_dir = 'exp/AL_{}/{}/{}/{}/{}'.format(opt.memo, cfg.MODEL.TYPE, opt.strategy, opt.video_id, timestamp)
    os.makedirs(opt.work_dir, exist_ok=False) # create unique experiment directory
    os.system('cp {} {}/'.format(opt.cfg, opt.work_dir)) # copy config file to experiment directory
    print(f'Result will be saved to: {opt.work_dir}!\n')

    al = ActiveLearning(cfg, opt) # Initialize active learning
    while True: # iterate until termination conditions have been met
        al.eval_and_query() # Evaluate pose estimator and get next query
        result = al.outcome()
        if result is not None: # The condition is met and break the loop.
            print('Active learning finished!')
            break
    return result

def save_result(cfg, opt, result):
    result_json = {}
    result_json['config_file'] = opt.cfg
    result_json['video_id'] = opt.video_id
    result_json['strategy'] = opt.strategy
    result_json['model'] = cfg.MODEL.TYPE
    result_json['percentages'] = result[0] # percentages
    result_json['performances'] = result[1] # performances
    result_json['area_under_learning_curve'] = compute_alc(result[0], result[1]) # area under learning curve
    result_json['query_list'] = result[2] # query list for each cycle
    result_json['uncertaity'] = result[3] # uncertainty dict
    result_json['influence'] = result[5] # influence dict
    result_json['combine_weight'] = result[6] # combine weight
    result_json['mean_uncertaity'] = result[4] # mean uncertainty
    result_json['learning_curve_path'] = plot_learning_curves(opt.work_dir, opt.video_id, opt.strategy, result[0], result[1]) # plot learning curve

    with open(os.path.join(opt.work_dir, 'result.json'), 'w') as f:
        json.dump(result_json, f) # save results to json file
    print('Result saved to: {}!'.format(os.path.join(opt.work_dir, 'result.json')))

"""---------------------------------- Execution ----------------------------------"""
if __name__ == '__main__':
    # try:
    #     set_start_method('spawn') # for multiprocessing
    # except RuntimeError:
    #     pass

    # Setting up experiment
    opt = parse_args() # get exp settings
    opt = setup_opt(opt) # setup option
    cfg = update_config(opt.cfg) # update config
    result = main(cfg, opt) # run active learning
    # print(result)
    save_result(cfg, opt, result)