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
import optuna

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
    parser.add_argument('--filter', type=str, default="None", help='filter type') # None, Random, Diversity, weighted
    parser.add_argument('--video_id', type=str, required=True, help = 'id of the video for test')
    parser.add_argument('--wunc', type=float, default=0.01, help='weight of uncertainty')
    parser.add_argument('--retrain_thresh', type=float, default=1, help='retrain threshold')
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
    parser.add_argument("--stopping", action="store_true", help="turn on stopping criteria")
    parser.add_argument("--continual", action="store_true", help="continual fine-tuning")
    parser.add_argument("--optimize", action="store_true", help="optimize hyperparameters by optuna")
    parser.add_argument("--PCIT", action="store_true", help="use PCIT dataset")
    parser.add_argument("--fixed_lambda", action="store_true", help="eliminate dynamic Gc and use fixed lambda in DUW.")
    parser.add_argument("--THCvsWPU", choices=["const", "increase", "decrease"], default="const", help="control the balance between THC and WPU when combining them. 'increase' means increasing the weight of THC linearly")
    parser.add_argument("--vis_thc", action="store_true", help="visualize THC")
    parser.add_argument("--vis_wpu", action="store_true", help="visualize WPU")
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
        SEED = 42 # 318
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed fixed for reproducibility')
    return opt

def set_dir(cfg, opt):
    """setting up directory for active learning, and make directory if not exist
    Args:
        cfg (_type_): experiment configuration
        opt (_type_): experiment options

    Raises:
        ValueError: check the validity of the given strategy arguments
    """
    # Check validity of strategy
    if opt.uncertainty == "None" and opt.representativeness == "None": # raise error
        if opt.filter == "None":
            raise ValueError('Uncertainty, representativeness, and filter cannot be None at the same time! \n --> Please specify one of them.')
        else: # filter only
            opt.strategy = ""
    elif opt.uncertainty == "None": # representativeness only
        opt.strategy = opt.representativeness
    elif opt.representativeness == "None": # uncertainty only
        opt.strategy = opt.uncertainty
    else: # both uncertainty and representativeness
        opt.strategy = opt.uncertainty + '+' + opt.representativeness
    if opt.filter != "None": # add filter
        opt.strategy = opt.strategy + '_' + opt.filter + 'filter' # add filter

    # set get_pre_next flag if uncertainty contains TPC or THC_L1
    opt.get_prenext = True if 'TPC' in opt.uncertainty or 'THC' in opt.uncertainty else False

    if os.uname()[1] in ["dl30"]: # set dataset root directory
        if cfg.DATASET.TRAIN.TYPE == "JRDB2022":
            cfg.DATASET.TRAIN.ROOT = '/home-local/halo/jrdb-pose/'
            cfg.DATASET.EVAL.ROOT = '/home-local/halo/jrdb-pose/'
        elif cfg.DATASET.TRAIN.TYPE == "Posetrack21":
            cfg.DATASET.TRAIN.ROOT = '/home-local/halo/PoseTrack21/' # set dataset root directory
            cfg.DATASET.EVAL.ROOT = '/home-local/halo/PoseTrack21/'
    if opt.PCIT: # set dataset root directory
        cfg.DATASET.TRAIN.ROOT = 'data/PCIT/' # set dataset root directory
        cfg.DATASET.EVAL.ROOT = 'data/PCIT/'

    # Set up experiment directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # get current date and time
    if opt.optimize: # optimize hyperparameters
        opt.work_dir = 'exp/AL_{}/{}/{}/{}/{}'.format(opt.memo, cfg.MODEL.TYPE, opt.strategy, "optimize", timestamp)
    else:
        opt.work_dir = 'exp/AL_{}/{}/{}/{}/{}'.format(opt.memo, cfg.MODEL.TYPE, opt.strategy, opt.video_id, timestamp)
    os.makedirs(opt.work_dir, exist_ok=False) # create unique experiment directory
    os.system('cp {} {}/'.format(opt.cfg, opt.work_dir)) # copy config file to experiment directory
    print(f'Result will be saved to: {opt.work_dir}!\n')
    return opt

def do_al(cfg, opt):
    al = ActiveLearning(cfg, opt) # Initialize active learning
    while True: # iterate until termination conditions have been met
        al.eval_and_query() # Evaluate pose estimator and get next query
        result = al.outcome()
        if result is not None: # The condition is met and break the loop.
            print('Active learning finished!')
            break
    return result

def hyper_objective(cfg, opt):
    def objective(trial):
        # randamize the video_id
        # opt.video_id = random.choice(opt.video_id_list)
        cfg.VAL.UNC_LAMBDA = trial.suggest_float('unc_lambda', 0.001, 100)
        # AE setting
        # cfg.AE.EPOCH = trial.suggest_int('epoch', 1, 25)
        # cfg.AE.LR = trial.suggest_float('lr', 0.00001, 0.0001)
        # print('video_id: {}'.format(opt.video_id), "unc_lambda: {}".format(cfg.VAL.UNC_LAMBDA))
        alc_list = []
        for video in opt.video_id_list:
            opt.video_id = video
            result = do_al(cfg, opt)
            ap95 = np.array(list(round_res["AP .95"] for round_res in result[2]))*100
            alc = compute_alc(result[0], ap95) # area under learning curve with annotation
            alc_list.append(alc)
        alc = np.mean(alc_list)
        # unc_mean = np.array(result[5]) # corrcoef_list
        # unc_norm = unc_mean/unc_mean[0]
        # unc_sum = np.sum(unc_norm)
        return alc
        # return unc_sum
    return objective

def optimize_alc(cfg, opt):
    search_space = {"unc_lambda": [0.001]}
    cfg.VAL.QUERY_RATIO = [0.05, 0.1, 0.2, 0.3, 0.4, 1]
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
    # study = optuna.create_study(direction='maximize')
    study.optimize(hyper_objective(cfg,opt), n_trials=30)
    print("Best ALC: {}".format(study.best_value), "Best params: {}".format(study.best_params))
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(opt.work_dir, 'optuna_history.png'))
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(os.path.join(opt.work_dir, 'optuna_slice.png'))

def save_result(cfg, opt, result):
    result_json = {}
    result_json['config_file'] = opt.cfg
    result_json['video_id'] = opt.video_id
    result_json['strategy'] = opt.strategy
    result_json['model'] = cfg.MODEL.TYPE
    result_json['percentages'] = result[0] # percentages
    result_json['performances'] = result[1] # performances
    result_json['performances_ann'] = result[2] # performances with annotation
    # result_json['area_under_learning_curve'] = compute_alc(result[0], result[1]) # area under learning curve
    # result_json['area_under_learning_curve_ann'] = compute_alc(result[0], result[2]) # area under learning curve with annotation
    result_json['query_list'] = result[3] # query list for each cycle
    result_json['uncertaity'] = result[4] # uncertainty dict
    result_json['influence'] = result[6] # influence dict
    result_json['combine_weight'] = result[7] # combine weight
    result_json['mean_uncertaity'] = result[5] # mean uncertainty
    result_json['spearmanr'] = result[8] # spearmanr_list
    result_json['corrcoef'] = result[9] # corrcoef_list
    result_json['true_labeled'] = result[10] # true_labeled
    result_json['true_unlabeled'] = result[11] # true_unlabeled
    result_json['false_labeled'] = result[12] # false_labeled
    result_json['false_unlabeled'] = result[13] # false_unlabeled
    result_json['actual_finish'] = result[14] # finished time
    result_json['finished_minerror'] = result[15] # finished time
    result_json['finished_oursc'] = result[16] # finished time
    result_json['ospa'] = result[17] # ospa
    result_json['ospa_ann'] = result[18] # ospa score with annotation
    result_json['moks_queried'] = result[19] # moks queried
    # result_json['learning_curve_path'] = plot_learning_curves(opt.work_dir, opt.video_id, opt.strategy, result[0], result[1]) # plot learning curve
    # result_json['learning_curve_path_ann'] = plot_learning_curves(opt.work_dir, opt.video_id, opt.strategy, result[0], result[2], ann=True) # plot learning curve with annotation

    with open(os.path.join(opt.work_dir, 'result.json'), 'w') as f:
        json.dump(result_json, f) # save results to json file
    print('Result saved to: {}!'.format(os.path.join(opt.work_dir, 'result.json')))

def main(cfg, opt):
    opt = set_dir(cfg, opt)
    if opt.optimize: # optimize hyperparameters
        opt.video_id_list = open("configs/trainval_video_list.txt").read().splitlines() # get video id list
        optimize_alc(cfg, opt)
    else: # standard setting
        result = do_al(cfg, opt)
        save_result(cfg, opt, result) # save results
    # result = do_al(cfg, opt)
    # save_result(cfg, opt, result) # save results
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
    cfg.VAL.UNC_LAMBDA = opt.wunc # set weight of uncertainty
    if opt.vis:
        cfg.RETRAIN.BASE = 0
        cfg.RETRAIN.ALPHA = 0
    main(cfg, opt) # run active learning