"""能動学習を実行する本体部分のスクリプト。全てはここをベースにして開発してゆく。
1. PoseTrack21の該当するディレクトリから動画の各フレーム読み込み
2. モデルを用意する（MS COCO で事前学習済みの姿勢推定器）
3. 

"""
##general library##
import argparse
import os
import platform
import sys
import time

##python general###
import numpy as np
import torch
from tqdm import tqdm


##
from skimage.feature import peak_local_max

def parse_args():
    """parse given arguments before active learning execution
    return: args parsed by parser"""

    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str, required=True, default="configs/active_learning/al_settings.yaml"
                        help='experiment configure file name')
    parser.add_argument('--sp', default=False, action='store_true',
                        help='Use single process for pytorch')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--detfile', dest='detfile',
                        help='detection result file', default="")
    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")
    parser.add_argument('--list', dest='inputlist',
                        help='image-list', default="")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default="")
    parser.add_argument('--outdir', dest='outputpath',
                        help='output-directory', default="examples/res/")
    parser.add_argument('--save_img', default=False, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--detbatch', type=int, default=5,
                        help='detection batch size PER GPU')
    parser.add_argument('--posebatch', type=int, default=64,
                        help='pose estimation maximum batch size PER GPU')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--debug', default=True, action='store_true',
                        help='print detail information')

    """----------------------------- Video options -----------------------------"""
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=True)

    args = parser.parse_args()
    return args


    


def main(): # rough flow of active learning
    """Setup, Active Learning iteration, and evaluation."""
    ###各種変数の設定と実験設定の読み込み###
    args = parse_args()
    cfg = update_config(args.cfg)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    


    ###能動学習イテレーション###
    FINISH_AL = False
    while not FINISH_AL: # 終了条件が満たされない限り続ける

        for sample in unlabeled_list:
            # 姿勢推定器によるUnlabeled dataの予測。index必ず取り出す

            # 予測結果のヒートマップから局所ピークを拾う 局所ピークの座標が返ってくる
            local_peaks = peak_local_max(hp, min_distance=7) # min_distance: filter size
            if len(local_peaks)>6: # 局所ピークいくつで不確実サンプルとするか
                # そのサンプルのindexをlabeledに追加。unlabeledから抜く。


        #　
    


if __name__ == '__main__': # Do active learninｇ
    main()
    print("finished!!")