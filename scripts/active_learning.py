"""能動学習を実行する本体部分のスクリプト。全てはここをベースにして開発してゆく。
1. 準備. PoseTrack21の該当するディレクトリから動画の各フレーム読み込み, 初期のデータローダーを作る
2. 準備. 初期姿勢推定器を用意する（PoseTrack21で事前学習済み) -> これを使って能動学習を行う
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
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'
import platform
import sys
import time
import random
import json

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from skimage.feature import peak_local_max
from cachetools import cached

# 3rd party libraries
from alipy.experiment import StoppingCriteria
from alipy.index import IndexCollection

# additional functions
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP, calc_accuracy, DataLogger
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord)
from alphapose.utils.vis import vis_frame_fast, vis_frame

"""---------------------------------- Functions Set ----------------------------------"""
def parse_args():
    """
    parse given arguments before active learning execution
    return: args parsed by parser
    """
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('--cfg', type=str, required=True, default="configs/active_learning/al_settings.yaml",
                        help='experiment configure file name')
    parser.add_argument('--exp-id', default='default', type=str, help='Experiment ID')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--verbose', default=False, action='store_true',
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
    opt.num_gpu = torch.cuda.device_count()
    print('Using GPU : {}'.format(opt.num_gpu))
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
        self.round_cnt = 0

        # Evaluation settings
        self.video_id = self.opt.video_id
        self.eval_dataset = builder.build_dataset(self.cfg.DATASET.EVAL, preset_cfg=self.cfg.DATA_PRESET, train=False, get_prenext=False)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.cfg.VAL.BATCH_SIZE*self.opt.num_gpu, shuffle=False, num_workers=os.cpu_count(),drop_last=False, pin_memory=True)

        # AL_settings
        self.strategy = self.opt.strategy
        self.stopping_criterion = StoppingCriteria(stopping_criteria=None)
        self.query_size = int(len(self.eval_dataset) * self.cfg.VAL.QUERY_RATIO)
        self.unlabeled_id = IndexCollection(list(range(len(self.eval_dataset))))
        self.eval_len = len(self.eval_dataset)
        self.labeled_id = IndexCollection()
        self.percentage = [] # number of query for each round
        self.performance = [] # list of mAP for each round

        # Training settings
        self.train_dataset = builder.build_dataset(self.cfg.DATASET.EVAL, preset_cfg=self.cfg.DATA_PRESET, train=True, get_prenext=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.RETRAIN.LR)
        self.criterion = builder.build_loss(self.cfg.LOSS).cuda()

        # Other settings
        if self.opt.verbose: # test dataset and dataloader
            self.test_dataset(self.eval_dataset)
            self.test_dataloader(self.eval_loader)
        self.eval_joints = self.eval_dataset.EVAL_JOINTS
        self.norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg) # function to get final prediction from heatmap
        self.show_info() # show information of active learning

    def outcome(self):
        """Check if the active learning is stopped"""
        print("[Judge]")
        print("Unlabeled items: {}".format(len(self.unlabeled_id.index)))
        print("Labeled items: {}".format(len(self.labeled_id.index)))
        self.round_cnt += 1
        if len(self.unlabeled_id.index) == 0:
            self.retrain_model()
            self.eval_and_query() # Evaluate final estimator performance
            print("Active Learning is finished!\n")
            return self.percentage, self.performance
        else:
            print(f'--> Continue')
            return None
        # return self.stopping_criterion.is_stop()

    def initialize_estimator(self): # Load an initial pose estimator
        """construct a initial pose estimator
        Returns:
            model: Initial pose estimator
        """
        model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        print(f'Loading model from {self.cfg.MODEL.PRETRAINED}...') # pretrained by PoseTrack21
        model.load_state_dict(torch.load(self.cfg.MODEL.PRETRAINED))
        if self.opt.device == torch.device('cuda'):
            model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).cuda()
        return model

    def show_info(self):
        print(f'\n[Active Learning Setting]')
        print(f'Input video: {self.video_id}')
        print(f'Model type: {self.cfg.MODEL.TYPE}')
        print(f'Active learning strategy: {self.strategy}')
        print(f'Query size: {self.query_size}')
        print(f'Unlabeled items: {len(self.unlabeled_id.index)}')
        print(f'Labeled items: {len(self.labeled_id.index)}\n')

    def eval_and_query(self):
        """Evaluate the current estimator and query the most uncertain samples"""
        print(f'[[Round {self.round_cnt}]]')
        kpt_json = []
        m = self.model
        m.eval()
        query_candidate = {}
        for i, (idxs, inps, _, _, img_ids, ann_ids, bboxes) in enumerate(tqdm(self.eval_loader, dynamic_ncols=True)):
            inps = [inp.cuda() for inp in inps] if isinstance(inps, list) else inps.cuda()
            # pdb.set_trace()
            with torch.no_grad():
                output = m(inps) # input inps into model and get heatmap
                assert output.dim() == 4
                pred = output[:, self.eval_joints, :, :]

            for j in range(output.shape[0]):
                bbox = bboxes[j].tolist()
                pose_coords, pose_scores = self.heatmap_to_coord(pred[j][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
                keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                keypoints = keypoints.reshape(-1).tolist()
                data = dict()
                data['bbox'] = bboxes[j].tolist()
                data['image_id'] = int(img_ids[j])
                data['ann_id'] = int(ann_ids[j])
                data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
                data['category_id'] = 1
                data['keypoints'] = keypoints
                kpt_json.append(data)

                # best criterion
                if (not self.strategy == 'Random' ) and (idxs[j] in self.unlabeled_id.index):
                    query_candidate[int(idxs[j])] = -np.sum(pose_scores)

        sysout = sys.stdout
        with open(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), 'w') as fid:
            json.dump(kpt_json, fid)
        res = evaluate_mAP(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), ann_type='keypoints', ann_file=os.path.join(self.cfg.DATASET.EVAL.ROOT, self.cfg.DATASET.EVAL.ANN))
        sys.stdout = sysout

        self.percentage.append((len(self.labeled_id.index)/self.eval_len)*100) # label percentage: 0~100
        if self.strategy=='Random':
            query_list = self.random_query(query_size=self.query_size)
        else:
            sorted_candidate = sorted(query_candidate.items(), key=lambda x: x[1], reverse=True)
            candidate_dict = dict((idx,uncertainty) for idx,uncertainty in sorted_candidate)
            query_list = list(candidate_dict.keys())[:self.query_size] # pick queries with high uncertainty
            self.labeled_id.update(query_list)
            self.unlabeled_id.difference_update(query_list)
        # pdb.set_trace()
        self.performance.append(res*100) # mAP performance: 0~100
        print(f'[Evaluation]\n mAP={res:.4f}\n Percentage: {self.percentage[-1]:.2f}')
        print(f'[Query Selection]\nindex: {sorted(query_list)}') # query list sorted by index


    def retrain_model(self):
        """Retrain the model with the labeled data"""
        print(f'[Retrain]')
        loss_logger = DataLogger()
        acc_logger = DataLogger()
        train_subset = Subset(self.train_dataset, self.labeled_id.index)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.cfg.RETRAIN.BATCH_SIZE*self.opt.num_gpu, shuffle=True, num_workers=os.cpu_count(),drop_last=False, pin_memory=True)
        train_loader = tqdm(train_loader, dynamic_ncols=True)
        self.model.train()
        for epoch in range(self.cfg.RETRAIN.EPOCH):
            for i, (idxs, inps, labels, label_masks, _, ann_ids, bboxes) in enumerate(train_loader):
                inps = [inp.cuda().requires_grad_() for inp in inps] if isinstance(inps, list) else inps.cuda().requires_grad_()
                labels = [label.cuda() for label in labels] if isinstance(labels, list) else labels.cuda()
                label_masks = [label_mask.cuda() for label_mask in label_masks] if isinstance(labels, list) else label_masks.cuda()

                output = self.model(inps) # input inps into model
                loss = 0.5 * self.criterion(output.mul(label_masks), labels.mul(label_masks)) # loss function
                acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks)) # accuracy function

                batch_size = inps[0].size(0) if isinstance(inps, list) else inps.size(0)
                loss_logger.update(loss.item(), batch_size)
                acc_logger.update(acc, batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # TQDM
                train_loader.set_description('loss: {loss:.6f} | acc: {acc:.3f}'.format(loss=loss_logger.avg, acc=acc_logger.avg))
        train_loader.close()
        # Save checkpoint
        torch.save(self.model.module.state_dict(), './{}/round_{}.pth'.format(self.opt.work_dir, self.round_cnt))
        print('--> Retrained!\n')

    def quality_eval(self):
        """Create animation of prediction using final estimator"""
        m = self.model
        m.eval()
        for i, (inps, _, _, img_ids, ann_ids, bboxes) in enumerate(tqdm(self.eval_loader, dynamic_ncols=True)):
            inps = [inp.cuda() for inp in inps] if isinstance(inps, list) else inps.cuda()
            # pdb.set_trace()
            with torch.no_grad():
                output = m(inps) # input inps into model and get heatmap
                assert output.dim() == 4
                pred = output[:, self.eval_joints, :, :]

            for j in range(output.shape[0]):
                bbox = bboxes[j].tolist()
                pose_coords, pose_scores = self.heatmap_to_coord(pred[j][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
                keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                keypoints = keypoints.reshape(-1).tolist()

                data = dict()
                data['bbox'] = bboxes[j].tolist()
                data['image_id'] = int(img_ids[j])
                data['ann_id'] = int(ann_ids[j])
                data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
                data['category_id'] = 1
                data['keypoints'] = keypoints
                kpt_json.append(data)

    def test_dataset(self, dataset):
        # pdb.set_trace()
        print(len(dataset))
        print(dataset[0])
    def test_dataloader(self, dataloader):
        print(len(dataloader))

    def random_query(self, query_size=10):
        # pdb.set_trace()
        query_list = []
        while (len(query_list) < query_size) and (len(self.unlabeled_id.index) > 0):
            query_index = int(np.random.choice(self.unlabeled_id.index)) # randomly select the query
            query_list.append(query_index)
            self.unlabeled_id.discard(query_index)
            self.labeled_id.add(query_index)
        return query_list


def plot_learning_curves(video_id, al_strategy, percentages, performances):
    """Plot the learning curves"""
    fig, ax = plt.subplots()
    c = ["blue","orange","green","red","black"]      # 各プロットの色
    ax.set_xlabel('Label Percentage (%)')  # x軸ラベル
    ax.set_ylabel('mAP Performance (%)')  # y軸ラベル
    ax.set_title(f'Active Learning Result on {video_id}') # グラフタイトル
    ax.grid()            # 罫線
    ax.set_xlim(0, 100)  # x軸の範囲
    ax.set_ylim(0, 100)  # y軸の範囲

    for i, strategy in enumerate(al_strategy):
        ax.plot(percentages[strategy], performances[strategy], label=strategy, color=c[i])

    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    savepath = os.path.join(opt.work_dir, 'learning_curve.png')
    plt.savefig(savepath)
    print(f'Experiment result saved to... {savepath}!')

"""---------------------------------- Execution ----------------------------------"""

if __name__ == '__main__': # Do active learning
    """Execution of active learning"""

    # Setting up experiment
    opt = parse_args() # get exp settings
    opt = setup_opt(opt) # setup option
    cfg = update_config(opt.cfg) # update config
    opt.video_id = os.path.splitext(os.path.basename(cfg.DATASET.EVAL.ANN))[0]
    opt.work_dir = '{}/{}_{}'.format(cfg.RESULT.OUTDIR, opt.exp_id, opt.video_id)
    print(f'Result will be saved to: {opt.work_dir}\n')
    if not os.path.exists(opt.work_dir):
        os.makedirs(opt.work_dir)

    al_strategy = cfg.VAL.STRATEGY
    percentages = {}
    performances = {}
    for strategy in al_strategy:
        print(f'\n[[AL strategy: {strategy}]]')
        opt.strategy = strategy
        # Initialize active learning
        al = ActiveLearning(cfg, opt)
        # active learning iteration continue until termination conditions have been met
        while True:
            al.eval_and_query() # Evaluate pose estimator and get next query
            result = al.outcome()
            if result==None: # The condition is met and break the loop.
                al.retrain_model() # Retrain the model with the labeled data
            else:
                # Save the results
                percentages[strategy] = result[0]
                performances[strategy] = result[1]
                break

    ## 最終性能の評価、実験結果のまとめ、保存
    plot_learning_curves(opt.video_id, al_strategy, percentages, performances)
    # al.quality_eval()


"""---------------------------------- Memo ----------------------------------"""
    # for sample in unlabeled_list:
        # 姿勢推定器によるUnlabeled dataの予測。index必ず取り出す

        # 予測結果のヒートマップから局所ピークを拾う 局所ピークの座標が返ってくる
        # local_peaks = peak_local_max(hp, min_distance=7) # min_distance: filter size
            # そのサンプルのindexをlabeledに追加。unlabeledから抜く。

    # evaluationの際にUnlabeledリストか見て、Unlabeledのもののみ不確実性を評価する。