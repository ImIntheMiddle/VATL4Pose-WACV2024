# general libraries
import argparse
import os
import io
import pdb # import python debugger
import platform
import sys
import time
import random
import json
import copy

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from skimage.feature import peak_local_max
from scipy.special import softmax
from scipy.stats import entropy
from cachetools import cached
import seaborn as sns
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances # used in coreset
import umap
import cv2

# 3rd party libraries
from alipy.experiment import StoppingCriteria
from alipy.index import IndexCollection
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP, calc_accuracy, DataLogger
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord, norm_heatmap)
from alphapose.utils.bbox import bbox_xyxy_to_xywh
from alphapose.utils.vis import vis_frame_fast, vis_frame
from JRDB_toolkit.pose_eval import ospa_for_loc

# original modules
from active_learning.local_peak import localpeak_mean
from active_learning.al_metric import compute_OKS, compute_Spearmanr, compute_corr
from .Whole_body_AE import WholeBodyAE
from .Whole_body_AE import Wholebody
from .Whole_body_AE.hybrid_feature import compute_hybrid
from .VL4Pose.AuxiliaryNet import AuxNet

"""---------------------------------- Main Class ----------------------------------"""
class ActiveLearning:
    def __init__(self, cfg, opt):
        self.round_cnt = 0
        self.cfg = cfg
        self.dataset = cfg.DATASET.EVAL.TYPE
        self.opt = opt
        self.one_by_one = self.opt.onebyone # if True, then only one image is selected per round
        self.is_early_stop = False
        if self.one_by_one:
            print("One by one annotation is enabled...")
        self.strategy = self.opt.strategy
        self.uncertainty = self.opt.uncertainty
        self.representativeness = self.opt.representativeness
        self.filter = self.opt.filter # type of filter
        self.video_id = self.opt.video_id
        self.get_prenext = self.opt.get_prenext
        if self.dataset == "Posetrack21":
            if self.opt.optimize:
                IMG_PREFIX = f'images/train/{self.video_id}_bonn_train/'
                ANN = f'activelearning/train_val/{self.video_id}_bonn_train.json'
            else:
                IMG_PREFIX = f'images/val/{self.video_id}_mpii_test/'
                ANN = f'activelearning/val/{self.video_id}_mpii_test.json'
        elif self.opt.PCIT:
            IMG_PREFIX = f'images/{self.video_id}_PCIT_eval/'
            ANN = f'annotations/eval/{self.video_id}.json'
        elif self.dataset == "JRDB2022":
            # line number is corresponding to the video_id
            if self.opt.optimize:
                with open("configs/jrdb-pose/jrdb_val.txt", "r") as f:
                    lines = f.readlines()
                    scene_name = lines[int(self.video_id)]
                IMG_PREFIX = f'images/image_stitched/{scene_name}/'
                ANN = f'activelearning/val/{self.video_id}_jrdb-pose.json'
            else:
                with open("configs/jrdb-pose/jrdb_test.txt", "r") as f:
                    lines = f.readlines()
                    scene_name = lines[int(self.video_id)]
                IMG_PREFIX = f'images/image_stitched/{scene_name}/'
                ANN = f'activelearning/test/{self.video_id}_jrdb-pose.json'
        self.cfg.DATASET.EVAL.IMG_PREFIX = IMG_PREFIX
        self.cfg.DATASET.EVAL.ANN = ANN
        self.cfg.DATASET.TRAIN.IMG_PREFIX = IMG_PREFIX
        self.cfg.DATASET.TRAIN.ANN = ANN

        # Evaluation settings
        self.eval_dataset = builder.build_dataset(self.cfg.DATASET.EVAL, preset_cfg=self.cfg.DATA_PRESET, train=False, get_prenext=self.opt.get_prenext)
        self.collate_fn = self.eval_dataset.my_collate_fn
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.cfg.VAL.BATCH_SIZE*self.opt.num_gpu, shuffle=False, num_workers=8, drop_last=False, pin_memory=True, collate_fn=self.collate_fn)
        self.cnt_early_stop = 0
        self.finish_acc = self.opt.retrain_thresh # finishing accuracy
        self.finish_margin = 0.05 # margin of finished accuracy
        self.actual_finish = 100 # finished percentage
        self.finished_minerror = 100 # finished percentage
        self.finished_oursc = 100 # finished percentage

        # AL_settings
        self.eval_len = len(self.eval_dataset)
        self.stopping_criterion = StoppingCriteria(stopping_criteria=None)
        self.query_ratio = self.cfg.VAL.QUERY_RATIO
        self.w_unc = self.cfg.VAL.W_UNC
        self.unc_lambda = self.cfg.VAL.UNC_LAMBDA # weight of uncertainty
        self.query_sizes = [int(self.eval_len*x) for x in self.query_ratio]
        self.query_size = self.query_sizes[0] # number of query for first round
        self.expected_performance = 0 # expected performance. It controls the balance between uncertainty and representativeness
        if self.one_by_one:
            self.query_size = 3 # query one by one
        self.plot_cluster = False # plot cluster
        self.unlabeled_id = IndexCollection(list(range(self.eval_len)))
        self.labeled_id = IndexCollection()
        self.percentage = [] # number of query for each round
        self.performance = [] # list of mAP for each round
        self.performance_ann = [] # list of mAP for each round (with annotation)
        self.ospa_list = [] # list of OSPA for each round
        self.ospa_list_ann = [] # list of OSPA for each round (with annotation)
        self.combine_weight = [] # list of combine weight for each round
        self.query_list_list = {} # list of query for each round
        self.uncertainty_dict = {} # dict of uncertainty and idx for each round
        self.uncertainty_mean = [] # list of mean of uncertainty for each round
        self.influence_dict = {} # dict of mean of influence for each round
        self.spearmanr_list = [] # list of spearmanr for each round
        self.corr_list = [] # list of correlation for each round
        self.true_labeled_dict = {} # dict of true labeled data for each round
        self.false_labeled_dict = {} # dict of false labeled data for each round
        self.true_unlabeled_dict = {} # dict of true unlabeled data for each round
        self.false_unlabeled_dict = {} # dict of false unlabeled data for each round
        self.moksQ_list = [] # list of mean OKS for each round

        # Training settings
        self.train_dataset = builder.build_dataset(self.cfg.DATASET.TRAIN, preset_cfg=self.cfg.DATA_PRESET, train=True, get_prenext=False) # get_prenext=False for training
        self.retrain_id = IndexCollection() # index of retraining dataset
        self.retrain_epoch = self.cfg.RETRAIN.BASE
        self.lr = self.cfg.RETRAIN.LR
        self.moks_queried = 0 # last mean OKS of unlabeled data
        self.model, self.optimizer, self.scheduler = self.initialize_estimator()
        self.criterion = builder.build_loss(self.cfg.LOSS).cuda()
        self.continual = True if self.opt.continual else False

        # WPU settings
        if "WPU" in self.strategy:
            self.AE, self.AE_dataset = self.initialize_AE()
            self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=self.cfg.AE.LR)
        if "VL4Pose" in self.strategy:
            self.auxnet = AuxNet()
            self.auxnet_optimizer = torch.optim.Adam(self.auxnet.parameters(), lr=self.cfg.AUXNET.LR)

        # Other settings
        if self.opt.verbose: # test dataset and dataloader
            self.test_dataset(self.eval_dataset)
        self.eval_joints = self.eval_dataset.EVAL_JOINTS
        self.norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg) # function to get final prediction from heatmap
        self.show_info() # show information of active learning

    def outcome(self): # Check if the active learning is stopped
        # self.is_early_stop = True # for test
        if self.is_early_stop or self.one_by_one:
            while len(self.performance) <= len(self.query_ratio): # fill the rest of performance (early stopping)
                self.round_cnt += 1
                self.performance.append(self.performance[-1])
                self.performance_ann.append(self.performance_ann[-1])
                self.ospa_list.append(self.ospa_list[-1])
                self.ospa_list_ann.append(self.ospa_list_ann[-1])
                self.uncertainty_mean.append(self.uncertainty_mean[-1])
                self.percentage.append(self.query_ratio[self.round_cnt-1]*100)
                self.combine_weight.append(self.combine_weight[-1])
                self.moksQ_list.append(self.moksQ_list[-1])
            finish = True
        else:
            if not self.continual:
                self.model, self.optimizer, self.scheduler = self.initialize_estimator() # initialize estimator for next round
                self.retrain_epoch = int(self.cfg.RETRAIN.BASE * len(self.labeled_id.index)/len(self.eval_dataset) + self.cfg.RETRAIN.ALPHA * (1-self.moks_queried)) # number of additional epoch
            else:
                self.retrain_epoch = int(self.cfg.RETRAIN.ALPHA * (1-self.moks_queried)) # number of additional epoch
            print(f'[Retrain Epoch]: {self.retrain_epoch}')
            self.retrain_model() # Retrain the model with the labeled data
            self.round_cnt += 1
            finish = False
            print(f"[Judge] Unlabeled items: {len(self.unlabeled_id.index)}, Labeled items: {len(self.labeled_id.index)}", end=' ')
            if len(self.unlabeled_id.index) == 0:
                print("\n --> Finished!")
                self.eval_and_query() # Evaluate final estimator performance
                finish = True
            else:
                print("--> Continue...")
                if self.round_cnt >= len(self.query_ratio): # last round
                    self.query_size = len(self.unlabeled_id.index) # query all unlabeled data
                else:
                    self.query_size = self.query_sizes[self.round_cnt]-len(self.labeled_id.index) # number of query for next round
                finish = False
        if finish:
            return self.percentage, self.performance, self.performance_ann, self.query_list_list, self.uncertainty_dict, self.uncertainty_mean, self.influence_dict, self.combine_weight, self.spearmanr_list, self.corr_list, self.true_labeled_dict, self.true_unlabeled_dict, self.false_labeled_dict, self.false_unlabeled_dict, self.actual_finish, self.finished_minerror, self.finished_oursc, self.ospa_list, self.ospa_list_ann, self.moksQ_list
        else:
            return None

    def initialize_estimator(self): # Load an initial pose estimator
        """construct a initial pose estimator
        Returns:
            model: Initial pose estimator
        """
        model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        if self.opt.from_scratch:
            print(f'Train the model from scratch!')
        elif self.cfg.MODEL.PRETRAINED:
            print(f'Loading model from {self.cfg.MODEL.PRETRAINED}...') # pretrained by PoseTrack21
            model.load_state_dict(torch.load(self.cfg.MODEL.PRETRAINED))
        else:
            raise ValueError('No pretrained model is given!')
        if self.cfg.RETRAIN.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        elif self.cfg.RETRAIN.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.cfg.RETRAIN.OPTIMIZER == 'AdamW':
            if self.cfg.MODEL.TYPE == 'SimplePose':
                optimizer = torch.optim.AdamW(params = [{"params": model.final_layer.parameters(), "lr": self.lr*10}, {"params": model.preact.parameters(), "lr": self.lr}, {"params": model.deconv_layers.parameters(), "lr": self.lr*5}], weight_decay=self.cfg.RETRAIN.WEIGHT_DECAY)
            elif self.cfg.MODEL.TYPE == 'FastPose':
                optimizer = torch.optim.AdamW(params = [{"params": model.conv_out.parameters(), "lr": self.lr*10}, {"params": model.preact.parameters(), "lr": self.lr}, {"params": model.suffle1.parameters(), "lr": self.lr*5}, {"params": model.duc1.parameters(), "lr": self.lr*5}, {"params": model.duc2.parameters(), "lr": self.lr*5}], weight_decay=self.cfg.RETRAIN.WEIGHT_DECAY)
        else:
            raise ValueError('Optimizer not supported!')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.RETRAIN.LR_GAMMA) # decay learning rate by 0.95 every epoch
        if self.opt.device == torch.device('cuda'):
            model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).cuda()
        else:
            print('Model is not on GPU!')
        return model, optimizer, scheduler

    def show_info(self):
        print(f'\n[[AL strategy: {self.opt.strategy}]]')
        print(f'[[Video ID: {self.opt.video_id}]]')
        print(f'[[Number of queries: {len(self.eval_dataset)}]]')
        print(f'[[Pose Estimator: {self.cfg.MODEL.TYPE}]]')
        print(f'[[eval_jounts: {self.eval_joints}]]')
        print(f'[[Uncertainty: {self.uncertainty}]]')
        print(f'[[Representativeness: {self.representativeness}]]')
        print(f'[[Filter: {self.filter}]]')
        assert(self.finish_acc <= 1.0 and self.finish_acc >= 0.0)
        print(f'[[Retraining Scheme: {self.continual}]]')
        print(f'[[Finishing Accuracy: {self.finish_acc}]]')
        assert(self.query_ratio[-1] <= 1.0 and self.query_ratio[0] >= 0.0)
        print(f'[[Query Ratio: {self.query_ratio}]]')

    def eval_and_query(self):
        """Evaluate the current estimator and query the most uncertain samples"""
        print(f'\n{self.video_id}[[Round{self.round_cnt}: {self.strategy}]]')
        GT_json = []
        kpt_json = []
        kpt_json_ann = []
        m = self.model
        m.eval()
        if "WPU" in self.strategy:
            self.AE.eval()
        if "VL4Pose" in self.strategy:
            self.auxnet.eval()
        query_candidate = {}
        OKS_dict = {}
        UNC_dict = {}
        total_uncertainty = 0 # total uncertainty initialized to 0
        combine_weight = 0
        fvecs_matrix = np.zeros((len(self.eval_dataset), 2048)) # 2048 is the dimension of the feature vector
        track_ids_list = np.zeros(len(self.eval_dataset))
        with torch.no_grad():
            for i, (idxs, inps, labels, label_masks, GTkpts, img_ids, ann_ids, bboxes_crop, bboxes_ann, isPrev, isNext) in enumerate(tqdm(self.eval_loader, dynamic_ncols=True, leave=False)):
                # pdb.set_trace()
                # print(inps.shape) # 62tensors, 3frames, 3channels, 256x192 each
                # print(inps[0][0].shape) # (3, 256, 192 # one current frame, 3 channels, 256x192
                output = m(inps[:, 0].cuda()) # input a batch of current frames (62, 3, 256, 192)
                # print(output.shape) # (62, 17, 64, 48) # 62 frames, 17 joints, 64x48 heatmap
                # print(output[0].shape) # (17, 64, 48) # one frame, 17 joints, 64x48 heatmap
                assert output.dim() == 4, 'the dimension of output must be 4'
                pred = output[:, self.eval_joints, :, :]

                if self.representativeness not in ['None', 'Random'] or self.filter not in ['None', 'Random'] or self.plot_cluster:
                    fvecs = self.model.module.get_embedding(inps[:, 0].to(self.opt.device))
                    # print(fvecs.shape) # (62, 2048) # 62 frames, 2048 features
                    fvecs_matrix[idxs] = fvecs.cpu().numpy()
                    # 下二桁 of each ann_ids
                    track_ids = [int(str(ann_id)[-2:]) for ann_id in ann_ids]
                    track_ids_list[idxs] = track_ids
                    # pdb.set_trace()

                # get prediction for previous/next frame
                if self.get_prenext:
                    output_prev = m(inps[:, 1].to(self.opt.device))
                    pred_prev = output_prev[:, self.eval_joints, :, :]
                    output_next = m(inps[:, 2].to(self.opt.device))
                    pred_next = output_next[:, self.eval_joints, :, :]

                for j in range(output.shape[0]): # for each input in batch (person)
                    heatmaps_dict = {"Round": [], "ann_ids": [], "heatmaps": [], "img": [], "img_ids": [], "bboxes": [], "uncertainty": [],"keypoints": [], "GTkpts": []} # for visualization
                    bbox_crop = bboxes_crop[j].tolist()
                    # xywh format
                    bbox_ann = bbox_xyxy_to_xywh(bboxes_ann[j].tolist())
                    pose_coords, pose_scores = self.heatmap_to_coord(pred[j][self.eval_joints], bbox_crop, hm_shape=self.hm_size, norm_type=self.norm_type)
                    # print(pose_coords) # (17, 2)
                    keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                    keypoints = keypoints.reshape(-1).tolist() # format: x1, y1, s1, x2, y2, s2, ..., shape: (51,)
                    GT_keypoints = GTkpts[j].reshape(-1).tolist() # ground truth keypoints
                    oks = float(compute_OKS(bbox_ann, keypoints, GT_keypoints))
                    data = dict()
                    data['bbox'] = bbox_ann # format: xyxy
                    data['image_id'] = int(img_ids[j])
                    data['id'] = int(ann_ids[j])
                    data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
                    data['category_id'] = 1
                    data['keypoints'] = keypoints
                    # pdb.set_trace()
                    data['GT_keypoints'] = GT_keypoints
                    data['OKS'] = oks
                    kpt_json.append(data)
                    data_ann = copy.deepcopy(data)
                    if idxs[j] in self.labeled_id.index:
                        data_ann['keypoints'] = GT_keypoints
                    kpt_json_ann.append(data_ann)
                    data_GT = copy.deepcopy(data)
                    data_GT['keypoints'] = GT_keypoints
                    GT_json.append(data_GT)

                    if (self.uncertainty == 'HP' ): # for unlabeled data, highest probability
                        uncertainty = float(-np.sum(pose_scores)) # calculate uncertainty
                        # print(type(-np.sum(pose_scores)))
                        # pdb.set_trace()
                    elif (self.uncertainty == 'TPC'): # temporal pose continuity
                        thresh = 0.01 * np.sqrt((bbox_crop[2] - bbox_crop[0]) * (bbox_crop[3] - bbox_crop[1])) # threshold for euclidean distance. 0.01 is a hyperparameter
                        tpc = 0
                        if isPrev[j]:
                            tpc += self.compute_tpc(pose_coords, pred_prev[j], bbox_crop, thresh)
                        if isNext[j]:
                            tpc += self.compute_tpc(pose_coords, pred_next[j], bbox_crop, thresh)
                            if not isPrev[j]:
                                tpc *= 2 # if only next frame exists, double the uncertainty
                        elif isPrev[j]:
                            tpc *= 2 # if only previous frame exists, double the uncertainty
                        uncertainty = float(tpc) # convert to float, tpc as uncertainty
                    elif "THC" in self.uncertainty: # temporal heatmap continuity
                        norm_type = 'L1'
                        thc = 0
                        hp_current = pred[j].cpu().numpy()
                        if isPrev[j]:
                            hp_prev = pred_prev[j].cpu().numpy()
                            thc += self.compute_thc(hp_current, hp_prev, norm_type=norm_type)
                        if isNext[j]:
                            hp_next = pred_next[j].cpu().numpy()
                            thc += self.compute_thc(hp_current, hp_next, norm_type=norm_type)
                            if not isPrev[j]:
                                thc *= 2
                            elif self.opt.vis_thc:
                                adj_hps = [pred_prev[j], pred[j], pred_next[j]]
                                adj_imgs = [inps[j, 1], inps[j, 0], inps[j, 2]] # current, previous, next frame image
                                self.visualize_thc(adj_imgs, adj_hps, thc, ann_ids[j], keypoints)
                        elif isPrev[j]:
                            thc *= 2
                        uncertainty = float(thc)
                        if "WPU" in self.uncertainty: # combine with WPU
                            bbox_crop = bbox_xyxy_to_xywh(bbox_crop)
                            # input = torch.tensor(compute_hybrid(bbox_crop, np.array(keypoints[:3*3]+keypoints[5*3:]))).float().cuda() # select 15 keypoints (exclude ears)
                            input = torch.tensor(compute_hybrid(bbox_crop, np.array(keypoints))).float().cuda() # select 17 keypoints (exclude ears)
                            output = self.AE(input) # output is the reconstructed keypoints
                            wpu = self.criterion(output, input) # calculate whole-body pose unnaturalness
                            uncertainty2 = float(wpu)
                    elif "WPU" in self.uncertainty:
                        if True: # use hybrid feature
                            bbox_crop = bbox_xyxy_to_xywh(bbox_crop)
                            # input = torch.tensor(compute_hybrid(bbox_crop, keypoints[:3*3]+keypoints[5*3:])).float().cuda()
                            input = torch.tensor(compute_hybrid(bbox_crop, keypoints)).float().cuda()
                        else: # use raw keypoint as feature
                            input = torch.tensor(keypoints).float()
                        output = self.AE(input) # output is the reconstructed keypoints
                        # to numpy array
                        input, output = input.cpu().numpy(), output.cpu().numpy()
                        input = np.concatenate([input[:3], input[5:20], input[22:]])
                        output = np.concatenate([output[:3], output[5:20], output[22:]])
                        wpu = self.criterion(torch.tensor(output).float().cuda(), torch.tensor(input).float().cuda())
                        if self.opt.vis_wpu:
                            self.visualize_wpu(input, output, wpu, ann_ids[j])
                        uncertainty = float(wpu)
                    elif (self.uncertainty == 'MPE'): # multiple peak entropy
                        hp_current = pred[j].cpu().numpy()
                        uncertainty = float(self.compute_mpe(hp_current))
                    elif (self.uncertainty == 'VL4Pose'): # VL4Pose
                        vl4pose_loss = self.vl4pose(pose_encodings=pose_features, heatmaps=pred, joint_exist=joint_exist, epoch=e) # TODO
                    elif (self.uncertainty == 'Entropy'): # entropy
                        hp_current = pred[j].cpu().numpy()
                        uncertainty = float(self.compute_entropy(hp_current))
                    elif (self.uncertainty == 'Margin'): # margin
                        hp_current = pred[j].cpu().numpy()
                        uncertainty = float(self.compute_margin(hp_current))
                    elif (self.uncertainty == "None"): # no uncertainty
                        uncertainty = 0
                    else: # error
                        raise ValueError("Uncertainty type is not supported")
                    total_uncertainty += uncertainty # add to total uncertainty
                    if self.uncertainty == "THC+WPU":
                        UNC_dict[int(idxs[j])] = [uncertainty, uncertainty2] # save uncertainty
                        uncertainty = [uncertainty, uncertainty2]
                    else: # single uncertainty
                        UNC_dict[int(idxs[j])] = uncertainty
                    OKS_dict[int(idxs[j])] = data['OKS']
                    # print(f'ID {int(idxs[j])}: OKS {data["OKS"]:.3f}, Uncertainty {uncertainty:.3f}')

                    if idxs[j] in self.unlabeled_id.index:
                        combine_weight += localpeak_mean(pred[j][self.eval_joints].cpu().numpy()) # find the minimum peak of the heatmap
                        # print(localpeak_mean(pred[j][self.eval_joints].cpu().numpy()))
                        query_candidate[int(idxs[j])] = uncertainty # add to candidate list

                    if self.opt.vis: # visualize every 4 rounds
                        heatmaps_dict["Round"] = self.round_cnt
                        heatmaps_dict["ann_ids"] = int(ann_ids[j])
                        heatmaps_dict["heatmaps"] = pred[j].cpu().numpy() # copy to cpu and convert to numpy array
                        heatmaps_dict["img"] = inps[j, 0].cpu().numpy()
                        heatmaps_dict["img_ids"] = int(img_ids[j])
                        heatmaps_dict["keypoints"] = keypoints
                        heatmaps_dict["uncertainty"] = uncertainty
                        heatmaps_dict["bboxes"].append(bbox_ann)
                        heatmaps_dict["GTkpts"] = GT_keypoints
                        hm_save_dir = os.path.join(self.opt.work_dir, 'heatmap', f'Round{self.round_cnt}')
                        if not os.path.exists(hm_save_dir):
                            os.makedirs(hm_save_dir)
                        np.save(os.path.join(hm_save_dir, f'{int(ann_ids[j])}.npy'), heatmaps_dict) # save heatmaps for visualization
        if self.uncertainty != "None" and False:
            Spearman = compute_Spearmanr(UNC_dict, OKS_dict)
            self.spearmanr_list.append(Spearman)
            corr = compute_corr(UNC_dict, OKS_dict)
            self.corr_list.append(corr)
            print(f'[Evaluation] Spearmanr: {Spearman:.3f}, Correlation: {corr:.3f}')
            print(f"Total uncertainty: {total_uncertainty:.3f}")

        sysout = sys.stdout
        GT_dict_path = self.save_GT_dict(GT_json)
        with open(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), 'w') as fid:
            json.dump(kpt_json, fid)
        res = evaluate_mAP(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), ann_type='keypoints', ann_file=GT_dict_path)
        ospa = ospa_for_loc(ann_json_path=GT_dict_path, pr_json_path=os.path.join(self.opt.work_dir, 'predicted_kpt.json'))
        with open(os.path.join(self.opt.work_dir, 'predicted_kpt_ann.json'), 'w') as fid_ann:
            json.dump(kpt_json_ann, fid_ann)
        res_ann = evaluate_mAP(os.path.join(self.opt.work_dir, 'predicted_kpt_ann.json'), ann_type='keypoints', ann_file=GT_dict_path)
        ospa_ann = ospa_for_loc(ann_json_path=GT_dict_path, pr_json_path=os.path.join(self.opt.work_dir, 'predicted_kpt_ann.json'))
        if self.opt.vis:
            pred_save_dir = os.path.join(self.opt.work_dir, 'prediction', f'Round{self.round_cnt}')
            if not os.path.exists(pred_save_dir):
                os.makedirs(pred_save_dir)
            with open(os.path.join(pred_save_dir, 'predicted_kpt.json'), 'w') as fid:
                json.dump(kpt_json, fid)
        sys.stdout = sysout

        self.percentage.append((len(self.labeled_id.index)/self.eval_len)*100) # label percentage: 0~100
        self.performance.append(res) # mAP performance: 0~100
        self.performance_ann.append(res_ann) # mAP performance: 0~100
        self.ospa_list.append(ospa)
        self.ospa_list_ann.append(ospa_ann)
        print(f'{self.video_id}[[Round{self.round_cnt}: {self.strategy}]]')
        print(f'[Evaluation] Percentage:{self.percentage[-1]:.1f}, mAP:{res["AP"]:.3f} (ANN:{res_ann["AP"]:.3f}), AP@0.5:{res["AP .5"]:.3f} (ANN:{res_ann["AP .5"]:.3f}), AP@0.6:{res["AP .6"]:.3f} (ANN:{res_ann["AP .6"]:.3f}), AP@0.75:{res["AP .75"]:.3f} (ANN:{res_ann["AP .75"]:.3f}), AP@0.95:{res["AP .95"]:.3f} (ANN:{res_ann["AP .95"]:.3f})')
        print(f'[Evaluation] OSPA:{ospa:.3f} (ANN:{ospa_ann:.3f})')

        # compute influence score for all unlabeled data
        self.uncertainty_mean.append(total_uncertainty/len(self.eval_dataset)) # mean uncertainty
        if self.representativeness != "None":
            if len(self.unlabeled_id.index) in [0, 1]:
                influence_score = np.zeros(len(self.unlabeled_id.index))
            elif self.representativeness == "Influence":
                fvecs_mat_inf = fvecs_matrix[self.unlabeled_id.index]
                # print("fvecs_matrix", fvecs_matrix.shape, fvecs_matrix)
                knn = KNeighborsTransformer(mode='distance', metric='cosine', n_neighbors=len(self.unlabeled_id.index)-1)
                dist_mat = knn.fit_transform(fvecs_mat_inf) # compute distance matrix
                influence_score = np.asarray(np.sum(dist_mat, axis=1)).flatten() # compute influence score
                # normalize influence score
                influence_score = (influence_score - np.min(influence_score)) / (np.max(influence_score) - np.min(influence_score))
            elif self.representativeness == "Random":
                influence_score = np.random.rand(len(self.unlabeled_id.index)) # random influence score. value: [0,1]
            else: # error
                raise ValueError("Representativeness type is not supported")
            # print("influence_score", influence_score.shape, type(influence_score))
            self.influence_dict['Round'+str(self.round_cnt)] = dict((idx,influence) for idx,influence in zip(list(query_candidate.keys()), influence_score)) # add to influence dictionary
            # print("influence_score:", influence_score)

        if len(self.unlabeled_id.index) > 0: # if no unlabeled data or only one unlabeled data
            combine_weight /= len(self.unlabeled_id.index) # normalize combine weight, value: [0,1]
            self.combine_weight.append(combine_weight)

        if len(self.unlabeled_id.index) in [0, 1]: # if no unlabeled data or only one unlabeled data
            total_score = np.zeros(len(self.unlabeled_id.index))
        elif self.uncertainty != "None": # If using both uncertainty and representativeness
            # combine uncertainty and representativeness, be careful to division by zero
            if self.uncertainty == "THC+WPU":
                uncertainty_thc = np.array(list(query_candidate.values()))[:,0]
                uncertainty_wpu = np.array(list(query_candidate.values()))[:,1]
                # normalize uncertainty score to [0,1] independently
                uncertainty_thc = (uncertainty_thc - np.min(uncertainty_thc)) / (np.max(uncertainty_thc) - np.min(uncertainty_thc))
                uncertainty_wpu = (uncertainty_wpu - np.min(uncertainty_wpu)) / (np.max(uncertainty_wpu) - np.min(uncertainty_wpu))
                # integrate uncertainty score
                if self.opt.THCvsWPU == "const":
                    uncertainty_ = uncertainty_thc + uncertainty_wpu
                else:
                    labeled_ratio = len(self.labeled_id.index)/len(self.eval_dataset)
                    if self.opt.THCvsWPU == "increase":
                        uncertainty_ = labeled_ratio * uncertainty_thc + (1-labeled_ratio) * uncertainty_wpu
                    elif self.opt.THCvsWPU == "decrease":
                        uncertainty_ = (1-labeled_ratio) * uncertainty_thc + labeled_ratio * uncertainty_wpu
                uncertainty_score = (uncertainty_ - np.min(uncertainty_)) / (np.max(uncertainty_) - np.min(uncertainty_)) # normalize uncertainty score to [0,1] again
                self.uncertainty_dict['Round'+str(self.round_cnt)] = UNC_dict # add to uncertainty dictionary
                # print("uncertainty_score:", uncertainty_score)
            else:
                uncertainty_score = np.array(list(query_candidate.values()))
                self.uncertainty_dict['Round'+str(self.round_cnt)] = UNC_dict # add to uncertainty dictionary
                uncertainty_score = (uncertainty_score - np.min(uncertainty_score)) / (np.max(uncertainty_score) - np.min(uncertainty_score)) # normalize uncertainty score to [0,1]
                # print("uncertainty_score:", uncertainty_score)
            if self.representativeness != "None":
                # print("combine_weight:", combine_weight)
                total_score = combine_weight * uncertainty_score + (1-combine_weight) * influence_score # combine uncertainty and representativeness
            else:
                total_score = uncertainty_score # only uncertainty
        elif self.representativeness == "None" and self.uncertainty == "None": # If using neither uncertainty nor representativeness
            total_score = np.zeros(len(self.unlabeled_id.index)) # no score
        else: # If using only representativeness
            total_score = influence_score # only representativeness
        # print("total_score:", total_score)
        score_dict = dict((idx,score) for idx,score in zip(self.unlabeled_id.index, total_score)) # add to candidate dictionary
        # print(f'[Score List]: {score_dict}')
        score_dict_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=True) # sort by score
        score_dict = dict((int(idx),score) for idx,score in score_dict_sorted) # convert to dictionary

        # pick top k candidates based on score. each element is index of unlabeled data
        if self.filter == "None": # no query selection
            candidate_list = sorted(list(score_dict.keys())[:self.query_size])
        elif self.filter in ["weighted", "K-Means", "Coreset"]: # weighted K-Means. filter all unlabeled data
            candidate_list = sorted(list(score_dict.keys()))
        else: # both uncertainty and representativeness
            candidate_list = sorted(list(score_dict.keys())[:8 *self.query_size]) # pick top k candidates
        # print(f'[Candidate List] index: {candidate_list}')
        # print(f'[Query Size]: {self.query_size}')
        # print(f"[Unlabeled Data Size]: {len(self.unlabeled_id.index)}")

        if (len(self.unlabeled_id.index) in [0, 1]) or self.filter=="None": # if no unlabeled data or only one unlabeled data, or no filter
            query_list = candidate_list
            if self.plot_cluster:
                plot_samples = sorted(list(query_candidate.keys()))
                embeddings = fvecs_matrix[plot_samples]
                track_ids_list = track_ids_list[plot_samples]
                weight = total_score
                cluster_idxs = np.zeros(len(plot_samples))
                self.pltcluster_and_save(cluster_idxs, embeddings, track_ids_list, query_list, weight)
        # filter query, select query from candidate list
        elif self.filter=="weighted": # weighted K-Means. use uncertainty as weight
            embeddings = fvecs_matrix[candidate_list]
            _, embed_idx = np.unique(embeddings, axis=0, return_index=True) # remove duplicate embeddings
            embeddings = embeddings[embed_idx] # remove duplicate embeddings
            track_ids_list = track_ids_list[candidate_list]
            # uncertainty_score = np.array(list(query_candidate.values()))
            # print(f"initial performance: {self.combine_weight[0]}, current performance: {self.combine_weight[-1]}")
            weight = 1 + self.w_unc * combine_weight * total_score # weight for each candidate
            # use the sample included in embedding as weight
            weight = weight[embed_idx] # remove duplicate embeddings
            # weight = 1 + np.power(total_score, self.w_unc*combine_weight) # instead of above

            # print("weight:", weight.shape, "embeddings:", embeddings.shape)
            if len(self.unlabeled_id.index) <= self.query_size:
                self.query_size = len(self.unlabeled_id.index) # set query size to number of unlabeled data
            if self.query_size > len(embeddings):
                self.query_size = len(embeddings) # set query size to number of unlabeled data
            cluster_learner = KMeans(n_clusters=self.query_size, random_state=318, verbose=0) # K-Means
            cluster_idxs = cluster_learner.fit_predict(embeddings, sample_weight=weight) # cluster unlabeled data. use uncertainty as weight
            cluster_num = len(np.unique(cluster_idxs)) # number of clusters
            centers = cluster_learner.cluster_centers_[cluster_idxs] # center for each candidate
            dis = (embeddings - centers)**2 # distance between each candidate and its center
            dis = dis.sum(axis=1)
            query_list = list([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(cluster_num)]) # pick query from each cluster, element of query_list is the index of candidate_list
            # pdb.set_trace()
            if self.plot_cluster:
                self.pltcluster_and_save(cluster_idxs, embeddings, track_ids_list, query_list, weight) # plot cluster result and save to file
            query_list = [int(candidate_list[i]) for i in query_list] # convert to list
        elif self.filter=="Diversity": # diversity query selection
            fvecs_mat_div = fvecs_matrix[candidate_list]
            knn = KNeighborsTransformer(mode='distance', metric='cosine', n_neighbors=len(candidate_list)-1)
            dist_mat = knn.fit_transform(fvecs_mat_div) # compute distance matrix
            diversity_score = np.asarray(np.sum(dist_mat, axis=1)).flatten() # compute diversity score
            # print("diversity_score:", diversity_score)
            diversity_dict = dict((idx,score) for idx,score in zip(candidate_list, diversity_score)) # add to candidate dictionary
            diversity_dict_sorted = sorted(diversity_dict.items(), key=lambda x: x[1]) # sort by score, ascending order
            diversity_dict = dict((int(idx),score) for idx,score in diversity_dict_sorted) # convert to dictionary
            query_list = list(diversity_dict.keys())[:self.query_size] # pick top k candidates
        elif self.filter=="Random": # random query selection
            query_list = self.random_query(candidate_list, self.query_size) # pick random query
        elif self.filter=="K-Means": # k-means query selection
            embeddings = fvecs_matrix[candidate_list]
            track_ids_list = track_ids_list[candidate_list]
            if len(self.unlabeled_id.index) < self.query_size: # if number of unlabeled data is less than query size
                self.query_size = len(self.unlabeled_id.index) # set query size to number of unlabeled data
            cluster_learner = KMeans(n_clusters=self.query_size, random_state=318)
            cluster_idxs = cluster_learner.fit_predict(embeddings) # cluster for each candidate
            cluster_num = len(np.unique(cluster_idxs)) # number of clusters
            centers = cluster_learner.cluster_centers_[cluster_idxs] # center for each candidate
            dis = (embeddings - centers)**2 # distance between each candidate and its center
            dis = dis.sum(axis=1)
            query_list = list([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(cluster_num)]) # pick query from each cluster, element of query_list is the index of candidate_list
            # pdb.set_trace()
            if self.plot_cluster:
                self.pltcluster_and_save(cluster_idxs, embeddings, track_ids_list, query_list) # plot cluster result and save to file
            query_list = [int(candidate_list[i]) for i in query_list] # convert to list
        elif self.filter=="Coreset": # coreset query selection
            unc_list = np.zeros(len(self.eval_dataset))
            unc_list[candidate_list] = np.array(list(total_score))
            # pdb.set_trace()
            coreset_unc = copy.deepcopy(unc_list)
            query_list = self.coreset_selection(fvecs_matrix, coreset_unc) # pick query indices from coreset
            if self.plot_cluster:
                embeddings = fvecs_matrix[candidate_list]
                self.pltcoreset_and_save(embeddings, query_list, unc_list) # plot cluster result and save to file
        else: # error
            raise ValueError("Filter type is not supported")

        tl, tu, fl, fu = self.get_corresponding_id(OKS_dict, true=True, labeled=True), self.get_corresponding_id(OKS_dict, true=True, labeled=False), self.get_corresponding_id(OKS_dict, true=False, labeled=True), self.get_corresponding_id(OKS_dict, true=False, labeled=False)
        assert len(self.eval_dataset) == len(tl) + len(tu) + len(fl) + len(fu)
        print(f'[Evaluation] Labeled: True={len(tl)}, False={len(fl)}, Unlabeled: True={len(tu)}, False={len(fu)}')
        self.true_labeled_dict['Round'+str(self.round_cnt)] = tl
        self.true_unlabeled_dict['Round'+str(self.round_cnt)] = tu
        self.false_labeled_dict['Round'+str(self.round_cnt)] = fl
        self.false_unlabeled_dict['Round'+str(self.round_cnt)] = fu

        if len(self.unlabeled_id.index) != 0: # if there is unlabeled data
            self.retrain_id = IndexCollection() # initialize retrain data
            retrain_id, self.moks_queried = self.get_retrain_id(query_list, OKS_dict) # get retrain data
            self.moksQ_list.append(self.moks_queried)
            self.retrain_id.update(retrain_id) # add to retrain data
            # print(f"[Retrain/Labeled]: {len(self.retrain_id.index)-len(query_list)}/{len(self.labeled_id.index)}")
            self.labeled_id.update(query_list) # add to labeled data
            self.unlabeled_id.difference_update(query_list) # remove
            self.query_list_list['Round'+str(self.round_cnt)] = list(map(int, query_list)) # add to query list dictionary
            print(f'{self.video_id}[[Round{self.round_cnt}: {self.strategy}]] Queried: {sorted(query_list)}')

            # if self.is_early_stop and time < self.finished_time:
            #     self.cnt_early_stop += 1
            #     if self.cnt_early_stop == 1:
            #         print(f'[Early Stop] Performance is higher than the threshold at {time:.1f}%!')
            #         self.finished_time = time # update finished time

            self.actual_finish, self.finished_minerror, self.finished_oursc = self.is_finished(query_list, OKS_dict)
            if self.actual_finish < 100:
                print(f'[Early Stop] Performance is higher than the threshold at {self.actual_finish:.1f}%!')
                self.is_early_stop = True

    def retrain_model(self):
        """Retrain the model with the labeled data"""
        print(f'{self.video_id}[[Round{self.round_cnt}: {self.strategy}]] Retraining...')
        loss_logger = DataLogger()
        acc_logger = DataLogger()
        train_subset = Subset(self.train_dataset, self.retrain_id.index) # get labeled data
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.cfg.RETRAIN.BATCH_SIZE*self.opt.num_gpu, shuffle=True, num_workers=2,drop_last=False, pin_memory=True, collate_fn=self.collate_fn)
        self.model.train()

        with tqdm(range(self.retrain_epoch), dynamic_ncols=True, leave=True) as progress_bar:
            for epoch in progress_bar:
                for i, (idxs, inps, labels, label_masks, GTkpts, img_ids, ann_ids, bboxes_crop, bboxes_ann, isPrev, isNext) in enumerate(train_loader):
                    inps = inps[:, 0].cuda().requires_grad_()
                    labels = [label.cuda() for label in labels] if isinstance(labels, list) else labels.cuda()
                    label_masks = [label_mask.cuda() for label_mask in label_masks] if isinstance(labels, list) else label_masks.cuda()

                    output = self.model(inps)

                    loss = 0.5 * self.criterion(output.mul(label_masks), labels.mul(label_masks)) # loss function
                    loss_logger.update(loss.item(), inps.size(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step() # update parameters

                    acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks), thr=0.5) # accuracy function
                    acc_logger.update(acc, inps.size(0))
                self.scheduler.step() # update learning rate
                # TQDM
                progress_bar.set_description('loss: {loss:.7f} | acc: {acc:.4f}'.format(loss=loss_logger.avg, acc=acc_logger.avg))
        # fine-tuning AE
        if 'WPU' in self.uncertainty:
            self.AE, self.AE_dataset = self.initialize_AE()
            self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=self.cfg.AE.LR)
            self.retrain_AE() # retrain AE
            torch.save(self.AE.state_dict(), './{}/latest_AE.pth'.format(self.opt.work_dir)) # AE
            # pass # no need to retrain AE

    def test_dataset(self, dataset):
        # pdb.set_trace()
        assert len(dataset) >= 1 # at least 1 image
        print(dataset[0])

    def save_GT_dict(self, GT_json: list) -> str:
        """save GT keypoints to json file, formatted as COCO format"""
        GT_dict = dict()
        GT_path = os.path.join(self.cfg.DATASET.EVAL.ROOT, self.cfg.DATASET.EVAL.ANN)
        with open(GT_path, 'r') as fid:
            GT_data = json.load(fid)
            GT_dict["images"] = GT_data["images"]
            GT_dict["categories"] = GT_data["categories"]
        GT_dict["annotations"] = GT_json

        with open(os.path.join(self.opt.work_dir, 'GT_kpt.json'), 'w') as fid:
            json.dump(GT_dict, fid)
        return os.path.join(self.opt.work_dir, 'GT_kpt.json')

    def is_finished(self, query_list: list, OKS_dict: dict) -> tuple:
        """judge if the active learning is finished based on the threshold"""
        time = (len(self.labeled_id.index)/self.eval_len)*100 # label percentage: 0~100
        # judgement of actual finish
        if np.all(np.array(list(OKS_dict.values())) >= self.finish_acc) and time<self.actual_finish: # if all OKS is higher than the threshold, return True
            print(f'[Finished] Actually finished at {time:.1f}%!')
            self.actual_finish = time
        # judgement of min-error
        OKS_q = np.array(list(OKS_dict[i] for i in query_list)) # queried OKS
        if np.mean(OKS_q) >= self.finish_acc and time<self.finished_minerror: # if mean OKS is higher than the threshold, return True
            print(f'[Finished] Min-error SC has met at {time:.1f}%!')
            self.finished_minerror = time
        # judgement of our sc
        idx_labeled_queried = self.labeled_id.index + query_list # get labeled and queried data
        OKS_lq = np.array(list(OKS_dict[i] for i in idx_labeled_queried)) # labeled and queried OKS
        if np.all(OKS_lq >= self.finish_acc) and time<self.finished_oursc: # if all OKS is higher than the threshold, return True
            print(f'[Finished] Our SC has met at {time:.1f}%!')
            self.finished_oursc = time
        return self.actual_finish, self.finished_minerror, self.finished_oursc

    def random_query(self, candidate_list, query_size):
        # pdb.set_trace()
        query_list = []
        while (len(query_list) < query_size) and (len(candidate_list) > 0):
            query_index = int(np.random.choice(candidate_list)) # randomly select the query
            query_list.append(query_index)
            candidate_list.remove(query_index) # remove the query from candidate list
        return query_list

    def compute_tpc(self, current_pose, pred_prev, bbox, thresh):
        """
        compute temporal pose continuity.
        Args:
            current_pose (_type_): _description_
        """
        adjacent_pose, _ = self.heatmap_to_coord(pred_prev[self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
        pose_dist = np.linalg.norm(current_pose - adjacent_pose, axis=1) # euclidean distance between current and previous
        tpc = np.count_nonzero(pose_dist > thresh) # count number of joints that move more than threshold
        return tpc

    def compute_thc(self, heatmaps, heatmaps_adj, norm_type='L1'):
        """compute temporal heatmap continuity between heatmaps of current frame and ones of adjacent frame.
        Args:
            heatmaps (ndarray): heatmaps of current frame. shape = (num_joints, height, width)
            heatmaps_adj (ndarray): heatmaps of adjacent frame (previous or next). shape = (num_joints, height, width)
        """
        keypoint_num = heatmaps.shape[0]
        if norm_type == 'L1':
            thc = np.sum(np.abs(heatmaps - heatmaps_adj)) / keypoint_num
            # print(f'[THC] L1 norm: {thc}')
        elif norm_type == 'L2':
            thc = np.sum(np.square(heatmaps - heatmaps_adj)) / keypoint_num
            # print(f'[THC] L2 norm: {thc}')
        return thc

    def compute_mpe(self, heatmaps):
        """compute multiple peak entropy.
        get local peaks from heatmaps and compute entropy of the number of peaks.

        Args:
            heatmaps (ndarray): heatmaps of current frame. shape = (num_joints, height, width)
        Returns:
            mpe (float): value of multiple peak entropy.
        """
        mpe = 0
        for heatmap in heatmaps:
            loc_peaks = peak_local_max(heatmap, min_distance=5, num_peaks=5)
            peaks = heatmap[loc_peaks[:, 0], loc_peaks[:, 1]]
            if peaks.shape[0] > 0:
                peaks = softmax(peaks)
                mpe += entropy(peaks)
        return mpe

    def compute_margin(self, heatmaps):
        """get local peaks from heatmaps and compute margin of the top2 peaks."""
        margin = 0
        for heatmap in heatmaps:
            loc_peaks = peak_local_max(heatmap, min_distance=5, num_peaks=5)
            peaks = heatmap[loc_peaks[:, 0], loc_peaks[:, 1]]
            if peaks.shape[0] > 1:
                margin += np.linalg.norm(peaks[0] - peaks[1])
        return margin

    def compute_entropy(self, heatmaps):
        """compute entropy of heatmaps."""
        entropy_value = 0
        for heatmap in heatmaps:
            heatmap = heatmap.flatten()
            entropy_value += entropy(heatmap)
        return entropy_value

    def coreset_selection(self, embeddings, uncertainty):
        """return: query idexs list sampled by k-center greedy algorithm"""
        labeled_idx = self.labeled_id.index
        query_list = []
        def update_distances(cluster_centers, encoding, min_distances=None):
            '''Update min distances given cluster centers.
            cluster_centers: indices of cluster centers'''
            if len(cluster_centers) != 0:
                # Update min_distances for all examples given new cluster center.
                x = encoding[cluster_centers]
                dist = pairwise_distances(encoding, x, metric='euclidean')

                if min_distances is None:
                    min_distances = np.min(dist, axis=1).reshape(-1, 1)
                else:
                    min_distances = np.minimum(min_distances, dist)
            return min_distances
        def _query_w_unc(min_distances, uncertainty):
            '''Select examples with highest uncertainty.'''
            if len(labeled_idx) == 0:
                ind = np.argmax(uncertainty)
            else:
                ind = np.argmax(((1-self.moks_queried)*min_distances.reshape(-1)) + (self.unc_lambda*self.moks_queried*uncertainty))
            return ind
        def _query_fixed_lambda(min_distances, uncertainty):
            if len(labeled_idx) == 0:
                ind = np.argmax(uncertainty)
            else:
                ind = np.argmax(min_distances.reshape(-1)+self.unc_lambda*uncertainty)
            return ind
        def _query(min_distances, _):
            if len(labeled_idx) == 0:
                ind = np.random.choice(np.arange(embeddings.shape[0]))
            else:
                ind = np.argmax(min_distances.reshape(-1))
            return ind
        if self.uncertainty == "None" or self.cfg.VAL.UNC_LAMBDA == 0:
            func_query = _query
        elif self.opt.fixed_lambda:
            func_query = _query_fixed_lambda
        else:
            func_query = _query_w_unc

        min_distances = update_distances(cluster_centers=labeled_idx, encoding=embeddings, min_distances=None) # initialize min_distances
        for _ in tqdm(range(self.query_size)):
            ind = func_query(min_distances, uncertainty)
            # print(f"{ind} was selected!")
            # New examples should not be in already selected since those points should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=embeddings, min_distances=min_distances)
            labeled_idx = np.concatenate([labeled_idx, [ind]], axis=0).astype(np.int32)
            uncertainty[ind] = 0 # set uncertainty of selected data to zero
            query_list.append(int(ind))
        return query_list

    def get_retrain_id(self: object, query_list: list, OKS_dict: dict) -> list:
        """return: the index of the index of datapoint to be retrained"""
        # 1. get the datapoint x_thr with highest OKS from newly-queried data
        OKS_dict_queried = dict((idx,OKS) for idx, OKS in OKS_dict.items() if idx in query_list) # get OKS of newly-queried data
        OKS_dict_queried_sorted = sorted(OKS_dict_queried.items(), key=lambda x: x[1], reverse=True) # sort by OKS, descending order
        # print(f"[OKS_dict_queried_sorted], len: {len(OKS_dict_queried_sorted)}, {OKS_dict_queried_sorted}")
        mOKS_queried = np.mean([OKS for idx, OKS in OKS_dict_queried_sorted]) # get the mean OKS of newly-queried data

        # 2. pick the labeled datapoint with lower OKS than x_thr
        OKS_labeled = dict((idx,OKS) for idx, OKS in OKS_dict.items() if idx in self.labeled_id.index) # get OKS of labeled data
        # print(f"[OKS_labeled], len: {len(OKS_labeled)}, {OKS_labeled}")
        mOKS_labeled = np.mean([OKS for idx, OKS in OKS_labeled.items()]) if len(OKS_labeled) > 0 else 0
        OKS_unlabeled = dict((idx,OKS) for idx, OKS in OKS_dict.items() if idx in self.unlabeled_id.index)
        mOKS_unlabeled = np.mean([OKS for idx, OKS in OKS_unlabeled.items()]) # get the mean OKS of unlabeled data
        print(f"[mOKS_queried]: {mOKS_queried:.3f}, [mOKS_labeled]: {mOKS_labeled:.3f}, [mOKS_unlabeled]: {mOKS_unlabeled:.3f}")

        retrain_id = [idx for idx, OKS in OKS_labeled.items() if OKS <= self.finish_acc + self.finish_margin] # get the index of labeled data with lower OKS than x_thr
        retrain_id += query_list # add newly-queried data to retrain data
        # print(f"[Retrain Data]: {sorted(retrain_id)}")
        return retrain_id, mOKS_queried

    def get_corresponding_id(self: object, OKS_dict: dict, true=True, labeled=True) -> list:
        """return: the index list of corresponding samples"""
        thresh = self.finish_acc + self.finish_margin
        if true and labeled: # get the index list of (labeled and acc>thresh) samples
            corresponding_id = [idx for idx, OKS in OKS_dict.items() if (idx in self.labeled_id.index) and (OKS >= thresh)]
        elif true and not labeled: # true unlabeled
            corresponding_id = [idx for idx, OKS in OKS_dict.items() if (idx in self.unlabeled_id.index) and (OKS >= thresh)]
        elif not true and labeled: # false labeled
            corresponding_id = [idx for idx, OKS in OKS_dict.items() if (idx in self.labeled_id.index) and (OKS < thresh)]
        elif not true and not labeled: # false unlabeled
            corresponding_id = [idx for idx, OKS in OKS_dict.items() if (idx in self.unlabeled_id.index) and (OKS < thresh)]
        return corresponding_id

    def initialize_AE(self):
        if True:
            AE = WholeBodyAE(z_dim=self.cfg.AE.Z_DIM)
            if self.opt.optimize:
                mode = 'train_val' if self.dataset == "Posetrack21" else 'val'
                AE_dataset = Wholebody(mode=mode, retrain_video_id=self.video_id, dataset_type=self.dataset)
            else:
                mode = 'val' if self.dataset == "Posetrack21" else 'test'
                AE_dataset = Wholebody(mode=mode, retrain_video_id=self.video_id, dataset_type=self.dataset)
            pretrained_path = os.path.join(self.cfg.AE.PRETRAINED_ROOT, 'Hybrid', f'WholeBodyAE_zdim{self.cfg.AE.Z_DIM}.pth')
        else: # use raw keypoint as feature
            AE = WholeBodyAE(z_dim=self.cfg.AE.Z_DIM, kp_direct=True)
            # AE_dataset = Wholebody(mode='val', retrain_video_id=self.video_id, kp_direct=True)
            pretrained_path = os.path.join(self.cfg.AE.PRETRAINED_ROOT, 'Raw', f'WholeBodyAE_zdim{self.cfg.AE.Z_DIM}.pth')
        AE.load_state_dict(torch.load(pretrained_path))
        AE.cuda()
        AE.eval()
        return AE, AE_dataset

    def retrain_AE(self):
        """Fine-tune the AE with the labeled data"""
        print(f'[Retrain AE]')
        loss_logger = DataLogger()
        train_subset = Subset(self.AE_dataset, self.labeled_id.index)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=10, shuffle=True, num_workers=3, drop_last=False, pin_memory=True)
        self.AE.train()
        train_loss = 0
        with tqdm(range(self.cfg.AE.EPOCH), dynamic_ncols=True) as progress_bar:
            for epoch in progress_bar:
                for i, feature in enumerate(train_loader):
                    input = feature.cuda()
                    output = self.AE(input)
                    loss = self.criterion(output, input)
                    train_loss += loss.item()
                    loss_logger.update(loss.item(), feature.size(0))
                    self.AE_optimizer.zero_grad()
                    loss.backward()
                    self.AE_optimizer.step()
                # TQDM
                progress_bar.set_description('loss: {loss:.6f}'.format(loss=loss_logger.avg))

    def visualize_thc(self, adjimgs, adjhps, thc, ann_id, keypoints):
        """visualize temporal heatmap continuity between current frame and adjacent frame.
        Args:
            adjimgs (list, Tensor): adjacent frames. shape = (3, height, width, channel)
            hp_current (list, Tensor): current frame heatmaps. shape = (3, num_joints, height, width)
            thc (float): temporal heatmap continuity value
            ann_id (int): annotation id of the sample
        """
        ann_id = ann_id % 10000
        joints_vis = [1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1] # 17 joints
        batch_images =[torch.Tensor(adjimg).view(3, 256, 192) for adjimg in adjimgs] # [3, 3, 256, 192]
        batch_heatmaps = [torch.Tensor(heatmaps).view(1, 17, 64, 48) for heatmaps in adjhps] # [3, 17, 64, 48]
        preds = [self.get_max_preds(batch_heatmaps[t].detach().cpu().numpy())[0] for t in range(3)] # [3, 17, 2]
        hm_width, hm_height  =int(adjhps[0].shape[2]), int(adjhps[0].shape[1]) # 64, 48

        batch_images = [batch_image.clone() for batch_image in batch_images]
        min = [float(batch_image.min()) for batch_image in batch_images]
        max = [float(batch_image.max()) for batch_image in batch_images]
        batch_images = [batch_images[t].add_(-min[t]).div_(max[t] - min[t] + 1e-5) for t in range(3)]
        for t in range(3):
            img = batch_images[t].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() # [256, 192, 3]
            img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img.copy(), (hm_width, hm_height)) # [64, 48, 3]
            batch_images[t] = img
            heatmaps = batch_heatmaps[t][0].mul(255).clamp(0, 255).byte().cpu().numpy() # [17, 64, 48]
            # pdb.set_trace()
            batch_heatmaps[t] = 255-heatmaps
        for j in range(len(joints_vis)):
            if joints_vis[j] == 0:
                continue
            grid_image = np.zeros((hm_height,3*hm_width,3),dtype=np.uint8) # [64, 144, 3]
            grid_original_image = np.zeros((hm_height,3*hm_width,3),dtype=np.uint8) # [64, 144, 3]
            for t in range(3):
                img = batch_images[t]
                heatmap = batch_heatmaps[t][j, :, :]
                # adjust heatmap color
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap*0.5 + img*0.5
                width_begin = hm_width * (t%3)
                width_end = hm_width * (t%3+1)
                # print(width_begin, width_end)
                grid_image[:, width_begin:width_end, :] = masked_image
                # original image with estimated joint position
                # convert color
                joint = preds[t][0][j]
                img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                cv2.circle(img=img, center=(int(joint[0]), int(joint[1])), radius=1, color=[0, 255, 0], thickness=2)
                grid_original_image[:, width_begin:width_end, :] = img

            # save image (add colorbar)
            save_path = os.path.join(self.opt.work_dir, "THC_vis", f"id{ann_id}_THC{thc}/joint{j}.png")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig, ax = plt.subplots()
            im = ax.imshow(grid_image, cmap='jet', vmin=0, vmax=1)
            cax = fig.colorbar(im, ax=ax) # add colorbar
            fig.savefig(io.BytesIO())
            ax_pos = ax.get_position()
            cax_pos0 = cax.ax.get_position()
            cax_pos1 = [cax_pos0.x0, ax_pos.y0, cax_pos0.x1 - cax_pos0.x0, ax_pos.y1 - ax_pos.y0]
            cax.ax.set_position(cax_pos1)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            # save original image with estimated joint position
            save_path = os.path.join(self.opt.work_dir, "THC_vis", f"id{ann_id}_THC{thc}/joint{j}_original.png")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig, ax = plt.subplots()
            img = ax.imshow(grid_original_image)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def visualize_wpu(self, input, output, wpu, ann_id, img, GT_keypoint):
        """visualize input, output hybrid features.
        Args:
            input : input hybrid feature
            output : output hybrid feature
            wpu : wpu score
            ann_id : annotation id of the sample
        """
        # joint_pairs = [[13, 11], [11, 9], [14, 12], [12, 10], [9, 10], [3, 9], [4, 10], [3, 5], [4, 6], [5, 7], [6, 8], [0, 1], [0, 2], [1, 3], [1, 4]]
        joint_vis = [1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1] # 17 joints
        joint_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 7], [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 5], [1, 6]]
        # print(f"input: {input[:30]}")
        # print(f"input: {output[:30]}")

        track_id = int(str(ann_id)[-2:])
        img_id = int(str(ann_id)[7:-2])

        # visualize input and output hybrid features
        # save independently also
        color = ['r', 'b']
        poses = [np.array(input), np.array(output)]
        result = ['input', 'output']
        for i in range(2):
            plt.figure()
            for j in range(len(joint_vis)):
                if joint_vis[j] == 0:
                    continue
                plt.scatter(poses[i][j], -poses[i][j+len(joint_vis)], c=color[i], s=100)
                # plt.scatter(poses[i][:len(joint_vi], -poses[i][len(joint_vis):2*len(joint_vis)], c=color[i])
            for pair in joint_pairs:
                plt.plot([poses[i][pair[0]], poses[i][pair[1]]], [-poses[i][pair[0]+len(joint_vis)], -poses[i][pair[1]+len(joint_vis)]], c=color[i], linewidth=5)
            plt.title(f'img_id: {img_id}, track_id: {track_id}')
            save_path = os.path.join(self.opt.work_dir, 'WPU_vis', f'img{img_id}', f'{wpu}_id{track_id}_{result[i]}.png')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close()

    def pltcluster_and_save(self, cluster_idxs, embeddings, track_ids_list, query_list, weight=None): # plot cluster result and save to file
        """plot cluster and save the figure to file.
        cluster_idxs (list): list of cluster index for each sample. use for color.
        embeddings (ndarray): embeddings of each sample.
        track_ids_list (list): list of track id for each sample. use for label.
        """
        # plot cluster result using densmap in UMAP
        densmap_model = umap.UMAP(densmap=True, random_state=318) # densmap=True
        densmap_emb = densmap_model.fit_transform(embeddings) # (num_samples, 2)

        # plot cluster result using DensMAP
        cluster_num = len(set(cluster_idxs))
        # color = [plt.cm.jet(i) for i in np.linspace(0, 1, cluster_num)]
        track_id = [int(track_id) for track_id in track_ids_list]
        if weight is None:
            weight = 0.5 * np.ones_like(cluster_idxs)
        weight = weight / np.max(weight) # normalize to [0, 1]

        plt.figure()
        markers = ['o', 'x', 's', 'D', 'v', 'p', 'h', '^', '<', '>', 'H', 'd', 'P', 'X']
        if len(markers) < len(set(track_ids_list)):
            markers = markers * (cluster_num // len(markers) + 1)
        for i in range(embeddings.shape[0]):
            s = 200 * weight[i] + 10
            c = cluster_idxs[i] / cluster_num if len(set(cluster_idxs)) > 1 else 0.5
            # marker = 'x' if i in query_list else 'o'
            # alpha = 1.0 if i in query_list else 0.6
            plt.scatter(densmap_emb[i, 0], densmap_emb[i, 1], s=s, marker=markers[track_id[i]], c=[plt.cm.jet(c)], alpha=0.6)
            if i in query_list:
                plt.annotate("selected", (densmap_emb[i, 0], densmap_emb[i, 1]+0.2), fontsize=12)
        # plt.colorbar(ticks=np.arange(0, 1, 0.1), label='weight')
        save_dict = os.path.join(self.opt.work_dir, 'cluster_result')
        if not os.path.exists(save_dict):
            os.makedirs(save_dict)
        plt.savefig(save_dict + f'/Round{self.round_cnt}_densmap.png') # save figure
        plt.savefig(save_dict + f'/Round{self.round_cnt}_densmap.SVG') # save figure
        plt.close()

    def pltcoreset_and_save(self, embeddings, query_list, unc_list): # plot cluster result and save to file
        """plot cluster and save the figure to file.
        embeddings (ndarray): embeddings of each sample.
        query_list: list of query index
        unc_list: list of uncertainty (normalized)
        """
        # plot cluster result using densmap in UMAP
        from matplotlib import colors as mcolors
        densmap_model = umap.UMAP(densmap=True, random_state=318) # densmap=True
        densmap_emb = densmap_model.fit_transform(embeddings) # (num_samples, 2)

        # plot cluster result using DensMAP
        # color = [plt.cm.jet(i) for i in np.linspace(0, 1, cluster_num)]

        plt.figure()
        print(np.array(unc_list)[query_list])
        print(unc_list)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        for sample in range(embeddings.shape[0]): # for every sample in candidate_list
            marker = 'o' if sample in query_list else 'x'
            size = 120 if sample in query_list else 40
            plt.scatter(densmap_emb[sample, 0], densmap_emb[sample, 1], s=size, marker=marker, c=(unc_list[sample]), cmap="jet", alpha=0.8, norm=norm) # change color according to uncertainty
            if sample in query_list:
                plt.annotate("query", (densmap_emb[sample, 0], densmap_emb[sample, 1]+0.2), fontsize=10)
        plt.colorbar(label='uncertainty', norm=norm)
        save_dict = os.path.join(self.opt.work_dir, 'coreset_result')
        if not os.path.exists(save_dict):
            os.makedirs(save_dict)
        plt.savefig(save_dict + f'/Round{self.round_cnt}_densmap.png') # save figure
        plt.savefig(save_dict + f'/Round{self.round_cnt}_densmap.pdf') # save figure
        plt.close()

    def vl4pose(self, pose_encodings, heatmaps, joint_exist, epoch):
        '''
            Train the auxiliary network for VL4Pose.

            - param pose_encodings: (Dict of tensors) output from the pose network
            - param heatmaps: Tensor of size (batch_size, num_joints, hm_size, hm_size)
            - param joint_exist: Tensor of size (batch_size, num_joints)
                - Joint status (1=Present, 0=Absent)
                - Present=1 may include occluded joints depending on configuration.yml setting
            - param epoch: (scalar) Epoch, used in auxiliary network loss warm start-up

            - return auxnet_loss: (Torch scalar tensor) auxiliary network Loss
        '''

        assert joint_exist.dim() == 2, "joint_exist should be BS x num_hm, received: {}".format(joint_exist.shape)
        j2i = self.dataset_obj.jnt_to_ind


        if self.conf.dataset['load'] == 'mpii':
            links = [[j2i['head'], j2i['neck']], [j2i['neck'], j2i['thorax']], [j2i['thorax'], j2i['pelvis']],
                     [j2i['thorax'], j2i['lsho']], [j2i['lsho'], j2i['lelb']], [j2i['lelb'], j2i['lwri']],
                     [j2i['thorax'], j2i['rsho']], [j2i['rsho'], j2i['relb']], [j2i['relb'], j2i['rwri']],
                     [j2i['pelvis'], j2i['lhip']], [j2i['lhip'], j2i['lknee']], [j2i['lknee'], j2i['lankl']],
                     [j2i['pelvis'], j2i['rhip']], [j2i['rhip'], j2i['rknee']], [j2i['rknee'], j2i['rankl']]]
        else:
            links = [[j2i['head'], j2i['neck']],
                     [j2i['neck'], j2i['lsho']], [j2i['lsho'], j2i['lelb']], [j2i['lelb'], j2i['lwri']],
                     [j2i['neck'], j2i['rsho']], [j2i['rsho'], j2i['relb']], [j2i['relb'], j2i['rwri']],
                     [j2i['lsho'], j2i['lhip']], [j2i['lhip'], j2i['lknee']], [j2i['lknee'], j2i['lankl']],
                     [j2i['rsho'], j2i['rhip']], [j2i['rhip'], j2i['rknee']], [j2i['rknee'], j2i['rankl']]]

        parameters = self._aux_net_inference(pose_encodings)
        parameters = parameters.reshape(self.cfg.VAL.BATCH_SIZE, len(links), 2)
        joint_exist = joint_exist.to(parameters.device)
        heatmaps = heatmaps.to(parameters.device)

        with torch.no_grad():

            joint_distances = get_pairwise_joint_distances(heatmaps)
            joint_exist = torch.matmul(joint_exist.unsqueeze(2).type(torch.float16), joint_exist.unsqueeze(1).type(torch.float16))

            # Batch Size x num_links
            skeleton_exist = torch.stack([joint_exist[:, u, v] for u,v in links], dim=1)
            skeleton_distances = torch.stack([joint_distances[:, u, v] for u,v in links], dim=1)

        ####
        residual = (parameters[:, :, 0].squeeze() - skeleton_distances)**2
        residual = 0.5 * residual * torch.exp(-parameters[:, :, 1]).squeeze()
        neg_log_likelihood = residual + (0.5 * parameters[:, :, 1].squeeze())
        neg_log_likelihood = neg_log_likelihood * skeleton_exist

        if 100 <= epoch:
            return torch.mean(neg_log_likelihood)

        else:
            return 0.0 * torch.mean(neg_log_likelihood)


    def _aux_net_inference(self, pose_features):
        """
        Common to VL4Pose, LearningLoss++ and Aleatoric which all use an auxiliary network
        """
        with torch.no_grad():
            depth = len(self.conf.architecture['aux_net']['spatial_dim'])
            encodings = torch.cat([pose_features['feature_{}'.format(i)].reshape(
                    self.cfg.VAL.BATCH_SIZE, pose_features['feature_{}'.format(i)].shape[1], -1) for i in range(depth, 0, -1)], dim=2)
        aux_out = self.aux_net(encodings)
        return aux_out

    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals