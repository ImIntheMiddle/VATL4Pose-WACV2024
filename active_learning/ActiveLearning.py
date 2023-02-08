# general libraries
import argparse
import os
import pdb # import python debugger
import platform
import sys
import time
import random
import json

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from skimage.feature import peak_local_max
from cachetools import cached
import seaborn as sns
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans

# 3rd party libraries
from alipy.experiment import StoppingCriteria
from alipy.index import IndexCollection
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP, calc_accuracy, DataLogger
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord, norm_heatmap)
from alphapose.utils.bbox import bbox_xyxy_to_xywh
from alphapose.utils.vis import vis_frame_fast, vis_frame

# original modules
from active_learning.local_peak import localpeak_mean
from .Whole_body_AE import WholeBodyAE
from .Whole_body_AE import Wholebody
from .Whole_body_AE.hybrid_feature import compute_hybrid

"""---------------------------------- Main Class ----------------------------------"""
class ActiveLearning:
    def __init__(self, cfg, opt):
        self.round_cnt = 0
        self.cfg = cfg
        self.opt = opt
        self.one_by_one = self.opt.onebyone # if True, then only one image is selected per round
        if self.one_by_one:
            print("One by one annotation is enabled...")
        self.model = self.initialize_estimator()
        self.strategy = self.opt.strategy
        self.uncertainty = self.opt.uncertainty
        self.representativeness = self.opt.representativeness
        self.filter = self.opt.filter # type of filter
        self.video_id = self.opt.video_id
        self.get_prenext = True if self.uncertainty in ['TPC', 'THC_L1', 'THC_L2'] else False
        IMG_PREFIX = f'images/val/{self.video_id}_mpii_test/'
        ANN = f'activelearning/val/{self.video_id}_mpii_test.json'
        self.cfg.DATASET.EVAL.IMG_PREFIX = IMG_PREFIX
        self.cfg.DATASET.EVAL.ANN = ANN
        self.cfg.DATASET.TRAIN.IMG_PREFIX = IMG_PREFIX
        self.cfg.DATASET.TRAIN.ANN = ANN

        # Evaluation settings
        self.eval_dataset = builder.build_dataset(self.cfg.DATASET.EVAL, preset_cfg=self.cfg.DATA_PRESET, train=False, get_prenext=self.get_prenext)
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.cfg.VAL.BATCH_SIZE*self.opt.num_gpu, shuffle=False, num_workers=os.cpu_count(), drop_last=False, pin_memory=True)

        # AL_settings
        self.stopping_criterion = StoppingCriteria(stopping_criteria=None)
        self.query_size = int(len(self.eval_dataset) * self.cfg.VAL.QUERY_RATIO)
        self.expected_performance = 0 # expected performance. It controls the balance between uncertainty and representativeness
        if self.one_by_one:
            self.query_size = 1 # query one by one
        self.unlabeled_id = IndexCollection(list(range(len(self.eval_dataset))))
        self.eval_len = len(self.eval_dataset)
        self.labeled_id = IndexCollection()
        self.percentage = [] # number of query for each round
        self.performance = [] # list of mAP for each round
        self.combine_weight = [] # list of combine weight for each round
        self.query_list_list = {} # list of query for each round
        self.uncertainty_dict = {} # dict of uncertainty and idx for each round
        self.uncertainty_mean = [] # list of mean of uncertainty for each round
        self.influence_dict = {} # dict of mean of influence for each round

        # Training settings
        self.train_dataset = builder.build_dataset(self.cfg.DATASET.TRAIN, preset_cfg=self.cfg.DATA_PRESET, train=True, get_prenext=False) # get_prenext=False for training
        self.retrain_epoch = 4 * self.cfg.RETRAIN.EPOCH
        if self.cfg.RETRAIN.OPTIMIZER == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.RETRAIN.LR, momentum=0.9, weight_decay=0.0005)
        elif self.cfg.RETRAIN.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.RETRAIN.LR)
        self.criterion = builder.build_loss(self.cfg.LOSS).cuda()

        # WPU settings
        if self.uncertainty == 'WPU_raw' or self.uncertainty == 'WPU_hybrid':
            self.AE, self.AE_dataset = self.initialize_AE()
            self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=0.001)
        # Other settings
        if self.opt.verbose: # test dataset and dataloader
            self.test_dataset(self.eval_dataset)
        self.eval_joints = self.eval_dataset.EVAL_JOINTS
        self.norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg) # function to get final prediction from heatmap
        self.show_info() # show information of active learning

    def outcome(self):
        """Check if the active learning is stopped"""
        print(f"[Judge] Unlabeled items: {len(self.unlabeled_id.index)}, Labeled items: {len(self.labeled_id.index)}")
        self.retrain_model() # Retrain the model with the labeled data
        self.round_cnt += 1
        if len(self.unlabeled_id.index) == 0:
            self.eval_and_query() # Evaluate final estimator performance
            print("--> Finished!\n")
            return self.percentage, self.performance, self.query_list_list, self.uncertainty_dict, self.uncertainty_mean, self.influence_dict, self.combine_weight
        else:
            # continue
            return None
        # return self.stopping_criterion.is_stop()

    def initialize_estimator(self): # Load an initial pose estimator
        """construct a initial pose estimator
        Returns:
            model: Initial pose estimator
        """
        model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        if self.cfg.MODEL.PRETRAINED:
            print(f'Loading model from {self.cfg.MODEL.PRETRAINED}...') # pretrained by PoseTrack21
            model.load_state_dict(torch.load(self.cfg.MODEL.PRETRAINED))
            # freeze the parameters of feature extractor
            # for param in model.parameters():
            #     param.requires_grad = False
            # # unfreeze the parameters of the last layer
            # # pdb.set_trace()
            # for param in model.final_layer.parameters():
            #     param.requires_grad = True
        if self.opt.device == torch.device('cuda'):
            model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).cuda()
        return model

    def show_info(self):
        print(f'\n[[AL strategy: {self.opt.strategy}]]')
        print(f'[[Video ID: {self.opt.video_id}]]')
        print(f'[[Number of queries: {len(self.eval_dataset)}]]')
        print(f'[[Pose Estimator: {self.cfg.MODEL.TYPE}]]')
        print(f'[[eval_jounts: {self.eval_joints}]]')

    def eval_and_query(self):
        """Evaluate the current estimator and query the most uncertain samples"""
        print(f'\n[[{self.strategy}: Round {self.round_cnt}]]')
        kpt_json = []
        m = self.model
        m.eval()
        query_candidate = {}
        total_uncertainty = 0 # total uncertainty initialized to 0
        combine_weight = 0
        fvecs_matrix = np.zeros((len(self.eval_dataset), 2048)) # 2048 is the dimension of the feature vector
        with torch.no_grad():
            for i, (idxs, inps, labels, label_masks, img_ids, ann_ids, bboxes, isPrev, isNext) in enumerate(tqdm(self.eval_loader, dynamic_ncols=True, leave=False)):
                # pdb.set_trace()
                # print(inps.shape) # 62tensors, 3frames, 3channels, 256x192 each
                # print(inps[0][0].shape) # (3, 256, 192 # one current frame, 3 channels, 256x192
                output = m(inps[:, 0].to(self.opt.device)) # input a batch of current frames (62, 3, 256, 192)
                # print(output.shape) # (62, 17, 64, 48) # 62 frames, 17 joints, 64x48 heatmap
                # print(output[0].shape) # (17, 64, 48) # one frame, 17 joints, 64x48 heatmap
                assert output.dim() == 4
                pred = output[:, self.eval_joints, :, :]

                if self.representativeness not in ['None', 'Random'] or self.filter not in ['None', 'Random']:
                    fvecs = self.model.module.get_embedding(inps[:, 0].to(self.opt.device))
                    # print(fvecs.shape) # (62, 2048) # 62 frames, 2048 features
                    fvecs_matrix[idxs] = fvecs.cpu().numpy()
                    # pdb.set_trace()

                # get prediction for previous/next frame
                if self.get_prenext:
                    output_prev = m(inps[:, 1].to(self.opt.device))
                    pred_prev = output_prev[:, self.eval_joints, :, :]
                    output_next = m(inps[:, 2].to(self.opt.device))
                    pred_next = output_next[:, self.eval_joints, :, :]

                for j in range(output.shape[0]): # for each input in batch
                    heatmaps_dict = {"Round": [], "ann_ids": [], "heatmaps": [], "img": [], "img_ids": [], "bboxes": [], "uncertainty": [],"keypoints": []} # for visualization
                    bbox = bboxes[j].tolist()
                    pose_coords, pose_scores = self.heatmap_to_coord(pred[j][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
                    # print(pose_coords) # (17, 2)
                    keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                    keypoints = keypoints.reshape(-1).tolist()
                    data = dict()
                    data['bbox'] = bbox # format: xyxy
                    data['image_id'] = int(img_ids[j])
                    data['id'] = int(ann_ids[j])
                    data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
                    data['category_id'] = 1
                    data['keypoints'] = keypoints
                    kpt_json.append(data)

                    if (self.uncertainty == 'HP' ): # for unlabeled data, highest probability
                        uncertainty = float(-np.sum(pose_scores)) # calculate uncertainty
                        # print(type(-np.sum(pose_scores)))
                        # pdb.set_trace()
                    elif (self.uncertainty == 'WPU_hybrid' or self.uncertainty == 'WPU_raw'):
                        if self.uncertainty == 'WPU_hybrid': # use hybrid feature
                            bbox = bbox_xyxy_to_xywh(bbox)
                            input = torch.tensor(compute_hybrid(bbox, keypoints)).float().cuda()
                        else: # use raw keypoint as feature
                            input = torch.tensor(keypoints).float().cuda()
                        output = self.AE(input) # output is the reconstructed keypoints
                        wpu = self.criterion(output, input) # calculate whole-body pose unnaturalness
                        uncertainty = float(wpu)
                    elif (self.uncertainty == 'TPC'): # temporal pose continuity
                        thresh = 0.01 * np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) # threshold for euclidean distance. 0.01 is a hyperparameter
                        tpc = 0
                        if isPrev[j]:
                            tpc += self.compute_tpc(pose_coords, pred_prev[j], bbox, thresh)
                        if isNext[j]:
                            tpc += self.compute_tpc(pose_coords, pred_next[j], bbox, thresh)
                            if not isPrev[j]:
                                tpc *= 2 # if only next frame exists, double the uncertainty
                        elif isPrev[j]:
                            tpc *= 2 # if only previous frame exists, double the uncertainty
                        uncertainty = float(tpc) # convert to float, tpc as uncertainty
                    elif (self.uncertainty == 'THC_L1') or (self.uncertainty == 'THC_L2'): # temporal heatmap continuity
                        norm_type = 'L1' if self.uncertainty == 'THC_L1' else 'L2'
                        thc = 0
                        hp_current = pred[j].cpu().numpy()
                        if isPrev[j]:
                            hp_pred = pred_prev[j].cpu().numpy()
                            thc += self.compute_thc(hp_current, hp_pred, norm_type=norm_type)
                        if isNext[j]:
                            hp_next = pred_next[j].cpu().numpy()
                            thc += self.compute_thc(hp_current, hp_next, norm_type=norm_type)
                            if not isPrev[j]:
                                thc *= 2
                        elif isPrev[j]:
                            thc *= 2
                        uncertainty = float(thc)
                    elif (self.uncertainty == "None"): # no uncertainty
                        uncertainty = 0
                    else: # error
                        raise ValueError("Uncertainty type is not supported")
                    total_uncertainty += uncertainty # add to total uncertainty
                    if idxs[j] in self.unlabeled_id.index:
                        combine_weight += localpeak_mean(pred[j][self.eval_joints].cpu().numpy()) # find the minimum peak of the heatmap
                        # print(localpeak_mean(pred[j][self.eval_joints].cpu().numpy()))
                        query_candidate[int(idxs[j])] = uncertainty # add to candidate list
                    if self.opt.vis and (self.round_cnt % 4 == 0): # visualize every 4 rounds
                        heatmaps_dict["Round"] = self.round_cnt
                        heatmaps_dict["ann_ids"] = int(ann_ids[j])
                        heatmaps_dict["heatmaps"] = pred[j].cpu().numpy() # copy to cpu and convert to numpy array
                        heatmaps_dict["img"] = inps[j, 0].cpu().numpy()
                        heatmaps_dict["img_ids"] = int(img_ids[j])
                        heatmaps_dict["keypoints"] = keypoints
                        heatmaps_dict["uncertainty"] = uncertainty
                        heatmaps_dict["bboxes"].append(bbox)
                        hm_save_dir = os.path.join(self.opt.work_dir, 'heatmap', f'Round{self.round_cnt}')
                        if not os.path.exists(hm_save_dir):
                            os.makedirs(hm_save_dir)
                        np.save(os.path.join(hm_save_dir, f'{int(ann_ids[j])}.npy'), heatmaps_dict) # save heatmaps for visualization

        sysout = sys.stdout
        with open(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), 'w') as fid:
            json.dump(kpt_json, fid)
        if self.opt.vis and (self.round_cnt % 4 == 0): # 20% each
            pred_save_dir = os.path.join(self.opt.work_dir, 'prediction', f'Round{self.round_cnt}')
            if not os.path.exists(pred_save_dir):
                os.makedirs(pred_save_dir)
            with open(os.path.join(pred_save_dir, 'predicted_kpt.json'), 'w') as fid:
                json.dump(kpt_json, fid)
        res = evaluate_mAP(os.path.join(self.opt.work_dir, 'predicted_kpt.json'), ann_type='keypoints', ann_file=os.path.join(self.cfg.DATASET.EVAL.ROOT, self.cfg.DATASET.EVAL.ANN))
        sys.stdout = sysout

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

        if len(self.unlabeled_id.index) in [0, 1]: # if no unlabeled data or only one unlabeled data
            total_score = np.zeros(len(self.unlabeled_id.index))
        elif self.uncertainty != "None": # If using both uncertainty and representativeness
            uncertainty_score = np.array(list(query_candidate.values()))
            self.uncertainty_dict['Round'+str(self.round_cnt)] = dict((idx,uncertainty) for idx,uncertainty in zip(list(query_candidate.keys()), uncertainty_score)) # add to uncertainty dictionary
            uncertainty_score = (uncertainty_score - np.min(uncertainty_score)) / (np.max(uncertainty_score) - np.min(uncertainty_score)) # normalize uncertainty score
            # print("uncertainty_score:", uncertainty_score)
            if self.representativeness != "None":
                # combine uncertainty and representativeness, be careful to division by zero
                combine_weight /= len(self.unlabeled_id.index) # normalize combine weight, value: [0,1]
                self.combine_weight.append(combine_weight)
                # print("combine_weight:", combine_weight)
                total_score = combine_weight * uncertainty_score + (1-combine_weight) * influence_score # combine uncertainty and representativeness
            else:
                total_score = uncertainty_score # only uncertainty
        else: # If using only representativeness
            total_score = influence_score # only representativeness
        # print("total_score:", total_score)
        score_dict = dict((idx,score) for idx,score in zip(self.unlabeled_id.index, total_score)) # add to candidate dictionary
        # print(f'[Score List]: {score_dict}')
        score_dict_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=True) # sort by score
        score_dict = dict((int(idx),score) for idx,score in score_dict_sorted) # convert to dictionary

        # set filter. filter_size is the number of queries to be picked
        if self.filter == "None": # no query selection
            self.filter_size = self.query_size
        else: # both uncertainty and representativeness
            self.filter_size = 8 * self.query_size # accept 8 times of query size
        candidate_list = sorted(list(score_dict.keys())[:self.filter_size]) # pick top k candidates
        # print(f'[Candidate List] index: {candidate_list}')

        if len(self.unlabeled_id.index) in [0, 1]: # if no unlabeled data or only one unlabeled data
            query_list = candidate_list
        # filter query, select query from candidate list
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
            if len(self.unlabeled_id.index) < self.query_size: # if number of unlabeled data is less than query size
                self.query_size = len(self.unlabeled_id.index) # set query size to number of unlabeled data
            cluster_learner = KMeans(n_clusters=self.query_size, random_state=318)
            cluster_idxs = cluster_learner.fit_predict(embeddings) # cluster for each candidate
            centers = cluster_learner.cluster_centers_[cluster_idxs] # center for each candidate
            dis = (embeddings - centers)**2 # distance between each candidate and its center
            dis = dis.sum(axis=1)
            query_list = list([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(self.query_size)]) # pick query from each cluster, element of query_list is the index of candidate_list
            # pdb.set_trace()
            query_list = [int(candidate_list[i]) for i in query_list] # convert to list
        elif self.filter=="None":
            query_list = candidate_list
        else: # error
            raise ValueError("Filter type is not supported")

        self.percentage.append((len(self.labeled_id.index)/self.eval_len)*100) # label percentage: 0~100
        self.labeled_id.update(query_list)
        self.unlabeled_id.difference_update(query_list)
        self.performance.append(res*100) # mAP performance: 0~100
        self.query_list_list['Round'+str(self.round_cnt)] = list(map(int, query_list)) # add to query list dictionary
        print(f'[Query Selection] index: {sorted(query_list)}')
        print(f'[Evaluation] mAP: {res:.3f}, Percentage: {self.percentage[-1]:.1f}')

    def retrain_model(self):
        """Retrain the model with the labeled data"""
        print(f'[Retrain]')
        loss_logger = DataLogger()
        acc_logger = DataLogger()
        train_subset = Subset(self.train_dataset, self.labeled_id.index)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.cfg.RETRAIN.BATCH_SIZE*self.opt.num_gpu, shuffle=True, num_workers=os.cpu_count(),drop_last=False, pin_memory=True)
        self.model.train()

        if self.round_cnt == 1: # define the optimizer again
            self.retrain_epoch = self.cfg.RETRAIN.EPOCH
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.RETRAIN.LR)
        elif len(self.unlabeled_id.index) == 0: # no unlabeled data
            # self.model = self.initialize_estimator() # reinitialize the model
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.RETRAIN.LR)
            self.cfg.RETRAIN.ACCURACY_THRES = 1
            # self.retrain_epoch = 20 * self.cfg.RETRAIN.EPOCH

        with tqdm(range(self.retrain_epoch), dynamic_ncols=True) as progress_bar:
            for epoch in progress_bar:
                for i, (idxs, inps, labels, label_masks, img_ids, ann_ids, bboxes, isPrev, isNext) in enumerate(train_loader):
                    inps = inps[:, 0].cuda().requires_grad_()
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
                progress_bar.set_description('loss: {loss:.6f} | acc: {acc:.3f}'.format(loss=loss_logger.avg, acc=acc_logger.avg))
                if acc_logger.avg >= self.cfg.RETRAIN.ACCURACY_THRES:
                    break
        # fine-tuning AE
        # if self.uncertainty == 'WPU_hybrid' or self.uncertainty == 'WPU_raw':
            # self.retrain_AE() # retrain AE
            # torch.save(self.AE.state_dict(), './{}/latest_AE.pth'.format(self.opt.work_dir)) # AE
            # pass # no need to retrain AE
        torch.save(self.model.module.state_dict(), './{}/latest_estimator.pth'.format(self.opt.work_dir)) # Save checkpoint

    def quality_eval(self):
        """Visualize prediction using final estimator"""
        m = self.model
        m.eval()
        for i, (idxs, inps, labels, label_masks, img_ids, bboxes) in enumerate(tqdm(self.eval_loader, dynamic_ncols=True)):
            inps = [inp.cuda() for inp in inps] if isinstance(inps, list) else inps.cuda()
            # pdb.set_trace()
            with torch.no_grad():
                output = m(inps) # input inps into model and get heatmap
                assert output.dim() == 4
                pred = output[:, self.eval_joints, :, :]

    def test_dataset(self, dataset):
        # pdb.set_trace()
        assert len(dataset) >= 1 # at least 1 image
        print(dataset[0])

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

    def initialize_AE(self):
        if self.uncertainty == 'WPU_hybrid':
            AE = WholeBodyAE(z_dim=self.cfg.AE.Z_DIM)
            AE_dataset = Wholebody(mode='val', retrain_video_id=self.video_id)
            pretrained_path = os.path.join(self.cfg.AE.PRETRAINED_ROOT, 'Hybrid', f'WholeBodyAE_zdim{self.cfg.AE.Z_DIM}.pth')
        else: # use raw keypoint as feature
            AE = WholeBodyAE(z_dim=self.cfg.AE.Z_DIM, kp_direct=True)
            AE_dataset = Wholebody(mode='val', retrain_video_id=self.video_id, kp_direct=True)
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
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=5000, shuffle=True, num_workers=os.cpu_count(),drop_last=False, pin_memory=True)
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