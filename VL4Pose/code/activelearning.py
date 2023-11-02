import os
import cv2
import torch
import torch.utils.data
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import pairwise_distances
from skimage.feature import peak_local_max
from scipy.special import softmax as softmax_fn
from scipy.stats import entropy as entropy_fn
from matplotlib import pyplot as plt
# EGL sampling
import autograd_hacks
from utils import heatmap_loss
from utils import shannon_entropy
from utils import heatmap_generator


class ActiveLearning(object):
    """
    Contains collection of active learning algorithms for human joint localization
    """

    def __init__(self, conf, pose_net, aux_net): #(self, conf, hg_network, learnloss_network):
        self.conf = conf
        self.pose_model = pose_net
        self.aux_net = aux_net
        self.num_images = conf.active_learning['num_images']

        self.pose_model.eval()
        if conf.active_learning['algorithm'] in ['learning_loss', 'aleatoric', 'vl4pose']:
            self.aux_net.eval()


    def base(self, train, dataset_size):
        """

        :param train:
        :param dataset_size:
        :return:
        """
        logging.info('Initializing base dataset.')

        # Set of indices not annotated
        unlabelled_idx = train['index']

        # Determine if per dataset sampling or overall
        if self.conf.dataset['load'] == 'mpii':
            selection = np.random.choice(unlabelled_idx, size=self.num_images, replace=False).astype(np.int32)

        elif self.conf.dataset['load'] == 'lsp':
            # First sample from lsp dataset only
            lsp_images = min(dataset_size['lsp']['train'], self.num_images)
            lspet_images = self.num_images - lsp_images

            selection_lsp = np.random.choice(np.arange(dataset_size['lsp']['train']), size=lsp_images, replace=False)
            selection_lspet = np.random.choice(np.arange(dataset_size['lsp']['train'],
                                                         dataset_size['lsp']['train'] + dataset_size['lspet']['train']),
                                               size=lspet_images, replace=False)

            selection = np.concatenate([selection_lsp, selection_lspet], axis=0).astype(np.int32)

        else:
            # Merged dataset
            selection = np.random.choice(unlabelled_idx, size=self.num_images, replace=False).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='base')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def random(self, train, dataset_size):
        """

        :param train:
        :param dataset_size: Maintain same method signature across all sampling methods, not used
        :return:
        """
        logging.info('Performing random sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Load previously annotated images indices
        assert self.conf.model['load'], "Use 'base' to train model from scratch."
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))
        selection = np.random.choice(unlabelled_idx, size=self.num_images, replace=False)
        selection = np.concatenate([annotated_idx, selection], axis=0).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='random')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def coreset_sampling(self, train, dataset_size):
        '''

        :return:
        '''
        logging.info('Performing Core-Set sampling.')

        def update_distances(cluster_centers, encoding, min_distances=None):
            '''
            Based on: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
            Update min distances given cluster centers.
            Args:
              cluster_centers: indices of cluster centers
              only_new: only calculate distance for newly selected points and update
                min_distances.
              rest_dist: whether to reset min_distances.
            '''

            if len(cluster_centers) != 0:
                # Update min_distances for all examples given new cluster center.
                x = encoding[cluster_centers]
                dist = pairwise_distances(encoding, x, metric='euclidean')

                if min_distances is None:
                    min_distances = np.min(dist, axis=1).reshape(-1, 1)
                else:
                    min_distances = np.minimum(min_distances, dist)

            return min_distances

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "Core-set requires a pretrained model."
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert np.all(train['index'] == np.arange(train['name'].shape[0]))

        dataset_ = ActiveLearningDataset(train, indices=np.arange(train['name'].shape[0]), conf=self.conf)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        pose_encoding = None

        # Part 1: Obtain embeddings for all (labelled, unlabelled images)
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(coreset_dataloader):

                _, pose_features = self.pose_model(images)

                try:
                    pose_encoding = torch.cat((pose_encoding, pose_features['penultimate'].cpu()), dim=0)
                except TypeError:
                    pose_encoding = pose_features['penultimate'].cpu()

        pose_encoding = pose_encoding.squeeze().numpy()
        logging.info('Core-Set encodings computed.')

        # Part 2: k-Centre Greedy
        core_set_budget = self.num_images
        min_distances = None

        assert len(annotated_idx) != 0, "No annotations for previous model found, core-set cannot proceeed."
        min_distances = update_distances(cluster_centers=annotated_idx, encoding=pose_encoding, min_distances=None)

        for _ in tqdm(range(core_set_budget)):
            if len(annotated_idx) == 0:  # Initial choice of point
                # Initialize center with a randomly selected datapoint
                ind = np.random.choice(np.arange(pose_encoding.shape[0]))
            else:
                ind = np.argmax(min_distances)

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=pose_encoding, min_distances=min_distances)

            annotated_idx = np.concatenate([annotated_idx, [ind]], axis=0).astype(np.int32)

        selection = annotated_idx
        self._uniquecounts(dataset=train, selection=selection, method='coreset')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def learning_loss_sampling(self, train, dataset_size):
        """

        :param train:
        :param dataset_size:
        :param hg_depth:
        :return:
        """
        logging.info('Performing learning loss sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "Learning loss requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        dataset_ = ActiveLearningDataset(train, indices=unlabelled_idx, conf=self.conf)
        learnloss_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        learnloss_pred = None

        # Prediction and concatenation of the learning loss network outputs
        with torch.no_grad():
            for images in tqdm(learnloss_dataloader):

                _, pose_features = self.pose_model(images)

                learnloss_pred_ = self._aux_net_inference(pose_features)
                learnloss_pred_ = learnloss_pred_.squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()

        # argsort defaults to ascending
        pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        indices = pred_with_index[-self.num_images:, 1]

        selection = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='learning_loss')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def expected_gradient_length_sampling(self, train, dataset_size):
        '''

        :return:
        '''

        def probability(pair_dist):
            '''
            Computes P(j|i) using Binary Search
            :param pairwise_dist: (2D Tensor) pairwise distances between samples --> actual dist, not squared
            :return: 2D Tensor containing conditional probabilities
            '''

            def calc_probs_perp(lower_bound, upper_bound, pair_dist):
                sigmas = (lower_bound + upper_bound) / 2
                variance = (sigmas ** 2).reshape(-1, 1)
                scaled_pair_dist_neg = -pair_dist / (2 * variance)
                probs_unnormalized = torch.exp(scaled_pair_dist_neg)
                probs_unnormalized = torch.clamp(probs_unnormalized, min=1e-20, max=1.)

                softmax = probs_unnormalized / torch.sum(probs_unnormalized, dim=1, keepdim=True)
                softmax = torch.clamp(softmax, min=1e-30, max=1.)

                entropy = shannon_entropy(softmax)
                perplexity_hat = torch.pow(2 * torch.ones(n_samples), entropy)

                return perplexity_hat, softmax

            def condition(perplexity_hat, perplexity):
                mask = torch.lt(torch.abs(perplexity_hat - perplexity), TOLERANCE)
                return False in mask

            global PERPLEXITY, TOLERANCE, n_samples

            tries = 100
            n_samples = pair_dist.shape[0]
            PERPLEXITY = self.conf.active_learning['egl']['perplexity']
            TOLERANCE = self.conf.active_learning['egl']['tolerance'] * torch.ones(n_samples)


            pair_dist = pair_dist ** 2
            lower = torch.zeros(n_samples)
            upper = (torch.max(torch.max(pair_dist), torch.max(pair_dist**0.5))) * torch.ones(n_samples) * 5
            perplexity = PERPLEXITY * torch.ones(n_samples)

            perplexity_hat, probs = calc_probs_perp(lower, upper, pair_dist)

            while condition(perplexity_hat, perplexity):
                if tries < 0:
                    break
                tries -= 1

                mask_gt = torch.gt(perplexity_hat - perplexity, TOLERANCE).type(torch.float32)
                upper_update = upper - torch.mul(mask_gt, (upper - lower) / 2)

                mask_lt = torch.lt(perplexity_hat - perplexity, -TOLERANCE).type(torch.float32)
                lower_update = lower + torch.mul(mask_lt, (upper - lower) / 2)

                upper = upper_update
                lower = lower_update

                perplexity_hat, probs = calc_probs_perp(lower, upper, pair_dist)

            del PERPLEXITY, TOLERANCE, n_samples
            return probs

        logging.info('Performing expected gradient length sampling.')
        # Setup --------------------------------------------------------------------------------------------------------

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Load indices of previously annotated data
        assert self.conf.model['load'], "Expected Gradient Length requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx)))

        # Part 1: Obtain embeddings and heatmaps for LABELLED data ----------------------------------------------------

        dataset_ = EGLpp_Dataset(dataset_dict=train, conf=self.conf, indices=annotated_idx)
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        logging.info('Computing heatmaps, embedding for labelled images.')

        # Disable autograd to speed up inference
        with torch.no_grad():

            pose_encoding_L = None
            pose_heatmap_L = None

            for images, og_heatmaps in tqdm(egl_dataloader):

                heatmaps, pose_features = self.pose_model(images)

                if self.conf.active_learning['egl']['og_heatmap']:
                    heatmaps = torch.stack([og_heatmaps, og_heatmaps], dim=1).to(heatmaps.device)

                try:
                    pose_encoding_L = torch.cat((pose_encoding_L, pose_features['penultimate'].cpu()), dim=0)  # GAP over the 4x4 lyr
                    pose_heatmap_L = torch.cat((pose_heatmap_L, heatmaps.cpu()), dim=0)
                except TypeError:
                    pose_encoding_L = pose_features['penultimate'].cpu()
                    pose_heatmap_L = heatmaps.cpu()

        # Part 2: Obtain embeddings ONLY for UNLABELLED data -----------------------------------------------------------

        logging.info('Computing embeddings (not heatmaps) for unlabelled data')

        dataset_ = ActiveLearningDataset(dataset_dict=train, indices=unlabelled_idx, conf=self.conf)
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        # Disable autograd to speed up inference
        with torch.no_grad():

            pose_encoding_U = None

            for images in tqdm(egl_dataloader):

                _, pose_features = self.pose_model(images)

                try:
                    pose_encoding_U = torch.cat((pose_encoding_U, pose_features['penultimate'].cpu()), dim=0)  # GAP over the 4x4 lyr
                except TypeError:
                    pose_encoding_U = pose_features['penultimate'].cpu()

        # Part 3: Compute the heatmap error between the unlabelled images and its neighbors ----------------------------
        with torch.no_grad():
            pair_dist = torch.cdist(pose_encoding_U, pose_encoding_L, p=2) # Unlabelled[i] to Labelled[j]
            p_i_given_j = probability(pair_dist)

            k = self.conf.active_learning['egl']['k']
            assert len(p_i_given_j.shape) == 2, "Not a 2-dimensional tensor"
            vals, idx = torch.topk(p_i_given_j, k=k, dim=1, sorted=True, largest=True)

            logging.info('Computing the gradient between the unlabelled and labelled images.')
            pose_gradients_nbrs = torch.zeros(size=(unlabelled_idx.shape[0], k), dtype=torch.float32).to(vals.device)
            assert vals.shape == pose_gradients_nbrs.shape

        autograd_hacks.add_hooks(self.pose_model)

        dataset_ = ActiveLearningDataset(dataset_dict=train, indices=unlabelled_idx, conf=self.conf)
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)
        i_unlabelled = 0

        # Obtain images in batches:
        for unlabelled_images in tqdm(egl_dataloader):
            # Iterate over each unlabelled image
            for i_ in range(unlabelled_images.shape[0]):
                self.pose_model.zero_grad()
                i_unlabelled_copies = torch.cat(k * [unlabelled_images[i_].unsqueeze(0)], dim=0)#.cuda()
                i_heatmaps, _ = self.pose_model(i_unlabelled_copies)

                loss = heatmap_loss(i_heatmaps, pose_heatmap_L[idx[i_unlabelled]], egl=True).mean()
                loss.backward()

                autograd_hacks.compute_grad1(model=self.pose_model, loss_type='mean')

                with torch.no_grad():
                    grads = torch.zeros((i_heatmaps.shape[0],), dtype=torch.float32)
                    for param in self.pose_model.parameters():
                        try:
                            # Sum of squared gradients for each batch element
                            grads = grads.to(param.grad1.device)
                            grads += (param.grad1 ** 2).sum(dim=list(range(len(param.grad1.shape)))[1:])

                        except AttributeError:
                            continue

                    pose_gradients_nbrs[i_unlabelled] = grads.to(pose_gradients_nbrs.device)

                # Removing gradients due to previous image
                self.pose_model.zero_grad()
                autograd_hacks.clear_backprops(self.pose_model)

                i_unlabelled += 1

        autograd_hacks.remove_hooks(self.pose_model)

        egl = (vals * pose_gradients_nbrs).sum(dim=1).squeeze()
        vals, idx = torch.topk(egl, k=self.num_images, sorted=False, largest=True)
        assert idx.dim() == 1, "'idx' should be a single dimensional array"

        selection = np.concatenate([annotated_idx, unlabelled_idx[idx.cpu().numpy()]], axis=0).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='egl')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def multipeak_entropy(self, train, dataset_size):
        """

        :param train:
        :param dataset_size:
        :return:
        """

        logging.info('Performing multi-peak entropy sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "Multipeak entropy was called without a pretrained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx)))

        # Multi-peak entropy only over the unlabelled set of images
        dataset_ = ActiveLearningDataset(dataset_dict=train, indices=unlabelled_idx, conf=self.conf)
        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        pose_heatmaps = None

        # Part 1: Obtain set of heatmaps
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(mpe_dataloader):

                pose_heatmaps_, _ = self.pose_model(images)

                try:
                    pose_heatmaps = torch.cat((pose_heatmaps, pose_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                except TypeError:
                    pose_heatmaps = pose_heatmaps_[:, -1, :, :, :].cpu()

        pose_heatmaps = pose_heatmaps.squeeze().numpy()
        logging.info('Heatmaps computed. Calculating multi-peak entropy')

        # Part 2: Multi-peak entropy
        mpe_budget = self.num_images
        mpe_value_per_img = np.zeros(pose_heatmaps.shape[0], dtype=np.float32)

        # e.g. shape of heatmap final is BS x 14 x 64 x 64
        for i in tqdm(range(pose_heatmaps.shape[0])):
            normalizer = 0
            entropy = 0
            for hm in range(pose_heatmaps.shape[1]):
                loc = peak_local_max(pose_heatmaps[i, hm], min_distance=5, num_peaks=5)
                peaks = pose_heatmaps[i, hm][loc[:, 0], loc[:, 1]]

                if peaks.shape[0] > 0:
                    normalizer += 1
                    peaks = softmax_fn(peaks)
                    entropy += entropy_fn(peaks)

            mpe_value_per_img[i] = entropy

        mpe_value_per_img = torch.from_numpy(mpe_value_per_img)
        vals, idx = torch.topk(mpe_value_per_img, k=mpe_budget, sorted=False, largest=True)
        assert idx.dim() == 1, "'idx' should be a single dimensional array"
        annotated_idx = np.concatenate([annotated_idx, unlabelled_idx[idx.numpy()]], axis=0).astype(np.int32)

        selection = annotated_idx
        self._uniquecounts(dataset=train, selection=selection, method='multipeak_entropy')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def aleatoric_uncertainty(self, train, dataset_size):
        """

        :param train:
        :param dataset_size:
        :param hg_depth:
        :return:
        """
        logging.info('Performing Uncertainty: Kendall and Gal sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "Aleatoric uncertainty requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        dataset_ = ActiveLearningDataset(train, indices=unlabelled_idx, conf=self.conf)
        aleatoric_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        aleatoric_pred = None

        # Prediction and concatenation of the aleatoric predictions
        with torch.no_grad():
            for images in tqdm(aleatoric_dataloader):

                _, pose_features = self.pose_model(images)

                aleatoric_pred_ = self._aux_net_inference(pose_features)
                aleatoric_pred_ = aleatoric_pred_.squeeze()

                try:
                    aleatoric_pred = torch.cat([aleatoric_pred, aleatoric_pred_.cpu()], dim=0)
                except TypeError:
                    aleatoric_pred = aleatoric_pred_.cpu()

        aleatoric_pred = aleatoric_pred.mean(dim=-1)
        # argsort defaults to ascending
        pred_with_index = np.concatenate([aleatoric_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        indices = pred_with_index[-self.num_images:, 1]

        selection = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='aleatoric')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection


    def vl4pose(self, train, dataset_size):
        """

        :param train:
        :param dataset_size:
        :param hg_depth:
        :return:
        """

        logging.info('Performing VL4Pose sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "VL4Pose requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        j2i = {'head': 0, 'neck': 1, 'lsho': 2, 'lelb': 3, 'lwri': 4, 'rsho': 5, 'relb': 6, 'rwri': 7, 'lhip': 8,
               'lknee': 9, 'lankl': 10, 'rhip': 11, 'rknee': 12, 'rankl': 13}
        
        i2j = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
               8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}
        
        if self.conf.dataset['load'] == 'mpii' or self.conf.dataset['load'] == 'merged':
            j2i['pelvis'] = 14
            j2i['thorax'] = 15

            i2j[14] = 'pelvis'
            i2j[15] = 'thorax'

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


        dataset_ = ActiveLearningDataset(train, indices=unlabelled_idx, conf=self.conf)
        vl4pose_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        pose_heatmaps = None
        likelihood_params = None

        # Part 1: Obtain set of heatmaps
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(vl4pose_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)
                likelihood_pred_ = self._aux_net_inference(pose_features_)

                try:
                    pose_heatmaps = torch.cat((pose_heatmaps, pose_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                    likelihood_params = torch.cat([likelihood_params, likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)], dim=0)
                except TypeError:
                    pose_heatmaps = pose_heatmaps_[:, -1, :, :, :].cpu()
                    likelihood_params = likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)

        pose_heatmaps = pose_heatmaps.squeeze().numpy()
        likelihood_params = likelihood_params.numpy()

        del vl4pose_dataloader
        
        logging.info('Heatmaps computed. Calculating likelihood of pose.')

        keypoint_compute = Keypoint_ParallelWrapper(hm=pose_heatmaps, param=likelihood_params, j2i=j2i, i2j=i2j,
                                                    links=links, vl4pose_config=self.conf.active_learning['vl4pose'])
        vl4pose_dataloader = torch.utils.data.DataLoader(keypoint_compute, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        max_likelihood = np.zeros(shape=pose_heatmaps.shape[0])
        ptr = 0

        for likelihoods, trace in tqdm(vl4pose_dataloader):
            max_likelihood[ptr: ptr + likelihoods.shape[0]] = likelihoods.squeeze().numpy()
            ptr += likelihoods.shape[0]


        loglikelihood_with_index = np.concatenate([max_likelihood.reshape(-1, 1),
                                                   unlabelled_idx.reshape(-1, 1)], axis=-1)

        loglikelihood_with_index = loglikelihood_with_index[loglikelihood_with_index[:, 0].argsort()]

        # Select the images with the lowest likelihood
        indices = loglikelihood_with_index[:self.num_images, 1]

        selection = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        self._uniquecounts(dataset=train, selection=selection, method='vl4pose')

        np.save(file=os.path.join(self.conf.model['save_path'], 'model_checkpoints/annotation.npy'), arr=selection)

        return selection



    def _uniquecounts(self, dataset, selection, method):
        """

        :param dataset:
        :param selection:
        :return:
        """
        # ['dataset'] is the dataset name such as mpii, lsp, lspet for an image
        unique, counts = np.unique(dataset['dataset'][selection], return_counts=True)
        proportion = {key: value for (key, value) in zip(unique, counts)}
        with open(os.path.join(self.conf.model['save_path'], 'model_checkpoints/sampling_proportion.txt'), "x") as file:
            file.write('{} Sampling\n'.format(method))
            [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]


    def _aux_net_inference(self, pose_features):
        extractor = self.conf.architecture['aux_net']['conv_or_avg_pooling']

        with torch.no_grad():
            if extractor == 'avg':
                # Transfer to GPU where auxiliary network is stored
                encodings = pose_features['penultimate']

            else:
                depth = len(self.conf.architecture['aux_net']['spatial_dim'])
                encodings = torch.cat(
                    [pose_features['feature_{}'.format(i)].reshape(
                        pose_features['feature_{}'.format(i)].shape[0], pose_features['feature_{}'.format(i)].shape[1], -1)
                        for i in range(depth, 0, -1)],
                    dim=2)

        aux_out = self.aux_net(encodings)
        return aux_out


class ActiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, indices, conf):
        '''

        :param dataset_dict:
        '''
        self.names = dataset_dict['name'][indices]
        self.bounding_box = dataset_dict['bbox_coords'][indices]
        self.dataset = dataset_dict['dataset'][indices]

        self.xy_to_uv = lambda xy: (xy[1], xy[0])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        '''

        :param item:
        :return:
        '''

        root = Path(os.getcwd()).parent
        mpii_path = os.path.join(root, 'data', 'mpii')
        lsp_path = os.path.join(root, 'data', 'lsp')
        lspet_path = os.path.join(root, 'data', 'lspet')

        bounding_box = self.bounding_box[item]
        dataset = self.dataset[item]

        name = self.names[item]
        if dataset == 'mpii':
            image = plt.imread(os.path.join(mpii_path, 'images', '{}.jpg'.format(name.split('_')[0])))
        elif dataset == 'lsp':
            image = plt.imread(os.path.join(lsp_path, 'images', name))
        else:
            image = plt.imread(os.path.join(lspet_path, 'images', name))

        # Determine crop
        img_shape = np.array(image.shape)

        # Bounding box for the first person
        [min_x, min_y, max_x, max_y] = bounding_box[0]

        tl_uv = self.xy_to_uv(np.array([min_x, min_y]))
        br_uv = self.xy_to_uv(np.array([max_x, max_y]))
        min_u = tl_uv[0]
        min_v = tl_uv[1]
        max_u = br_uv[0]
        max_v = br_uv[1]

        centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        height = max_u - min_u
        width = max_v - min_v

        scale = 1.75

        top_left = np.array([centre[0] - (scale * height / 2), centre[1] - (scale * width / 2)])
        bottom_right = np.array([centre[0] + (scale * height / 2), centre[1] + (scale * width / 2)])

        top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # Cropping the image
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

        # Resize the image
        image = self.resize_image(image, target_size=[256, 256, 3])

        return torch.tensor(data=image / 256.0, dtype=torch.float32, device='cpu')

    def resize_image(self, image_=None, target_size=None):
        '''

        :return:
        '''
        # Compute the aspect ratios
        image_aspect_ratio = image_.shape[0] / image_.shape[1]
        tgt_aspect_ratio = target_size[0] / target_size[1]

        # Compare the original and target aspect ratio
        if image_aspect_ratio > tgt_aspect_ratio:
            # If target aspect ratio is smaller, scale the first dim
            scale_factor = target_size[0] / image_.shape[0]
        else:
            # If target aspect ratio is bigger or equal, scale the second dim
            scale_factor = target_size[1] / image_.shape[1]

        # Compute the padding to fit the target size
        pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
        pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

        output_img = np.zeros(target_size, dtype=image_.dtype)

        # Write scaled size in reverse order because opencv resize
        scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

        padding_u = int(pad_u / 2)
        padding_v = int(pad_v / 2)

        im_scaled = cv2.resize(image_, scaled_size)
        # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

        output_img[padding_u : im_scaled.shape[0] + padding_u,
                   padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

        return output_img


class EGLpp_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, conf, indices=None):
        """

        :param dataset_dict:
        :param conf:
        :param indices:
        """

        self.names = dataset_dict['name'][indices]

        self.bounding_box = dataset_dict['bbox_coords'][indices]
        self.dataset = dataset_dict['dataset'][indices]
        self.gt = dataset_dict['gt'][indices]

        self.occlusion = conf.experiment_settings['occlusion']
        self.hm_shape = [64, 64]
        self.hm_peak = conf.experiment_settings['hm_peak']

        self.xy_to_uv = lambda xy: (xy[1], xy[0])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        '''

        :param item:
        :return:
        '''
        root = Path(os.getcwd()).parent
        mpii_path = os.path.join(root, 'data', 'mpii')
        lsp_path = os.path.join(root, 'data', 'lsp')
        lspet_path = os.path.join(root, 'data', 'lspet')
        dataset = self.dataset[item]

        name = self.names[item]
        if dataset == 'mpii':
            image = plt.imread(os.path.join(mpii_path, 'images', '{}.jpg'.format(name.split('_')[0])))
        elif dataset == 'lsp':
            image = plt.imread(os.path.join(lsp_path, 'images', name))
        else:
            image = plt.imread(os.path.join(lspet_path, 'images', name))

        bounding_box = self.bounding_box[item]
        gt = self.gt[item]

        # Determine crop
        img_shape = np.array(image.shape)

        # Bounding box for the first person
        [min_x, min_y, max_x, max_y] = bounding_box[0]

        tl_uv = self.xy_to_uv(np.array([min_x, min_y]))
        br_uv = self.xy_to_uv(np.array([max_x, max_y]))
        min_u = tl_uv[0]
        min_v = tl_uv[1]
        max_u = br_uv[0]
        max_v = br_uv[1]

        centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        height = max_u - min_u
        width = max_v - min_v

        scale = 2.0

        window = max(scale * height, scale * width)
        top_left = np.array([centre[0] - (window / 2), centre[1] - (window / 2)])
        bottom_right = np.array([centre[0] + (window / 2), centre[1] + (window / 2)])

        top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # Cropping the image and adjusting the ground truth
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        for person in range(gt.shape[0]):
            for joint in range(gt.shape[1]):
                gt_uv = self.xy_to_uv(gt[person][joint])
                gt_uv = gt_uv - top_left
                gt[person][joint] = np.concatenate([gt_uv, np.array([gt[person][joint][2]])], axis=0)

        # Resize the image
        image, gt = self.resize_image(image, gt, target_size=[256, 256, 3])

        heatmaps, joint_exist = heatmap_generator(
            joints=np.copy(gt), occlusion=self.occlusion, hm_shape=self.hm_shape, img_shape=image.shape)

        heatmaps = self.hm_peak * heatmaps

        return torch.tensor(data=image / 256.0, dtype=torch.float32, device='cpu'),\
               torch.tensor(data=heatmaps, dtype=torch.float32, device='cpu')

    def resize_image(self, image_=None, gt=None, target_size=None):
        '''

        :return:
        '''
        # Compute the aspect ratios
        image_aspect_ratio = image_.shape[0] / image_.shape[1]
        tgt_aspect_ratio = target_size[0] / target_size[1]

        # Compare the original and target aspect ratio
        if image_aspect_ratio > tgt_aspect_ratio:
            # If target aspect ratio is smaller, scale the first dim
            scale_factor = target_size[0] / image_.shape[0]
        else:
            # If target aspect ratio is bigger or equal, scale the second dim
            scale_factor = target_size[1] / image_.shape[1]

        # Compute the padding to fit the target size
        pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
        pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

        output_img = np.zeros(target_size, dtype=image_.dtype)

        # Write scaled size in reverse order because opencv resize
        scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

        padding_u = int(pad_u / 2)
        padding_v = int(pad_v / 2)

        im_scaled = cv2.resize(image_, scaled_size)
        # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

        output_img[padding_u : im_scaled.shape[0] + padding_u,
                   padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

        gt *= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)
        gt[:, :, 0] += padding_u
        gt[:, :, 1] += padding_v

        return output_img, gt


class Keypoint_ParallelWrapper(torch.utils.data.Dataset):
    def __init__(self, hm, param, j2i, i2j, links, vl4pose_config):
        self.hm = hm
        self.param = param
        self.j2i = j2i
        self.i2j = i2j
        self.links = links
        self.config = vl4pose_config

    def __len__(self):
        return self.hm.shape[0]

    def __getitem__(self, i):
        joints = {}

        heatmaps = self.hm[i]
        parameters = self.param[i]

        # Initialize keypoints for each node
        for key in self.j2i.keys():

            heatmap = heatmaps[self.j2i[key]]
            loc = peak_local_max(heatmap, min_distance=self.config['min_distance'], num_peaks=self.config['num_peaks'])
            peaks = heatmap[loc[:, 0], loc[:, 1]]
            peaks = softmax_fn(peaks)
            joints[key] = Keypoint(name=key, loc=loc, peaks=peaks)

        # Initialize parent-child relations
        for k, l in enumerate(self.links):

            joints[self.i2j[l[0]]].parameters.append(parameters[k])
            joints[self.i2j[l[0]]].children.append(joints[self.i2j[l[1]]])

        max_ll, trace = joints['head'].run_likelihood()
        return max_ll, trace



class Keypoint(object):
    def __init__(self, name, loc, peaks):
        self.name = name
        self.loc = loc
        self.peaks = peaks
        self.children = []
        self.parameters = []

    def run_likelihood(self):
        """

        :return:
        """
        assert self.name == 'head'

        likelihood_per_location = []
        per_location_trace = []

        for location in range(self.loc.shape[0]):
            log_ll = np.log(self.peaks[location])

            per_child_trace = []
            for child in range(len(self.children)):
                child_ll, joint_trace = self.children[child].compute_likelihood_given_parent(self.loc[location], self.parameters[child])
                log_ll += child_ll
                per_child_trace.append(joint_trace)

            likelihood_per_location.append(log_ll)
            per_location_trace.append(per_child_trace)

        likelihood_per_location = np.array(likelihood_per_location)

        return_trace = {}
        for child_trace in per_location_trace[np.argmax(likelihood_per_location)]:
            return_trace.update(child_trace)

        return_trace[self.name] = np.argmax(likelihood_per_location)
        return_trace['{}_uv'.format(self.name)] = self.loc[np.argmax(likelihood_per_location)]
        return np.sum(likelihood_per_location), return_trace


    def compute_likelihood_given_parent(self, parent_location, gaussian_params):
        """

        :param parent_location:
        :param gaussian_params:
        :return:
        """

        likelihood_per_location = []
        per_location_trace = []

        for location in range(self.loc.shape[0]):
            log_ll = np.log(2 * np.pi) + gaussian_params[1]
            log_ll += (gaussian_params[0] - np.linalg.norm(parent_location - self.loc[location]))**2 * np.exp(-gaussian_params[1])
            log_ll *= -0.5
            log_ll += np.log(self.peaks[location])

            if len(self.children) == 0:
                likelihood_per_location.append(log_ll)

            else:
                per_child_trace = []
                for child in range(len(self.children)):
                    child_ll, joint_trace = self.children[child].compute_likelihood_given_parent(self.loc[location], self.parameters[child])
                    log_ll += child_ll
                    per_child_trace.append(joint_trace)

                likelihood_per_location.append(log_ll)
                per_location_trace.append(per_child_trace)

        likelihood_per_location = np.array(likelihood_per_location)

        if len(self.children) == 0:
            return np.sum(likelihood_per_location), {self.name: np.argmax(likelihood_per_location),
                                                         '{}_uv'.format(self.name): self.loc[np.argmax(likelihood_per_location)]}

        return_trace = {}
        for child_trace in per_location_trace[np.argmax(likelihood_per_location)]:
            return_trace.update(child_trace)

        return_trace[self.name] = np.argmax(likelihood_per_location)
        return_trace['{}_uv'.format(self.name)] = self.loc[np.argmax(likelihood_per_location)]
        return np.sum(likelihood_per_location), return_trace