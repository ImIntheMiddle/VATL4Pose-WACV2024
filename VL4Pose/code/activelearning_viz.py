import os
import cv2
import math
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
import copy
# EGL sampling
import autograd_hacks
from utils import heatmap_loss
from utils import shannon_entropy
from utils import heatmap_generator
from utils import uv_from_heatmap
from utils import visualize_image

plt.style.use('ggplot')

class ActiveLearning_Visualization(object):
    """
    Contains collection of active learning algorithms for human joint localization
    """

    def __init__(self, conf, pose_net, aux_net):
        self.conf = conf
        self.pose_model = pose_net
        self.aux_net = aux_net

        self.pose_model.eval()
        if conf.active_learning['algorithm'] in ['learning_loss', 'aleatoric', 'vl4pose']:
            self.aux_net.eval()

        self.j2i = {'head': 0, 'neck': 1, 'lsho': 2, 'lelb': 3, 'lwri': 4, 'rsho': 5, 'relb': 6, 'rwri': 7, 'lhip': 8,
                    'lknee': 9, 'lankl': 10, 'rhip': 11, 'rknee': 12, 'rankl': 13}

        self.i2j = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
                    8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}

        # update j2i, i2j with new joints
        if conf.dataset['load'] == 'mpii' or conf.dataset['load'] == 'merged':
            self.j2i['pelvis'] = 14
            self.j2i['thorax'] = 15

            self.i2j[14] = 'pelvis'
            self.i2j[15] = 'thorax'

        # I don't know why I have this assertion, need to check
        assert self.conf.dataset['load'] != 'merged'



    def base(self, train, dataset_size):
        """
        Do no visualization
        """
        raise Exception('Base method cannot be visualized')


    def random(self, train, dataset_size):
        """
        Do no visualization
        """
        raise Exception('Random method cannot be visualized')


    def coreset_sampling(self, train, dataset_size):
        """
        Sener and Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
        ICLR 2018
        https://arxiv.org/abs/1708.00489
        """

        logging.info('\nVisualizing Core-Set.')

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
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        assert np.all(train['index'] == np.arange(train['name'].shape[0]))

        dataset_ = ActiveLearningVizDataLoader(train, indices=np.arange(train['name'].shape[0]), conf=self.conf)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        ################################################################################################################
        # Part 1: Obtain Pose Embeddings
        ################################################################################################################
        
        pose_encoding = None

        logging.info('\nComputing Core-Set embeddings.')
        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(coreset_dataloader):

                pose_heatmaps_, pose_features = self.pose_model(images)
                try:
                    pose_encoding = torch.cat((pose_encoding, pose_features['penultimate'].cpu()), dim=0)
                except TypeError:
                    pose_encoding = pose_features['penultimate'].cpu()
                    
        pose_encoding = pose_encoding.squeeze().numpy()
        logging.info('Core-Set embeddings computed.\n')

        ################################################################################################################
        # Part 2: K-Centre Greedy
        ################################################################################################################
        
        logging.info('\nComputing k-Centre Greedy')
        core_set_budget = 15 # Hardcoded, we want to see the first 15 diverse samples
        min_distances = None 

        assert len(annotated_idx) != 0, "No annotations for previous model found, core-set cannot proceeed."
        min_distances = update_distances(cluster_centers=annotated_idx, encoding=pose_encoding, min_distances=None)

        display_idx = []
        distances_over_time = []

        for _ in tqdm(range(unlabelled_idx.shape[0])):
            ind = np.argmax(min_distances)
            distances_over_time.append(np.max(min_distances))

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=pose_encoding, min_distances=min_distances)

            annotated_idx = np.concatenate([annotated_idx, [ind]], axis=0).astype(np.int32)
            display_idx.append(ind)

        logging.info('Computed k-Centre Greedy.\n')
        del pose_encoding

        ################################################################################################################
        # Part 3: Plot max distances over time
        ################################################################################################################

        plt.plot(np.arange(unlabelled_idx.shape[0]), distances_over_time, label='Maximum distance')
        plt.title('Core-Set: Maximum Distances over n-selections')
        os.makedirs(os.path.join(self.conf.model['save_path'], 'coreset_visualizations'), exist_ok=True)
        plt.savefig(fname=os.path.join(self.conf.model['save_path'], 'coreset_visualizations/Distances_CoreSet.jpg'), 
                    facecolor='black', edgecolor='black', bbox_inches='tight', dpi=300)
        plt.close()
        
        ################################################################################################################
        # Part 4: Visualize Core-Set images
        ################################################################################################################
        
        dataset_ = ActiveLearningVizDataLoader(train, indices=display_idx[:core_set_budget], conf=self.conf)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []
        logging.info('Obtaining images and data for selected samples.')
        
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(coreset_dataloader):
                
                pose_heatmaps_, _ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()

        logging.info('Images loaded.\n')
        
        logging.info('Visualizing images (GT and Prediction)')
        self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all,
                                    gt=gts_all, pred=hm_uv_stack, string=names_all)


    def learning_loss_sampling(self, train, dataset_size):
        """
        Yoo and Kweon, "Learning Loss for Active Learning"
        CVPR 2019
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf

        Shukla and Ahmed, "A Mathematical Analysis of Learning Loss for Active Learning in Regression"
        CVPR-W 2021
        https://openaccess.thecvf.com/content/CVPR2021W/TCV/papers/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.pdf
        """

        logging.info('Visualizing learning loss.')

        assert self.conf.model['load'], "Learning loss requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx, conf=self.conf)
        learnloss_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        ################################################################################################################
        # Part 1: Obtain Learning Loss predictions
        ################################################################################################################
        
        learnloss_pred = None
        
        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(learnloss_dataloader):
                pose_heatmaps_, pose_features_ = self.pose_model(images)

                learnloss_pred_ = self._aux_net_inference(pose_features_).squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()
        
        learnloss_pred = learnloss_pred.squeeze().numpy()

        # argsort defaults to ascending
        new_index = np.arange(learnloss_pred.shape[0]).reshape(-1, 1)
        learnloss_with_index = np.concatenate([learnloss_pred.reshape(-1, 1),
                                               new_index], axis=-1)

        learnloss_with_index = learnloss_with_index[learnloss_with_index[:, 0].argsort()]

        # Slice aleatoric for top-5 and bottom-5 images
        min_learnloss_idx = learnloss_with_index[:15, 1].astype(np.int32)
        max_learnloss_idx = learnloss_with_index[-15:, 1].astype(np.int32)

        ################################################################################################################
        # Part 2: Visualize images with low loss
        ################################################################################################################
        
        idx = min_learnloss_idx

        # Part 2.a: Loading images
        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx[idx], conf=self.conf)
        learnloss_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []
        
        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(learnloss_dataloader):
                pose_heatmaps_, pose_features_ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])
                
                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()

        names_modified = []
        for i in range(15):
            temp = 'LearningLoss_{}_andName_{}'.format(learnloss_pred[min_learnloss_idx[i]], names_all[i])
            names_modified.append(temp)

        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)

        ################################################################################################################
        # Part 3: Visualize images with high loss
        ################################################################################################################

        idx = max_learnloss_idx

        # Part 3.a: Loading images
        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx[idx], conf=self.conf)
        learnloss_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []
        
        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(learnloss_dataloader):
                pose_heatmaps_, pose_features_ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])
                
                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()

        names_modified = []
        for i in range(15):
            temp = 'LearningLoss_{}_andName_{}'.format(learnloss_pred[max_learnloss_idx[i]], names_all[i])
            names_modified.append(temp)

        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)


    def expected_gradient_length_sampling(self, train, dataset_size):
        """
        Megh Shukla, "Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?"
        WACV 2022
        https://openaccess.thecvf.com/content/WACV2022/papers/Shukla_Bayesian_Uncertainty_and_Expected_Gradient_Length_-_Regression_Two_Sides_WACV_2022_paper.pdf

        """

        def _probability(pair_dist):
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

        logging.info('VISUALIZING expected gradient length sampling.')
        # Setup --------------------------------------------------------------------------------------------------------

        # Load indices of previously annotated data
        assert self.conf.model['load'], "Expected Gradient Length requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx)))

        ################################################################################################################
        # Part 1: Obtain embeddings for labelled data
        ################################################################################################################

        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=annotated_idx)
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        logging.info('Computing embeddings for labelled images.')

        # Disable autograd to speed up inference
        with torch.no_grad():

            pose_encoding_L = None

            for images, _, names, gts, datasets in tqdm(egl_dataloader):

                _, pose_features_ = self.pose_model(images)

                try:
                    pose_encoding_L = torch.cat((pose_encoding_L, pose_features_['penultimate'].cpu()), dim=0)

                except TypeError:
                    pose_encoding_L = pose_features_['penultimate'].cpu()


        ################################################################################################################
        # Part 2: Obtain embeddings for unlabelled data
        ################################################################################################################

        logging.info('Computing embeddings for unlabelled data')

        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=unlabelled_idx)
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        # Disable autograd to speed up inference
        with torch.no_grad():

            pose_encoding_U = None
            names_U = []
            gts_U = None
            hm_uv_stack_U = []
            datasets_U = []

            for images, _, names, gts, datasets in tqdm(egl_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack_U.append(hm_uv)

                    names_U.append(names[i])
                    datasets_U.append(datasets[i])

                try:
                    pose_encoding_U = torch.cat((pose_encoding_U, pose_features_['penultimate'].cpu()), dim=0)  # GAP over the 4x4 lyr
                    gts_U = torch.cat([gts_U, gts], dim=0)

                except TypeError:
                    pose_encoding_U = pose_features_['penultimate'].cpu()
                    gts_U = gts


            hm_uv_stack_U = np.stack(hm_uv_stack_U, axis=0)
            names_U = np.array(names_U)
            datasets_U = np.array(datasets_U)
            gts_U = gts_U.numpy()

        ################################################################################################################
        # Part 3: Compute pairwise distances
        ################################################################################################################
        
        logging.info('Computing pairwise probabilities data')
        with torch.no_grad():
            pair_dist = torch.cdist(pose_encoding_U, pose_encoding_L, p=2) # Unlabelled[i] to Labelled[j]
            p_i_given_j = _probability(pair_dist)

            k = self.conf.active_learning['egl']['k']
            assert len(p_i_given_j.shape) == 2, "Not a 2-dimensional tensor"
            vals_L, idx_L = torch.topk(p_i_given_j, k=k, dim=1, sorted=True, largest=True)


        del pose_encoding_L, pose_encoding_U, pair_dist, p_i_given_j

        ################################################################################################################
        # Part 4: Compute expected gradient length
        ################################################################################################################

        logging.info('Computing the gradient between the unlabelled and labelled images.')
        pose_gradients_nbrs = torch.zeros(size=(unlabelled_idx.shape[0], k), dtype=torch.float32).to(vals_L.device)
        assert vals_L.shape == pose_gradients_nbrs.shape
        
        autograd_hacks.add_hooks(self.pose_model)
        unlabelled_dataset = ActiveLearningVizDataLoader(dataset_dict=train, indices=unlabelled_idx, conf=self.conf)

        for i in tqdm(range(len(unlabelled_dataset))):
            neighbors_ = ActiveLearningVizDataLoader(dataset_dict=train, indices=annotated_idx[idx_L[i]], conf=self.conf)
            neighbors_dataloader = torch.utils.data.DataLoader(neighbors_,
                                                               batch_size=self.conf.experiment_settings['batch_size'],
                                                               shuffle=False,
                                                               num_workers=2)

            # Keep on GPU, collect heatmaps for the neighbours
            with torch.no_grad():
                hm_L = None
                for images_l, _, _, _, _ in neighbors_dataloader:
                    hm_l, _ = self.pose_model(images_l)
                    try:
                        hm_L = torch.cat((hm_L, hm_l), dim=0)
                    except TypeError:
                        hm_L = hm_l
            
            # Compute gradient wrt these neighbors
            image, _, _, _, _ = unlabelled_dataset.__getitem__(i)
            images = torch.cat(k * [image.unsqueeze(0)], dim=0)
            hm_U, _ = self.pose_model(images)

            loss = heatmap_loss(hm_U, hm_L, egl=True).mean()
            loss.backward()

            autograd_hacks.compute_grad1(model=self.pose_model, loss_type='mean')

            with torch.no_grad():
                grads = torch.zeros((k,), dtype=torch.float32)
                for param in self.pose_model.parameters():
                    try:
                        # Sum of squared gradients for each batch element
                        grads = grads.to(param.grad1.device)
                        grads += (param.grad1 ** 2).sum(dim=list(range(len(param.grad1.shape)))[1:])

                    except AttributeError:
                        continue

                pose_gradients_nbrs[i] = grads.to(pose_gradients_nbrs.device)

            # Removing gradients due to previous image
            self.pose_model.zero_grad()
            autograd_hacks.clear_backprops(self.pose_model)
        
        autograd_hacks.remove_hooks(self.pose_model)

        egl = (vals_L * pose_gradients_nbrs).sum(dim=1).squeeze()

        ################################################################################################################
        # Part 5: Visualize top-K images and their nearest neighbors
        ################################################################################################################

        vals_topK, idx_topK = torch.topk(egl, k=15, sorted=True, largest=True)

        idx_topK = idx_topK.numpy()
        vals_topK = vals_topK.numpy()
        
        # Part 5.a: topk actual imgs visualization
        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=unlabelled_idx[idx_topK])
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        logging.info('Generating top-K images')

        images_U = None

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, _, _, datasets in tqdm(egl_dataloader):
                try:
                    images_U = torch.cat([images_U, images], dim=0)

                except TypeError:
                    images_U = images

        images_U = images_U.numpy()

        name_modified = []

        for i in range(15):
            #name_ = train['name'][unlabelled_idx_topK[i]]
            name_ = names_U[idx_topK[i]]
            name_modified.append('top_{}_egl_{}_name_{}.png'.format(i + 1, vals_topK[i], name_))

        self._visualize_predictions(image=images_U, name=names_U[idx_topK],
                                    dataset=datasets_U[idx_topK], gt=gts_U[idx_topK],
                                    pred=hm_uv_stack_U[idx_topK], string=np.array(name_modified))


        # Part 5.b: topk nbrs
        logging.info('Generating neighbors for top-K images')
        images_L = None
        gts_L = None
        names_L = []
        hm_uv_stack_L = []
        datasets_L = []
        name_modified = []


        for i in tqdm(range(15)): # Top fifteen samples with highest EGL

            # Top 5 neighbors for each joint
            dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=annotated_idx[idx_L[idx_topK[i], :5]])
            egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                         shuffle=False, num_workers=2, drop_last=False)
            
            
            nbr = 0
            with torch.no_grad():
                for images, _, names, gts, datasets in egl_dataloader:

                    pose_heatmaps_, pose_features_ = self.pose_model(images)

                    for j in range(images.shape[0]):
                        hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[j])
                        hm_uv_stack_L.append(hm_uv)

                        names_L.append(names[j])
                        datasets_L.append(datasets[j])
                        name_modified.append(
                                'top_{}_nbr_{}_pji_{}_name_{}.png'.format(i + 1, nbr + j, vals_L[idx_topK[i], nbr+j], names[j]))
                    nbr += images.shape[0]

                    try:
                        images_L = torch.cat([images_L, images], dim=0)
                        gts_L = torch.cat([gts_L, gts], dim=0)

                    except TypeError:
                        images_L = images
                        gts_L = gts

                
        images_L = images_L.numpy()
        hm_uv_stack_L = np.stack(hm_uv_stack_L, axis=0)
        names_L = np.array(names_L)
        datasets_L = np.array(datasets_L)
        gts_L = gts_L.numpy()
        name_modified = np.array(name_modified)

        self._visualize_predictions(image=images_L, name=names_L, dataset=datasets_L, gt=gts_L,
                                    pred=hm_uv_stack_L, string=name_modified)

        ################################################################################################################
        # Part 6: Visualize bottom-K images and their nearest neighbors
        ################################################################################################################

        vals_topK, idx_topK = torch.topk(-egl, k=15, sorted=True, largest=True)
        
        idx_topK = idx_topK.numpy()
        vals_topK = vals_topK.numpy()

        # Generate images
        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=unlabelled_idx[idx_topK])
        egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        logging.info('Generating Bottom-K images')

        images_U = None

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, _, _, datasets in tqdm(egl_dataloader):
                try:
                    images_U = torch.cat([images_U, images], dim=0)

                except TypeError:
                    images_U = images

        images_U = images_U.numpy()

        name_modified = []

        for i in range(15):
            #name_ = train['name'][unlabelled_idx_topK[i]]
            name_ = names_U[idx_topK[i]]
            name_modified.append('bottom_{}_egl_{}_name_{}.png'.format(i + 1, vals_topK[i], name_))

        self._visualize_predictions(image=images_U, name=names_U[idx_topK],
                                    dataset=datasets_U[idx_topK], gt=gts_U[idx_topK],
                                    pred=hm_uv_stack_U[idx_topK], string=np.array(name_modified))


        # Part 5.d: bottom-K nbrs
        logging.info('Generating neighbors for bottom-K images')

        images_L = None
        gts_L = None
        names_L = []
        hm_uv_stack_L = []
        datasets_L = []
        name_modified = []


        for i in tqdm(range(15)): # Top fifteen samples with highest EGL

            # Top 5 neighbors for each joint
            dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, conf=self.conf, indices=annotated_idx[idx_L[idx_topK[i], :5]])
            egl_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                         shuffle=False, num_workers=2, drop_last=False)
            
            nbr = 0
            with torch.no_grad():
                for images, _, names, gts, datasets in egl_dataloader:

                    pose_heatmaps_, pose_features_ = self.pose_model(images)

                    for j in range(images.shape[0]):
                        hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[j])
                        hm_uv_stack_L.append(hm_uv)

                        names_L.append(names[j])
                        datasets_L.append(datasets[j])
                        name_modified.append(
                                'bottom_{}_nbr_{}_pji_{}_name_{}.png'.format(i + 1, nbr + j, vals_L[idx_topK[i], nbr+j], names[j]))
                    nbr += images.shape[0]

                    try:
                        images_L = torch.cat([images_L, images], dim=0)
                        gts_L = torch.cat([gts_L, gts], dim=0)

                    except TypeError:
                        images_L = images
                        gts_L = gts

                
        images_L = images_L.numpy()
        hm_uv_stack_L = np.stack(hm_uv_stack_L, axis=0)
        names_L = np.array(names_L)
        datasets_L = np.array(datasets_L)
        gts_L = gts_L.numpy()
        name_modified = np.array(name_modified)

        self._visualize_predictions(image=images_L, name=names_L, dataset=datasets_L, gt=gts_L,
                                    pred=hm_uv_stack_L, string=name_modified)


    def multipeak_entropy(self, train, dataset_size):
        """
        Liu and Ferrari, "Active Learning for Human Pose Estimation"
        ICCV 2017
        https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Active_Learning_for_ICCV_2017_paper.pdf

        """

        logging.info('VISUALIZING multi-peak entropy sampling.')

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        assert self.conf.model['load'], "Multipeak entropy was called without a base model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx)))

        # Multi-peak entropy only over the unlabelled set of images
        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, indices=unlabelled_idx, conf=self.conf)
        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        
        ################################################################################################################
        # Part 1: Computing entropy for unlabelled data
        ################################################################################################################
        
        mpe_value_per_img = []

        logging.info('Computing entropy.')
        
        with torch.no_grad():
            for images, _, _, _, _ in tqdm(mpe_dataloader):

                pose_heatmaps_, _ = self.pose_model(images)
                pose_heatmaps_ = pose_heatmaps_.detach().cpu().numpy()[:, -1, :, :, :]

                for i in range(pose_heatmaps_.shape[0]):
                    entropy = 0
                    normalize = 0
                    for hm in range(pose_heatmaps_.shape[1]):
                        loc = peak_local_max(pose_heatmaps_[i, hm], min_distance=5, num_peaks=5, exclude_border=False)
                        peaks = pose_heatmaps_[i, hm][loc[:, 0], loc[:, 1]]

                        if peaks.shape[0] > 0:
                            peaks = softmax_fn(peaks)
                            entropy += entropy_fn(peaks)
                            normalize += 1

                    mpe_value_per_img.append(entropy / normalize)


        ################################################################################################################
        # Part 2: Finding images with lowest and highest entropy
        ################################################################################################################
                    
        mpe_value_per_img = np.array(mpe_value_per_img)
        mpe_with_index = np.concatenate([mpe_value_per_img.reshape(-1, 1), unlabelled_idx.reshape(-1, 1)], axis=-1)
        mpe_with_index = mpe_with_index[mpe_with_index[:, 0].argsort()]
        # Slice multipeak entropy for top-15 and bottom-15 images
        min_mpe_idx = mpe_with_index[:15, 1].astype(np.int32)
        max_mpe_idx = mpe_with_index[-15:, 1].astype(np.int32)

        
        ################################################################################################################
        # Part 3: Visualizing images with lowest entropy
        ################################################################################################################

        logging.info('Visualizing samples with low entropy')
        idx = min_mpe_idx
        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, indices=idx, conf=self.conf)
        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        pose_heatmaps = None
        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, names, gts, datasets in mpe_dataloader:

                pose_heatmaps_, _ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    pose_heatmaps = torch.cat((pose_heatmaps, pose_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    pose_heatmaps = pose_heatmaps_[:, -1, :, :, :].cpu()
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()
        pose_heatmaps = pose_heatmaps.squeeze().numpy()


        for i in range(15):
            img = images_all[i]
            hm = pose_heatmaps[i]
            name = names_all[i]
            for j in range(hm.shape[0]):
                plt.imshow(img)
                plt.imshow(cv2.resize(hm[j], dsize=(256, 256), interpolation=cv2.INTER_CUBIC), alpha=.5)
                plt.title('{}'.format(self.i2j[j]))
                os.makedirs(os.path.join(self.conf.model['save_path'], 'entropy_visualizations/images_entropy/minimum'), exist_ok=True)
                plt.savefig(fname=os.path.join(self.conf.model['save_path'], 'entropy_visualizations/images_entropy/minimum/{}_{}.jpg'.format(name, self.i2j[j])),
                            facecolor='black', edgecolor='black', bbox_inches='tight', dpi=300)
                plt.close()


        names_modified = []
        for i in range(15):
            names_modified.append('MPE_{}_andName_{}'.format(mpe_with_index[i, 0], names_all[i]))
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)

        ################################################################################################################
        # Part 4: Visualizing images with highest entropy
        ################################################################################################################

        logging.info('Visualizing samples with high entropy')
        idx = max_mpe_idx
        dataset_ = ActiveLearningVizDataLoader(dataset_dict=train, indices=idx, conf=self.conf)
        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.experiment_settings['batch_size'],
                                                     shuffle=False, num_workers=2)

        pose_heatmaps = None
        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, names, gts, datasets in mpe_dataloader:

                pose_heatmaps_, _ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    pose_heatmaps = torch.cat((pose_heatmaps, pose_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    pose_heatmaps = pose_heatmaps_[:, -1, :, :, :].cpu()
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()
        pose_heatmaps = pose_heatmaps.squeeze().numpy()


        for i in range(15):
            img = images_all[i]
            hm = pose_heatmaps[i]
            name = names_all[i]
            for j in range(hm.shape[0]):
                plt.imshow(img)
                plt.imshow(cv2.resize(hm[j], dsize=(256, 256), interpolation=cv2.INTER_CUBIC), alpha=.5)
                plt.title('{}'.format(self.i2j[j]))
                os.makedirs(os.path.join(self.conf.model['save_path'], 'entropy_visualizations/images_entropy/maximum'), exist_ok=True)
                plt.savefig(fname=os.path.join(self.conf.model['save_path'], 'entropy_visualizations/images_entropy/maximum/{}_{}.jpg'.format(name, self.i2j[j])),
                            facecolor='black', edgecolor='black', bbox_inches='tight', dpi=300)
                plt.close()


        names_modified = []
        for i in range(15):
            names_modified.append('MPE_{}_andName_{}'.format(mpe_with_index[-15 + i, 0], names_all[i]))
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)
        

    def aleatoric_uncertainty(self, train, dataset_size):
        """
        Kendall and Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
        NeurIPS 2017
        https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html
        """
        logging.info('VISUALIZING Uncertainty: Kendall and Gal sampling.')

        assert self.conf.model['load'], "Aleatoric uncertainty requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx, conf=self.conf)
        aleatoric_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        ################################################################################################################
        # Part 1: Computing aleatoric uncertainty for unlabelled data
        ################################################################################################################

        aleatoric_pred = None

        # Part 1: Active Learning
        logging.info('Computing aleatoric uncertainty.')

        with torch.no_grad():
            for images, _, names, gts, datasets in tqdm(aleatoric_dataloader):
                _, pose_features_ = self.pose_model(images)

                aleatoric_pred_ = self._aux_net_inference(pose_features_)
                aleatoric_pred_ = aleatoric_pred_.squeeze()

                try:
                    aleatoric_pred = torch.cat([aleatoric_pred, aleatoric_pred_.cpu()], dim=0)

                except TypeError:
                    aleatoric_pred = aleatoric_pred_.cpu()

        aleatoric_pred_copy = aleatoric_pred.mean(dim=-1)
        
        ################################################################################################################
        # Part 2: Sort images based on aleatoric uncertainty
        ################################################################################################################

        # argsort defaults to ascending
        aleatoric_with_index = np.concatenate([aleatoric_pred_copy.numpy().reshape(-1, 1),
                                               unlabelled_idx.reshape(-1, 1)], axis=-1)

        aleatoric_with_index = aleatoric_with_index[aleatoric_with_index[:, 0].argsort()]

        # Slice aleatoric for top-5 and bottom-5 images
        min_aleatoric_idx = aleatoric_with_index[:15, 1].astype(np.int32)
        max_aleatoric_idx = aleatoric_with_index[-15:, 1].astype(np.int32)

        ################################################################################################################
        # Part 3: Visualize skeletons for samples with low aleatoric uncertainty
        ################################################################################################################

        logging.info('Visualizing samples with low aleatoric uncertainty')

        idx = min_aleatoric_idx

        dataset_ = ActiveLearningVizDataLoader(train, indices=idx, conf=self.conf)
        aleatoric_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        # Compile all together
        aleatoric_pred = None
        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, names, gts, datasets in aleatoric_dataloader:
                pose_heatmaps_, pose_features_ = self.pose_model(images)

                aleatoric_pred_ = self._aux_net_inference(pose_features_)
                aleatoric_pred_ = aleatoric_pred_.squeeze()

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    aleatoric_pred = torch.cat([aleatoric_pred, aleatoric_pred_.cpu()], dim=0)
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    aleatoric_pred = aleatoric_pred_.cpu()
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()
        aleatoric_pred_copy = aleatoric_pred.mean(dim=-1)

        names_modified = []
        for i in range(15):
            temp = 'Aleatoric_{}andName_{}'.format(aleatoric_pred_copy[i], names_all[i])
            names_modified.append(temp)
        
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all,
                                    name=names_all,
                                    dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)

        ################################################################################################################
        # Part 4: Visualize heatmaps for samples with low aleatoric uncertainty
        ################################################################################################################

        aleatoric_hm_all = []
        for i in range(15):
            aleatoric_hm = np.zeros([64, 64])
            for j in range(self.conf.experiment_settings['num_hm']):
                # after idx slicing: hm_uv_stack = 5 x 1 x 14 x 3
                pred = hm_uv_stack[i, 0, j]
                log_var = aleatoric_pred[i, j]

                # print('Mean: {}'.format(mean))
                # print('Var: {}'.format(np.exp(log_var)))

                target = pred[:2] / np.array([(256 - 1) / (64 - 1), ((256 - 1) / (64 - 1))])
                # print('Target: {}'.format(target))
                aleatoric_hm_jnt = self._draw_gaussian(target, log_var)
                aleatoric_hm = np.maximum(aleatoric_hm, aleatoric_hm_jnt)
                # print()
            # RESIZE HM
            aleatoric_hm_all.append(cv2.resize(aleatoric_hm, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))

            
        aleatoric_hm_all = np.stack(aleatoric_hm_all, axis=0)
        names_modified = []
        for i in range(15):
            names_modified.append('SkeletonViz_Name_{}'.format(names_all[i]))
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=aleatoric_hm_all,
                                    name=names_all,
                                    dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)

        ################################################################################################################
        # Part 5: Visualize skeletons for samples with high aleatoric uncertainty
        ################################################################################################################

        logging.info('Visualizing samples with high aleatoric uncertainty')

        idx = max_aleatoric_idx

        dataset_ = ActiveLearningVizDataLoader(train, indices=idx, conf=self.conf)
        aleatoric_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

        # Compile all together
        aleatoric_pred = None
        images_all = None
        gts_all = None
        names_all = []
        hm_uv_stack = []
        datasets_all = []

        # Disable autograd to speed up inference
        with torch.no_grad():
            for images, _, names, gts, datasets in aleatoric_dataloader:
                pose_heatmaps_, pose_features_ = self.pose_model(images)

                aleatoric_pred_ = self._aux_net_inference(pose_features_)
                aleatoric_pred_ = aleatoric_pred_.squeeze()

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    aleatoric_pred = torch.cat([aleatoric_pred, aleatoric_pred_.cpu()], dim=0)
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    aleatoric_pred = aleatoric_pred_.cpu()
                    images_all = images
                    gts_all = gts

        images_all = images_all.numpy()
        hm_uv_stack = np.stack(hm_uv_stack, axis=0)
        names_all = np.array(names_all)
        datasets_all = np.array(datasets_all)
        gts_all = gts_all.numpy()
        aleatoric_pred_copy = aleatoric_pred.mean(dim=-1)

        names_modified = []
        for i in range(15):
            temp = 'Aleatoric_{}andName_{}'.format(aleatoric_pred_copy[i], names_all[i])
            names_modified.append(temp)
        
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=images_all,
                                    name=names_all,
                                    dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)

        ################################################################################################################
        # Part 4: Visualize heatmaps for samples with high aleatoric uncertainty
        ################################################################################################################

        aleatoric_hm_all = []
        for i in range(15):
            aleatoric_hm = np.zeros([64, 64])
            for j in range(self.conf.experiment_settings['num_hm']):
                # after idx slicing: hm_uv_stack = 5 x 1 x 14 x 3
                pred = hm_uv_stack[i, 0, j]
                log_var = aleatoric_pred[i, j]

                # print('Mean: {}'.format(mean))
                # print('Var: {}'.format(np.exp(log_var)))

                target = pred[:2] / np.array([(256 - 1) / (64 - 1), ((256 - 1) / (64 - 1))])
                # print('Target: {}'.format(target))
                aleatoric_hm_jnt = self._draw_gaussian(target, log_var)
                aleatoric_hm = np.maximum(aleatoric_hm, aleatoric_hm_jnt)
                # print()
            # RESIZE HM
            aleatoric_hm_all.append(cv2.resize(aleatoric_hm, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))

            
        aleatoric_hm_all = np.stack(aleatoric_hm_all, axis=0)
        names_modified = []
        for i in range(15):
            names_modified.append('SkeletonViz_Name_{}'.format(names_all[i]))
        names_modified = np.array(names_modified)

        self._visualize_predictions(image=aleatoric_hm_all,
                                    name=names_all,
                                    dataset=datasets_all, gt=gts_all,
                                    pred=hm_uv_stack, string=names_modified)


    def vl4pose(self, train, dataset_size):
        """
        Shukla et al., "VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation"
        BMVC 2022
        https://bmvc2022.mpi-inf.mpg.de/610/
        """

        logging.info('Visualizing: VL4Pose Sampling.')

        assert self.conf.model['load'], "VL4Pose requires a previously trained model"
        annotated_idx = np.load(os.path.join(self.conf.model['load_path'], 'model_checkpoints/annotation.npy'))

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        # links definition
        if self.conf.dataset['load'] == 'mpii':
            links = [[self.j2i['head'], self.j2i['neck']], [self.j2i['neck'], self.j2i['thorax']], [self.j2i['thorax'], self.j2i['pelvis']],
                     [self.j2i['thorax'], self.j2i['lsho']], [self.j2i['lsho'], self.j2i['lelb']], [self.j2i['lelb'], self.j2i['lwri']],
                     [self.j2i['thorax'], self.j2i['rsho']], [self.j2i['rsho'], self.j2i['relb']], [self.j2i['relb'], self.j2i['rwri']],
                     [self.j2i['pelvis'], self.j2i['lhip']], [self.j2i['lhip'], self.j2i['lknee']], [self.j2i['lknee'], self.j2i['lankl']],
                     [self.j2i['pelvis'], self.j2i['rhip']], [self.j2i['rhip'], self.j2i['rknee']], [self.j2i['rknee'], self.j2i['rankl']]]
        else:
            links = [[self.j2i['head'], self.j2i['neck']],
                     [self.j2i['neck'], self.j2i['lsho']], [self.j2i['lsho'], self.j2i['lelb']], [self.j2i['lelb'], self.j2i['lwri']],
                     [self.j2i['neck'], self.j2i['rsho']], [self.j2i['rsho'], self.j2i['relb']], [self.j2i['relb'], self.j2i['rwri']],
                     [self.j2i['lsho'], self.j2i['lhip']], [self.j2i['lhip'], self.j2i['lknee']], [self.j2i['lknee'], self.j2i['lankl']],
                     [self.j2i['rsho'], self.j2i['rhip']], [self.j2i['rhip'], self.j2i['rknee']], [self.j2i['rknee'], self.j2i['rankl']]]

        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx, conf=self.conf)

        vl4pose_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=1)
        
        ################################################################################################################
        # Part 1: Computing images where VL4Pose has changed the pose estimator's prediction
        ################################################################################################################

        logging.info('Computing images where poses have changed.')

        with torch.no_grad():
            has_pose_changed = None
            vl4pose_refinement = None

            for images, _, names, gts, datasets in tqdm(vl4pose_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)
                likelihood_pred_ = self._aux_net_inference(pose_features_).reshape(images.shape[0], len(links), 2)

                keypoint_compute = Keypoint_ParallelWrapper(
                        hm=pose_heatmaps_[:, -1, :, :, :].cpu().numpy(), param=likelihood_pred_.cpu().numpy(), j2i=self.j2i, i2j=self.i2j,
                        links=links, vl4pose_config=self.conf.active_learning['vl4pose'], function=np.max)
                
                keypoint_dataloader = torch.utils.data.DataLoader(
                        keypoint_compute, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

                # Poses evaluated according to VL4Pose
                for likelihoods, vl4pose_refinement_, has_pose_changed_ in keypoint_dataloader:

                    try:
                        vl4pose_refinement = torch.cat((vl4pose_refinement, vl4pose_refinement_), dim=0)
                        has_pose_changed.extend(has_pose_changed_.tolist())

                    except TypeError:
                        vl4pose_refinement = vl4pose_refinement_
                        has_pose_changed = has_pose_changed_.tolist()

            vl4pose_refinement = vl4pose_refinement.numpy()
            vl4pose_refinement *= np.array([4.0476, 4.0476, 1])
        
            # Part 1.b: Collect the corresponding images
            logging.info('Collecting data where {} poses have changed.'.format(sum(has_pose_changed)))
            dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx[has_pose_changed], conf=self.conf)

            vl4pose_dataloader = torch.utils.data.DataLoader(
                dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=1)

            images_all = None
            gts_all = None
            names_all = []
            hm_uv_stack = []
            datasets_all = []

            for images, _, names, gts, datasets in tqdm(vl4pose_dataloader):

                pose_heatmaps_, _ = self.pose_model(images)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts

            images_all = images_all.numpy()
            hm_uv_stack = np.stack(hm_uv_stack, axis=0)
            names_all = np.array(names_all)
            datasets_all = np.array(datasets_all)
            gts_all = gts_all.numpy()

            # Part 1.c: For these images visualize the poses
            logging.info('Visualizing images where pose has changed.')
    
            self._visualize_predictions(image=images_all,
                                        name=names_all,
                                        dataset=datasets_all,
                                        gt=hm_uv_stack,
                                        pred=vl4pose_refinement[has_pose_changed],
                                        string=names_all)

        del vl4pose_refinement, has_pose_changed, images_all, names_all, datasets_all, hm_uv_stack

        
        ################################################################################################################
        # Part 2: Computing minimum and maximum expected likelihood 
        ################################################################################################################

        logging.info('Computing images with maximum and minimum likelihood.')
        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx, conf=self.conf)

        vl4pose_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=1)
        
        with torch.no_grad():
            max_likelihood = None
            
            for images, _, names, gts, datasets in tqdm(vl4pose_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)
                likelihood_pred_ = self._aux_net_inference(pose_features_).reshape(images.shape[0], len(links), 2)

                keypoint_compute = Keypoint_ParallelWrapper(
                        hm=pose_heatmaps_[:, -1, :, :, :].cpu().numpy(), param=likelihood_pred_.cpu().numpy(), j2i=self.j2i, i2j=self.i2j,
                        links=links, vl4pose_config=self.conf.active_learning['vl4pose'], function=np.sum)
                
                keypoint_dataloader = torch.utils.data.DataLoader(
                        keypoint_compute, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=2)

                # Poses evaluated according to VL4Pose
                for likelihoods, _, _ in keypoint_dataloader:
                    try:
                        max_likelihood = torch.cat((max_likelihood, likelihoods.squeeze()), dim=0)

                    except TypeError:
                        max_likelihood = likelihoods.squeeze()

            max_likelihood = max_likelihood.numpy()
            new_index = np.arange(max_likelihood.shape[0])
            loglikelihood_with_index = np.concatenate([max_likelihood.reshape(-1, 1), new_index.reshape(-1, 1)], axis=-1)
            loglikelihood_with_index = loglikelihood_with_index[loglikelihood_with_index[:, 0].argsort()]

            # Slice images, heatmaps, likelihood and parameters for top-5 and bottom-5 images
            min_likelihood_idx = loglikelihood_with_index[:15, 1].astype(np.int32)
            max_likelihood_idx = loglikelihood_with_index[-15:, 1].astype(np.int32)
        

        ################################################################################################################
        # Part 3: Visualizing images with minimum expected likelihood
        ################################################################################################################
            
        logging.info('Visualizing images with minimum likelihood.')
        idx = min_likelihood_idx

        # Collect images for these indices
        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx[idx], conf=self.conf)

        vl4pose_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=1)

        with torch.no_grad():
            images_all = None
            gts_all = None
            names_all = []
            hm_uv_stack = []
            datasets_all = []
            likelihood_params = None

            for images, _, names, gts, datasets in tqdm(vl4pose_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)
                likelihood_pred_ = self._aux_net_inference(pose_features_)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)
                    likelihood_params = torch.cat([likelihood_params, likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts
                    likelihood_params = likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)

            images_all = images_all.numpy()
            hm_uv_stack = np.stack(hm_uv_stack, axis=0)
            names_all = np.array(names_all)
            datasets_all = np.array(datasets_all)
            gts_all = gts_all.numpy()
            likelihood_params = likelihood_params.numpy()

            names_modified = []
            for i in range(15):
                temp = 'LogLikelihood_{}_andName_{}'.format(max_likelihood[min_likelihood_idx[i]], names_all[i])
                names_modified.append(temp)
            names_modified = np.array(names_modified)
        
            self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                        pred=hm_uv_stack, string=names_modified)
            
            del images_all
        
            # Part 3.b: Next, prepare conditional heatmaps
            logging.info('Visualizing heatmaps for minimum likelihood.')
            conditional_hm_all = []
            for i in range(15):
                conditional_hm = np.zeros([64, 64])
                for j, link in enumerate(links):
                    pred1 = hm_uv_stack[i, 0, link[0]]
                    pred2 = hm_uv_stack[i, 0, link[1]]

                    mean = likelihood_params[i, j, 0]
                    log_var = likelihood_params[i, j, 1]

                    target = self._find_point_along_line(source_pt=pred1, dest_pt=pred2, magnitude=mean)
                    conditional_hm_link = self._draw_gaussian(target, log_var)
                    conditional_hm = np.maximum(conditional_hm, conditional_hm_link)
                #RESIZE HM
                conditional_hm_all.append(cv2.resize(conditional_hm, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))

            conditional_hm_all = np.stack(conditional_hm_all, axis=0)
            names_modified = []
            for i in range(15):
                names_modified.append('SkeletonVizandName{}'.format(names_all[i]))
            names_modified = np.array(names_modified)

            self._visualize_predictions(image=conditional_hm_all, name=names_all, dataset=datasets_all,
                                        gt=gts_all, pred=hm_uv_stack, string=names_modified)
        

        ################################################################################################################
        # Part 4: Visualizing images with maximum expected likelihood
        ################################################################################################################
            
        logging.info('Visualizing images with maximum likelihood.')
        idx = max_likelihood_idx

        # Collect images for these indices
        dataset_ = ActiveLearningVizDataLoader(train, indices=unlabelled_idx[idx], conf=self.conf)

        vl4pose_dataloader = torch.utils.data.DataLoader(
            dataset_, batch_size=self.conf.experiment_settings['batch_size'], shuffle=False, num_workers=1)

        with torch.no_grad():
            images_all = None
            gts_all = None
            names_all = []
            hm_uv_stack = []
            datasets_all = []
            likelihood_params = None

            for images, _, names, gts, datasets in tqdm(vl4pose_dataloader):

                pose_heatmaps_, pose_features_ = self.pose_model(images)
                likelihood_pred_ = self._aux_net_inference(pose_features_)

                for i in range(images.shape[0]):
                    hm_uv = self._estimate_uv(hm_array=pose_heatmaps_[:, -1].cpu().numpy()[i])
                    hm_uv_stack.append(hm_uv)

                    names_all.append(names[i])
                    datasets_all.append(datasets[i])

                try:
                    images_all = torch.cat([images_all, images], dim=0)
                    gts_all = torch.cat([gts_all, gts], dim=0)
                    likelihood_params = torch.cat([likelihood_params, likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)], dim=0)

                except TypeError:
                    images_all = images
                    gts_all = gts
                    likelihood_params = likelihood_pred_.cpu().reshape(images.shape[0], len(links), 2)

            images_all = images_all.numpy()
            hm_uv_stack = np.stack(hm_uv_stack, axis=0)
            names_all = np.array(names_all)
            datasets_all = np.array(datasets_all)
            gts_all = gts_all.numpy()
            likelihood_params = likelihood_params.numpy()

            names_modified = []
            for i in range(15):
                temp = 'LogLikelihood_{}_andName_{}'.format(max_likelihood[max_likelihood_idx[i]], names_all[i])
                names_modified.append(temp)
            names_modified = np.array(names_modified)
        
            self._visualize_predictions(image=images_all, name=names_all, dataset=datasets_all, gt=gts_all,
                                        pred=hm_uv_stack, string=names_modified)
            
            del images_all
        
            # Part 4.b: Next, prepare conditional heatmaps
            logging.info('Visualizing heatmaps for maximum likelihood.')
            conditional_hm_all = []
            for i in range(15):
                conditional_hm = np.zeros([64, 64])
                for j, link in enumerate(links):
                    pred1 = hm_uv_stack[i, 0, link[0]]
                    pred2 = hm_uv_stack[i, 0, link[1]]

                    mean = likelihood_params[i, j, 0]
                    log_var = likelihood_params[i, j, 1]

                    target = self._find_point_along_line(source_pt=pred1, dest_pt=pred2, magnitude=mean)
                    conditional_hm_link = self._draw_gaussian(target, log_var)
                    conditional_hm = np.maximum(conditional_hm, conditional_hm_link)
                #RESIZE HM
                conditional_hm_all.append(cv2.resize(conditional_hm, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))

            conditional_hm_all = np.stack(conditional_hm_all, axis=0)
            names_modified = []
            for i in range(15):
                names_modified.append('SkeletonVizandName{}'.format(names_all[i]))
            names_modified = np.array(names_modified)

            self._visualize_predictions(image=conditional_hm_all, name=names_all, dataset=datasets_all,
                                        gt=gts_all, pred=hm_uv_stack, string=names_modified)


    def _draw_gaussian(self, pt, log_var, hm_shape=(64, 64)):
        """

        :param pt:
        :param log_var:
        :param hm_shape:
        :return:
        """

        im = np.zeros(hm_shape, dtype=np.float32)
        pt_rint = np.rint(pt).astype(int)
        sigma = (np.exp(log_var))**0.5

        # Size of 2D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Generate gaussian, with window=size and variance=sigma
        u = np.arange(pt_rint[0] - (size // 2), pt_rint[0] + (size // 2) + 1)
        v = np.arange(pt_rint[1] - (size // 2), pt_rint[1] + (size // 2) + 1)
        uu, vv = np.meshgrid(u, v, sparse=True)
        z = (np.exp(-((uu - pt[0]) ** 2 + (vv - pt[1]) ** 2) / (2 * (sigma ** 2)))) * (1/((2*np.pi*sigma*sigma)**0.5))
        z = z.T

        # Identify indices in im that will define the crop area
        top = max(0, pt_rint[0] - (size//2))
        bottom = min(hm_shape[0], pt_rint[0] + (size//2) + 1)
        left = max(0, pt_rint[1] - (size//2))
        right = min(hm_shape[1], pt_rint[1] + (size//2) + 1)

        im[top:bottom, left:right] = \
            z[top - (pt_rint[0] - (size//2)): top - (pt_rint[0] - (size//2)) + (bottom - top),
            left - (pt_rint[1] - (size//2)): left - (pt_rint[1] - (size//2)) + (right - left)]

        return im


    def _visualize_predictions(self, image=None, name=None, dataset=None, gt=None, pred=None, string=None):

        dataset_viz = {}
        dataset_viz['img'] = image
        dataset_viz['name'] = name
        dataset_viz['dataset'] = dataset
        dataset_viz['gt'] = gt
        dataset_viz['pred'] = pred
        dataset_viz['string'] = string

        dataset_viz = self._recreate_images(gt=True, pred=True, external=True, ext_data=dataset_viz)
        visualize_image(dataset_viz, save_dir=self.conf.model['save_path'], bbox=False)


    def _recreate_images(self, gt=False, pred=False, external=False, ext_data=None):
        '''

        :return:
        '''
        assert gt + pred != 0, "Specify atleast one of GT or Pred"
        assert external
        assert ext_data, "ext_dataset can't be none to recreate external datasets"

        data_split = ext_data

        # Along with the below entries, we also pass bbox coordinates for each dataset
        img_dict = {'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'dataset': [], 'display_string': []},
                    'lspet': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'dataset': [], 'display_string': []},
                    'lsp': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'dataset': [], 'display_string': []}}

        for i in range(len(data_split['img'])):
            dataset = data_split['dataset'][i]
            img_dict[dataset]['img'].append(data_split['img'][i])
            img_dict[dataset]['img_name'].append(data_split['name'][i])
            img_dict[dataset]['dataset'].append(data_split['dataset'][i])
            img_dict[dataset]['display_string'].append(data_split['string'][i])


            joint_dict = dict([(self.i2j[i], []) for i in range(self.conf.experiment_settings['num_hm'])])
            gt_dict = copy.deepcopy(joint_dict)
            pred_dict = copy.deepcopy(joint_dict)

            if gt:
                for person in range(1):
                    for joint in range(self.conf.experiment_settings['num_hm']):
                        gt_dict[self.i2j[joint]].append(data_split['gt'][i, person, joint])

            if pred:
                for person in range(1):
                    for joint in range(self.conf.experiment_settings['num_hm']):
                        pred_dict[self.i2j[joint]].append(data_split['pred'][i, person, joint])

            img_dict[dataset]['img_gt'].append(gt_dict)
            img_dict[dataset]['img_pred'].append(pred_dict)

        return img_dict


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


    def _estimate_uv(self, hm_array):
        '''
        Assumes single person
        :param hm_array:
        :param pred_placeholder:
        :return:
        '''
        threshold = 0
        joint = np.empty(shape=[1, hm_array.shape[0], 3], dtype=np.float32)
        # Iterate over each heatmap
        for jnt_id in range(hm_array.shape[0]):
            joint[0, jnt_id, :] = uv_from_heatmap(hm=hm_array[jnt_id], threshold=threshold)
        return joint


    def _find_point_along_line(self, source_pt, dest_pt, magnitude):
        downscale = [(256 - 1) / (64 - 1), ((256 - 1) / (64 - 1))]
        # Ignore visibility flag
        source_pt = source_pt[:2] / np.array(downscale)
        dest_pt = dest_pt[:2] / np.array(downscale)

        direction = dest_pt - source_pt
        t = magnitude / np.linalg.norm(source_pt - dest_pt)

        return source_pt + (t * direction)



class ActiveLearningVizDataLoader(torch.utils.data.Dataset):
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
        bounding_box = self.bounding_box[item]
        gt = self.gt[item]


        if self.load_all_imgs:
            image = self.images[item]

        else:
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
               torch.tensor(data=heatmaps, dtype=torch.float32, device='cpu'), name, gt, dataset


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
    def __init__(self, hm, param, j2i, i2j, links, vl4pose_config, function):
        self.hm = hm
        self.param = param
        self.j2i = j2i
        self.i2j = i2j
        self.links = links
        self.config = vl4pose_config
        self.function = function

    def __len__(self):
        return self.hm.shape[0]

    def __getitem__(self, i):
        joints = {}

        heatmaps = self.hm[i]
        parameters = self.param[i]

        # Initialize keypoints for each node
        for key in self.j2i.keys():

            heatmap = heatmaps[self.j2i[key]]
            loc = peak_local_max(heatmap, min_distance=self.config['min_distance'], num_peaks=self.config['num_peaks'], exclude_border=False)
            peaks = heatmap[loc[:, 0], loc[:, 1]]

            peaks = softmax_fn(peaks)
            joints[key] = Keypoint(name=key, loc=loc, peaks=peaks, function=self.function)

        # Initialize parent-child relations
        for k, l in enumerate(self.links):

            joints[self.i2j[l[0]]].parameters.append(parameters[k])
            joints[self.i2j[l[0]]].children.append(joints[self.i2j[l[1]]])

        max_ll, trace = joints['head'].run_likelihood()


        vl4pose_image = []
        for j in range(heatmaps.shape[0]):
            vl4pose_image.append(torch.from_numpy(trace['{}_uv'.format(self.i2j[j])]))
        vl4pose_image = torch.stack(vl4pose_image, dim=0)
        vl4pose_image = torch.cat([vl4pose_image, torch.ones(heatmaps.shape[0]).view(-1, 1)], dim=1)
        vl4pose_image = vl4pose_image.unsqueeze(0)

        string = ''
        for jnt in self.j2i.keys():
            string += str(trace[jnt].item())

        if string != ('0'*len(self.j2i.keys())): has_pose_changed = True
        else: has_pose_changed = False
    

        return max_ll, vl4pose_image, has_pose_changed



class Keypoint(object):
    def __init__(self, name, loc, peaks, function):
        self.name = name
        self.loc = loc
        self.peaks = peaks
        self.children = []
        self.parameters = []
        self.function = function


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
        return self.function(likelihood_per_location), return_trace


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
            return self.function(likelihood_per_location), {self.name: np.argmax(likelihood_per_location),
                                                         '{}_uv'.format(self.name): self.loc[np.argmax(likelihood_per_location)]}

        return_trace = {}
        for child_trace in per_location_trace[np.argmax(likelihood_per_location)]:
            return_trace.update(child_trace)

        return_trace[self.name] = np.argmax(likelihood_per_location)
        return_trace['{}_uv'.format(self.name)] = self.loc[np.argmax(likelihood_per_location)]
        return self.function(likelihood_per_location), return_trace
