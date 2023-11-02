import os
import copy
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau

from config import ParseConfig
from utils import fast_argmax
from utils import visualize_image
from utils import heatmap_loss
from utils import count_parameters
from utils import get_pairwise_joint_distances
from activelearning import ActiveLearning
from activelearning_viz import ActiveLearning_Visualization
from dataloader import load_hp_dataset
from dataloader import HumanPoseDataLoader
from evaluation import PercentageCorrectKeypoint
from models.auxiliary.AuxiliaryNet import AuxNet
from models.hrnet.pose_hrnet import PoseHighResolutionNet as HRNet
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass

# Global declarations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))


class Train(object):
    def __init__(self, pose_model, aux_net, hyperparameters, dataset_obj, conf, tb_writer):
        """
        Class for training the human pose and aux_net model

        :param pose_model: (torch.nn) Human pose model
        :param aux_net: (torch.nn) Auxiliary network
        :param hyperparameters: (dict) Various hyperparameters used in training
        :param dataset_obj: (torch.utils.data.Dataset)
        :param conf: (Object of ParseConfig) Contains the configurations for the model
        :param tb_writer: (Object of SummaryWriter) Tensorboard writer to log values
        """

        self.conf = conf
        self.aux_net = aux_net
        self.network = pose_model
        self.tb_writer = tb_writer
        self.dataset_obj = dataset_obj
        self.hyperparameters = hyperparameters

        # Experiment Settings
        self.batch_size = conf.experiment_settings['batch_size']
        self.epoch = hyperparameters['num_epochs']
        self.optimizer = hyperparameters['optimizer']  # Adam / SGD
        self.loss_fn = hyperparameters['loss_fn']  # MSE
        self.learning_rate = hyperparameters['optimizer_config']['lr']
        self.start_epoch = hyperparameters['start_epoch']  # Used in case of resume training
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        self.joint_names = self.dataset_obj.ind_to_jnt
        self.model_save_path = conf.model['save_path']
        self.train_aux_net = conf.model['aux_net']['train']


        # Stacked Hourglass scheduling
        if self.train_aux_net:
            min_lr = [0.000003, conf.experiment_settings['lr']]
        else:
            min_lr = 0.000003

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5, cooldown=2, min_lr=min_lr, verbose=True)

        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, self.batch_size,
                                                            shuffle=True, num_workers=2, drop_last=True)


    def train_model(self):
        """
        Training loop
        :return:
        """

        print("Initializing training: Epochs - {}\tBatch Size - {}".format(
            self.hyperparameters['num_epochs'], self.batch_size))

        # 'mean_loss_validation': {'Pose': validation_loss_pose, 'AuxNet': validation_aux_net}
        if self.conf.resume_training:
            best_val_pose = self.hyperparameters['mean_loss_validation']['Pose']
            best_val_auxnet = self.hyperparameters['mean_loss_validation']['AuxNet']
            best_epoch_pose = self.hyperparameters['start_epoch']
            best_epoch_auxnet = -1
            global_step = 0

        else:
            best_val_pose = np.inf
            best_val_auxnet = np.inf
            best_epoch_pose = -1
            best_epoch_auxnet = -1
            global_step = 0

        # Variable to store all the loss values for logging
        loss_across_epochs = []
        validation_across_epochs = []

        for e in range(self.start_epoch, self.epoch):
            epoch_loss = []
            epoch_loss_aux_net = []

            # Network alternates between train() and validate()
            self.network.train()
            if self.train_aux_net:
                self.aux_net.train()

            self.dataset_obj.input_dataset(train=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_image, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 0, "Training split should be 0."

                self.optimizer.zero_grad()
                outputs, pose_features = self.network(images)          # images.cuda() done internally within the model
                loss = heatmap_loss(outputs, heatmaps)                 # heatmaps transferred to GPU within the function

                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'learning_loss':
                    learning_loss_ = loss.clone().detach().to('cuda:{}'.format(torch.cuda.device_count() - 1))
                    learning_loss_ = torch.mean(learning_loss_, dim=[1])

                    loss_learnloss = self.learning_loss(pose_encodings=pose_features, true_loss=learning_loss_,
                                                        gt_per_img=gt_per_image, epoch=e)
                    loss_learnloss.backward()
                    epoch_loss_aux_net.append(loss_learnloss.item())


                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'aleatoric':
                    loss_aleatoric = self.aleatoric_uncertainty(pose_encodings=pose_features, outputs=outputs,
                                                                heatmaps=heatmaps, joint_exist=joint_exist, epoch=e)
                    loss_aleatoric.backward()
                    epoch_loss_aux_net.append(loss_aleatoric.item())

                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'vl4pose':
                    loss_vl4pose = self.vl4pose(pose_encodings=pose_features, heatmaps=heatmaps, 
                                                joint_exist=joint_exist, epoch=e)
                    loss_vl4pose.backward()
                    epoch_loss_aux_net.append(loss_vl4pose.item())


                if self.conf.model['aux_net']['train_auxnet_only']:
                    loss = torch.mean(loss) * 0
                else:
                    loss = torch.mean(loss)

                loss.backward()
                if self.conf.tensorboard:
                    self.tb_writer.add_scalar('Train/Loss_batch', torch.mean(loss), global_step)
                epoch_loss.append(loss.item())

                # Weight update
                self.optimizer.step()
                global_step += 1


            # Epoch training ends -------------------------------------------------------------------------------------
            epoch_loss = np.mean(epoch_loss)

            if self.conf.model['aux_net']['train']:
                epoch_loss_aux_net = np.mean(epoch_loss_aux_net)
                validation_loss_pose, validation_aux_net = self.validation(e)
            else:
                validation_loss_pose = self.validation(e)
                validation_aux_net = 0.0

            # Learning rate scheduler on the Human Pose validation loss
            self.scheduler.step(validation_loss_pose)

            # TensorBoard Summaries
            if self.conf.tensorboard:
                self.tb_writer.add_scalar('Train', torch.tensor([epoch_loss]), global_step)
                self.tb_writer.add_scalar('Validation/HG_Loss', torch.tensor([validation_loss_pose]), global_step)
                if self.conf.model['aux_net']['train']:
                    self.tb_writer.add_scalar('Validation/Learning_Loss', torch.tensor([validation_aux_net]), global_step)


            # Save if best model
            if best_val_pose > validation_loss_pose:
                torch.save(self.network.state_dict(),
                           os.path.join(self.model_save_path, 'model_checkpoints/pose_net.pth'))

                if self.conf.model['aux_net']['train']:
                    torch.save(
                        self.aux_net.state_dict(),
                        os.path.join(self.model_save_path,
                                     'model_checkpoints/aux_net_{}_BestPose.pth'.format(self.conf.model['aux_net']['method'])))

                best_val_pose = validation_loss_pose
                best_epoch_pose = e + 1

                torch.save({'epoch': e + 1,
                            'optimizer_load_state_dict': self.optimizer.state_dict(),
                            'mean_loss_train': epoch_loss,
                            'mean_loss_validation': {'Pose': validation_loss_pose, 'AuxNet': validation_aux_net},
                            'aux_net': self.conf.model['aux_net']['train']},
                           os.path.join(self.model_save_path, 'model_checkpoints/optim_best_model.tar'))


            if self.conf.model['aux_net']['train']:
                if (best_val_auxnet > validation_aux_net) and (validation_aux_net != 0.0):
                    torch.save(self.aux_net.state_dict(),
                               os.path.join(self.model_save_path, 'model_checkpoints/aux_net_{}.pth'.format(self.conf.model['aux_net']['method'])))

                    best_val_auxnet = validation_aux_net
                    best_epoch_auxnet = e + 1

            print("Loss at epoch {}/{}: (train:Pose) {}\t"
                  "(train:AuxNet) {}\t"
                  "(validation:Pose) {}\t"
                  "(Validation:AuxNet) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epoch,
                epoch_loss,
                epoch_loss_aux_net,
                validation_loss_pose,
                validation_aux_net,
                best_epoch_pose))

            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_pose)

            # Save the loss values
            f = open(os.path.join(self.model_save_path, 'model_checkpoints/loss_data.txt'), "w")
            f_ = open(os.path.join(self.model_save_path, 'model_checkpoints/validation_data.txt'), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_.close()

        if self.conf.tensorboard:
            self.tb_writer.close()
        logging.info("Model training completed\nBest validation loss (Pose): {}\tBest Epoch: {}"
                     "\nBest validation loss (AuxNet): {}\tBest Epoch: {}".format(
            best_val_pose, best_epoch_pose, best_val_auxnet, best_epoch_auxnet))


    def validation(self, e):
        """

        :param e: Epoch
        :return:
        """
        with torch.no_grad():
            # Stores the loss for all batches
            epoch_val_pose = []
            self.network.eval()

            if self.conf.model['aux_net']['train']:
                epoch_val_auxnet = []
                self.aux_net.eval()

            # Augmentation only needed in Training
            self.dataset_obj.input_dataset(validate=True)

            # Compute and store batch-wise validation loss in a list
            logging.info('Validation for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_img, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."

                outputs, pose_features = self.network(images)
                loss_val_pose = heatmap_loss(outputs, heatmaps)

                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'learning_loss':
                    learning_loss_val = loss_val_pose.clone().detach().to('cuda:{}'.format(torch.cuda.device_count() - 1))
                    learning_loss_val = torch.mean(learning_loss_val, dim=[1])
                    loss_val_auxnet = self.learning_loss(pose_features, learning_loss_val, gt_per_img, e)
                    epoch_val_auxnet.append(loss_val_auxnet.item())

                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'aleatoric':
                    loss_val_aleatoric = self.aleatoric_uncertainty(pose_encodings=pose_features, outputs=outputs,
                                                                heatmaps=heatmaps, joint_exist=joint_exist, epoch=e)
                    epoch_val_auxnet.append(loss_val_aleatoric.item())

                if self.conf.model['aux_net']['train'] and self.conf.model['aux_net']['method'] == 'vl4pose':
                    vl4pose_loss = self.vl4pose(pose_encodings=pose_features, heatmaps=heatmaps, joint_exist=joint_exist, epoch=e)
                    epoch_val_auxnet.append(vl4pose_loss.item())


                loss_val_pose = torch.mean(loss_val_pose)
                epoch_val_pose.append(loss_val_pose.item())

            if self.conf.model['aux_net']['train']:
                return np.mean(epoch_val_pose), np.mean(epoch_val_auxnet)

            else:
                return np.mean(epoch_val_pose)


    def learning_loss(self, pose_encodings, true_loss, gt_per_img, epoch):
        '''
        Learning Loss module
        Based on the paper: "Learning Loss For Active Learning, CVPR 2019" and "A Mathematical Analysis of Learning Loss for Active Learning in Regression, CVPR-W 2021"

        :param pose_encodings: (Dict of tensors) Intermediate (Hourglass) and penultimate layer output of the M10 network
        :param true_loss: (Tensor of shape [Batch Size]) Loss computed from M10 prediction and ground truth
        :param gt_per_img: (Tensor, shape [Batch Size]) Number of ground truth per image
        :param epoch: (scalar) Epoch, used in learning loss warm start-up
        :return: (Torch scalar tensor) Learning Loss
        '''

        learnloss_margin = self.conf.active_learning['learningLoss']['margin']
        learnloss_objective = self.conf.active_learning['learningLoss']['objective']
        learnloss_warmup = self.conf.model['aux_net']['warmup']

        emperical_loss = self._aux_net_inference(pose_encodings)
        emperical_loss = emperical_loss.squeeze()

        assert emperical_loss.shape == true_loss.shape, "Mismatch in Batch size for true and emperical loss"

        with torch.no_grad():
            # Scale the images as per the number of joints
            # To prevent DivideByZero. PyTorch does not throw an exception to DivideByZero
            gt_per_img = torch.sum(gt_per_img, dim=1)
            gt_per_img += 0.1
            true_loss = true_loss / gt_per_img.to(true_loss.device)

            # Splitting into pairs: (i, i+half)
            half_split = true_loss.shape[0] // 2

            true_loss_i = true_loss[: half_split]
            true_loss_j = true_loss[half_split: 2 * half_split]

        emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
        emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]

        # Pair wise loss as mentioned in the original paper
        if learnloss_objective == 'YooAndKweon':
            loss_sign = torch.sign(true_loss_i - true_loss_j)
            loss_emp = (emp_loss_i - emp_loss_j)

            # Learning Loss objective
            llal_loss = torch.max(torch.zeros(half_split, device=loss_sign.device), (-1 * (loss_sign * loss_emp)) + learnloss_margin)

        # Computing loss over the entire batch using softmax.
        elif learnloss_objective == 'KLdivergence':
            # Removed the standardization-KL Divergence parts
            with torch.no_grad():
                true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
                true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)

            emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
            emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)

            # Scaling the cross entropy loss with respect to true loss values
            llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)
            #llal_loss = torch.sum((-true_loss_scaled * torch.log(emp_loss_logsftmx)), dim=1, keepdim=True)
        else:
            raise NotImplementedError('Currently only "YooAndKweon" or "KLdivergence" supported. ')

        if learnloss_warmup <= epoch:
            return torch.mean(llal_loss)
        else:
            return 0.0 * torch.mean(llal_loss)


    def aleatoric_uncertainty(self, pose_encodings, outputs, heatmaps, joint_exist, epoch):
        """
        Extension of Kendall and Gal's method for calculating uncertainty to human pose estimation
        Auxiliary Network module to train sigmas for the HG/HRN network's heatmaps directly

        :param pose_encodings: (Dict of tensors) Intermediate and penultimate layer outputs of the pose model
        :param outputs: Tensor of size (batch_size, num_joints, hm_size, hm_size) Outputs of the main HG/HRN network
        :param heatmaps: Tensor of size (batch_size, num_joints, hm_size, hm_size) Ground truth heatmaps
        :param joint_exist: Tensor of size (batch_size, num_joints) Joint status (1=Present, 0=Absent)
                            Present=1 may include occluded joints depending on configuration.yml setting
        :param epoch: (scalar) Epoch, used in auxiliary network loss to compare warmp-up
        :return: auxnet_loss: (Torch scalar tensor) auxiliary network Loss
        """

        parameters = self._aux_net_inference(pose_encodings)

        # Final stack of hourglass / output of HRNet
        outputs = outputs[:, -1].clone().detach().to('cuda:{}'.format(torch.cuda.device_count() - 1))
        assert heatmaps.shape == outputs.shape  # Batch Size x num_joints x 64 x 64

        joint_exist = joint_exist.float().to(device=parameters.device)

        residual = torch.sum((fast_argmax(outputs) - fast_argmax(heatmaps).to(outputs.device))**2, dim=-1) # along axis representing u,v
        residual = 0.5 * residual * torch.exp(-parameters)
        neg_log_likelihood = residual + (0.5 * parameters)
        neg_log_likelihood = neg_log_likelihood * joint_exist


        if self.conf.model['aux_net']['warmup'] <= epoch:
            return torch.mean(neg_log_likelihood)

        else:
            return 0.0 * torch.mean(neg_log_likelihood)


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
        parameters = parameters.reshape(self.batch_size, len(links), 2)
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

        if self.conf.model['aux_net']['warmup'] <= epoch:
            return torch.mean(neg_log_likelihood)

        else:
            return 0.0 * torch.mean(neg_log_likelihood)


    def _aux_net_inference(self, pose_features):
        """
        Common to VL4Pose, LearningLoss++ and Aleatoric which all use an auxiliary network
        """
        extractor = self.conf.architecture['aux_net']['conv_or_avg_pooling']

        with torch.no_grad():
            if extractor == 'avg':
                # Transfer to GPU where auxiliary network is stored
                encodings = pose_features['penultimate']

            else:
                depth = len(self.conf.architecture['aux_net']['spatial_dim'])
                encodings = torch.cat(
                    [pose_features['feature_{}'.format(i)].reshape(
                        self.batch_size, pose_features['feature_{}'.format(i)].shape[1], -1)
                        for i in range(depth, 0, -1)],
                    dim=2)

        aux_out = self.aux_net(encodings)
        return aux_out


class Metric(object):
    def __init__(self, network, dataset_obj, conf):
        '''
        Class for Testing the model:
            1. Compute ground truth and predictions
            2. Computing metrics: PCK@0.x
        :param network: (torch.nn) Hourglass network to compute predictions
        :param dataset_obj: (Dataset object) Handles data to be fed to PyTorch DataLoader
        :param conf: (Object of ParseConfig) Configuration for the experiment
        '''

        self.dataset_obj = dataset_obj
        self.dataset_obj.input_dataset(validate=True)

        self.network = network
        self.viz=conf.viz                                                                       # Controls visualization
        self.conf = conf
        self.batch_size = conf.experiment_settings['batch_size']
        self.ind_to_jnt = self.dataset_obj.ind_to_jnt


        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, batch_size=self.batch_size,
                                                            shuffle=False, num_workers=2)


    def inference(self):
        '''
        Obtains model inference
        :return: None
        '''

        self.network.eval()
        logging.info("Starting model inference")

        outputs_ = None
        scale_ = None
        num_gt_ = None
        dataset_ = None
        name_ = None
        gt_ = None
        normalizer_ = None

        with torch.no_grad():
            for (images, _, gt, name, dataset, num_gt, split, _, scale_params, normalizer, joint_exist) in tqdm(
                    self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."
                outputs, pose_features = self.network(images)
                outputs = outputs[:, -1]

                try:
                    outputs_ = torch.cat((outputs_, outputs.cpu().clone()), dim=0)
                    scale_['scale_factor'] = torch.cat((scale_['scale_factor'], scale_params['scale_factor']), dim=0)
                    scale_['padding_u'] = torch.cat((scale_['padding_u'], scale_params['padding_u']), dim=0)
                    scale_['padding_v'] = torch.cat((scale_['padding_v'], scale_params['padding_v']), dim=0)
                    num_gt_ = torch.cat((num_gt_, num_gt), dim=0)
                    dataset_ = dataset_ + dataset
                    name_ = name_ + name
                    gt_ = torch.cat((gt_, gt), dim=0)
                    normalizer_ = torch.cat((normalizer_, normalizer), dim=0)

                except TypeError:
                    outputs_ = outputs.cpu().clone()
                    scale_ = copy.deepcopy(scale_params)
                    num_gt_ = num_gt
                    dataset_ = dataset
                    name_ = name
                    gt_ = gt
                    normalizer_ = normalizer

                # Generate visualizations (256x256) for that batch of images
                if self.conf.viz:
                    hm_uv_stack = []
                    # Compute u,v values from heatmap
                    for i in range(images.shape[0]):
                        hm_uv = self.dataset_obj.estimate_uv(hm_array=outputs.cpu().numpy()[i],
                                                             pred_placeholder=-np.ones_like(gt[i].numpy()))
                        hm_uv_stack.append(hm_uv)
                    hm_uv = np.stack(hm_uv_stack, axis=0)
                    self.visualize_predictions(image=images.numpy(), name=name, dataset=dataset, gt=gt.numpy(), pred=hm_uv)


        scale_['scale_factor'] = scale_['scale_factor'].numpy()
        scale_['padding_u'] = scale_['padding_u'].numpy()
        scale_['padding_v'] = scale_['padding_v'].numpy()

        model_inference = {'heatmap': outputs_.numpy(), 'scale': scale_, 'dataset': dataset_,
                           'name': name_, 'gt': gt_.numpy(), 'normalizer': normalizer_.numpy()}

        return model_inference


    def keypoint(self, infer):
        '''

        :param infer:
        :return:
        '''

        heatmap = infer['heatmap']
        scale = infer['scale']
        dataset = infer['dataset']
        name = infer['name']
        gt = infer['gt']
        normalizer = infer['normalizer']

        hm_uv_stack = []

        csv_columns = ['name', 'dataset', 'normalizer', 'joint', 'uv']

        gt_csv = []
        pred_csv = []

        # Iterate over all heatmaps to obtain predictions
        for i in range(gt.shape[0]):

            heatmap_ = heatmap[i]

            gt_uv = gt[i]
            hm_uv = self.dataset_obj.estimate_uv(hm_array=heatmap_, pred_placeholder=-np.ones_like(gt_uv))
            hm_uv_stack.append(hm_uv)

            # Scaling the point ensures that the distance between gt and pred is same as the scale of normalization
            scale_factor = scale['scale_factor'][i]
            padding_u = scale['padding_u'][i]
            padding_v = scale['padding_v'][i]

            # Scaling ground truth
            gt_uv_correct = np.copy(gt_uv)
            hm_uv_correct = np.copy(hm_uv)

            gt_uv_correct[:, :, 1] -= padding_v
            gt_uv_correct[:, :, 0] -= padding_u
            gt_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            # Scaling predictions
            hm_uv_correct[:, :, 1] -= padding_v
            hm_uv_correct[:, :, 0] -= padding_u
            hm_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            assert gt_uv_correct.shape == hm_uv_correct.shape, "Mismatch in gt ({}) and prediction ({}) shape".format(
                gt_uv_correct.shape, hm_uv_correct.shape)

            # Iterate over joints
            for jnt in range(gt_uv_correct.shape[1]):
                gt_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': gt_uv_correct[:, jnt, :].astype(np.float32)
                }

                pred_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': hm_uv_correct[:, jnt, :].astype(np.float32)
                }

                gt_csv.append(gt_entry)
                pred_csv.append(pred_entry)


        pred_csv = pd.DataFrame(pred_csv, columns=csv_columns)
        gt_csv = pd.DataFrame(gt_csv, columns=csv_columns)

        pred_csv.sort_values(by='dataset', ascending=True, inplace=True)
        gt_csv.sort_values(by='dataset', ascending=True, inplace=True)

        assert len(pred_csv.index) == len(gt_csv.index), "Mismatch in number of entries in pred and gt dataframes."

        pred_csv.to_csv(os.path.join(self.conf.model['save_path'], 'model_checkpoints/pred.csv'), index=False)
        gt_csv.to_csv(os.path.join(self.conf.model['save_path'], 'model_checkpoints/gt.csv'), index=False)
        logging.info('Pandas dataframe saved successfully.')

        return gt_csv, pred_csv


    def visualize_predictions(self, image=None, name=None, dataset=None, gt=None, pred=None):

        dataset_viz = {}
        dataset_viz['img'] = image
        dataset_viz['name'] = name
        dataset_viz['display_string'] = name
        dataset_viz['split'] = np.ones(image.shape[0])
        dataset_viz['dataset'] = dataset
        dataset_viz['bbox_coords'] = np.zeros([image.shape[0], 4, 4])
        dataset_viz['num_persons'] = np.ones([image.shape[0], 1])
        dataset_viz['gt'] = gt
        dataset_viz['pred'] = pred

        dataset_viz = self.dataset_obj.recreate_images(gt=True, pred=True, external=True, ext_data=dataset_viz)
        visualize_image(dataset_viz, save_dir=self.conf.model['save_path'], bbox=False)


    def compute_metrics(self, gt_df=None, pred_df=None):
        '''
        Loads the ground truth and prediction CSVs into memory.
        Evaluates Precision, FPFN metrics for the prediction and stores them into memory.
        :return: None
        '''

        # Ensure that same datasets have been loaded
        assert all(pred_df['dataset'].unique() == gt_df['dataset'].unique()), \
            "Mismatch in dataset column for gt and pred"

        logging.info('Generating evaluation metrics for dataset:')
        # Iterate over unique datasets
        for dataset_ in gt_df['dataset'].unique():
            logging.info(str(dataset_))

            # Separate out images based on dataset
            pred_ = pred_df.loc[pred_df['dataset'] == dataset_]
            gt_ = gt_df.loc[gt_df['dataset'] == dataset_]

            # Compute scores
            pck_df = PercentageCorrectKeypoint(
                pred_df=pred_, gt_df=gt_, config=self.conf, jnts=list(self.ind_to_jnt.values()))

            # Save the tables
            if dataset_ == 'mpii':
                metric_ = 'PCKh'
            else:
                metric_ = 'PCK'

            pck_df.to_csv(os.path.join(self.conf.model['save_path'],
                                       'model_checkpoints/{}_{}.csv'.format(metric_, dataset_)),
                          index=False)

        print("Metrics computation completed.")


    def eval(self):
        '''

        :return:
        '''
        model_inference = self.inference()
        gt_csv, pred_csv = self.keypoint(model_inference)
        self.compute_metrics(gt_df=gt_csv, pred_df=pred_csv)



def load_models(conf, load_pose, load_aux, model_dir):
    """

    :param conf:
    :param load_pose:
    :param load_aux:
    :param model_dir:
    :return:
    """

    # Initialize AuxNet, Hourglass/HRNet
    # Elsewhere, resume training ensures the code creates a copy of the best models from the interrupted run.

    if conf.use_auxnet:
        logging.info('Initializing Auxiliary Network')
        aux_net = AuxNet(arch=conf.architecture['aux_net'])
    else:
        aux_net = None

    if conf.model['type'] == 'hourglass':
        logging.info('Initializing Hourglass Network')
        pose_net = Hourglass(arch=conf.architecture['hourglass'],
                             auxnet=conf.use_auxnet,
                             intermediate_features=conf.architecture['aux_net']['conv_or_avg_pooling'])
        print('Number of parameters (Hourglass): {}\n'.format(count_parameters(pose_net)))

    else:
        logging.info('Initializing HRNet')
        assert conf.model['type'] == 'hrnet', "Currently support 'hourglass' and 'hrnet'."
        pose_net = HRNet(arch=conf.architecture['hrnet'],
                         auxnet=conf.use_auxnet,
                         intermediate_features=conf.architecture['aux_net']['conv_or_avg_pooling'])
        print('Number of parameters (HRNet): {}\n'.format(count_parameters(pose_net)))

    # Load AuxNet modules (Best Model / resume training)
    if load_aux:
        if conf.resume_training:
            logging.info('\n-------------- Resuming training (Loading AuxNet) --------------\n')

            # Load and save the previous best model
            aux_net.load_state_dict(torch.load(
                os.path.join(model_dir, 'model_checkpoints/aux_net_{}.pth'.format(conf.active_learning['algorithm'])),
                map_location='cpu'))
            torch.save(
                aux_net.state_dict(),
                os.path.join(
                    conf.model['save_path'],
                    'model_checkpoints/aux_net_{}.pth'.format(conf.active_learning['algorithm'])))

            # Load the model corresponding to best pose model
            aux_net.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir, 'model_checkpoints/aux_net_{}_BestPose.pth'.format(conf.active_learning['algorithm'])),
                    map_location='cpu'))

        else:
            logging.info('Loading AuxNet Best Model')
            aux_net.load_state_dict(
                torch.load(os.path.join(
                    model_dir, 'model_checkpoints/aux_net_{}.pth'.format(conf.active_learning['algorithm'])), map_location='cpu'))

    # Load Pose model (code is independent of architecture)
    if load_pose:

        # Load model
        logging.info('Loading Pose model from: ' + model_dir)
        pose_net.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/pose_net.pth'), map_location='cpu'))
        logging.info("Successfully loaded Pose model.")

        if conf.resume_training:
            logging.info('\n-------------- Resuming training (Loading PoseNet) --------------\n')
            torch.save(pose_net.state_dict(), os.path.join(conf.model['save_path'], 'model_checkpoints/pose_net.pth'))

    # CUDA support: Single/Multi-GPU
    # Hourglass net and HRNet have CUDA definitions inside __init__(), specify only for aux_net
    if conf.model['aux_net']['train'] or load_aux:
        aux_net.cuda(torch.device('cuda:{}'.format(torch.cuda.device_count()-1)))

    logging.info('Successful: Model transferred to GPUs.\n')

    return pose_net, aux_net


def define_hyperparams(conf, pose_model, aux_net):#(conf, net, learnloss):
    """

    :param conf:
    :param pose_model:
    :param aux_net:
    :return:
    """
    logging.info('Initializing the hyperparameters for the experiment.')
    hyperparameters = dict()
    hyperparameters['optimizer_config'] = {
                                           'lr': conf.experiment_settings['lr'],
                                           'weight_decay': conf.experiment_settings['weight_decay']
                                          }
    hyperparameters['loss_params'] = {'size_average': True}
    hyperparameters['num_epochs'] = conf.experiment_settings['epochs']
    hyperparameters['start_epoch'] = 0  # Used for resume training

    # Parameters declared to the optimizer
    if conf.model['aux_net']['train']:
        logging.info('Parameters of AuxNet and PoseNet passed to Optimizer.')
        params_list = [{'params': pose_model.parameters()},
                       {'params': aux_net.parameters()}]
    else:
        logging.info('Parameters of PoseNet passed to Optimizer')
        params_list = [{'params': pose_model.parameters()}]

    hyperparameters['optimizer'] = torch.optim.Adam(params_list, **hyperparameters['optimizer_config'])

    if conf.resume_training:
        logging.info('Loading optimizer state dictionary')
        optim_dict = torch.load(os.path.join(conf.model['load_path'], 'model_checkpoints/optim_best_model.tar'))

        # If the previous experiment trained aux_net, ensure the flag is true for the current experiment
        assert optim_dict['aux_net'] == conf.model['aux_net']['train'], "AuxNet model needed to resume training"

        hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict'])
        logging.info('Optimizer state loaded successfully.\n')

        logging.info('Optimizer and Training parameters:\n')
        for key in optim_dict:
            if key == 'optimizer_load_state_dict':
                logging.info('Param group length: {}'.format(len(optim_dict[key]['param_groups'])))
            else:
                logging.info('Key: {}\tValue: {}'.format(key, optim_dict[key]))

        logging.info('\n')
        hyperparameters['start_epoch'] = optim_dict['epoch']
        hyperparameters['mean_loss_validation'] = optim_dict['mean_loss_validation']

    hyperparameters['loss_fn'] = torch.nn.MSELoss(reduction='none')

    return hyperparameters


def main():
    """
    Control flow for the code
    """

    # 1. Load configuration file --------------------------------------------------------------------------------------
    logging.info('Loading configurations.\n')
    conf  = ParseConfig()




    # 2. Loading datasets ---------------------------------------------------------------------------------------------
    logging.info('Loading pose dataset(s)\n')
    dataset_dict = load_hp_dataset(dataset_conf=conf.dataset, model_conf=conf.model)




    # 3. Defining the network -----------------------------------------------------------------------------------------
    logging.info('Initializing (and loading) human pose network and auxiliary network for Active Learning.\n')
    pose_model, aux_net = load_models(conf=conf, load_pose=conf.model['load'], load_aux=conf.model['aux_net']['load'],
                                      model_dir=conf.model['load_path'])




    # 4. Defining the Active Learning library --------------------------------------------------------------------------
    logging.info('Importing active learning object.\n')
    if conf.activelearning_viz:
        activelearning = ActiveLearning_Visualization(conf=conf, pose_net=pose_model, aux_net=aux_net)
    else:
        activelearning = ActiveLearning(conf=conf, pose_net=pose_model, aux_net=aux_net)




    # 5. Defining DataLoader -------------------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    datasets = HumanPoseDataLoader(dataset_dict=dataset_dict, activelearning=activelearning, conf=conf)


    # 5.a: Delete models, activelearning object to remove stray computational graphs (esp. for EGL)
    if conf.activelearning_viz:
        exit()

    del activelearning
    del pose_model, aux_net
    torch.cuda.empty_cache()
    
    logging.info('Re-Initializing (and loading) human pose network and auxiliary network.\n')
    pose_model, aux_net = load_models(conf=conf, load_pose=conf.model['load'], load_aux=conf.model['aux_net']['load'],
                                      model_dir=conf.model['load_path'])




    # 6. Defining Hyperparameters, TensorBoard directory ---------------------------------------------------------------
    logging.info('Initializing experiment settings.')
    hyperparameters = define_hyperparams(conf=conf, pose_model=pose_model, aux_net=aux_net)

    if conf.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(conf.model['save_path'], 'tensorboard'))
    else:
        writer = None




    # 7. Train the model
    if conf.train:
        train_obj = Train(pose_model=pose_model, aux_net=aux_net, hyperparameters=hyperparameters,
                          dataset_obj=datasets, conf=conf, tb_writer=writer)
        train_obj.train_model()

        del train_obj
        # Reload the best model for metric evaluation
        conf.resume_training = False
        pose_model, _ = load_models(conf=conf, load_pose=True, load_aux=False, model_dir=conf.model['save_path'])

    if conf.metric:
        metric_obj = Metric(network=pose_model, dataset_obj=datasets, conf=conf)
        metric_obj.eval()


if __name__ == "__main__":
    main()
