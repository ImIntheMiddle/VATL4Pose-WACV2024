import logging

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter


class AuxNet(nn.Module):
    def __init__(self, arch):#num_feat, pose_num_channels, convolution=False, pose_feat_shape=(64, 32, 16, 8, 4)):
        """

        :param num_feat:
        :param pose_num_channels: Pose estimator number of channels
        :param convolution: (bool) Does the AuxNet need a convolutional feature extractor?
        :param pose_feat_shape: Spatial dimension of intermediate pose estimator features
        """
        super(AuxNet, self).__init__()

        # From configuration
        self.fc_arch = arch['fc']
        self.is_conv = True if arch['conv_or_avg_pooling'] == 'conv' else False

        # Derived from Hourglass / HRNet
        self.conv_arch_spatial = arch['spatial_dim']
        self.conv_arch_channels = arch['channels']

        # List that houses the network
        self.pytorch_layers = []

        if self.is_conv:
            self.pytorch_layers.append(ConvolutionFeatureExtractor(channels=self.conv_arch_channels, spatial=self.conv_arch_spatial))

        # Initializing for input-output chaining across layers
        input_nodes_fc_network = arch['channels'][-1]

        in_feat = input_nodes_fc_network
        for out_feat in self.fc_arch:
            self.pytorch_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            self.pytorch_layers.append(nn.ReLU())
            in_feat = out_feat

        self.pytorch_layers = self.pytorch_layers[:-1]  # Removing the ReLU after the output layer
        self.pytorch_layers = nn.ModuleList(self.pytorch_layers)


    def forward(self, x):
        """

        :param x:
        :return:
        """

        # Conv feature extractor
        if self.is_conv:
            # Restoring heatmaps
            with torch.no_grad():
                conv_x = []
                border = 0

                for size in self.conv_arch_spatial:
                    conv_x.append(x[:, :, border: border + (size**2)].reshape(x.shape[0], x.shape[1], size, size))
                    border += (size**2)

            x = self.pytorch_layers[0](conv_x)
            # [1:] skips the ConvFeatExtract layer
            for layer in self.pytorch_layers[1:]:
                x = layer(x)

            return x

        # GAP feature extractor
        else:
            for layer in self.pytorch_layers:
                x = layer(x)
            return x


class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, channels, spatial):
        super(ConvolutionFeatureExtractor, self).__init__()

        self.hg_conv_feat_extract = []
        self.depth = len(channels)

        # Down from 64 to 4
        for i in range(self.depth-1):    # 32 --> 16, 16 --> 8, 8 --> 4
            self.hg_conv_feat_extract.append(torch.nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                                             kernel_size=(2, 2), stride=2, padding=0))
            self.hg_conv_feat_extract.append(torch.nn.ReLU())

        # Down from 4 to 1
        self.hg_conv_feat_extract.append(torch.nn.Conv2d(in_channels=channels[-1],
                                                         out_channels=channels[-1],
                                                         kernel_size=(spatial[-1], spatial[-1]), stride=1, padding=0))
        self.hg_conv_feat_extract.append(torch.nn.ReLU())
        self.hg_conv_feat_extract = nn.ModuleList(self.hg_conv_feat_extract)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x_ = x[0]
        for i in range(self.depth - 1):
            x_ = self.hg_conv_feat_extract[2 * i](x_)
            x_ = self.hg_conv_feat_extract[(2 * i) + 1](x_)
            x_ = x[i+1] + x_

        out = self.hg_conv_feat_extract[-2](x_).squeeze()
        out = self.hg_conv_feat_extract[-1](out)

        return out
