'''
Baseline Architecture: Stacked Hourglass
https://github.com/princeton-vl/pytorch_stacked_hourglass
'''
import torch
from torch import nn
from .layers import Conv, Hourglass, Pool, Residual


class Merge(nn.Module):
    '''

    '''
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, arch, auxnet, intermediate_features):#, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        '''

        :param nstack: (int) Number of stacks
        :param inp_dim: (int) Number of input channels for the Stacked Hourglass
        :param oup_dim: (int) Number of output channels for the Stacked Hourglass
        :param bn: (bool) Whether to perform Batch Normalization
        :param increase:
        :param kwargs:
        '''
        super(PoseNet, self).__init__()

        self.auxnet = auxnet   # Whether to compute features for auxnet
        self.intermediate_features = intermediate_features  # Whether extractor is conv or avg
        self.nstack = arch['nstack']
        inp_dim = arch['channels']
        oup_dim = arch['num_hm']

        if torch.cuda.device_count() > 1:
            # We don't need 8 GPUs for 3 stacks lol.
            n_gpus = min(torch.cuda.device_count(), self.nstack)
            stacks_per_gpu = torch.zeros(size=(n_gpus,), dtype=torch.int16)
            # 1. Equal allocation to all
            stacks_per_gpu += (self.nstack // n_gpus)
            # 2. Distribute the remaining (max 1) among all the GPUs
            temp_tensor = torch.zeros(size=(n_gpus,), dtype=torch.int16)
            for i in range(self.nstack % n_gpus):
                temp_tensor[i] = 1
            stacks_per_gpu += temp_tensor

            cuda_devices = []
            for i in range(stacks_per_gpu.shape[0]):
                for _ in range(stacks_per_gpu[i]):
                    cuda_devices.append(torch.device('cuda:{}'.format(i)))

        else:
            cuda_devices = [torch.device('cuda:0')] * self.nstack

        self.cuda_devices = cuda_devices


        self.pre = nn.Sequential(
            Conv(inp_dim=3, out_dim=64, kernel_size=7, stride=2, bn=True, relu=True),
            Residual(inp_dim=64, out_dim=128),
            Pool(2, 2),
            Residual(inp_dim=128, out_dim=128),
            Residual(inp_dim=128, out_dim=inp_dim)).cuda(cuda_devices[0])
        
        self.hgs = nn.ModuleList(
            [nn.Sequential(Hourglass(n=4, f=inp_dim, bn=False, increase=0,
                                     intermediate_features=intermediate_features, auxnet=auxnet)
                           ).cuda(cuda_devices[i]) for i in range(self.nstack)])
        
        self.features = nn.ModuleList([nn.Sequential(Residual(inp_dim, inp_dim),
                                                     Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
                                                     ).cuda(cuda_devices[i]) for i in range(self.nstack)])
        
        self.outs = nn.ModuleList(
            [Conv(inp_dim=inp_dim, out_dim=oup_dim, kernel_size=1, relu=False, bn=False).cuda(cuda_devices[i])
             for i in range(self.nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim).cuda(cuda_devices[i]) for i in range(self.nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim).cuda(cuda_devices[i]) for i in range(self.nstack-1)])

        self.global_avg_pool = nn.ModuleList([nn.AvgPool2d(kernel_size=(64, 64), stride=1).cuda(cuda_devices[i])
                                              for i in range(self.nstack)])

    def forward(self, imgs):
        '''
        Constructing the Stacked Hourglass Posenet Model
        :param imgs:
        :return:
        '''
        # x is of shape: (BatchSize, #channels == 3, input_dim1, input_dim2)
        x = imgs.permute(0, 3, 1, 2).cuda(self.cuda_devices[0])
        x = self.pre(x)
        combined_hm_preds = []
        hourglass_dict= {}

        for i in range(self.nstack):
            x = x.cuda(self.cuda_devices[i])
            hourglass_dict = self.hgs[i](x)
            x = hourglass_dict['out']

            # Hourglass parameters
            if self.intermediate_features == 'conv' and self.auxnet:
                hourglass_dict['feature_5'] = x.clone().detach().to(
                    'cuda:{}'.format(torch.cuda.device_count() - 1))

            x = self.features[i](x)

            hourglass_dict['penultimate'] = self.global_avg_pool[i](x).clone().detach().to(
                'cuda:{}'.format(torch.cuda.device_count() - 1)).reshape(x.shape[0], -1)

            preds = self.outs[i](x)
            combined_hm_preds.append(preds.cuda(self.cuda_devices[-1]))
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](x)

        del hourglass_dict['out']
        # ip_learn_loss_dict is a dictionary containing intermediate outputs of hourglass
        return torch.stack(combined_hm_preds, 1), hourglass_dict
