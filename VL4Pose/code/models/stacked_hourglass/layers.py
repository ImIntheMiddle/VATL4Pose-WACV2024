import torch
from torch import nn

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    '''
    Initializes: Conv, Conv-Relu or Conv-Relu-BN combinaton
    '''
    def __init__(self, inp_dim, out_dim, kernel_size, stride=1, bn=False, relu=True):
        '''
        :param inp_dim: (int) Number of input channels
        :param out_dim: (int) Number of output channels
        :param kernel_size: (int) Kernel size
        :param stride: (int) Convolution stride
        :param bn: (bool) Whether to perform Batch Normalization
        :param relu: (bool) Whether to perform ReLU
        '''
        assert type(inp_dim) == type(out_dim) == type(kernel_size) == type(stride) == int, "[Conv]: Wrong typing"
        assert type(bn) == type(relu) == bool, "[Conv]: Wrong typing"

        super(Conv, self).__init__()

        self.bn = None
        self.relu = None
        self.inp_dim = inp_dim
        # Input spatial dim is same as output spatial dim for stride == 1
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)

        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(num_features=out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "Passed: {}\tExpected: {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)

        return x


class Residual(nn.Module):
    '''

    '''
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=inp_dim)
        self.conv1 = Conv(inp_dim=inp_dim, out_dim=int(out_dim/2), kernel_size=1, relu=False)
        self.bn2 = nn.BatchNorm2d(num_features=int(out_dim/2))
        self.conv2 = Conv(inp_dim=int(out_dim/2), out_dim=int(out_dim/2), kernel_size=3, relu=False)
        self.bn3 = nn.BatchNorm2d(num_features=int(out_dim/2))
        self.conv3 = Conv(inp_dim=int(out_dim/2), out_dim=out_dim, kernel_size=1, relu=False)
        self.skip_layer = Conv(inp_dim=inp_dim, out_dim=out_dim, kernel_size=1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Hourglass(nn.Module):
    '''

    '''
    def __init__(self, n, f, bn=None, increase=0, intermediate_features=None, auxnet=None):
        super(Hourglass, self).__init__()

        self.auxnet = auxnet
        self.intermediate_features = intermediate_features

        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n=n-1, f=nf, bn=bn, increase=0,
                                  intermediate_features=intermediate_features, auxnet=auxnet)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.avg_pool = nn.AvgPool2d(kernel_size=2 ** (self.n + 1), stride=1)

    def forward(self, x):
        upper_1 = self.up1(x)
        x = self.pool1(x)
        x = self.low1(x)

        if self.n > 1:
            hourglass_dict = self.low2(x)
            x = hourglass_dict['out']
        else:
            x = self.low2(x)
        x = self.low3(x)

        if self.intermediate_features == 'conv' and self.auxnet:
            hourglass_feature_map = x.clone().detach().to('cuda:{}'.format(torch.cuda.device_count() - 1))

        upper_2 = self.up2(x)

        if self.n > 1:

            hourglass_dict['out'] = upper_1 + upper_2
            if self.intermediate_features == 'conv' and self.auxnet:
                hourglass_dict['feature_{}'.format(self.n)] = hourglass_feature_map

            return hourglass_dict
        else:
            if self.intermediate_features == 'conv' and self.auxnet:
                return {'out': upper_1 + upper_2, 'feature_1': hourglass_feature_map}
            else:
                return {'out': upper_1 + upper_2}