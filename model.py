import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from HILL import HILL_costfunction
from cfg import *

class HILLCOST(nn.Module):
    def __init__(self, img_size=img_size):
        super(HILLCOST, self).__init__()
        self.img_size = img_size
        self.HILL = HILL_costfunction.HILL(img_size=self.img_size)

    def forward(self, cover):
        costvalue = self.HILL(cover)
        return costvalue


class HILLSTC(nn.Module):
    def __init__(self, img_size=img_size):
        super(HILLSTC, self).__init__()
        self.img_size = img_size
        self.HILL = HILL_costfunction.HILL(img_size=self.img_size)
        self.STC = stc_simulation.stc(img_size=self.img_size)

    def forward(self, cover, payload, num_pixel, randChange):
        costvalue = self.HILL(cover)
        rhoP1 = costvalue
        rhoM1 = costvalue
        stego_noise = self.STC(cover, rhoP1, rhoM1, payload, num_pixel, randChange)
        return (stego_noise + cover)


SRM_npy = np.load('SRM_Kernels.npy')


class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()
        self.criterion = torch.nn.L1Loss()
        self.Padding = torch.nn.ReplicationPad2d(2)

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        t1 = self.Padding(input)
        t2 = F.conv2d(t1, self.weight, self.bias, \
                      self.stride, self.padding, self.dilation, \
                      self.groups)
        t2 = t2.permute(1, 0, 2, 3)
        return t2

class HPF_SRM(nn.Module):
    def __init__(self):
        super(HPF_SRM, self).__init__()
        self.SRM = torch.nn.Conv2d(1, 30, 5, 1, 0, bias=False)
        self.Weight = torch.from_numpy(SRM_npy)
        self.SRM.weight.data = self.Weight
        self.SRM.weight.requires_grad = False
        self.Padding = torch.nn.ReplicationPad2d(2)
    def forward(self, input):
        t1 = self.SRM(self.Padding(input))
        return t1.permute(1, 0, 2, 3)


class HPF(torch.nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        self.preprocessing = torch.nn.Conv2d(1, 1, 5, 1, 2, bias=False)
        self.preprocessing.weight.data = torch.tensor(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            dtype=torch.float).view(1, 1, 5, 5)
        self.preprocessing.weight.requires_grad = False
        self.cost = HILLCOST()

    def forward(self, input):
        t1 = self.preprocessing(input)
        return t1

if __name__=='__main__':
    net1 = HPF_SRM()
    net2 = SRM_conv2d()
    input = torch.rand(1,1,128,128)
    out1 = net1(input)
    out2 = net2(input)
    pass
