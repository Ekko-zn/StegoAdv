import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import  os
from cfg import *
import sys
img_size = 256
criterionl1 = torch.nn.L1Loss()
SRM_npy = np.load('SRM_Kernels.npy')
class HILL(nn.Module):
    def __init__(self,img_size):
        super(HILL,self).__init__()
        self.img_size = img_size
        self.pad_3 = nn.ReplicationPad2d(3)
        self.pad = nn.ReplicationPad2d(7)
        self.conv1 = nn.Conv2d(1,1,3,1,padding=1,bias=False)
        self.avepool1 = nn.AvgPool2d(3,stride = 1,padding=1)
        self.avepool2 = nn.AvgPool2d(15,stride = 1)
        self.eps = 1e-10
        self.res()
    def res(self):
        self.conv1.weight.data = torch.tensor([[-1,2,-1],[2,-4,2],[-1,2,-1]],dtype = torch.float).view(1,1,3,3)
    def forward(self, x):
        t1 = self.pad_3(x)
        t2 = self.conv1(t1)
        t3 = self.avepool1(torch.abs(t2))
        t4 = 1 / (t3[:,:,3:self.img_size+3,3:self.img_size+3]+self.eps)
        t5 = self.avepool2(self.pad(t4))
        return t5

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
        return t1.permute(0, 1, 2, 3)

HILLCOST = HILL(img_size=img_size).to(device)
HPF = HPF_SRM().to(device)

def HPFLOSS(img,target):
    img = img.view(-1,1,img_size,img_size)
    target = target.view(-1,1,img_size,img_size)
    HPFimg = HPF(img).view(-1,1,img_size,img_size)
    HPFtarget = HPF(target).detach().view(-1,1,img_size,img_size)
    cost = HILLCOST(HPFtarget)
    cost[cost > 1] = 1
    cost = cost.detach()
    loss = criterionl1(cost * HPFimg, cost * HPFtarget)
    return loss



if __name__ == '__main__':
    t1 = torch.rand(2,3,256,256)
    t1.requires_grad = True
    t2 = torch.rand(2,3,256,256)
    m = HPFLOSS(t1,t2)
    m.backward()
    pass