import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
criterionl1 = torch.nn.L1Loss()
num_workers = 0
cel = nn.CrossEntropyLoss()
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#    "cpu")
device = torch.device('cuda')
normalize=transforms.Normalize(mean=[0.485, 0.456,
                                  0.406],std=[0.229, 0.224,
                                 0.225])

cft_transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
batch_size = 1
img_size = 256

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std



norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])