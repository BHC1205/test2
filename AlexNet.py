import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


net=nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(96,256)
)