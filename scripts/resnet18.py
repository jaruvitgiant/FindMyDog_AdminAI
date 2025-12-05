import torch
import math
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
# from pytorch_metric_learning import losses
# from pytorch_metric_learning import miners
# import numpy as np
# import matplotlib.pyplot as plt

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from pytorch_metric_learning import distances, losses, miners, reducers, testers, trainers
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
# from pytorch_metric_learning.utils import logging_presets
# from pytorch_metric_learning.distances import LpDistance
# from pytorch_metric_learning.utils.inference import CustomKNN
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(512, 128)
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def extract_features(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
    
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
    
            out = self.avgpool(out)
            out = out.view(out.size(0), -1) 
            return out # นี่คือ 512-dim vector

    def forward(self, x):
            out = self.extract_features(x) # 512-dim
            out = self.embedding(out) # 128-dim
            # การ Normalize ทำที่นี่เพื่อให้ได้ Embedding Vector
            out = nn.functional.normalize(out, p=2, dim=1) 
            return out