import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.norm2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.norm3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.norm4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    
    def forward(self, x):
         x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm4(x)
        x = self.fc3(x)
        return x

def create_model():
    return ConvNet()
