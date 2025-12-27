import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time
import sys
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

# VGG-16 architecture adapted for segmentation
# based on VGG model in demo code
class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.3, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return x


def vgg16_layer(input_channels, out_channels, num, dropout=[0.3, 0.4]):
    layers = []
    for i in range(num):
        layers.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
        input_channels = out_channels
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class ModifiedVGG16Segmentation(nn.Module):
    
    def __init__(self, in_channels=8, num_classes=7):
        super(ModifiedVGG16Segmentation, self).__init__()
        
        # Encoder - using demo code VGG architecture
        self.enc1 = vgg16_layer(in_channels, 16, 2)
        self.enc2 = vgg16_layer(16, 32, 2)
        self.enc3 = vgg16_layer(32, 64, 2)
        self.enc4 = vgg16_layer(64, 128, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            vgg16_conv_block(128, 256, drop=True),
            vgg16_conv_block(256, 256, drop=True)
        )
        
        # Decoder - upsampling path for segmentation
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            vgg16_conv_block(128, 128, drop=True),
            vgg16_conv_block(128, 128, drop=False)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            vgg16_conv_block(64, 64, drop=True),
            vgg16_conv_block(64, 64, drop=False)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            vgg16_conv_block(32, 32, drop=True),
            vgg16_conv_block(32, 32, drop=False)
        )
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            vgg16_conv_block(16, 16, drop=False),
            vgg16_conv_block(16, 16, drop=False)
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)    # 16 channels
        x2 = self.enc2(x1)   # 32 channels
        x3 = self.enc3(x2)   # 64 channels
        x4 = self.enc4(x3)   # 128 channels
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(x)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = self.dec4(x)
        
        # Classification
        x = self.classifier(x)
        
        return x