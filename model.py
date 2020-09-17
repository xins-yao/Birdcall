import os
import gc
import cv2
import time
import math
import random
import logging
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils import data

# import librosa
# import librosa.display
# from librosa.core import spectrum

from glob import glob
from tqdm import tqdm_notebook
from datetime import datetime
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# === model block
class ConvBlockBase(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size, pool_stride):
        super(ConvBlockBase, self).__init__()
        assert out_channel % 3 == 0
        self.conv3x3 = nn.Conv2d(in_channel, out_channel // 3, 3, 1, 1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channel, out_channel // 3, 5, 1, 2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channel, out_channel // 3, 7, 1, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
      
    def forward(self, x):
        x_3 = self.conv3x3(x)
        x_5 = self.conv5x5(x)
        x_7 = self.conv7x7(x)
        x = F.relu_(self.bn(torch.cat([x_3, x_5, x_7], dim=1)))
        x = self.maxpool(x)
        return x

    
class ConvBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size, pool_stride, n_layer=2):
        super(ConvBlock2D, self).__init__()
        self.n_layer = n_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
      
    def forward(self, x):
        x = self.conv1(x)
        if self.n_layer > 1:
            x = self.conv2(x)
        x = self.maxpool(x)
        return x
    

class ConvBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
               
       
# Backbone1: CRNN_3x5x7    
class BirdCRNN(nn.Module):
    def __init__(self, n_class):
        super(BirdCRNN, self).__init__()    
        self.conv_base = ConvBlockBase(1, 48, 2, 2)        
        self.conv2d = nn.Sequential( 
            ConvBlock2D(48, 64, (2, 2), (2, 2)),
            ConvBlock2D(64, 96, (2, 2), (2, 2)),
            ConvBlock2D(96, 128, (2, 1), (2, 1)),
            ConvBlock2D(128, 256, (2, 1), (2, 1), n_layer=1)
        )
        self.conv1d = ConvBlock1D(1024, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, n_class)
        )   
                
    def forward(self, x):
        x = self.conv_base(x)
        x = self.conv2d(x)
        
        n_batch, n_channel, n_freq, n_time = x.shape
        x = torch.reshape(x, [n_batch, n_channel * n_freq, n_time])
        x = self.conv1d(x)
        
        # (batch, feature, time) -> (batch, time, feature)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        
        # global pooling
        x_avg = torch.mean(x, dim=1)
        x_max, _ = torch.max(x, dim=1)
        x = torch.cat([x_avg, x_max], dim=1)  
        
        x = self.fc(x)
        return x
       
       
# Backbone2: CNN_3x5x7_Encoder
class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_head=8):
        super(MultiHeadAttnBlock, self).__init__()
        assert out_channel % n_head == 0
        
        self.n_head = n_head
        self.d_head = out_channel // n_head
        self.scale = self.d_head ** 0.5
        
        self.q_fc = nn.Linear(in_channel, out_channel, bias=True)
        self.k_fc = nn.Linear(in_channel, out_channel, bias=True)
        self.v_fc = nn.Linear(in_channel, out_channel, bias=True)
        
        self.fc = nn.Sequential(
            nn.Linear(out_channel, out_channel, bias=True),
            nn.Dropout(p=0.2)
        )
        
        self.ln = nn.LayerNorm(out_channel)
      
    def forward(self, x):
        residual = x
        
        # (batch, time, d_head x n_head)
        q = self.q_fc(x)
        k = self.k_fc(x)
        v = self.v_fc(x)
        
        # (batch, time, d_head x n_head) -> (batch, time, n_head, d_head) -> (batch, n_head, time, d_head)
        n_batch, n_time = k.shape[0], k.shape[1]
        q = q.view(n_batch, n_time, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(n_batch, n_time, self.n_head, self.d_head).permute(0, 2, 1, 3)
        v = v.view(n_batch, n_time, self.n_head, self.d_head).permute(0, 2, 1, 3)
        
        weight = torch.softmax(torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale, dim=-1)
        x = torch.matmul(weight, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n_batch, n_time, -1)        
        
        x = self.ln(self.fc(x) + residual)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, n_head=8, ratio=4, n_layer=1):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.attn1 = MultiHeadAttnBlock(in_channel, out_channel, n_head)
        self.fc1 = nn.Sequential(
            nn.Linear(out_channel, out_channel*ratio, bias=True), nn.ReLU(inplace=True),
            nn.Linear(out_channel*ratio, out_channel, bias=True), nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.ln1 = nn.LayerNorm(out_channel)
      
    def forward(self, x):
        x = self.attn1(x)
        x = self.ln1(self.fc1(x) + x)
        return x
        
        
class BirdCNNEncoder(nn.Module):
    def __init__(self, n_class):
        super(BirdCNNEncoder, self).__init__()   
        self.conv_base = ConvBlockBase(1, 48, (2, 2), (2, 2))
        self.conv2d = nn.Sequential(
            ConvBlock2D(48, 64, (2, 2), (2, 2)),
            ConvBlock2D(64, 96, (2, 2), (2, 2)),
            ConvBlock2D(96, 128, (2, 1), (2, 1)),
            ConvBlock2D(128, 256, (2, 1), (2, 1))
        )
        self.conv1d = ConvBlock1D(1024, 512)
        self.encoder = Encoder(512, 512, n_head=8, n_layer=1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, n_class)
        )   
                
    def forward(self, x):
        x = self.conv_base(x)
        x = self.conv2d(x)
        
        n_batch, n_channel, n_freq, n_time = x.shape
        x = torch.reshape(x, [n_batch, n_channel * n_freq, n_time])
        x = self.conv1d(x)
        
        # (batch, feature, time) -> (batch, time, feature)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        # global pooling
        x_avg = torch.mean(x, dim=1)
        x_max, _ = torch.max(x, dim=1)
        x = torch.cat([x_avg, x_max], dim=1)     
        x = self.fc(x)
        return x


# Backbone3: CNN_SE
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.avgpool(x)
        attn = self.conv(attn)
        return x * attn
        

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size, pool_stride, is_pool=True):
        super(ConvBlock, self).__init__()
        self.is_pool = is_pool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            SqueezeExcitationBlock(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            SqueezeExcitationBlock(out_channel)
        )
        # pool over freq
        self.f_avgpool = nn.AvgPool2d(kernel_size=(pool_size, 1), stride=(pool_stride, 1))
        # pool over time
        self.t_maxpool = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
      
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.is_pool:
            x = self.t_maxpool(self.f_avgpool(x))           
        return x
    
    
class BirdCNNSE(nn.Module):
    def __init__(self, n_class):
        super(BirdConvNet, self).__init__()
        self.conv1 = ConvBlockBase(1, 48, 2, 2)
        self.conv2 = nn.Sequential(
            ConvBlock(48, 64, 2, 2),
            ConvBlock(64, 96, 2, 2),
            ConvBlock(96, 128, 2, 2),
            ConvBlock(128, 256, 0, 0, is_pool=False)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((None, 1)),
            nn.AdaptiveAvgPool2d((1, None))
        )        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(1024, n_class)
        )
        
    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        
        n_batch, n_channel, n_freq, n_time = x.shape
        x = torch.reshape(x, [n_batch, n_channel * 4, n_freq // 4, n_time])
        x = self.pool(x)
        
        x = self.fc(x.view(-1, 1024))
        return x
