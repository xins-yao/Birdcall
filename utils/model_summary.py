!pip install torchsummary
from torchsummary import summary


# === lib
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
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data

# import librosa
# import librosa.display
# from librosa.core import spectrum

from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# === model
def init_layer(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_uniform_(layer.weight)
        layer.weight.data.fill_(1 / layer.weight.shape[1])
    elif classname.find('Linear') != -1:
        # nn.init.xavier_uniform_(layer.weight)
        nn.init.eye_(layer.weight)


class Stacker(nn.Module):
    def __init__(self, in_channel, n_class):
        super(Stacker, self).__init__()

        self.fc1 = nn.Conv1d(in_channel, n_class, 1, bias=False)
        self.fc2 = nn.Linear(n_class, n_class, bias=False)

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        x = torch.diagonal(self.fc1(x), dim1=-2, dim2=-1)
        x = self.fc2(x)
        return x


# summary
model = Stacker(6, 264)
summary(model, (6, 264), device='cpu')
