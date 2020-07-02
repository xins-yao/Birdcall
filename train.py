import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils import data
from torchsummary import summary
from torchvision.models import resnet34

import librosa
from librosa.feature import melspectrogram

from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ================================================================= Loading
seed = 1011

train = pd.read_csv('./input/train.csv')
train = train[train['duration'] <= 30]

n_class = train['ebird_code'].nunique()
code2bird = dict(enumerate(train['ebird_code'].unique()))
bird2code = {v: k for k, v in code2bird.items()}


# ================================================================= Feature
class MelDataset(data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sr = 44100
        folder = r'D:\Workspace\Python\BirdCall\input\train_audio'

        ebird_code = self.df.loc[idx, 'ebird_code']
        filename = self.df.loc[idx, 'filename']
        path_audio = os.path.join(folder, ebird_code, filename)

        duration = self.df.loc[idx, 'duration']
        t_start = np.random.randint(0 ,duration, 1)[0]

        dur_clip = 5
        len_signal = dur_clip * sr
        signal, _ = librosa.load(path_audio, sr=sr, mono=True, offset=t_start, duration=dur_clip)
        signal = np.aapend(signal, np.zeros(len_signal - len(signal)))

        paras_mel = {
            'n_mel': 128,
            'fmin': 20,
            'fmax': sr // 2
        }
        mel_spec = melspectrogram(signal, sr=sr, **paras_mel)
        mel_spec = librosa.power_to_db(mel_spec)

        label = to_categorical(bird2code[ebird_code], n_class)

        return [mel_spec], label


train, valid = train_test_split(train, test_size=0.2, random_state=seed)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

train_set = MelDataset(train)
valid_set = MelDataset(valid)

train_loader = data.DataLoader(train_set, batch_size=16)
valid_loader = data.DataLoader(valid_set, batch_size=16)

# ================================================================= Metric
def cel(y_true, y_pred):
    y_true = torch.argmax(y_true, axis=-1)
    return nn.CrossEntropyLoss()(y_pred, y_true.squeeze())

def accuracy(y_true, y_pred):
    y_true = torch.argmax(y_true, axis=-1).squeeze()
    y_pred = torch.argmax(y_pred, axis=-1).squeeze()
    return (y_true == y_pred).float().sum()/len(y_true)


# ================================================================= Model
class BirdNet(nn.Module):
    def __init__(self, n_class):
        super(BirdNet, self).__init__()
        self.resnet = resnet34(pretrained=False)
        self.resnet_head = list(self.resnet.children())[:-1]
        self.resnet_head = nn.Sequential(*self.resnet_head)
        self.dense1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(1024, n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet_head(x)
        x = self.dense1(x.view(-1, 512))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


model = BirdNet(n_class)
optimizer = Adam(model.parameters(), lr=1e-3)

start = time.time()
print("STARTING TRAINING ...\n")

n_epoch = 20
for epoch in range(n_epoch):
    print('EPOCH {}'.format(epoch+1))

    batch = 1
    for x_trn_batch, y_trn_batch in train_loader:
        y_pred_batch = model.forward(x_trn_batch)
        loss_trn = cel(y_trn_batch, y_pred_batch)
        acc_trn = accuracy(y_trn_batch, y_pred_batch)

        optimizer.zero_grad()
        loss_trn.backward()

        optimizer.step()
        end = time.time()
        batch += 1
        is_print = batch % 100 == 1
        acc = np.round(acc_trn.item(), 3)
        if is_print:
            print('BATCH {}  Train Acc: {}  Time: {}'.format(batch, acc, end-start))

