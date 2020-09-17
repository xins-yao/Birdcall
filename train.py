import os
import gc
import cv2
import time
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
from pytorch_lightning.metrics.sklearns import F1

warnings.filterwarnings("ignore")


# === logger
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


seed = 1011
set_seed(seed)

today = datetime.today()
now = today.strftime('%m%d-%H%M')

output_folder = './20200914_cnn_fold4'
os.makedirs(output_folder, exist_ok=True)

log_name = now + '.log'
logger = get_logger(os.path.join(output_folder, log_name))


# === loading
train = pd.read_csv('./input/train.csv')
train_extend  = pd.read_csv('./input/train_extended.csv')
train = pd.concat([train, train_extend], axis=0)

# corrupted file
file_corrupt = ['XC195038.mp3']
train = train[~train['filename'].isin(file_corrupt)].reset_index(drop=True)

# map
n_class = train['ebird_code'].nunique()
code2bird = dict(enumerate(train['ebird_code'].unique()))
code2bird[-1] = 'nocall'
bird2code = {v: k for k, v in code2bird.items()}
name2bird = dict(zip(train['sci_name'], train['ebird_code']))

# secondary label
def secondary_label_to_bird(x):
    labels = []
    for l in x:
        labels += l.strip().split('_')    
    labels = np.array(labels)
    labels = labels[np.isin(labels, list(name2bird.keys()))]
    return ' '.join([name2bird[l] for l in labels])

train['labels'] = train['secondary_labels'].str.replace('\'', '').str.strip('[]').str.split(',')
train['labels'] = train['labels'].map(secondary_label_to_bird)
train['labels'] = train['ebird_code'].str.cat(train['labels'], sep=' ').str.strip()


# === preprocess spectrogram
spec_image = np.concatenate(
    [np.load('../data/spectrogram/spectrogram{}.npy'.format(seg)) for seg in range(8)],
    axis=0
)
spec_image = np.expand_dims(spec_image, 1)


# === dataset split
# 80% sample from each bird group
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
target = np.array([bird2code[b] for b in train['ebird_code'].values])
idx_train, idx_valid = dict(enumerate(kfold.split(target, target, groups=target)))[4]

# train set
n_train = len(idx_train)

# label smoothing (set alpha->0 to disable)
alpha_smooth = 0.05
target_smooth = np.ones([len(target), n_class]) * alpha_smooth / n_class
for i in range(len(target)):
    target_smooth[i, target[i]] += 1 - alpha_smooth

# valid set
multi_target = np.tile(train['labels'].values, n_augment)
n_valid = len(idx_valid)
x_valid = spec_image[idx_valid]
y_valid = np.zeros([n_valid, n_class], dtype=np.uint8)
for i in range(n_valid):
    labels = multi_target[idx_valid[i]].split(' ')
    y_valid[i, [bird2code[b] for b in labels]] =  1


# === dataloader
# train set
class MelDataset(data.Dataset):
    def __init__(self, image, target, mix=True):
        self.image = image
        self.target = target
        self.mix = mix
        self.len = len(target)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.mix:
            mel_spec = self.image[idx]
            label = self.target[idx]
        else:
            idx_mix = np.random.randint(self.len)
            alpha = np.random.beta(8.0, 8.0)
            # alpha = np.random.uniform(0.3, 0.7)
            mel_spec = self.image[idx] * alpha + self.image[idx_mix] * (1 - alpha)
            label = self.target[idx] * alpha + self.target[idx_mix] * (1 - alpha)
        
        mel_spec = mel_spec / 255.0 
        return torch.FloatTensor(mel_spec.astype(np.float32)), torch.FloatTensor(label.astype(np.float32))
    
# valid set   
class ValDataset(data.Dataset):
    def __init__(self, image, target):
        self.image = image
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):       
        mel_spec = self.image[idx]
        label = self.target[idx]
        mel_spec = mel_spec / 255.0
        return torch.FloatTensor(mel_spec.astype(np.float32)), torch.FloatTensor(label.astype(np.float32))

valid_set = ValDataset(x_valid, y_valid)
valid_loader = data.DataLoader(valid_set, batch_size=64, num_workers=2)

# test set
class TestDataset(data.Dataset):
    def __init__(self, image, target):
        self.image = image
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):       
        mel_spec = self.image[idx]
        label = self.target[idx]
        mel_spec = mel_spec / 255.0
        return torch.FloatTensor(mel_spec.astype(np.float32)), torch.FloatTensor(label.astype(np.float32))
    
# shape: (batch, n_mel, n_timeframe)
test = pd.read_csv('../input/birdcall-test-image/test.csv')
x_test = np.load('../input/birdcall-test-image/test_image.npy')
x_test = np.expand_dims(x_test, 1)

y_test = np.zeros([len(test), n_class])
for i, birds in enumerate(test['ebird_code'].values):
    if birds != 'nocall':
        y_test[i, [bird2code[b] for b in birds.split(' ')] ] = 1
    
idx_positive = np.where(test['ebird_code'].values != 'nocall')[0]
idx_negative = np.where(test['ebird_code'].values == 'nocall')[0]

test_set = TestDataset(x_test, y_test)
test_loader = data.DataLoader(test_set, batch_size=64, num_workers=2)


# === loss function
transformer = 'Sigmoid'

# binary crossentorpy
def bce(y_true, y_pred, activation=transformer):
    if activation == 'Sigmoid':
        return nn.BCEWithLogitsLoss()(y_pred, y_true)
    elif activation == 'Linear':
        return nn.BCELoss()(y_pred, y_true)

# focal loss
def f1(y_true, y_pred, activation=transformer):
    threshold = 0.5
    if activation == 'Sigmoid':
        y_prob = nn.Sigmoid()(y_pred)
    elif activation == 'Linear':
        y_prob = y_pred
    return F1(average='samples')(y_prob > threshold, y_true)


# crossentropy loss
def cel(y_true, y_pred):
    y_t = torch.argmax(y_true, dim=1)
    return nn.CrossEntropyLoss()(y_pred, y_t)

loss_func = bce


# === optimizer
model = BirdCNNSE(n_class)
# model = BirdCRNN(n_class)
# model = BirdCNNEncoder(n_class)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, min_lr=1e-4, verbose=True)

# discontinous training
DISCONTINOUS = False
if DISCONTINOUS:
    state_dict = torch.load('../input/birdcall-crnn/model39.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
   
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
logger.info(model.to(device))


# === balance sampler
# up to 150 sample per bird
def bird_sampler(x, n_limit=150):
    if len(x) > n_limit:
        return x.sample(n_limit)
    else:
        return x
  
df_sampler = train.loc[idx_train, ['ebird_code', 'duration']]


# === training
start = time.time()
logger.info("STARTING TRAINING ...")

seed_list = glob('../input/birdcall-noisy-*')
seed_list = np.sort(seed_list)
logger.info(f'Dataset List\n{pd.Series(seed_list)}')

n_epoch = 100
for epoch in range(n_epoch):
    logger.info(f'EPOCH {epoch+1}')

    # random noise spectrogram
    noise_seed = seed_list[epoch % len(seed_list)]
    logger.info(f'Dataset {noise_seed}')
    spec_image = np.concatenate(
        [np.load(f'{noise_seed}/spectrogram{seg}.npy') for seg in range(8)],
        axis=0
    )
    spec_image = np.expand_dims(spec_image, 1)
    
    # balance sampler
    idx_train = df_sampler.groupby('ebird_code').apply(bird_sampler).reset_index(level=0, drop=True).index.values
    x_train = spec_image[idx_train]
    y_train = target_smooth[idx_train]
    train_set = MelDataset(x_train, y_train)
    train_loader = data.DataLoader(train_set, batch_size=360, shuffle=True, drop_last=True)
    
    # train
    batch = 0
    model.train()
    for x_trn, y_trn in train_loader:        
        # forward
        x_trn = x_trn.to(device)
        y_trn = y_trn.to(device)
        y_pred = model.forward(x_trn)
        
        # update
        loss_trn = loss_func(y_trn, y_pred)
        
        optimizer.zero_grad()
        loss_trn.backward()
        optimizer.step()
        scheduler.step()
        
        batch += 1
    end = time.time()
    logger.info('BATCH {}  Train Loss: {:.5f}  Time: {:.1f}'.format(batch, loss_trn.item(), end-start))
     
    
    # valid   
    model.eval()
    with torch.no_grad():  
        y_pred = []
        y_true = []
        for x_val, y_val in valid_loader:     
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_pred.append(model.forward(x_val))
            y_true.append(y_val)
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
                        
    end = time.time()
    
    loss_val = loss_func(y_true, y_pred).item()
    f1_val = f1(y_true, y_pred).item()
    
    logger.info('BATCH {}  Valid Loss: {:.5f}  Valid F1: {:.3f}  Time: {:.1f}'.format(batch, loss_val, f1_val, end-start))    
    # scheduler.step(loss_val)
    
    
    # test       
    model.eval()
    with torch.no_grad():  
        y_pred = []
        y_true = []
        for x_tst, y_tst in test_loader:     
            x_tst = x_tst.to(device)
            y_tst = y_tst.to(device)
            y_pred.append(model.forward(x_tst))
            y_true.append(y_tst)
            
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
     
    # loss
    loss_test = loss_func(y_true, y_pred).item()
    f1_test = f1(y_true, y_pred)
    
    loss_positive = loss_func(y_true[idx_positive], y_pred[idx_positive]).item()
    f1_positive = f1(y_true[idx_positive], y_pred[idx_positive]).item()
    
    loss_negative = loss_func(y_true[idx_negative], y_pred[idx_negative]).item()
    f1_negative = (nn.Sigmoid()(y_pred[idx_negative]) <= 0.4).all(dim=1).float().mean().item()
    
    logger.info('Loss: {:.5f}  Positive Loss: {:.5f}  Negative Loss: {:.5f}'.format(loss_test, loss_positive, loss_negative))    
    logger.info('F1  : {:.5f}  Positive F1  : {:.5f}  Negative F1  : {:.5f}'.format(f1_test, f1_positive, f1_negative))
    
    if epoch > 50:
        torch.save(model.state_dict(), f'model{epoch}.pt')
    gc.collect()
    
logger.info("ENDING TRAINING ...")
