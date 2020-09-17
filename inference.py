import os
import gc
import cv2
import time
import random

import numpy as np
import pandas as pd
import multiprocess as mp

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

import librosa
from librosa.core import spectrum

from scipy import stats
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import multiprocess as mp
n_thread = mp.cpu_count()

import warnings
warnings.filterwarnings("ignore")


# === utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
 
def to_image(x, eps=1e-6):
    mean = x.mean()
    x = x - mean
    std = x.std()
    x_std = x / (std + eps)
    norm_min, norm_max = x_std.min(), x_std.max()
    if norm_max - norm_min > eps:
        x_norm = (x_std - norm_min) / (norm_max - norm_min)
    else:
        x_norm = np.zeros_like(x_std)
    return np.expand_dims(x_norm, 0)


def transformer(x, activation='Sigmoid'):
    if activation == 'Sigmoid':
        output = nn.Sigmoid()(x)
    elif activation == 'Softmax':
        output = nn.Softmax(dim=1)(x)
    elif activation == 'Linear':
        output = x
    return output.detach().cpu().numpy()
    

# === loading
seed = 1011
set_seed(seed)

train = pd.read_csv('../input/birdsong-recognition/train.csv', usecols=['ebird_code'])

n_class = train['ebird_code'].nunique()
code2bird = dict(enumerate(train['ebird_code'].unique()))
code2bird[-1] = 'nocall'
bird2code = {v: k for k, v in code2bird.items()}


# === paras
sr = 32000
fmin = 20
n_channel = 128
n_fft = 2048
hop_length = 512
win_length = n_fft

paras_mel_filterbank = {
    'n_mels': n_channel,
    'fmin': fmin,
    'fmax': sr // 2,
    'n_fft': n_fft
}

paras_mel_spectrogram = {
    'n_fft': n_fft,
    'hop_length': hop_length,
    'win_length': win_length
}

dur_frame = 5
len_frame = dur_frame * sr
n_frame = len_frame // hop_length + 1


# === mel spectrogram
mel_filterbank = librosa.filters.mel(sr, **paras_mel_filterbank)

def melspectrogram(y, sr, filterbank, n_fft, hop_length, win_length):
    S, _ = spectrum._spectrogram(y=y, power=2.0, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.dot(filterbank, S)
    

# === model
model_list = []

# model1
pretrained_list = [    
    '../input/crnn-3x5x7-5fold/CRNN_3x5x7_Fold0_model105.pt', 
    '../input/crnn-3x5x7-5fold/CRNN_3x5x7_Fold1_model103.pt',
    '../input/crnn-3x5x7-5fold/CRNN_3x5x7_Fold2_model94.pt',
    '../input/crnn-3x5x7-5fold/CRNN_3x5x7_Fold3_model105.pt',
    '../input/crnn-3x5x7-5fold/CRNN_3x5x7_Fold4_model103.pt',
]

for path_pt in pretrained_list:
    model = BirdCRNN(n_class)
    
    state_dict = torch.load(path_pt, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model_list.append(model)
    
# model2 ... 
# model3 ... 
    

# === load test
TEST = os.path.exists('../input/birdsong-recognition/test_audio')

if TEST:
    test = pd.read_csv('../input/birdsong-recognition/test.csv')
    path_folder = '../input/birdsong-recognition/test_audio'
else:
    test = pd.read_csv('../input/birdcall-check/test.csv')
    path_folder = '../input/birdcall-check/test_audio'  
    
   
# === feature extract
def worker(args):
    df, site = args
    
    files = df['audio_id'].unique()
    spec_image = []
    audio_id = []
    
    if site == 'site_1' or site == 'site_2':   
        for f in files:
            path_audio = os.path.join(path_folder, f + '.mp3')
            df = test[test['audio_id'] == f]
            
            # duration = df['seconds'].max()            
            n_chunk = int(np.ceil(df['seconds'].max() / dur_frame))
            len_expect = n_chunk * n_frame * hop_length - 1
            duration = len_expect / sr
            
            signal, _ = librosa.load(path_audio, sr, duration=duration, mono=True, res_type="kaiser_fast")

            delta = len_expect - len(signal)
            if delta > 0:
                signal = np.append(signal, np.zeros(delta))
            
            mel_spec = melspectrogram(signal, sr, mel_filterbank, **paras_mel_spectrogram)
            mel_spec = librosa.power_to_db(mel_spec)

            for i, sample in df.iterrows():
                i_chunk = int(np.ceil(sample['seconds'] / 5)) - 1
                i_start = i_chunk * n_frame
                spec_slice = mel_spec[:, i_start: i_start + n_frame]
                spec_slice = to_image(spec_slice)  

                spec_image.append(spec_slice)
                audio_id.append(f)                
                
    elif site == 'site_3':
        for f in files:
            path_audio = os.path.join(path_folder, f + '.mp3')    
            signal, _ = librosa.load(path_audio, sr, mono=True, res_type="kaiser_fast")
            
            duration = librosa.get_duration(signal, sr)
            n_chunk = int(duration // 5)
            len_expect = n_chunk * n_frame * hop_length - 1
            delta = len_expect - len(signal)
            if delta > 0:
                signal = np.append(signal, np.zeros(delta))
            
            mel_spec = melspectrogram(signal, sr, mel_filterbank, **paras_mel_spectrogram)
            mel_spec = librosa.power_to_db(mel_spec)
            
            for i in range(n_chunk):
                i_start = i * n_frame
                spec_slice = mel_spec[:, i_start: i_start + n_frame]
                spec_slice = to_image(spec_slice)
                
                spec_image.append(spec_slice)
                audio_id.append(f)
    
    print('Load {} done'.format(site) )
    return spec_image, audio_id
    

# === multi-process
n_thread = mp.cpu_count()
pool = mp.Pool(n_thread)

args = [ 
    [test[test['site'] == 'site_1'], 'site_1'],
    [test[test['site'] == 'site_2'], 'site_2'],
    [test[test['site'] == 'site_3'], 'site_3'],
]
results = pool.map(worker, args)

pool.close()
pool.join()

spec_image = np.array(results[0][0] + results[1][0])
audio_list = np.array(results[0][1] + results[1][1])

spec_image3 = np.array(results[2][0])
audio_list3 = results[2][1]


# === dataloader    
class TestDataset(data.Dataset):
    def __init__(self, image):
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):       
        mel_spec = self.image[idx]        
        return torch.FloatTensor(mel_spec.astype(np.float32))

test_set = TestDataset(spec_image)
test_loader = data.DataLoader(test_set, batch_size=32)

test_set3 = TestDataset(spec_image3)
test_loader3 = data.DataLoader(test_set3, batch_size=32)


# === inference
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# prediction for site_1 and site_2
n_model = len(model_list)
weight = np.ones(n_model) / n_model
y_average = np.zeros([len(spec_image), n_class])
y_average3 = np.zeros([len(spec_image3), n_class])
for i, model in enumerate(model_list):
    if weight[i] == 0:
        continue
    model.to(device)
    model.eval()
    y_output = []
    y_output3 = []
    with torch.no_grad():
        for x_test in tqdm_notebook(test_loader):
            y_pred = model.forward(x_test.to(device))
            y_output.extend(transformer(y_pred))

        for x_test in tqdm_notebook(test_loader3):
            y_pred = model.forward(x_test.to(device))
            y_output3.extend(transformer(y_pred))

    y_output = np.array(y_output)
    y_output3 = np.array(y_output3) 
    
    y_average += weight[i] * y_output
    y_average3 += weight[i] * y_output3
    

y_preds = np.zeros([len(test), n_class])
# site_1, site_2
y_preds[:len(y_average), :] = y_average   
# site_3
for i, audio_id in enumerate(np.unique(audio_list3)):
    y_preds[i + len(y_average), :] = np.max(y_average3[np.isin(audio_list3, audio_id)], axis=0)
    
 
# === post-process
thr_dict = {
    'high': 0.5,
    'median': 0.3,
    'low' : 0.1,
    'bottom': 0.05
}
threshold = np.ones_like(y_preds) * thr_dict['median']
# high threshold for site_3
# threshold[len(y_output):] = thr_dict['high']

#  dynamic threshold
DYNAMIC = True
if DYNAMIC:
    thr_matrix = np.ones_like(y_output) * thr_dict['median']
    for i, audio_id in enumerate(np.unique(audio_list)):
        idx = np.isin(audio_list, audio_id)
        is_confident = np.sum(y_output[idx] > thr_dict['high'], axis=0).astype(bool)
        thr_slice = thr_matrix[idx]
        thr_slice[:, is_confident] = thr_dict['low']
        
        code_confident = np.where(is_confident)[0]
        for rank in range(10):
            col_max = np.argsort(y_output[idx], axis=1)[:, -rank-1]
            row_max = np.isin(col_max, code_confident)
            thr_slice[row_max, col_max[row_max]] = thr_dict['bottom'] * (1 - rank / 10)
        
        thr_matrix[idx] = thr_slice
    threshold[:len(y_output)] = thr_matrix
    
# map probability to class
y_class = dict( enumerate([[-1]] * len(y_preds)) )
rows, values = np.where(y_preds > threshold)
for r in np.unique(rows):
    y_class[r] = values[np.where(rows == r)]
    
y_class = list(y_class.values())
y_class = [[code2bird[c] for c in c_list] for c_list in y_class]


# === output
test['birds'] = [' '.join(c_list) for c_list in y_class]
submission = test[['row_id', 'birds']]
submission.to_csv('submission.csv', index=False)
for i, model in enumerate(model_list):
    torch.save(model.cpu().state_dict(), f'model{i}.pt')

submission
