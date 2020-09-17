import os
import gc
import cv2
import time
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess as sp
import multiprocess as mp

import warnings
warnings.filterwarnings('ignore')

import librosa
import librosa.display
from librosa.core import spectrum

from scipy.ndimage.morphology import binary_dilation, binary_erosion

from tqdm import tqdm_notebook


# === loading
seed = 1011
np.random.seed(seed)

train = pd.read_csv('../input/birdsong-recognition/train.csv')
train['folder'] = '../input/birdsong-recognition/train_audio'

# train_extend = pd.read_csv('../input/train-extended-form/train_extended.csv')
train_extend = pd.read_csv('../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv')
train_extend.loc[train_extend['ebird_code'] < 'n', 'folder'] = '../input/xeno-canto-bird-recordings-extended-a-m/A-M'
train_extend.loc[train_extend['ebird_code'] > 'n', 'folder'] = '../input/xeno-canto-bird-recordings-extended-n-z/N-Z'
train = pd.concat([train, train_extend], ignore_index=True)


# === paras
snr_threshold = 0.001
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


# === utils
def signal_noise_split(audio):
    S, _ = spectrum._spectrogram(y=audio, power=1.0, n_fft=2048, hop_length=512, win_length=2048)
    
    col_median = np.median(S, axis=0, keepdims=True)
    row_median = np.median(S, axis=1, keepdims=True)
    S[S < row_median * 3] = 0.0
    S[S < col_median * 3] = 0.0
    S[S > 0] = 1
    
    S = binary_erosion(S, structure=np.ones((4, 4)))
    S = binary_dilation(S, structure=np.ones((4, 4)))
    
    indicator = S.any(axis=0)
    indicator = binary_dilation(indicator, structure=np.ones(4), iterations=2)
    
    mask = np.repeat(indicator, hop_length)
    mask = binary_dilation(mask, structure=np.ones(win_length - hop_length), origin=-(win_length - hop_length)//2)
    mask = mask[:len(audio)]
    signal = audio[mask]
    noise = audio[~mask]
    return signal, noise
    
    
# === split audio
def worker(args):
    df, ebird_code = args
    os.makedirs(ebird_code, exist_ok=True)
    n_sample = len(df)
        
    print(f'processing {ebird_code}')
    for i, sample in df.iterrows():
        filename = sample['filename']
        title = filename.split('.')[0]
        path_folder = sample['folder']
        path_audio = os.path.join(path_folder, ebird_code, filename)
                
        try:
            signal, _ = librosa.load(path_audio, sr=sr, mono=True, res_type='kaiser_fast')
        except:
            print(f'file {filename} corrupted.')
            continue
        signal, noise = signal_noise_split(signal)
        
        # signal = signal[:dur_chunk * sr]
        # noise = noise[:dur_chunk * sr]
        
        sf.write(f'./{ebird_code}/{title}.wav', signal, sr)
        sf.write(f'./noise/{title}.wav', noise, sr)
        
    gc.collect()
    

# === multi-process
os.makedirs('noise', exist_ok=True)

n_thread = mp.cpu_count()
pool = mp.Pool(n_thread)

birds = train['ebird_code'].unique()
# birds = birds[(birds > 'a') & (birds < 'c')]
args = [ [train[train['ebird_code'] == b], b] for b in birds ]
pool.map(worker, args)

pool.close()
pool.join()


# === output
!zip -q -r noise.zip noise
!rm -r ./noise
