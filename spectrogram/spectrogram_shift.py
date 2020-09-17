# === prerequisite
!apt-get -y install libsox-fmt-all
!apt-get -y install sox
!pip install git+https://github.com/rabitt/pysox.git
import sox


# === lib
import os
import gc
import cv2
import time
import numpy as np
import pandas as pd
import subprocess as sp
import multiprocess as mp

import warnings
warnings.filterwarnings('ignore')

import librosa
from librosa.core import spectrum

from tqdm import tqdm_notebook


# === loading 
seed = 1011
np.random.seed(seed)

train = pd.read_csv('../input/birdsong-recognition/train.csv')
train['folder'] = '../input/birdsong-recognition/train_audio'

train_extend = pd.read_csv('../input/train-extended-form/train_extended.csv')
train_extend.loc[train_extend['ebird_code'] < 'n', 'folder'] = '../input/xeno-canto-bird-recordings-extended-a-m/A-M'
train_extend.loc[train_extend['ebird_code'] > 'n', 'folder'] = '../input/xeno-canto-bird-recordings-extended-n-z/N-Z'
train = pd.concat([train, train_extend], ignore_index=True)

train.loc[train['filename'] == 'XC534249.mp3', 'ebird_code'] = 'casvir'
train.loc[train['filename'] == 'XC534249.mp3', 'folder'] = '../input/xeno-canto-bird-recordings-extended-a-m/A-M'


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
mel_filterbank = librosa.filters.mel(sr, **paras_mel_filterbank)


def melspectrogram(y, sr, filterbank, n_fft, hop_length, win_length):
    S, _ = spectrum._spectrogram(y=y, power=2.0, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.dot(filterbank, S)


def to_image(x, eps=1e-6):
    mean = x.mean()
    x = x - mean
    std = x.std()
    x_std = x / (std + eps)
    norm_min, norm_max = x_std.min(), x_std.max()
    x_norm = (x_std - norm_min) / (norm_max - norm_min)
    return x_norm


def signal_noise_ratio(spec):
    # Get working copy
    spec = spec.copy()
    
    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)

    spec[spec < row_median * 1.25] = 0.0
    spec[spec < col_median * 1.15] = 0.0
    spec[spec > 0] = 1.0
    
    spec = cv2.medianBlur(spec, 3)
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))
    
    spec_sum = spec.sum()
    try:
        snr = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
    except:
        snr = spec_sum / (spec.shape[0] * spec.shape[1])

    return snr
    

# === feature extraction
def worker(args):
    df, seg = args
    n_sample = len(df)
    
    # transformer
    tfm = sox.Transformer()
    # pitch-up / pitch-down
    # tfm.pitch(10)
    # treble-up / treble-down
    # tfm.treble(20)
    # bass-up / bass-down
    # tfm.bass(20)
    # fade-in / fade-out
    # tfm.fade(fade_in_len=2, fade_out_len=2)
   
    output = []
    output_idx = []
    for i, sample in df.iterrows():
        filename, ebird_code, duration = sample[['filename', 'ebird_code', 'duration']]  
        path_folder = sample['folder']
        path_audio = os.path.join(path_folder, ebird_code, filename)
                
        try:
            signal, _ = librosa.load(path_audio, sr=sr, mono=True, res_type='kaiser_fast')
        except:
            print('file {} corrupted.'.format(filename))
            continue
        signal = librosa.effects.trim(signal)[0]
        signal = tfm.build_array(input_array=signal, sample_rate_in=sr)
        len_signal = len(signal)

        max_attemp = 100
        cnt_attemp = 0
        max_snr = -1
        tmp_spec = None
        tmp_idx = None
        while cnt_attemp < max_attemp:
            cnt_attemp += 1
            
            chunk = np.zeros(len_frame)
            if len_signal > len_frame:
                i_start = np.random.randint(len_signal - len_frame)
                chunk[:] = signal[i_start: i_start + len_frame]
            elif len_signal < len_frame:
                i_start = np.random.randint(len_frame - len_signal)
                chunk[i_start: i_start + len_signal] = signal
            else:
                chunk[:] = signal

            mel_spec = melspectrogram(chunk, sr, mel_filterbank, **paras_mel_spectrogram)
            mel_spec = librosa.power_to_db(mel_spec)
            mel_spec = to_image(mel_spec)
                    
            snr = signal_noise_ratio(mel_spec)
            if (snr > snr_threshold) & (cnt_attemp < max_attemp):
                output.append((mel_spec * 255).astype(np.uint8))
                break
            elif snr > max_snr:
                tmp_spec = (mel_spec * 255).astype(np.uint8)
                max_snr = snr
        
        if cnt_attemp == max_attemp:
            output.append(tmp_spec)
            print('file {} reach max attemp'.format(filename))
            
        gc.collect()

    output = np.array(output)
    np.save('spectrogram{}.npy'.format(seg), output)    
    
    print('segment {} complete'.format(seg))
    
 
# === multi-process
n_thread = mp.cpu_count()
pool = mp.Pool(n_thread)

n_seg = 8
step = len(train) // n_seg
rows = [i*step for i in range(n_seg)] + [len(train)]
args = [ [train[rows[i]:rows[i+1]], i] for i in range(n_seg) ]

pool.map(worker, args)

pool.close()
pool.join()
