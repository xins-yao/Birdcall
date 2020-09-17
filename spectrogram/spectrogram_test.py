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

sr = 32000
n_mels = 128
paras_mel_spectrogram = {
    'n_mels': n_mels,
    'fmin': 20,
    'fmax': sr // 2
}

path_files = glob('../input/birdclef-2020/gt/*.csv')
path_folder = '../input/birdclef-2020/audio'

test = pd.DataFrame(columns=['site', 'row_id', 'seconds', 'audio_id'])
for f in tqdm_notebook(path_files):
    audio_id = f.rsplit('/')[-1].split('.')[0]
    path_audio = os.path.join(path_folder, audio_id + '.wav')
    
    signal, _ = librosa.load(path_audio, sr=sr, res_type='kaiser_fast')
    duration = librosa.get_duration(signal, sr)
    
    # test metadata
    df = pd.DataFrame({
        'site': 'site_1',
        'seconds': list(range(5, int(duration + 5), 5)),
        'audio_id': audio_id,
        'row_id': ['site_1_{}_{}'.format(audio_id, s) for s in range(5, int(duration + 5), 5)]
    })
    
    # ground truth
    gt = pd.read_csv(f, names=['seconds', 'ebird_code'])
    gt['seconds'] = pd.to_timedelta(gt['seconds'].str[-8:].values).seconds
    gf_agg = gt.groupby('seconds')['ebird_code'].agg(lambda x: ' '.join(x.unique())).reset_index()
    df = pd.merge(df, gf_agg, how='left', on='seconds')
    
    test = pd.concat([test ,df], axis=0, ignore_index=True)
    
test['ebird_code'] = test['ebird_code'].fillna('nocall')
test = test[(test['audio_id'].str.startswith('SSW')) | (test['ebird_code'] == 'nocall')]
test.to_csv('test.csv', index=False)
test.shape


# feature extract
output = []
for i in tqdm_notebook(test.index):
    filename = test.loc[i, 'audio_id'] + '.wav'
    path_audio = os.path.join(path_folder, filename)

    t_start = test.loc[i, 'seconds'] - 5

    # pad to (sample_rate * duration)
    len_signal = 5 * sr
    try:
        signal, _ = librosa.load(path_audio,
                                 sr=sr,
                                 mono=True,
                                 offset=t_start,
                                 duration=5,
                                 res_type="kaiser_fast")
    except:
        signal = np.zeros(len_signal)
    signal = np.append(signal, np.zeros(len_signal - len(signal)))

    # mel spectrogram
    mel_spec = melspectrogram(signal, sr=sr, **paras_mel_spectrogram)
    mel_spec = librosa.power_to_db(mel_spec)
    mel_spec = to_image(mel_spec).astype(np.float32)

    output.append((mel_spec * 255.0).astype(np.uint8))
    
output = np.array(output)
np.save('test_image.npy', output)
