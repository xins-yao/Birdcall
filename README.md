Cornell Birdcall Identification 
https://www.kaggle.com/c/birdsong-recognition

Feature Engineering
- Log MEL Spectrogram
'''sr = 32000
fmin = 20
fmax = sr // 2

n_channel = 128
n_fft = 2048
hop_length = 512
win_length = n_fft
