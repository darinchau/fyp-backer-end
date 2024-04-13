import os
import numpy as np
import sys
import base64
import json

import tensorflow as tf
from librosa.core import stft
from scipy.signal.windows import hann
from contextlib import contextmanager
import random
import warnings
import librosa
import torch
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

import os
from .DilatedTransformer import Demixed_DilatedTransformerModel

PARAM_PATH = os.path.join(__file__, "../checkpoint/model.pt")

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # My GPU is not good enough to run this model

import random
import tempfile

# Audio separator
separator = Separator('spleeter:5stems')

# The audio model
model = Demixed_DilatedTransformerModel(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True)

#Initialize DBN Beat Tracker to locate beats from beat activation
beat_tracker = DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=44100/1024, transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2)

#Initialize DBN Downbeat Tracker to locate downbeats from downbeat activation
downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=44100/1024, transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2)

# Performs STFT with the deezer nutzer separator
def separator_stft(separator, data: np.ndarray):
    data = np.asfortranarray(data)
    N = separator._params["frame_length"]
    H = separator._params["frame_step"]
    win = hann(N, sym=False)
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
        s = stft(d, hop_length=H, window=win, center=False, n_fft = N)
        s = np.expand_dims(s.T, 2)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2)

# Split the audio into 5 stems
def split(audio_dir: str):
    audio = AudioAdapter.default()
    waveform, _ = audio.load(audio_dir, sample_rate=44100)
    mel_f = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
    x = separator.separate(waveform)
    x = np.stack([np.dot(np.abs(np.mean(separator_stft(separator, x[key]), axis=-1))**2, mel_f) for key in x])
    x = np.transpose(x, (0, 2, 1))
    x = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))])
    x = np.transpose(x, (0, 2, 1))
    return x

# Predict beats from the audio
def predict_beats(x):
    model.load_state_dict(torch.load(PARAM_PATH, map_location=torch.device('cpu'))['state_dict'])

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        model_input = torch.from_numpy(x).unsqueeze(0).float().to(device)
        activation, _ = model(model_input)

    beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
    downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()
    dbn_beat_pred = beat_tracker(beat_activation)

    combined_act = np.concatenate((np.maximum(beat_activation - downbeat_activation,
                                            np.zeros(beat_activation.shape)
                                            )[:, np.newaxis],
                                downbeat_activation[:, np.newaxis]
                                ), axis=-1)   #(T, 2)
    dbn_downbeat_pred = downbeat_tracker(combined_act)
    dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1] == 1][:, 0]
    return dbn_downbeat_pred, dbn_beat_pred

# The main function
def beats_prediction(b: bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'temp.wav')
        with open(temp_file, 'wb') as f:
            f.write(b)
        x = split(temp_file)
    dbn_downbeat_pred, dbn_beat_pred = predict_beats(x)
    downbeat_frames = np.array(dbn_downbeat_pred * 44100, dtype = np.int32).tolist()
    beat_frames = np.array(dbn_beat_pred * 44100, dtype = np.int32).tolist()
    data = {
        "downbeat_frames": downbeat_frames,
        "beat_frames": beat_frames
    }
    return data
