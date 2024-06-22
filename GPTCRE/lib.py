import numpy as np
import librosa

def load_data(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def preprocess_audio(audio, sr, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram
