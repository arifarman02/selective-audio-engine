import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

class RemixDetector:
    def __init__(self, database_path=None):
        
        self.database = {}
        if database_path and Path(database_path).exists():
            with open(database_path, 'rb') as f:
                self.database = pickle.load(f)
    
    def save_database(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.database, f)
    
    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        fingerprint= {
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_var': np.var(chroma, axis=1),
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_var': np.var(mfcc, axis=1),
            'tempo': tempo[0]
        }
        return fingerprint
    
    def add_original_song(self, song_path, song_id):
        fingerprint = self.extract_features(song_path)
        self.database[song_id] = fingerprint
        return fingerprint