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
            