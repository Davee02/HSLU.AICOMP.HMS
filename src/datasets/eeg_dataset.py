import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

from src.utils.constants import Constants


class EEGDataset(Dataset):

    def __init__(self, df, data_path, mode="test", specs=None, eeg_specs=None):
        self.df = df
        self.data_path = data_path
        self.mode = mode

        self.sos = butter(N=4, Wn=[0.5, 40.0], btype="band", fs=200.0, output="sos")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_path = f"{self.data_path}/processed/eegs_parquet/{row.eeg_id}.parquet"
        eeg_data = pd.read_parquet(eeg_path).values.astype(np.float32)

        filtered_eeg = sosfiltfilt(self.sos, eeg_data, axis=0)

        mean = np.mean(filtered_eeg, axis=0, keepdims=True)
        std = np.std(filtered_eeg, axis=0, keepdims=True)
        epsilon = 1e-6  # Add epsilon to prevent division by zero

        normalized_eeg = (filtered_eeg - mean) / (std + epsilon)

        signals = torch.from_numpy(normalized_eeg).float()

        if self.mode == "test":
            return signals
        else:
            numeric_values = row[Constants.TARGETS].values.astype(np.float32)
            labels = torch.tensor(numeric_values, dtype=torch.float32)
            return signals, labels
