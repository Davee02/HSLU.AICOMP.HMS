from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.constants import Constants
from src.utils.utils import fill_nan_with_mean


class CBraModDataset(Dataset):
    def __init__(self, df, data_path, window_size_seconds, eeg_frequency, mode, normalization="naive"):
        self.df = df
        self.data_path = Path(data_path)
        self.window_size_seconds = window_size_seconds
        self.eeg_frequency = eeg_frequency
        self.mode = mode  # 'train' or 'test'
        self.normalization = normalization

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.mode == "train":
            eeg_path = self.data_path / "processed" / "eegs_parquet" / f"{row.eeg_id}.parquet"
        else:
            eeg_path = self.data_path / "test_eegs" / f"{row.eeg_id}.parquet"

        eeg_df = pd.read_parquet(eeg_path)
        eeg_data = eeg_df[Constants.EEG_FEATURES].values.astype(np.float32)

        # take middle window
        rows = len(eeg_data)
        offset = (rows - self.window_size_seconds * self.eeg_frequency) // 2
        eeg_data = eeg_data[offset : offset + self.window_size_seconds * self.eeg_frequency]
        eeg_data = fill_nan_with_mean(eeg_data)

        if self.normalization == "naive":
            # normalize to ~[-1, 1] by dividing by 100uV
            eeg_data = eeg_data / 100.0
        else:
            # z-score normalization to zero mean and unit variance
            mean = np.mean(eeg_data, axis=0, keepdims=True)
            std = np.std(eeg_data, axis=0, keepdims=True)
            eeg_data = (eeg_data - mean) / (std + 1e-6)

        signals = torch.from_numpy(eeg_data.copy()).float()

        assert signals.shape == (self.window_size_seconds * self.eeg_frequency, len(Constants.EEG_FEATURES))
        signals = signals.transpose(0, 1)  # transpose from [time_steps, num_channels] to [num_channels, time_steps]
        assert signals.shape == (len(Constants.EEG_FEATURES), self.window_size_seconds * self.eeg_frequency)
        signals = signals.reshape(
            signals.shape[0], self.window_size_seconds, self.eeg_frequency
        )  # then reshape into [num_channels, seq_len, patch_size]
        assert signals.shape == (len(Constants.EEG_FEATURES), self.window_size_seconds, self.eeg_frequency)

        if self.mode == "test":
            return signals
        else:
            labels = torch.tensor(row[Constants.TARGETS].values.astype(np.float32), dtype=torch.float32)
            return signals, labels
