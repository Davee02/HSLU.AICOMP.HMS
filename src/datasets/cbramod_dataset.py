from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.constants import Constants
from src.utils.utils import fill_nan_with_mean


class CBraModDataset(Dataset):
    def __init__(self, df, data_path, window_size_seconds, eeg_frequency, mode):
        self.df = df
        self.data_path = Path(data_path)
        self.window_size_seconds = window_size_seconds
        self.eeg_frequency = eeg_frequency
        self.mode = mode  # 'train' or 'test'

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
        middle_offset = (self.window_size_seconds // 2) * self.eeg_frequency
        start_idx = (rows - middle_offset) // 2
        eeg_data = eeg_data[start_idx : start_idx + middle_offset]
        eeg_data = fill_nan_with_mean(eeg_data)

        # normalize to ~[-1, 1] by dividing by 100uV
        eeg_data = eeg_data / 100.0

        signals = torch.from_numpy(eeg_data.copy()).float()

        if self.mode == "test":
            return signals
        else:
            labels = torch.tensor(row[Constants.TARGETS].values.astype(np.float32), dtype=torch.float32)
            return signals, labels
