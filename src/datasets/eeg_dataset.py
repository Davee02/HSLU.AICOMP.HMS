import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

from src.utils.constants import Constants


class EEGDataset(Dataset):

    def __init__(self, df, data_path, mode="train", downsample_factor=1):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.downsample_factor = downsample_factor

        original_fs = 200.0
        self.sos = butter(N=4, Wn=[0.5, 40.0], btype="band", fs=original_fs, output="sos")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.mode in ["train", "valid"]:
            eeg_path = f"{self.data_path}/processed/eegs_parquet/{row.eeg_id}.parquet"
        else:
            eeg_path = f"{self.data_path}/test_eegs/{row.eeg_id}.parquet"

        eeg_df = pd.read_parquet(eeg_path)
        eeg_data = eeg_df[Constants.EEG_FEATURES].values.astype(np.float32)

        if self.mode == "test":
            rows = len(eeg_data)
            offset = (rows - 10_000) // 2
            eeg_data = eeg_data[offset : offset + 10_000]

            if np.isnan(eeg_data).any():
                for j in range(eeg_data.shape[1]):
                    col_data = eeg_data[:, j]
                    if np.isnan(col_data).mean() < 1.0:
                        mean_val = np.nanmean(col_data)
                        eeg_data[:, j] = np.nan_to_num(col_data, nan=mean_val)
                    else:
                        eeg_data[:, j] = 0

        filtered_eeg = sosfiltfilt(self.sos, eeg_data, axis=0)

        if self.downsample_factor > 1:
            eeg_to_process = filtered_eeg[:: self.downsample_factor, :]
        else:
            eeg_to_process = filtered_eeg

        mean = np.mean(eeg_to_process, axis=0, keepdims=True)
        std = np.std(eeg_to_process, axis=0, keepdims=True)
        normalized_eeg = (eeg_to_process - mean) / (std + 1e-6)

        signals = torch.from_numpy(normalized_eeg.copy()).float()

        if self.mode == "test":
            return signals
        else:
            labels = torch.tensor(row[Constants.TARGETS].values.astype(np.float32), dtype=torch.float32)
            return signals, labels
