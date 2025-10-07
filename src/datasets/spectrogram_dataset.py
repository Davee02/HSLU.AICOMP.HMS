import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, df, targets, data_path, img_size, mode="train"):
        self.df = df
        self.targets = targets
        self.data_path = data_path
        self.img_size = img_size
        self.mode = mode  # 'train' or 'test'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectrogram_id = row["spectrogram_id"]

        if self.mode == "train":
            spec_path = os.path.join(self.data_path, "train_spectrograms", f"{spectrogram_id}.parquet")
        else:
            spec_path = os.path.join(self.data_path, "test_spectrograms", f"{spectrogram_id}.parquet")

        spec_df = pd.read_parquet(spec_path)

        processed_channels = []
        # we treat each of the montage blocks as seperate channels
        for k in range(4):
            # Extract 100 rows for the current EEG channel and transpose
            # The +1 is to skip the 'time' column
            start_col = k * 100 + 1
            end_col = (k + 1) * 100 + 1
            img = spec_df.iloc[:, start_col:end_col].values.T  # Shape: (100, time)
            # Log transform (to make the differences between values less extreme)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # standardize per channel to focus on
            ep = 1e-6
            m, s = np.nanmean(img.flatten()), np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            # Crop time axis to 256 and pad frequency axis to 128
            time_dim = img.shape[1]
            crop_start = (time_dim - 256) // 2
            img_cropped = img[:, crop_start : crop_start + 256]  # Shape: (100, 256)

            padded_img = np.zeros((128, 256), dtype=np.float32)
            padded_img[14:-14, :] = img_cropped  # Center the 100 rows in a 128-row image
            processed_channels.append(padded_img)

        # Stack 4 channels to create [4, 128, 256] image
        spectrogram = np.stack(processed_channels, axis=0)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)

        if self.mode == "train":
            labels = torch.tensor(row[self.targets], dtype=torch.float32)
            return spectrogram_tensor, labels
        else:
            return spectrogram_tensor
