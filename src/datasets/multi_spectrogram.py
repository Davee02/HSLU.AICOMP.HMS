import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiSpectrogramDataset(Dataset):
    def __init__(self, df, targets, data_path, img_size, eeg_spec_path, mode="train"):
        self.df = df
        self.targets = targets
        self.data_path = data_path
        self.img_size = img_size
        self.eeg_spec_path = eeg_spec_path
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        kaggle_spec_tensor = self._load_kaggle_spectrogram(row)

        eeg_spec_tensor = self._load_eeg_spectrogram(row)

        spectrogram_tensor = torch.cat([kaggle_spec_tensor, eeg_spec_tensor], dim=0)

        if self.mode == "train":
            labels = torch.tensor(row.loc[self.targets].values.astype(np.float32))
            return spectrogram_tensor, labels
        else:
            return spectrogram_tensor

    def _load_kaggle_spectrogram(self, row):
        spectrogram_id = row["spectrogram_id"]
        spec_path = os.path.join(self.data_path, f"{self.mode}_spectrograms", f"{spectrogram_id}.parquet")
        spec_df = pd.read_parquet(spec_path)

        processed_channels = []
        for k in range(4):
            start_col = k * 100 + 1
            end_col = (k + 1) * 100 + 1
            img = spec_df.iloc[:, start_col:end_col].values.T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            m, s = np.nanmean(img.flatten()), np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            time_dim = img.shape[1]
            crop_start = max(0, (time_dim - self.img_size[1]) // 2)
            img_cropped = img[:, crop_start : crop_start + self.img_size[1]]

            padded_img = np.zeros(self.img_size, dtype=np.float32)
            pad_start = max(0, (self.img_size[0] - img_cropped.shape[0]) // 2)
            padded_img[pad_start : pad_start + img_cropped.shape[0], :] = img_cropped
            processed_channels.append(padded_img)

        return torch.tensor(np.stack(processed_channels, axis=0), dtype=torch.float32)

    def _load_eeg_spectrogram(self, row):
        eeg_id = row["eeg_id"]
        eeg_spec_path = os.path.join(self.eeg_spec_path, f"{eeg_id}.npy")

        img = np.load(eeg_spec_path)

        img = img[:, :, :-1]

        img = np.transpose(img, (2, 0, 1))

        ep = 1e-6
        m, s = np.mean(img.flatten()), np.std(img.flatten())
        img = (img - m) / (s + ep)

        return torch.tensor(img, dtype=torch.float32)
