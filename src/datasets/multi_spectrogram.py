import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiSpectrogramDataset(Dataset):
    def __init__(
        self,
        df,
        targets,
        data_path,
        img_size,
        eeg_spec_path,
        mode="train",
        apply_augmentations=["gaussian_noise", "time_reversal", "time_masking", "frequency_masking"],
    ):
        self.df = df
        self.targets = targets
        self.data_path = data_path
        self.img_size = img_size
        self.eeg_spec_path = eeg_spec_path
        self.mode = mode
        self.apply_augmentations = apply_augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        kaggle_spec_tensor = self._load_kaggle_spectrogram(row)
        eeg_spec_tensor = self._load_eeg_spectrogram(row)

        # Concatenate both spectrograms along the channel dimension (dim=0)
        # This combines the 4 Kaggle channels with the 4 EEG channels into a single tensor of shape (8, height, width)
        spectrogram_tensor = torch.cat([kaggle_spec_tensor, eeg_spec_tensor], dim=0)

        if self.mode == "train":
            if self.apply_augmentations:
                spectrogram_tensor = self._augment(spectrogram_tensor)
            labels = torch.tensor(row.loc[self.targets].values.astype(np.float32))
            return spectrogram_tensor, labels
        else:
            return spectrogram_tensor

    def _load_kaggle_spectrogram(self, row):
        spectrogram_id = row["spectrogram_id"]
        spec_path = os.path.join(self.data_path, f"{self.mode}_spectrograms", f"{spectrogram_id}.parquet")
        spec_df = pd.read_parquet(spec_path)

        processed_channels = []
        # Process 4 separate channels from the spectrogram
        for k in range(4):
            # Extract columns for this channel (100 columns per channel)
            # Column 0 is the time index, so we start from column 1
            start_col = k * 100 + 1
            end_col = (k + 1) * 100 + 1
            # Transpose to get (frequency, time) shape
            img = spec_df.iloc[:, start_col:end_col].values.T

            # Clip values to range [exp(-4), exp(8)] to remove outliers
            img = np.clip(img, np.exp(-4), np.exp(8))
            # Apply log transform to compress dynamic range
            img = np.log(img)

            # Standardize the data: subtract mean and divide by standard deviation
            ep = 1e-6  # Small epsilon to prevent division by zero
            m, s = np.nanmean(img.flatten()), np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            # Replace any NaN values with 0
            img = np.nan_to_num(img, nan=0.0)

            # Center crop along the time dimension (take the middle part)
            time_dim = img.shape[1]
            crop_start = max(0, (time_dim - self.img_size[1]) // 2)
            img_cropped = img[:, crop_start : crop_start + self.img_size[1]]

            # Pad the frequency dimension to match target size
            # Creates a zero-filled array of target size
            padded_img = np.zeros(self.img_size, dtype=np.float32)
            # Center the cropped image within the padded array
            pad_start = max(0, (self.img_size[0] - img_cropped.shape[0]) // 2)
            padded_img[pad_start : pad_start + img_cropped.shape[0], :] = img_cropped
            processed_channels.append(padded_img)

        # Stack all 4 channels along axis 0 to create shape (4, height, width)
        return torch.tensor(np.stack(processed_channels, axis=0), dtype=torch.float32)

    def _load_eeg_spectrogram(self, row):
        eeg_id = row["eeg_id"]
        eeg_spec_path = os.path.join(self.eeg_spec_path, f"{eeg_id}.npy")
        img = np.load(eeg_spec_path)

        # Remove the last channel (central chain electrodes)
        img = img[:, :, :-1]

        # Transpose to have channels first: (channels, height, width)
        img = np.transpose(img, (2, 0, 1))

        # Standardize the EEG data across all dimensions
        ep = 1e-6  # Small epsilon to prevent division by zero
        m, s = np.mean(img.flatten()), np.std(img.flatten())
        img = (img - m) / (s + ep)

        return torch.tensor(img, dtype=torch.float32)

    def _augment(self, tensor):
        if "gaussian_noise" in self.apply_augmentations:
            if np.random.rand() < 0.3:
                noise = torch.randn_like(tensor) * 0.05
                tensor = tensor + noise

        if "time_reversal" in self.apply_augmentations:
            # Horizontal flip (time reversal)
            if np.random.rand() < 0.5:
                tensor = torch.flip(tensor, dims=[2])

        if "time_masking" in self.apply_augmentations:
            # Time masking (p=0.5)
            if np.random.rand() < 0.5:
                time_mask_width = np.random.randint(10, 30)
                time_mask_start = np.random.randint(0, max(1, tensor.shape[2] - time_mask_width))
                tensor[:, :, time_mask_start : time_mask_start + time_mask_width] = 0

        if "frequency_masking" in self.apply_augmentations:
            # Frequency masking (p=0.5)
            if np.random.rand() < 0.5:
                freq_mask_height = np.random.randint(5, 15)
                freq_mask_start = np.random.randint(0, max(1, tensor.shape[1] - freq_mask_height))
                tensor[:, freq_mask_start : freq_mask_start + freq_mask_height, :] = 0

        return tensor
