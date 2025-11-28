import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

from src.utils.constants import Constants


class EEGDatasetMontage(Dataset):
    """
    A PyTorch Dataset class for the HMS Kaggle competition.
    Updates:
    - Supports dynamic augmentation selection via a list of strings.
    - Uses Max pooling for downsampling to preserve spikes.
    """

    def __init__(
        self,
        df,
        data_path,
        mode="train",
        specs=None,
        eeg_specs=None,
        downsample_factor=1,
        augmentations=None,  # <--- NEW PARAMETER
    ):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.downsample_factor = downsample_factor

        # Store the list of active augmentations (e.g. ['time_shift', 'channel_mask'])
        # If None, default to empty list (no augmentations)
        self.augmentations = augmentations if augmentations is not None else []

        original_fs = 200.0
        self.sos = butter(N=4, Wn=[0.5, 50.0], btype="band", fs=original_fs, output="sos")

        # 16 bipolar montage pairs

        self.MONTAGES = {
            # Left Temporal
            "Fp1-F7": ["Fp1", "F7"],
            "F7-T3": ["F7", "T3"],
            "T3-T5": ["T3", "T5"],
            "T5-O1": ["T5", "O1"],
            # Right Temporal
            "Fp2-F8": ["Fp2", "F8"],
            "F8-T4": ["F8", "T4"],
            "T4-T6": ["T4", "T6"],
            "T6-O2": ["T6", "O2"],
            # Left Parasagittal
            "Fp1-F3": ["Fp1", "F3"],
            "F3-C3": ["F3", "C3"],
            "C3-P3": ["C3", "P3"],
            "P3-O1": ["P3", "O1"],
            # Right Parasagittal
            "Fp2-F4": ["Fp2", "F4"],
            "F4-C4": ["F4", "C4"],
            "C4-P4": ["C4", "P4"],
            "P4-O2": ["P4", "O2"],
            # Midline Chain -
            "Fz-Cz": ["Fz", "Cz"],
            "Cz-Pz": ["Cz", "Pz"],
        }

        self.EKG_LEAD = ["EKG"]

        self.channel_indices = {name: i for i, name in enumerate(Constants.EEG_FEATURES)}
        self.new_feature_names = list(self.MONTAGES.keys()) + self.EKG_LEAD

        self.specs = specs
        self.eeg_specs = eeg_specs

    def __len__(self):
        return len(self.df)

    def apply_augmentations(self, eeg_data):
        """
        Refined augmentations for EEG Physics.
        eeg_data shape: (Time_Steps, Channels)
        """

        if "gaussian_noise" in self.augmentations:
            if np.random.rand() < 0.5:
                noise_factor = np.random.uniform(0.01, 0.05)
                std = eeg_data.std(axis=0, keepdims=True)
                noise = np.random.normal(0, std * noise_factor, eeg_data.shape)
                eeg_data = eeg_data + noise

        if "time_shift" in self.augmentations:
            if np.random.rand() < 0.5:
                max_shift = int(eeg_data.shape[0] * 0.1)
                shift = np.random.randint(-max_shift, max_shift)
                eeg_data = np.roll(eeg_data, shift, axis=0)

                if shift > 0:
                    eeg_data[:shift, :] = 0
                elif shift < 0:
                    eeg_data[shift:, :] = 0

        if "channel_mask" in self.augmentations:
            if np.random.rand() < 0.5:
                n_channels = eeg_data.shape[1]

                # Decide how many channels to mask (between 1 and 3)
                n_mask = np.random.randint(1, min(4, n_channels + 1))

                mask_indices = np.random.choice(n_channels, size=n_mask, replace=False)

                eeg_data[:, mask_indices] = 0.0

        return eeg_data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.mode in ["train", "valid"]:
            eeg_path = f"{self.data_path}/train_eegs/{row.eeg_id}.parquet"
        else:
            eeg_path = f"{self.data_path}/test_eegs/{row.eeg_id}.parquet"

        eeg_df = pd.read_parquet(eeg_path)
        eeg_data = eeg_df[Constants.EEG_FEATURES].values.astype(np.float32)

        # Time cropping
        if self.mode in ["train", "valid", "test"]:
            rows = len(eeg_data)
            offset = (rows - 10_000) // 2
            eeg_data = eeg_data[offset : offset + 10_000]

        # Handle NaNs
        if np.isnan(eeg_data).any():
            for j in range(eeg_data.shape[1]):
                col_data = eeg_data[:, j]
                if np.isnan(col_data).mean() < 1.0:
                    mean_val = np.nanmean(col_data)
                    eeg_data[:, j] = np.nan_to_num(col_data, nan=mean_val)
                else:
                    eeg_data[:, j] = 0

        num_rows = eeg_data.shape[0]
        num_new_features = len(self.new_feature_names)
        montage_data = np.zeros((num_rows, num_new_features), dtype=np.float32)

        # Calculate 18 bipolar differences
        for i, (name, electrodes) in enumerate(self.MONTAGES.items()):
            idx1 = self.channel_indices[electrodes[0]]
            idx2 = self.channel_indices[electrodes[1]]
            montage_data[:, i] = eeg_data[:, idx1] - eeg_data[:, idx2]

        # Add EKG
        for i, name in enumerate(self.EKG_LEAD):
            idx = self.channel_indices[name]
            montage_data[:, i + len(self.MONTAGES)] = eeg_data[:, idx]

        filtered_eeg = sosfiltfilt(self.sos, montage_data, axis=0)

        # Downsampling logic
        if self.downsample_factor > 1:
            n_timesteps, n_features = filtered_eeg.shape
            factor = self.downsample_factor

            pad_size = (factor - n_timesteps % factor) % factor
            if pad_size > 0:
                padded_eeg = np.pad(filtered_eeg, ((0, pad_size), (0, 0)), "constant", constant_values=0)
            else:
                padded_eeg = filtered_eeg

            eeg_to_process = padded_eeg.reshape(-1, factor, n_features).mean(axis=1)
        else:
            eeg_to_process = filtered_eeg

        if self.mode == "train" and self.augmentations:
            eeg_to_process = self.apply_augmentations(eeg_to_process)

        # 1. Calculate global mean
        mean = np.mean(eeg_to_process)

        # 2. Center the data
        centered_eeg = eeg_to_process - mean

        # 3. Calculate MAD for each channel
        channel_mads = np.mean(np.abs(centered_eeg), axis=0)

        # 4. Get the median of these MADs as the global scale
        global_scale = np.median(channel_mads) + 1e-6

        # 5. Normalize
        normalized_eeg = centered_eeg / global_scale

        # 6. Clip
        normalized_eeg = np.clip(normalized_eeg, -10.0, 10.0)

        signals = torch.from_numpy(normalized_eeg.copy()).float()

        if self.mode == "test":
            return signals
        else:
            labels = torch.tensor(row[Constants.TARGETS].values.astype(np.float32), dtype=torch.float32)
            return signals, labels
