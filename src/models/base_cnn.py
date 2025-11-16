import timm
import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=6, dropout_p=0.1, image_alignment="stacked"):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            drop_rate=dropout_p,
        )

        if image_alignment not in ["paired", "stacked"]:
            raise ValueError("image_alignment must be either 'paired' or 'stacked'")
        self.image_alignment = image_alignment  # "paired" combines Kaggle and EEG spectrograms side by side; "stacked" stacks all spectrograms vertically

    def forward(self, x):
        if self.image_alignment == "paired":
            # Input x has shape [batch_size, 8, 128, 256]
            # 8 channels: 4 Kaggle spectrograms + 4 EEG spectrograms

            # Split into Kaggle (first 4) and EEG (last 4) spectrograms
            # Each has shape [batch_size, 4, 128, 256]
            kaggle_spectrograms = x[:, :4, :, :]  # Channels 0-3
            eeg_spectrograms = x[:, 4:, :, :]  # Channels 4-7

            # Split each group into individual spectrograms
            # Each list contains 4 tensors of shape [batch_size, 1, 128, 256]
            kaggle_channels = torch.split(kaggle_spectrograms, 1, dim=1)
            eeg_channels = torch.split(eeg_spectrograms, 1, dim=1)

            # Pair up spectrograms: concatenate each Kaggle with its corresponding EEG horizontally
            # This creates 4 tensors, each of shape [batch_size, 1, 128, 512]
            paired_spectrograms = []
            for kaggle_ch, eeg_ch in zip(kaggle_channels, eeg_channels):
                # Concatenate along width dimension (dim=3)
                paired = torch.cat([kaggle_ch, eeg_ch], dim=3)
                paired_spectrograms.append(paired)

            # Stack the 4 paired spectrograms vertically
            # This creates a tensor of shape [batch_size, 1, 512, 512]
            # (4 pairs × 128 height = 512)
            x_stacked = torch.cat(paired_spectrograms, dim=2)

            # Repeat the single channel 3 times to create a 3-channel RGB image
            # Final shape is [batch_size, 3, 512, 512]
            x_3_channel = x_stacked.repeat(1, 3, 1, 1)

        elif self.image_alignment == "stacked":
            # Input x has shape [batch_size, num_channels, 128, 256], where num_channels is either 4 or 8 (4 Kaggle + 4 EEG)
            # Split the channels into a list of num_channels tensors
            # Each tensor has shape [batch_size, 1, 128, 256]
            channels = torch.split(x, 1, dim=1)

            # Concatenate along the height dimension
            # This creates a tensor of shape [batch_size, 1, num_channels * 128, 256]
            x_reshaped = torch.cat(channels, dim=2)

            # Repeat the single channel 3 times to create a 3-channel image
            # Final shape is [batch_size, 3, num_channels * 128, 256]
            x_3_channel = x_reshaped.repeat(1, 3, 1, 1)

        return self.model(x_3_channel)
