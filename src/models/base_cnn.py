import timm
import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=6, dropout_p=0.1):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            drop_rate=dropout_p,
            drop_path_rate=dropout_p,
        )

    def forward(self, x):
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
