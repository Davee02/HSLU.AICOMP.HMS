import timm
import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, model_name, pretrained=True, in_channels=4, num_classes=6):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=num_classes)

    def forward(self, x):
        # Input x has shape [batch_size, 4, 128, 256]
        # Split the 4 channels into a list of 4 tensors
        # Each tensor has shape [batch_size, 1, 128, 256]
        channels = torch.split(x, 1, dim=1)

        # Concatenate along the height dimension
        # This creates a tensor of shape [batch_size, 1, 512, 256]
        x_reshaped = torch.cat(channels, dim=2)

        # Repeat the single channel 3 times to create a 3-channel image
        # Final shape is [batch_size, 3, 512, 256]
        x_3_channel = x_reshaped.repeat(1, 3, 1, 1)

        return self.model(x_3_channel)
