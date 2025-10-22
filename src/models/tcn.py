import torch.nn as nn
from pytorch_tcn import TCN


class TCNModel(nn.Module):
    def __init__(
        self, num_inputs, num_outputs, channel_sizes, kernel_size, dropout, causal=False, use_skip_connections=True
    ):
        super(TCNModel, self).__init__()
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=channel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            use_skip_connections=use_skip_connections,
        )
        # The output of the TCN is the number of channels in the last layer
        self.fc = nn.Linear(channel_sizes[-1], num_outputs)

    def forward(self, x):
        # TCN expects input of shape (batch_size, num_channels, sequence_length)
        # Our dataloader provides (batch_size, sequence_length, num_channels)
        # So we need to permute the dimensions
        x = x.permute(0, 2, 1)

        tcn_output = self.tcn(x)

        # We take the output of the last time step for classification
        last_time_step_output = tcn_output[:, :, -1]

        output = self.fc(last_time_step_output)
        return output
