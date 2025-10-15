import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.classifier(last_hidden_state)
        return output
