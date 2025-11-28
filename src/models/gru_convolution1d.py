import torch
import torch.nn.functional as F
from torch import nn


class AttentionPool(nn.Module):
    """
    A simple self-attention pooling layer.

    Learns a weighted average of the input sequence.
    """

    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, in_features]
        energy = self.attention_net(x)
        # weights shape: [batch, seq_len, 1]
        weights = F.softmax(energy, dim=1)
        # context shape: [batch, in_features]
        context_vector = torch.sum(x * weights, dim=1)
        return context_vector


class GRUConvModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_cnn_blocks=3, dropout=0.2):

        super().__init__()

        channels = [input_size] + [64 * (2**i) for i in range(num_cnn_blocks)]
        cnn_layers = []
        for i in range(num_cnn_blocks):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding="same"))
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.cnn_frontend = nn.Sequential(*cnn_layers)

        gru_input_size = channels[num_cnn_blocks]
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        gru_output_features = hidden_size * 2

        self.attention = AttentionPool(gru_output_features, hidden_size)

        # classifier with 3 inputs concatenated:
        # 1. Attention-pooled output
        # 2. Average-pooled output
        # 3. Max-pooled output
        classifier_input_dim = gru_output_features * 3

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.cnn_frontend(x)
        x = x.permute(0, 2, 1)

        gru_out, _ = self.gru(x)

        attn_pool = self.attention(gru_out)  # shape: [batch, hidden_size * 2]

        avg_pool = torch.mean(gru_out, dim=1)  # shape: [batch, hidden_size * 2]

        max_pool, _ = torch.max(gru_out, dim=1)  # shape: [batch, hidden_size * 2]

        pooled_output = torch.cat((attn_pool, avg_pool, max_pool), dim=1)

        output = self.classifier(pooled_output)
        return output
