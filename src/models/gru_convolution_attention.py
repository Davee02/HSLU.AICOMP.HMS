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
        # x shape: [batch, seq_len (nodes), in_features]
        energy = self.attention_net(x)
        # weights shape: [batch, seq_len (nodes), 1]
        weights = F.softmax(energy, dim=1)
        # context shape: [batch, in_features]
        context_vector = torch.sum(x * weights, dim=1)
        return context_vector


class NodeAttentionModel(nn.Module):
    def __init__(self, num_nodes, node_embed_size, hidden_size, num_layers, num_classes, num_cnn_blocks=3, dropout=0.2):

        super().__init__()

        self.num_nodes = num_nodes
        self.node_embed_size = node_embed_size

        channels = [1] + [64 * (2**i) for i in range(num_cnn_blocks)]
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

        self.temporal_pool = AttentionPool(gru_output_features, gru_output_features // 2)

        self.node_embedding_head = nn.Linear(gru_output_features, node_embed_size)

        self.pos_encoding = nn.Parameter(torch.randn(1, num_nodes, node_embed_size))

        self.attn_layer_norm1 = nn.LayerNorm(node_embed_size)
        self.mha = nn.MultiheadAttention(embed_dim=node_embed_size, num_heads=4, dropout=0.1, batch_first=True)

        self.attn_layer_norm2 = nn.LayerNorm(node_embed_size)
        self.fc = nn.Sequential(
            nn.Linear(node_embed_size, node_embed_size * 2), nn.ReLU(), nn.Linear(node_embed_size * 2, node_embed_size)
        )

        self.node_pool = AttentionPool(node_embed_size, node_embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(node_embed_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        B, S, N = x.shape

        x = x.permute(0, 2, 1)

        x = x.reshape(B * N, 1, S)

        x_cnn = self.cnn_frontend(x)

        x_cnn = x_cnn.permute(0, 2, 1)

        gru_out, _ = self.gru(x_cnn)

        x_pooled = self.temporal_pool(gru_out)

        x_embedded = self.node_embedding_head(x_pooled)

        node_embeddings = x_embedded.reshape(B, N, -1)

        x = node_embeddings + self.pos_encoding

        x_norm = self.attn_layer_norm1(x)

        attn_out, _ = self.mha(x_norm, x_norm, x_norm)

        x = x + attn_out

        x_norm = self.attn_layer_norm2(x)

        fc_out = self.fc(x_norm)
        x = x + fc_out

        pooled_output = self.node_pool(x)

        output = self.classifier(pooled_output)  # -> [B, 6]
        return output
