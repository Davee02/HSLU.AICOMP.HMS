import torch
import torch.nn.functional as F
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        energy = self.attention_net(x)
        weights = F.softmax(energy, dim=1)
        context_vector = torch.sum(x * weights, dim=1)
        return context_vector


class ResNet1DBlock(nn.Module):
    """
    ResNet 1D Block.
    Includes: Conv -> BN -> ReLU -> Conv -> BN -> Add(Skip) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip Connection
        self.shortcut = nn.Sequential()

        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(x)

        out += identity
        out = self.act2(out)

        return out


class NodeAttentionModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_embed_size,
        hidden_size,
        num_layers,
        num_classes,
        num_cnn_blocks=4,
        dropout=0.2,
        use_inception=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_size = node_embed_size
        self.use_inception = use_inception

        channels = [32, 64, 64, 128, 128]

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        layers = []
        for i in range(num_cnn_blocks):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            # Block 0: Stride 2 (Total 8x)
            # Block 1: Stride 2 (Total 16x)
            # Block 2: Stride 2 (Total 32x) -> Sequence length ~312
            # Block 3: Stride 1 (Keep resolution)

            if i < 3:
                current_stride = 2
            else:
                current_stride = 1

            if self.use_inception:
                layers.append(ResNet1DBlock(in_channels, out_channels, stride=current_stride))
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=current_stride, padding=3),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                    )
                )

        self.cnn_frontend = nn.Sequential(*layers)

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

        x = self.stem(x)
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
        output = self.classifier(pooled_output)
        return output
