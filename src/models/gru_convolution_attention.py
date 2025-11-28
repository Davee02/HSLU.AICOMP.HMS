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


# --- NEW BLOCK DEFINITIONS ---


class StandardConvBlock(nn.Module):
    """
    Your original block: Conv1d(k=7) -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class Inception1DBlock(nn.Module):
    """
    Inception-style block for EEG.
    Captures features at multiple scales (Short, Medium, Long range) simultaneously.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Ensure out_channels is divisible by 4 for equal branching
        # If not, we add the remainder to the first branch
        base_channels = out_channels // 4
        remainder = out_channels % 4

        # Branch 1: Small kernel (High frequency/Local details)
        self.branch1 = nn.Conv1d(in_channels, base_channels + remainder, kernel_size=3, padding="same")

        # Branch 2: Medium kernel (Original scale)
        self.branch2 = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding="same")

        # Branch 3: Large kernel (Low frequency/Broad trends)
        self.branch3 = nn.Conv1d(in_channels, base_channels, kernel_size=15, padding="same")

        # Branch 4: MaxPool branch (Preserves dominant spikes)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, base_channels, kernel_size=1),  # 1x1 conv to reduce depth
        )

        # Standard post-processing (Batch Norm, ReLU, Downsampling)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # 1. Apply branches in parallel
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 2. Concatenate along channel dimension
        x = torch.cat([b1, b2, b3, b4], dim=1)

        # 3. Post-process
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        return x


class NodeAttentionModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_embed_size,
        hidden_size,
        num_layers,
        num_classes,
        num_cnn_blocks=3,
        dropout=0.2,
        use_inception=False,
    ):

        super().__init__()

        self.num_nodes = num_nodes
        self.node_embed_size = node_embed_size
        self.use_inception = use_inception

        # Channel definitions: Start with 1 input channel, scale up
        channels = [1] + [64 * (2**i) for i in range(num_cnn_blocks)]

        layers = []
        for i in range(num_cnn_blocks):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            if self.use_inception:
                # Use the new Multi-Scale Inception Block
                layers.append(Inception1DBlock(in_channels, out_channels))
            else:
                # Use the Original Block
                layers.append(StandardConvBlock(in_channels, out_channels))

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

        # [B, S, N] -> [B, N, S]
        x = x.permute(0, 2, 1)

        # Merge Batch and Nodes to treat every electrode as an independent sample
        # [B*N, 1, S]
        x = x.reshape(B * N, 1, S)

        # Apply CNN (Standard or Inception)
        x_cnn = self.cnn_frontend(x)

        # [B*N, Channels, New_S] -> [B*N, New_S, Channels]
        x_cnn = x_cnn.permute(0, 2, 1)

        gru_out, _ = self.gru(x_cnn)

        x_pooled = self.temporal_pool(gru_out)

        x_embedded = self.node_embedding_head(x_pooled)

        # Unmerge Batch and Nodes
        # [B, N, Embed_Dim]
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
