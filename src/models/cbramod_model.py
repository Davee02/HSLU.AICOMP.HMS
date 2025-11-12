import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from src.models.cbramod.cbramod import CBraMod


class CBraModModel(nn.Module):
    def __init__(
        self,
        pretrained_weights_path,
        classifier_type,
        num_of_classes,
        dropout_prob,
        num_eeg_channels,
        seq_len_seconds,
        device,
    ):
        super(CBraModModel, self).__init__()

        self.sampling_rate = 200  # fixed at 200Hz as in the original implementation
        self.num_eeg_channels = num_eeg_channels
        self.seq_len_seconds = seq_len_seconds

        self.backbone = CBraMod(  # use the same hyperparameters as in the original implementation
            in_dim=self.sampling_rate,
            out_dim=self.sampling_rate,
            d_model=self.sampling_rate,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )

        if pretrained_weights_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

        self.backbone.proj_out = nn.Identity()

        if classifier_type == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, num_of_classes),
            )
        elif classifier_type == "all_patch_reps_onelayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(num_eeg_channels * seq_len_seconds * self.sampling_rate, num_of_classes),
            )
        elif classifier_type == "all_patch_reps_twolayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(num_eeg_channels * seq_len_seconds * self.sampling_rate, self.sampling_rate),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.sampling_rate, num_of_classes),
            )
        elif classifier_type == "all_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(
                    num_eeg_channels * seq_len_seconds * self.sampling_rate, seq_len_seconds * self.sampling_rate
                ),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(seq_len_seconds * self.sampling_rate, self.sampling_rate),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.sampling_rate, num_of_classes),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        batch_size, num_channels, seq_len, patch_size = x.shape
        assert (
            num_channels == self.num_eeg_channels
        ), f"Expected number of channels {self.num_eeg_channels}, but got {num_channels}"
        assert seq_len == self.seq_len_seconds, f"Expected sequence length {self.seq_len_seconds}, but got {seq_len}"
        assert patch_size == self.sampling_rate, f"Expected patch size {self.sampling_rate}, but got {patch_size}"

        feats = self.backbone(x)
        out = self.classifier(feats)
        return out
