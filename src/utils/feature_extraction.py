import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))

from src.datasets.eeg_dataset import EEGDataset  # noqa: E402
from src.datasets.multi_spectrogram import MultiSpectrogramDataset  # noqa: E402
from src.models.base_cnn import BaseCNN  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402


class FeatureExtraction:

    def __init__(self, cfg, device):

        self.CFG = cfg
        self.device = device

    def get_eeg_inference_loader(self, df, data_path, downsample_factor):
        dataset = EEGDataset(df, data_path, mode="train", downsample_factor=downsample_factor)
        loader = DataLoader(
            dataset,
            batch_size=self.CFG.INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=self.CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def get_spec_inference_loader(self, df, targets, data_path, img_size, eeg_spec_path):
        dataset = MultiSpectrogramDataset(df, targets, data_path, img_size, eeg_spec_path, mode="train")
        loader = DataLoader(
            dataset,
            batch_size=self.CFG.INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=self.CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def build_tcn_feature_extractor(self, model_path):
        model = TCNModel(
            num_inputs=self.CFG.TCN_NUM_CHANNELS,
            num_outputs=self.CFG.TARGET_SIZE,
            channel_sizes=self.CFG.TCN_CHANNEL_SIZES,
            kernel_size=self.CFG.TCN_KERNEL_SIZE,
            dropout=self.CFG.TCN_DROPOUT,
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.fc = nn.Identity()
        model.eval()
        return model

    def build_cnn_feature_extractor(self, model_path):
        model = BaseCNN(
            self.CFG.CNN_MODEL_NAME,
            pretrained=False,
            num_classes=self.CFG.TARGET_SIZE,
            in_channels=self.CFG.CNN_IN_CHANNELS,
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.model.classifier = nn.Identity()
        model.eval()
        return model

    def extract_features(self, model, loader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for signals, labels in tqdm(loader, desc="Extracting Features"):
                signals = signals.to(self.device)
                features = model(signals)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)
