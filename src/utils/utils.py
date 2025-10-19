import logging
import os
import random
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

from src.utils.constants import Constants

RandomState = Union[np.random.Generator, np.random.RandomState, int, None]


def get_raw_data_dir() -> Path:
    if running_in_kaggle():
        return Constants.KAGGLE_DATA_BASE_PATH
    else:
        return get_library_root() / "data"


def get_processed_data_dir() -> Path:
    if running_in_kaggle():
        return Path("/kaggle/temp/processed")
    else:
        return get_library_root() / "data" / "processed"


def get_submission_csv_path() -> Path:
    if running_in_kaggle():
        return Path("/kaggle/working/submission.csv")
    else:
        return get_library_root() / "data" / "submission.csv"


def get_models_save_path(kaggle_hsm_models_dataset_id: str = "") -> Path:
    if running_in_kaggle():
        if kaggle_hsm_models_dataset_id:
            # if a dataset name is provided, use it to load existing model checkpoints
            return Path("/kaggle/input") / kaggle_hsm_models_dataset_id
        else:
            # if no dataset name is provided, use temp path
            # can be the case when we want to train and save models in Kaggle and not load existing checkpoints
            return Path("/kaggle/temp/models")
    else:
        return get_library_root() / "models"


def get_eeg_spectrogram_path(split: str, kaggle_eeg_spectrogram_dataset_id: str) -> Path:
    if running_in_kaggle():
        if split == "train":
            # for train, load the preprocessed spectrograms from input dataset
            return Path("/kaggle/input") / kaggle_eeg_spectrogram_dataset_id
        else:
            # if not, use temp path
            return Path("/kaggle/temp/eeg_spectrograms") / split
    else:
        return get_processed_data_dir() / "eeg_spectrograms" / split


def running_in_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_URL_BASE", ""))


def set_seeds(seed: int = Constants.SEED):
    """
    Set seed for various random generators.

    RandomGenerators affected: ``HASHSEED``, ``random``, ``torch``, ``torch.cuda``,
    ``numpy.random``
    :param seed: Integer seed to set random generators to
    :raises ValueError: If the seed is not an integer
    """
    if not isinstance(seed, int):
        raise ValueError(f"Expect seed to be an integer, but got {type(seed)}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_library_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def fill_nan_with_mean(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        col_mean = np.nanmean(X)
        X = np.nan_to_num(X, nan=col_mean)
    else:
        col_means = np.nanmean(X, axis=0)
        X = np.nan_to_num(X, nan=col_means)
    return X


def walk_and_collect(base_path: str, extensions: Sequence[str]):
    if not isinstance(base_path, str) or not isinstance(extensions, Sequence):
        raise TypeError(
            f"Expected base_path of type str or extensions of type sequence of"
            f" strings, but got {type(base_path)} and {type(extensions)}."
        )
    return [
        os.path.join(path, name)
        for path, _, files in os.walk(base_path)
        for name in files
        if any(name.endswith(s) for s in extensions)
    ]


def pad_sequence(batch, batch_as_features: bool = False):
    # Make all tensor in a batch the same length by padding with zeros
    if batch_as_features:
        permute_tuple = (2, 1, 0)
    else:
        permute_tuple = (1, 0)
    batch = [item.permute(*permute_tuple) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    if batch_as_features:
        permute_tuple = (0, 3, 2, 1)
    else:
        permute_tuple = (0, 2, 1)
    return batch.permute(*permute_tuple)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, label, (optional info)
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, label, *_ in batch:
        tensors += [waveform]
        targets += [label]
    # check if the waveform contains features
    # len(shape) == 2: waveform
    # len(shape) > 2: features
    batch_as_features = len(tensors[0].shape) > 2
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors, batch_as_features=batch_as_features)
    targets = torch.Tensor(targets)
    return tensors, targets


def collate_segments(batch):
    # A data tuple with segments has the form:
    # waveforms, labels, (optional info)
    tensors, targets = None, None
    # Gather in lists, and encode labels as indices
    # As we have segments and labels as a return of __get_item__ we need to concatenate
    # instead of appending
    for waveforms, labels, *_ in batch:
        if tensors is not None and targets is not None:
            tensors = torch.cat([tensors, waveforms], dim=0)
            targets = torch.cat([targets, labels], dim=0)
        else:
            tensors = waveforms
            targets = labels
    # check if the waveform contains features
    # len(shape) == 2: waveform
    # len(shape) > 2: features
    batch_as_features = len(tensors[0].shape) > 2
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors, batch_as_features=batch_as_features)
    targets = torch.Tensor(targets)
    return tensors, targets


def collate_tuples(batch):
    return torch.cat(batch, dim=0)


def initialise_logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
    )


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
