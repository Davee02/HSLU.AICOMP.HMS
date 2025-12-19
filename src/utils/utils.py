import logging
import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
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


def fill_nan_with_zero(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0)

def initialise_logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
    )

def load_middle_50_seconds_of_eeg(data_path, eeg_id):
    eeg = pd.read_parquet(data_path / f"{eeg_id}.parquet")
    middle = (len(eeg) - 10000) // 2  # 10000 samples = 25 seconds
    return eeg.iloc[middle : middle + 10_000]