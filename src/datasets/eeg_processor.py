import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.constants import Constants


class EEGDataProcessor:
    """
    A class to process EEG data from raw Parquet files to aggregated NumPy arrays
    and a processed CSV file for machine learning tasks.
    """

    def __init__(
        self,
        raw_data_path: Path,
        processed_data_path: Path,
    ):
        """
        Initializes the processor with specified data paths.
        """
        self.RAW_DATA_PATH = raw_data_path
        self.PROCESSED_DATA_PATH = processed_data_path

        self.RAW_EEG_PATH = os.path.join(self.RAW_DATA_PATH, "train_eegs/")
        self.PROCESSED_EEG_PATH = os.path.join(self.PROCESSED_DATA_PATH, "eegs_parquet/")
        self.TRAIN_CSV = os.path.join(self.RAW_DATA_PATH, "train.csv")
        self.OUTPUT_CSV = os.path.join(self.PROCESSED_DATA_PATH, "train_processed.csv")

        print("Processor initialized.")
        print(f"Raw data path: '{os.path.abspath(self.RAW_DATA_PATH)}'")
        print(f"Processed data path: '{os.path.abspath(self.PROCESSED_DATA_PATH)}'")

    @staticmethod
    def _process_single_eeg(parquet_path: str) -> np.ndarray:
        """
        Reads a single Parquet file, extracts the middle 50 seconds (10,000 rows),
        handles NaN values, and returns the data as a NumPy array.
        """
        eeg = pd.read_parquet(parquet_path, columns=Constants.EEG_FEATURES)

        rows = len(eeg)
        offset = (rows - 10_000) // 2
        eeg = eeg.iloc[offset : offset + 10_000]

        processed_data = np.zeros((10_000, len(Constants.EEG_FEATURES)), dtype=np.float32)

        for j, col in enumerate(Constants.EEG_FEATURES):
            signal = eeg[col].values.astype("float32")

            if np.isnan(signal).mean() < 1.0:
                mean_val = np.nanmean(signal)
                signal = np.nan_to_num(signal, nan=mean_val)
            else:
                signal[:] = 0

            processed_data[:, j] = signal

        return processed_data

    @staticmethod
    def _create_eeg_parquet_files(df, raw_eeg_path, processed_eeg_path):
        """
        Reads raw individual EEG parquet files, processes them,
        and saves new, smaller parquet files.
        """
        os.makedirs(processed_eeg_path, exist_ok=True)

        for eeg_id in tqdm(df["eeg_id"].unique(), desc="Processing EEG files to Parquet"):
            source_path = os.path.join(raw_eeg_path, f"{eeg_id}.parquet")

            processed_signals = EEGDataProcessor._process_single_eeg(source_path)

            signals_df = pd.DataFrame(processed_signals, columns=Constants.EEG_FEATURES)

            output_path = os.path.join(processed_eeg_path, f"{eeg_id}.parquet")

            signals_df.to_parquet(output_path, index=False)

    def _aggregate_sum_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates votes and spectrogram info for each unique eeg_id.
        """
        print("Using 'sum_and_normalize' vote aggregation strategy with spectrogram info.")
        agg_dict = {
            "spectrogram_id": "first",
            "spectrogram_label_offset_seconds": ["min", "max"],
            "patient_id": "first",
            "expert_consensus": "first",
        }
        agg_df = df.groupby(Constants.EEG_ID_COL).agg({**agg_dict, **{t: "sum" for t in Constants.TARGETS}})
        agg_df.columns = [
            "spectrogram_id",
            "min_offset",
            "max_offset",
            "patient_id",
            "expert_consensus",
        ] + Constants.TARGETS
        vote_values = agg_df[Constants.TARGETS].values
        agg_df[Constants.TARGETS] = vote_values / (vote_values.sum(axis=1, keepdims=True) + 1e-9)
        return agg_df.reset_index()

    def _aggregate_max_vote_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each eeg_id, finds the window with the maximum total votes.
        """
        print("Using 'max_vote_window' vote aggregation strategy.")
        df["total_votes"] = df[Constants.TARGETS].sum(axis=1)
        idx = df.groupby(Constants.EEG_ID_COL)["total_votes"].idxmax()
        max_vote_df = df.loc[idx].copy()
        offset_agg = df.groupby(Constants.EEG_ID_COL)["spectrogram_label_offset_seconds"].agg(["min", "max"])
        offset_agg.columns = ["min_offset", "max_offset"]
        max_vote_df = max_vote_df.merge(offset_agg, left_on=Constants.EEG_ID_COL, right_index=True, how="left")
        vote_values = max_vote_df[Constants.TARGETS].values
        max_vote_df[Constants.TARGETS] = vote_values / (vote_values.sum(axis=1, keepdims=True) + 1e-9)
        return max_vote_df.drop(
            columns=[
                # "total_votes",
                "eeg_sub_id",
                "eeg_label_offset_seconds",
                "spectrogram_sub_id",
                "spectrogram_label_offset_seconds",
                "label_id",
            ]
        ).reset_index(drop=True)

    def process_data(self, vote_method: str = "sum_and_normalize", skip_parquet: bool = False) -> pd.DataFrame:
        """
        Main function to run the EEG data processing pipeline.
        """
        print("=" * 50)
        print("Starting EEG Data Processing Pipeline")
        print("=" * 50)

        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        raw_df = pd.read_csv(self.TRAIN_CSV)

        if not skip_parquet:
            self._create_eeg_parquet_files(raw_df, self.RAW_EEG_PATH, self.PROCESSED_EEG_PATH)
        else:
            print("Skipping Parquet file creation as requested.")

        if vote_method == "sum_and_normalize":
            processed_df = self._aggregate_sum_and_normalize(raw_df)
        elif vote_method == "max_vote_window":
            processed_df = self._aggregate_max_vote_window(raw_df)
        else:
            raise ValueError("Invalid vote_method. Choose 'sum_and_normalize' or 'max_vote_window'.")

        assert processed_df[Constants.EEG_ID_COL].nunique() == len(processed_df)

        processed_df.to_csv(self.OUTPUT_CSV, index=False)
        print(f"\nProcessed train data saved to '{self.OUTPUT_CSV}'.")
        print(f"Shape of the final dataframe: {processed_df.shape}")
        print("\nPipeline finished successfully!")
        print("=" * 50)

        return processed_df
