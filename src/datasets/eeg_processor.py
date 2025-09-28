import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm  

class EEGDataProcessor:
    """
    A class to process EEG data from raw Parquet files to aggregated NumPy arrays
    and a processed CSV file for machine learning tasks.
    """
    
    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    EEG_FEATURES = [
        'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 
        'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 
        'EKG', 'Fz', 'Cz', 'Pz'
    ]
    
    def __init__(self, raw_data_path: str = '../data/', processed_data_path: str = '../data/processed/'):
        """
        Initializes the processor with specified data paths.

        Args:
            raw_data_path (str): The path to the directory containing raw data (e.g., 'train.csv', 'train_eegs/').
            processed_data_path (str): The path where processed files will be saved.
        """
        self.RAW_DATA_PATH = raw_data_path
        self.PROCESSED_DATA_PATH = processed_data_path
        
        self.RAW_EEG_PATH = os.path.join(self.RAW_DATA_PATH, 'train_eegs/')
        self.PROCESSED_EEG_PATH = os.path.join(self.PROCESSED_DATA_PATH, 'eegs_npy/')
        self.TRAIN_CSV = os.path.join(self.RAW_DATA_PATH, 'train.csv')
        self.OUTPUT_CSV = os.path.join(self.PROCESSED_DATA_PATH, 'train_processed.csv')
        
        print(f"Processor initialized.")
        print(f"Raw data path: '{os.path.abspath(self.RAW_DATA_PATH)}'")
        print(f"Processed data path: '{os.path.abspath(self.PROCESSED_DATA_PATH)}'")

    @staticmethod
    def _process_single_eeg(parquet_path: str) -> np.ndarray:
        """
        Reads a single Parquet file, extracts the middle 50 seconds (10,000 rows),
        handles NaN values, and returns the data as a NumPy array.
        """
        eeg = pd.read_parquet(parquet_path, columns=EEGDataProcessor.EEG_FEATURES)
        
        rows = len(eeg)
        offset = (rows - 10_000) // 2
        eeg = eeg.iloc[offset:offset + 10_000]
        
        processed_data = np.zeros((10_000, len(EEGDataProcessor.EEG_FEATURES)), dtype=np.float32)
        
        for j, col in enumerate(EEGDataProcessor.EEG_FEATURES):
            signal = eeg[col].values.astype('float32')
            
            if np.isnan(signal).mean() < 1.0:
                mean_val = np.nanmean(signal)
                signal = np.nan_to_num(signal, nan=mean_val)
            else: 
                signal[:] = 0
                
            processed_data[:, j] = signal
            
        return processed_data

    def _create_eeg_npy_files(self, eeg_ids: np.ndarray):
        """
        Converts all specified EEG Parquet files to NumPy arrays and saves them.
        """
        print(f"Creating NumPy EEG files in '{self.PROCESSED_EEG_PATH}'...")
        os.makedirs(self.PROCESSED_EEG_PATH, exist_ok=True)
        
        for eeg_id in tqdm(eeg_ids, desc="Converting Parquet to NumPy"):
            parquet_file = os.path.join(self.RAW_EEG_PATH, f'{eeg_id}.parquet')
            data = self._process_single_eeg(parquet_file)
            np.save(os.path.join(self.PROCESSED_EEG_PATH, f'{eeg_id}.npy'), data)
            
        print("NumPy file creation complete.")

    def _aggregate_sum_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Aggregates votes and spectrogram info for each unique eeg_id.
            It sums the votes and normalizes them to create a probability distribution.
            It also captures the spectrogram ID and the min/max offsets.
            """
            print("Using 'sum_and_normalize' vote aggregation strategy with spectrogram info.")
            
            agg_dict = {
                'spectrogram_id': 'first',
                'spectrogram_label_offset_seconds': ['min', 'max'],
                'patient_id': 'first',
                'expert_consensus': 'first'
            }
            
            agg_df = df.groupby('eeg_id').agg({**agg_dict, **{t: 'sum' for t in self.TARGETS}})
            
            agg_df.columns = ['spectrogram_id', 'min_offset', 'max_offset', 'patient_id', 'expert_consensus'] + self.TARGETS
            
            vote_values = agg_df[self.TARGETS].values
            agg_df[self.TARGETS] = vote_values / (vote_values.sum(axis=1, keepdims=True) + 1e-9)
            
            return agg_df.reset_index()

    def _aggregate_max_vote_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each eeg_id, finds the window (row) with the maximum total votes
        and uses only that window's votes.
        """
        print("Using 'max_vote_window' vote aggregation strategy.")
        
        df['total_votes'] = df[self.TARGETS].sum(axis=1)
        idx = df.groupby('eeg_id')['total_votes'].idxmax()
        max_vote_df = df.loc[idx].copy()
        
        vote_values = max_vote_df[self.TARGETS].values
        normalized_votes = vote_values / (vote_values.sum(axis=1, keepdims=True) + 1e-9)
        max_vote_df[self.TARGETS] = normalized_votes
        
        return max_vote_df.drop(columns=['total_votes']).reset_index(drop=True)

    def process_data(self, vote_method: str = 'sum_and_normalize', skip_npy: bool = False) -> pd.DataFrame:
        """
        Main function to run the EEG data processing pipeline.

        Args:
            vote_method (str): The strategy for aggregating votes. 
                               Choices: 'sum_and_normalize' or 'max_vote_window'.
            skip_npy (bool): If True, skips the creation of .npy files, assuming they already exist.

        Returns:
            pd.DataFrame: The final processed and aggregated dataframe.
        """
        print("=" * 50)
        print("Starting EEG Data Processing Pipeline")
        print("=" * 50)
        
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        raw_df = pd.read_csv(self.TRAIN_CSV)
        
        if not skip_npy:
            eeg_ids = raw_df.eeg_id.unique()
            self._create_eeg_npy_files(eeg_ids)
        else:
            print("Skipping NumPy file creation as requested.")
                
        if vote_method == 'sum_and_normalize':
            processed_df = self._aggregate_sum_and_normalize(raw_df)
        elif vote_method == 'max_vote_window':
            processed_df = self._aggregate_max_vote_window(raw_df)
        else:
            raise ValueError("Invalid vote_method. Choose 'sum_and_normalize' or 'max_vote_window'.")
            
        processed_df.to_csv(self.OUTPUT_CSV, index=False)
        print(f"\nProcessed train data saved to '{self.OUTPUT_CSV}'.")
        print(f"Shape of the final dataframe: {processed_df.shape}")
        
        print("\nPipeline finished successfully!")
        print("=" * 50)
        
        return processed_df