import pandas as pd


def load_middle_50_seconds_of_eeg(data_path, eeg_id):
    eeg = pd.read_parquet(data_path / f"{eeg_id}.parquet")
    middle = (len(eeg) - 10000) // 2  # 10000 samples = 25 seconds
    return eeg.iloc[middle : middle + 10_000]
