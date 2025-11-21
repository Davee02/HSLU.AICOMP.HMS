import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
from src.utils.data_utils import load_middle_50_seconds_of_eeg  # noqa: E402
from src.utils.eeg_spectrogram_creator import EEGSpectrogramGenerator  # noqa: E402

if __name__ == "__main__":

    def list_of_strings(arg):
        return arg.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_eegs_dir",
        help="Directory containing the raw EEG files",
        required=False,
        type=Path,
    )
    parser.add_argument(
        "--data_csv",
        help="CSV file containing the EEG IDs",
        required=False,
        type=Path,
    )
    parser.add_argument(
        "--target_spectrograms_dir",
        help="Directory to save the generated spectrograms",
        required=False,
        type=Path,
    )
    parser.add_argument(
        "--spectrogram_types",
        help="Types of spectrograms to create ('mel', 'stft', 'cwt)",
        required=True,
        type=list_of_strings,
    )

    args = parser.parse_args()

    data_path = Path(__file__).parent.parent / Path("data")
    raw_eegs_dir = args.raw_eegs_dir if args.raw_eegs_dir else data_path / "train_eegs"
    data_csv = args.data_csv if args.data_csv else data_path / "train.csv"
    target_spectrograms_dir = (
        args.target_spectrograms_dir
        if args.target_spectrograms_dir
        else data_path / "processed" / "eeg_spectrograms" / "train"
    )

    spectrogram_creator = EEGSpectrogramGenerator(args.spectrogram_types)

    for type in args.spectrogram_types:
        type_save_path = target_spectrograms_dir / type
        print(f"Creating directory for {type} spectrograms at {type_save_path}")
        type_save_path.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_csv)
    eeg_ids = data["eeg_id"].unique()

    print(f"Found {len(eeg_ids)} unique EEG IDs in {data_csv}")

    for i, eeg_id in enumerate(tqdm(eeg_ids, desc="Generating spectrograms")):
        eeg = load_middle_50_seconds_of_eeg(raw_eegs_dir, eeg_id)
        spectrograms = spectrogram_creator.generate(eeg)

        for spectrogram_type, spectrogram in spectrograms.items():
            out_path = target_spectrograms_dir / spectrogram_type / f"{eeg_id}.npy"
            np.save(out_path, spectrogram)
