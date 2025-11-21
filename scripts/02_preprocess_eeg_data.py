import argparse
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
from src.datasets.eeg_processor import EEGDataProcessor  # noqa: E402

if __name__ == "__main__":

    def list_of_strings(arg):
        return arg.split(",")

    parser = argparse.ArgumentParser(description="Process EEG data using different voting methods")
    parser.add_argument(
        "--raw_data_path",
        help="Path to the raw EEG data directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--processed_data_path",
        help="Path to save processed data",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--vote_methods",
        help="Voting methods to use (comma-separated: 'max_vote_window', 'sum_and_normalize')",
        type=list_of_strings,
        default="max_vote_window,sum_and_normalize",
    )
    parser.add_argument(
        "--skip_parquet",
        help="Skip creating parquet files for processed EEGs",
        action="store_true",
    )

    args = parser.parse_args()

    # Use provided paths or fall back to utility functions
    data_path = Path(__file__).parent.parent / Path("data")
    raw_data_path = args.raw_data_path if args.raw_data_path else data_path
    processed_data_path = args.processed_data_path if args.processed_data_path else data_path / "processed"

    # Initialize the processor
    processor = EEGDataProcessor(raw_data_path=raw_data_path, processed_data_path=processed_data_path)

    # Process data for each voting method
    for i, vote_method in enumerate(args.vote_methods):
        print(f"\n{'=' * 60}")
        print(f"Processing with vote_method: {vote_method}")
        print(f"{'=' * 60}")

        # Skip parquet creation for all methods except the first one if --skip_parquet is set
        skip_parquet = args.skip_parquet or (i > 0)

        processed_df = processor.process_data(vote_method=vote_method, skip_parquet=skip_parquet)

        print(f"\nFinal Data Head ({vote_method}):")
        print(processed_df.head())
