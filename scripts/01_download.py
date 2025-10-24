import argparse
import zipfile
from pathlib import Path

import kaggle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        help="Target directory to download data into",
        required=False,
        type=Path,
    )
    args = parser.parse_args()

download_path = args.target_dir if args.target_dir else Path(__file__).parent.parent / Path("data")
if not download_path.exists():
    print("The target download directory {download_path} does not exist yet. Creating it now.")
    download_path.mkdir(parents=True, exist_ok=True)

kaggle.api.authenticate()

print("Starting download of zip file")
kaggle.api.competition_download_files(
    "hms-harmful-brain-activity-classification",
    path=download_path,
    quiet=False,
    force=False,
)
print("Finished download")

print("Unzipping zip archive")
zip_path = download_path / Path("hms-harmful-brain-activity-classification.zip")
with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(download_path)
print("Finished unzipping")

print("Deleting zip archive")
zip_path.unlink()
