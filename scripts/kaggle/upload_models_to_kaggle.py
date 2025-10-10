"""
Script to upload models/ folder to Kaggle as a dataset.
Creates a new dataset if it doesn't exist, or updates it with a new version if it does.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from kaggle_utils import (
    create_dataset,
    create_dataset_metadata,
    dataset_exists,
    update_dataset,
    validate_dataset_id,
)


def copy_models_to_temp(models_path, temp_dir):
    """
    Copy the contents of the models/ folder to the temporary directory.
    Subfolders of models/ are placed at the root of the dataset.

    Args:
        models_path: Path to the source models/ directory
        temp_dir: Path to the temporary directory

    Returns:
        Path to the temporary directory
    """
    if not models_path.exists():
        print(f"ERROR: Source directory {models_path} does not exist!")
        sys.exit(1)

    print(f"Copying contents of {models_path} to {temp_dir}")

    file_count = 0
    total_size = 0

    # Copy all items from models/ to the root of temp_dir
    for item in models_path.iterdir():
        dest = Path(temp_dir) / item.name

        if item.is_dir():
            shutil.copytree(item, dest)
            # Count files in subdirectory
            subdir_files = sum(1 for _ in dest.rglob("*") if _.is_file())
            file_count += subdir_files
            print(f"Copied folder: {item.name}/ ({subdir_files} files)")
        else:
            shutil.copy2(item, dest)
            file_count += 1
            print(f"Copied file: {item.name}")

        # Calculate size
        if dest.is_dir():
            total_size += sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        else:
            total_size += dest.stat().st_size

    size_mb = total_size / (1024 * 1024)
    print(f"\nTotal: {file_count} files, {size_mb:.2f} MB")

    return Path(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Upload models/ folder to Kaggle as a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_models_to_kaggle.py --title "HSM trained models" --id "username/hsm-models"
  python upload_models_to_kaggle.py -t "My Models" -i "username/my-models"
        """,
    )

    parser.add_argument("-t", "--title", help="Title of the dataset", default="HSM trained models")

    parser.add_argument("-i", "--id", help="Dataset ID in format 'username/dataset-name'", default="dedeif/hsm-models")

    args = parser.parse_args()

    # Validate dataset ID format
    if not validate_dataset_id(args.id):
        sys.exit(1)

    # Get the models/ path (assuming script is in scripts/kaggle/)
    models_path = Path(__file__).parent.parent.parent / "models"
    models_path = models_path.resolve()

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        # Copy models/ folder contents to temp directory (subfolders go to root)
        copy_models_to_temp(models_path, temp_dir)

        # Create dataset metadata
        create_dataset_metadata(args.title, args.id, temp_dir)

        # Check if dataset exists and create or update accordingly
        if dataset_exists(args.id):
            update_dataset(temp_dir, args.id)
        else:
            create_dataset(temp_dir, args.id)

        print("\nOperation completed successfully!")
        print(f"Dataset URL: https://www.kaggle.com/datasets/{args.id}")


if __name__ == "__main__":
    main()
