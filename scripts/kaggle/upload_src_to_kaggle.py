"""
Script to upload src/ folder to Kaggle as a dataset.
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


def create_placeholder_in_src(temp_src_path):
    """
    Create a placeholder file in the src/ directory to ensure it's preserved in the dataset.

    Args:
        temp_src_path: Path to the temporary src/ directory
    """
    placeholder_path = temp_src_path / ".gitkeep"
    placeholder_path.write_text("# This file ensures the src/ directory structure is preserved\n")
    print(f"Created placeholder file: {placeholder_path}")


def copy_src_to_temp(src_path, temp_dir):
    """
    Copy the src/ folder to a temporary directory.

    Args:
        src_path: Path to the source src/ directory
        temp_dir: Path to the temporary directory

    Returns:
        Path to the copied src/ directory in temp
    """
    temp_src_path = Path(temp_dir) / "src"

    if src_path.exists():
        print(f"Copying {src_path} to {temp_src_path}")
        shutil.copytree(src_path, temp_src_path)

        # Count files copied
        file_count = sum(1 for _ in temp_src_path.rglob("*") if _.is_file())
        print(f"Copied {file_count} files from src/")
    else:
        print(f"ERROR: Source directory {src_path} does not exist!")
        sys.exit(1)

    return temp_src_path.parent


def main():
    parser = argparse.ArgumentParser(
        description="Upload src/ folder to Kaggle as a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_src_to_kaggle.py --title "My Project Source" --id "username/my-project-src"
  python upload_src_to_kaggle.py -t "ML Pipeline" -i "username/ml-pipeline-code"
        """,
    )

    parser.add_argument("-t", "--title", help="Title of the dataset", default="HSM source files")

    parser.add_argument(
        "-i", "--id", help="Dataset ID in format 'username/dataset-name'", default="dedeif/hsm-source-files"
    )

    args = parser.parse_args()

    # Validate dataset ID format
    if not validate_dataset_id(args.id):
        sys.exit(1)

    # Get the src/ path
    src_path = Path(__file__).parent.parent.parent / "src"
    src_path = src_path.resolve()

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        # Copy src/ folder to temp directory
        temp_src_path = copy_src_to_temp(src_path, temp_dir)

        # Create placeholder file in src/ subdirectory
        create_placeholder_in_src(temp_src_path)

        # Create dataset metadata
        create_dataset_metadata(args.title, args.id, temp_dir)

        # Check if dataset exists and create or update accordingly
        if dataset_exists(args.id):
            update_dataset(temp_dir, args.id)
        else:
            create_dataset(temp_dir, args.id)

        shutil.rmtree(temp_dir)

        print("\nOperation completed successfully!")
        print(f"Dataset URL: https://www.kaggle.com/datasets/{args.id}")


if __name__ == "__main__":
    main()
