"""
Script to upload src/ folder to Kaggle as a dataset.
Creates a new dataset if it doesn't exist, or updates it with a new version if it does.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def create_placeholder_in_src(temp_src_path):
    """
    Create a placeholder file in the src/ directory to ensure it's preserved in the dataset.

    Args:
        temp_src_path: Path to the temporary src/ directory
    """
    placeholder_path = temp_src_path / ".gitkeep"
    placeholder_path.write_text("# This file ensures the src/ directory structure is preserved\n")
    print(f"Created placeholder file: {placeholder_path}")


def create_dataset_metadata(dataset_title, dataset_id, temp_dir):
    """
    Create the dataset-metadata.json file required by Kaggle.

    Args:
        dataset_title: Title of the dataset
        dataset_id: Dataset ID in format 'username/dataset-name'
        temp_dir: Temporary directory where metadata will be created
    """
    metadata = {"title": dataset_title, "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}

    metadata_path = Path(temp_dir) / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata file: {metadata_path}")
    return metadata_path


def dataset_exists(dataset_id):
    """
    Check if a dataset already exists on Kaggle.

    Args:
        dataset_id: Dataset ID in format 'username/dataset-name'

    Returns:
        bool: True if dataset exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "status", dataset_id], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking dataset existence: {e}")
        return False


def create_dataset(temp_dir, dataset_id):
    """
    Create a new dataset on Kaggle.

    Args:
        temp_dir: Directory containing the dataset files and metadata
        dataset_id: Dataset ID in format 'username/dataset-name'
    """
    print(f"\nCreating new dataset: {dataset_id}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", temp_dir, "-r", "tar", "-t"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("Dataset created successfully!")
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to create dataset")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def update_dataset(temp_dir, dataset_id):
    """
    Update an existing dataset on Kaggle (create a new version).

    Args:
        temp_dir: Directory containing the dataset files and metadata
        dataset_id: Dataset ID in format 'username/dataset-name'
    """
    print(f"\nUpdating existing dataset: {dataset_id}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", temp_dir, "-m", "Updated via upload script", "-r", "tar", "-t"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("Dataset updated successfully!")
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to update dataset")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


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
  python upload_to_kaggle.py --title "My Project Source" --id "username/my-project-src"
  python upload_to_kaggle.py -t "ML Pipeline" -i "username/ml-pipeline-code"
        """,
    )

    parser.add_argument("-t", "--title", help="Title of the dataset", default="HSM source files")

    parser.add_argument(
        "-i", "--id", help="Dataset ID in format 'username/dataset-name'", default="dedeif/hsm-source-files"
    )

    args = parser.parse_args()

    # Validate dataset ID format
    if "/" not in args.id:
        print("ERROR: Dataset ID must be in format 'username/dataset-name'")
        sys.exit(1)

    # Get the src/ path
    src_path = Path(__file__).parent.parent / "src"
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
