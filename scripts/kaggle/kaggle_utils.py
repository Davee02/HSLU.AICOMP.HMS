"""
Shared utilities for Kaggle dataset upload scripts.
Contains common functions used across multiple upload scripts.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_kaggle_username():
    """
    Extract Kaggle username from the API credentials.

    Returns:
        str: Kaggle username, or None if not found
    """
    try:
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json_path.exists():
            with open(kaggle_json_path, "r") as f:
                credentials = json.load(f)
                return credentials.get("username")
        else:
            print("ERROR: Could not find kaggle.json credentials file")
            return None
    except Exception as e:
        print(f"ERROR: Failed to read Kaggle credentials: {e}")
        return None


def create_dataset_metadata(dataset_title, dataset_id, temp_dir):
    """
    Create the dataset-metadata.json file required by Kaggle.

    Args:
        dataset_title: Title of the dataset
        dataset_id: Dataset ID in format 'username/dataset-name'
        temp_dir: Temporary directory where metadata will be created

    Returns:
        Path: Path to the created metadata file
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


def validate_dataset_id(dataset_id):
    """
    Validate that the dataset ID is in the correct format.

    Args:
        dataset_id: Dataset ID to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if "/" not in dataset_id:
        print("ERROR: Dataset ID must be in format 'username/dataset-name'")
        return False
    return True


def kernel_exists(username, kernel_slug):
    """
    Check if a kernel already exists on Kaggle.

    Args:
        username: Kaggle username
        kernel_slug: Kernel slug (URL-friendly name)

    Returns:
        bool: True if kernel exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", f"{username}/{kernel_slug}"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking kernel existence: {e}")
        return False


def create_kernel_slug(kernel_title):
    """
    Create a URL-friendly slug from the kernel title.

    Args:
        kernel_title: Title of the kernel

    Returns:
        str: URL-friendly slug
    """
    import re

    # Convert to lowercase
    slug = kernel_title.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Replace multiple consecutive hyphens with single hyphen
    slug = re.sub(r"-+", "-", slug)
    return slug


def parse_sources(sources_str):
    """
    Parse comma-separated sources string into a list.

    Args:
        sources_str: Comma-separated string of sources

    Returns:
        list: List of sources, or empty list if None
    """
    if not sources_str:
        return []
    return [s.strip() for s in sources_str.split(",") if s.strip()]
