"""
Script to push a Jupyter notebook to Kaggle as a kernel.
Creates a new kernel if it doesn't exist, or updates it if it does.
Supports attaching datasets and competitions as data sources.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_kaggle_username():
    """Extract Kaggle username from the API credentials."""
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


def create_kernel_metadata(
    notebook_path, kernel_title, kernel_slug, username, dataset_sources, competition_sources, temp_dir
):
    """
    Create the kernel-metadata.json file required by Kaggle.

    Args:
        notebook_path: Path to the notebook file
        kernel_title: Title of the kernel
        kernel_slug: Kernel slug (URL-friendly name)
        username: Kaggle username
        dataset_sources: List of dataset sources (format: 'username/dataset-name')
        competition_sources: List of competition sources (format: 'competition-name')
        temp_dir: Temporary directory where metadata will be created
    """
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_title,
        "code_file": Path(notebook_path).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources if dataset_sources else [],
        "competition_sources": competition_sources if competition_sources else [],
        "kernel_sources": [],
    }

    metadata_path = Path(temp_dir) / "kernel-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata file: {metadata_path}")
    print("\nKernel configuration:")
    print(f"  ID: {metadata['id']}")
    print(f"  Title: {metadata['title']}")
    print(f"  Type: {metadata['kernel_type']}")
    print(f"  Language: {metadata['language']}")
    print(f"  Private: {metadata['is_private']}")
    print(f"  GPU: {metadata['enable_gpu']}")
    print(f"  Internet: {metadata['enable_internet']}")

    if dataset_sources:
        print(f"  Dataset sources: {', '.join(dataset_sources)}")
    if competition_sources:
        print(f"  Competition sources: {', '.join(competition_sources)}")

    return metadata_path


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


def push_kernel(temp_dir, is_new):
    """
    Push or update a kernel on Kaggle.

    Args:
        temp_dir: Directory containing the notebook and metadata
        is_new: Boolean indicating if this is a new kernel or an update
    """
    print(f"\n{'Creating new' if is_new else 'Updating'} kernel on Kaggle...")
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", temp_dir], capture_output=True, text=True, check=True
        )
        print(result.stdout)

        if "successfully" in result.stdout.lower() or result.returncode == 0:
            print(f"Kernel {'created' if is_new else 'updated'} successfully!")
        else:
            print("Warning: Command completed but response unclear:")
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to push kernel")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def validate_notebook(notebook_path):
    """
    Validate that the notebook file exists and is a valid JSON file.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        bool: True if valid, False otherwise
    """
    if not notebook_path.exists():
        print(f"ERROR: Notebook file not found: {notebook_path}")
        return False

    if not notebook_path.suffix == ".ipynb":
        print(f"ERROR: File must have .ipynb extension: {notebook_path}")
        return False

    try:
        with open(notebook_path, "r") as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        print(f"ERROR: Invalid notebook file (not valid JSON): {notebook_path}")
        return False


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


def main():
    parser = argparse.ArgumentParser(
        description="Push a Jupyter notebook to Kaggle as a kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (create or update kernel)
  python push_kernel.py --notebook my_analysis.ipynb --title "My Analysis"

  # With dataset sources
  python push_kernel.py -n analysis.ipynb -t "Analysis" -d "username/dataset1,username/dataset2"

  # With competition source
  python push_kernel.py -n submission.ipynb -t "My Submission" -c "titanic"

  # With both datasets and competition
  python push_kernel.py -n kernel.ipynb -t "Full Analysis" \\
    -d "username/dataset1,username/dataset2" -c "house-prices-advanced-regression-techniques"

  # Custom slug
  python push_kernel.py -n notebook.ipynb -t "My Kernel" -s "custom-kernel-slug"

Note:
  - Dataset sources format: 'username/dataset-name'
  - Competition sources format: 'competition-name'
  - Multiple sources can be comma-separated
        """,
    )

    parser.add_argument("-n", "--notebook", required=True, help="Path to the local Jupyter notebook (.ipynb file)")

    parser.add_argument(
        "-t",
        "--title",
        required=True,
        help="Title of the kernel on Kaggle.",
    )

    parser.add_argument(
        "-d",
        "--dataset-sources",
        help="Comma-separated list of dataset sources (format: 'username/dataset-name')",
        default="dedeif/hsm-source-files",
    )

    parser.add_argument(
        "-c",
        "--competition-sources",
        help="Comma-separated list of competition sources (format: 'competition-name')",
        default="hms-harmful-brain-activity-classification",
    )

    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")

    parser.add_argument("--tpu", action="store_true", help="Enable TPU for the kernel")

    args = parser.parse_args()

    # Get Kaggle username
    username = get_kaggle_username()
    if not username:
        sys.exit(1)

    print(f"Kaggle username: {username}")

    # Validate notebook path
    notebook_path = Path(args.notebook).resolve()
    if not validate_notebook(notebook_path):
        sys.exit(1)

    print(f"Notebook path: {notebook_path}")

    # Generate or use provided slug
    kernel_slug = create_kernel_slug(args.title)
    print(f"Kernel slug: {kernel_slug}")

    # Parse sources
    dataset_sources = parse_sources(args.dataset_sources)
    competition_sources = parse_sources(args.competition_sources)

    # Check if kernel exists
    is_new = not kernel_exists(username, kernel_slug)

    if is_new:
        print("\nKernel does not exist. Will create new kernel.")
    else:
        print("\nKernel exists. Will update existing kernel.")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        # Copy notebook to temp directory
        temp_notebook_path = Path(temp_dir) / notebook_path.name
        shutil.copy2(notebook_path, temp_notebook_path)
        print(f"Copied notebook to: {temp_notebook_path}")

        # Create kernel metadata with custom parameters
        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": args.title,
            "code_file": notebook_path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": args.gpu,
            "enable_tpu": args.tpu,
            "enable_internet": False,
            "dataset_sources": dataset_sources,
            "competition_sources": competition_sources,
            "kernel_sources": [],
        }

        metadata_path = Path(temp_dir) / "kernel-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nCreated metadata file: {metadata_path}")
        print("\nKernel configuration:")
        print(f"  ID: {metadata['id']}")
        print(f"  Title: {metadata['title']}")
        print(f"  Type: {metadata['kernel_type']}")
        print(f"  Language: {metadata['language']}")
        print(f"  Private: {metadata['is_private']}")
        print(f"  GPU: {metadata['enable_gpu']}")
        print(f"  TPU: {metadata['enable_tpu']}")
        print(f"  Internet: {metadata['enable_internet']}")

        if dataset_sources:
            print(f"  Dataset sources ({len(dataset_sources)}):")
            for ds in dataset_sources:
                print(f"    - {ds}")
        if competition_sources:
            print(f"  Competition sources ({len(competition_sources)}):")
            for cs in competition_sources:
                print(f"    - {cs}")

        # Push kernel
        push_kernel(temp_dir, is_new)

        print("\nOperation completed successfully!")
        print(f"Kernel URL: https://www.kaggle.com/code/{username}/{kernel_slug}")


if __name__ == "__main__":
    main()
