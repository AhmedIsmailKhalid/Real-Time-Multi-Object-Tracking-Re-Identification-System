"""
Download Market-1501 dataset from Google Drive.
"""

import subprocess
import sys
from pathlib import Path


def download_market1501():
    """Download Market-1501 dataset."""

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "Market-1501-v15.09.15.zip"

    if output_file.exists():
        print(f"File already exists: {output_file}")
        return

    print("Downloading Market-1501 dataset from Google Drive...")
    print("This will download ~1.2GB, may take 5-10 minutes")

    # Google Drive file ID for Market-1501
    file_id = "0B8-rUzbwVRk0c054eEozWG9COHM"

    try:
        # Use gdown to download from Google Drive
        cmd = f"gdown {file_id} -O {output_file}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"\nDownload complete: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view")
        sys.exit(1)


if __name__ == "__main__":
    download_market1501()
