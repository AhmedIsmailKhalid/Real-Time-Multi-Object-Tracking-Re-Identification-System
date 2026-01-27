"""
PyTorch Dataset for Market-1501 person re-identification.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Market1501Dataset(Dataset):
    """Market-1501 person re-identification dataset."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Callable | None = None,
    ):
        """
        Initialize Market-1501 dataset.

        Args:
            data_dir: Path to processed Market-1501 data
            split: "train", "val", "query", or "gallery"
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file) as f:
            self.metadata = json.load(f)

        # Load labels for train/val splits
        if split in ["train", "val"]:
            labels_file = self.data_dir / f"{split}_labels.csv"
            self.labels_df = pd.read_csv(labels_file)

            # Get image paths
            self.image_paths = []
            self.person_ids = []
            self.camera_ids = []

            for _, row in self.labels_df.iterrows():
                person_id = row["person_id"]
                img_path = self.data_dir / split / f"{person_id:04d}" / row["filename"]

                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.person_ids.append(person_id)
                    self.camera_ids.append(row["camera_id"])

            # Create person_id to label mapping
            unique_ids = sorted(set(self.person_ids))
            self.id_to_label = {pid: idx for idx, pid in enumerate(unique_ids)}

            logger.info(
                f"Loaded {split} split: {len(self.image_paths)} images, "
                f"{len(unique_ids)} identities"
            )

        elif split in ["query", "gallery"]:
            # For evaluation splits
            split_dir = self.data_dir / split
            self.image_paths = sorted(split_dir.glob("*.jpg"))

            # Parse person_ids and camera_ids from filenames
            self.person_ids = []
            self.camera_ids = []

            for img_path in self.image_paths:
                filename = img_path.name
                parts = filename.replace(".jpg", "").split("_")
                person_id = int(parts[0])
                camera_id = int(parts[1][1])  # Remove 'c' prefix

                self.person_ids.append(person_id)
                self.camera_ids.append(camera_id)

            logger.info(f"Loaded {split} split: {len(self.image_paths)} images")

        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        """Dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            image: Transformed image tensor
            label: Person ID label (for train/val) or person_id (for query/gallery)
            camera_id: Camera ID
            person_id: Original person ID
        """
        img_path = self.image_paths[idx]
        person_id = self.person_ids[idx]
        camera_id = self.camera_ids[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Get label (mapped ID for train/val, original ID for query/gallery)
        if self.split in ["train", "val"]:
            label = self.id_to_label[person_id]
        else:
            label = person_id

        return img, label, camera_id, person_id

    def get_num_classes(self) -> int:
        """
        Get number of unique person identities.

        Returns:
            Number of classes
        """
        if self.split in ["train", "val"]:
            return len(self.id_to_label)
        else:
            return len(set(self.person_ids))
