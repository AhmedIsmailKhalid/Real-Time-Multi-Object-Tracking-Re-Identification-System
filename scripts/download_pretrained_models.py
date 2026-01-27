"""
Download pre-trained models.
"""
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_yolov8():
    """Download pre-trained YOLOv8s model."""

    output_dir = Path("models/detection/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "yolov8s.pt"

    if output_file.exists():
        logger.info(f"YOLOv8s model already exists: {output_file}")
        return

    logger.info("Downloading YOLOv8s pre-trained model...")

    # Ultralytics downloads to current directory
    temp_model = Path("yolov8s.pt")

    if not temp_model.exists():
        # This will download the model
        model = YOLO("yolov8s")  # Will download yolov8s.pt  # noqa: F841

    # Move to our models directory
    if temp_model.exists():
        shutil.move(str(temp_model), str(output_file))
        logger.info(f"YOLOv8s model saved to {output_file}")
        logger.info(f"Model size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error("Failed to download YOLOv8s model")


def download_resnet50_imagenet():
    """Download pre-trained ResNet50 ImageNet weights."""

    output_dir = Path("models/reid/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "resnet50_imagenet.pth"

    if output_file.exists():
        logger.info(f"ResNet50 model already exists: {output_file}")
        return

    logger.info("Downloading ResNet50 ImageNet pre-trained model...")

    import torchvision.models as models

    # Use weights parameter instead of pretrained
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    torch.save(model.state_dict(), output_file)

    logger.info(f"ResNet50 model downloaded to {output_file}")
    logger.info(f"Model size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Download all pre-trained models."""

    logger.info("Downloading pre-trained models...")

    download_yolov8()
    download_resnet50_imagenet()

    logger.info("All models downloaded successfully!")


if __name__ == "__main__":
    main()
