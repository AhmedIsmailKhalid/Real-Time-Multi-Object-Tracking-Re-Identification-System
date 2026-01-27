"""
Test that all dependencies are installed correctly.
"""

import sys


def test_imports():
    """Test importing all major dependencies."""

    print("Testing imports...\n")

    try:
        import torch

        print(f"PyTorch {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    except ImportError as e:
        print(f"PyTorch import failed: {e}")
        return False

    try:
        import torchvision

        print(f"torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"torchvision import failed: {e}")
        return False

    try:
        import cv2

        print(f"OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"OpenCV import failed: {e}")
        return False

    try:
        from ultralytics import YOLO  # noqa: F401

        print("Ultralytics YOLOv8")
    except ImportError as e:
        print(f"Ultralytics import failed: {e}")
        return False

    try:
        import fastapi

        print(f"FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"FastAPI import failed: {e}")
        return False

    try:
        import mlflow

        print(f"MLflow {mlflow.__version__}")
    except ImportError as e:
        print(f"MLflow import failed: {e}")
        return False

    try:
        from src.utils.logger import get_logger  # noqa: F401

        print("src.utils.logger import works")
    except ImportError as e:
        print(f"Cannot import from src.utils: {e}")
        return False

    print("\nAll imports successful!")
    return True


if __name__ == "__main__":
    if test_imports():
        print("\nENVIRONMENT SETUP SUCCESSFUL")
        sys.exit(0)
    else:
        print("\nENVIRONMENT SETUP FAILED")
        sys.exit(1)
