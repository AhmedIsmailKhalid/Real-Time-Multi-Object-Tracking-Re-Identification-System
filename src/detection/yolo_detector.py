"""
YOLOv8 object detector wrapper.
Provides clean interface to Ultralytics YOLOv8.
"""

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.detection.detector_base import DetectorBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YOLODetector(DetectorBase):
    """YOLOv8 object detector."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: list[int] | None = None,
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path: Path to YOLOv8 model (.pt file)
            device: Device to run inference on ("cuda" or "cpu")
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            target_classes: list of class IDs to keep (None = all classes)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []

        # Load model
        self.model = self._load_model()
        logger.info(
            f"YOLOv8 detector initialized on {device} "
            f"(conf={conf_threshold}, iou={iou_threshold})"
        )

    def _load_model(self) -> YOLO:
        """Load YOLOv8 model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            # If you trust the checkpoint file, force legacy torch.load behavior.
            # This fixes PyTorch 2.6+ default weights_only=True incompatibility in ultralytics.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

            model = YOLO(str(self.model_path))
            model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float, float, int]]:
        """
        Run detection on image.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            list of detections: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detector")
            return []

        # Run inference
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Filter by target classes if specified
                if self.target_classes and class_id not in self.target_classes:
                    continue

                detections.append((x1, y1, x2, y2, confidence, class_id))

        return detections

    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for detections.

        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        self.conf_threshold = threshold
        logger.info(f"Confidence threshold set to {threshold}")

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "target_classes": self.target_classes,
            "model_type": "YOLOv8",
        }
