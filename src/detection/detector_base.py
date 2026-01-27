"""
Abstract base class for object detectors.
All detectors must implement this interface.
"""

from abc import ABC, abstractmethod

import numpy as np


class DetectorBase(ABC):
    """Base class for object detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float, float, int]]:
        """
        Run object detection on input image.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            List of detections: [(x1, y1, x2, y2, confidence, class_id), ...]
            where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner
        """
        pass

    @abstractmethod
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for detections.

        Args:
            threshold: Confidence threshold (0.0-1.0)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata (name, version, parameters, etc.)
        """
        pass
