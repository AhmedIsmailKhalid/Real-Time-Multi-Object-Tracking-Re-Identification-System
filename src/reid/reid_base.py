"""
Abstract base class for Re-ID models.
"""

from abc import ABC, abstractmethod

import torch


class ReIDBase(ABC):
    """Base class for person re-identification models."""

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract appearance features from person crops.

        Args:
            images: Batch of person crops (N, 3, H, W)

        Returns:
            Feature vectors (N, feature_dim)
        """
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (includes classification head).

        Args:
            images: Batch of person crops (N, 3, H, W)

        Returns:
            Class logits (N, num_classes)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        pass
