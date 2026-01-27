"""
ResNet50-based person re-identification network.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.reid.reid_base import ReIDBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResNet50ReID(nn.Module, ReIDBase):
    """ResNet50 backbone for person Re-ID."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        """
        Initialize ResNet50 Re-ID model.

        Args:
            num_classes: Number of person identities (for training)
            pretrained: Use ImageNet pre-trained weights
            feature_dim: Dimension of output feature vector
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Load ResNet50 backbone
        if pretrained:
            from torchvision.models import ResNet50_Weights

            resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet50 = models.resnet50(weights=None)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])

        # Bottleneck layer
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)

        # Feature projection
        self.fc = nn.Linear(2048, feature_dim, bias=False)
        self.bn_feat = nn.BatchNorm1d(feature_dim)
        self.bn_feat.bias.requires_grad_(False)

        # Classification head (for training)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        self._init_params()

        logger.info(
            f"ResNet50ReID initialized (num_classes={num_classes}, "
            f"feature_dim={feature_dim}, pretrained={pretrained})"
        )

    def _init_params(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.constant_(self.bn_feat.weight, 1)
        nn.init.constant_(self.bn_feat.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (training mode - returns class logits).

        Args:
            x: Input images (N, 3, H, W)

        Returns:
            Class logits (N, num_classes)
        """
        # Extract features
        global_feat = self.backbone(x)  # (N, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.size(0), -1)  # (N, 2048)

        # Bottleneck
        feat = self.bottleneck(global_feat)  # (N, 2048)

        # Feature projection
        feat = self.fc(feat)  # (N, feature_dim)
        feat = self.bn_feat(feat)  # (N, feature_dim)

        # Classification
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score
        else:
            return feat

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized feature vectors (inference mode).

        Args:
            images: Batch of person crops (N, 3, H, W)

        Returns:
            L2-normalized feature vectors (N, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            # Extract features
            global_feat = self.backbone(images)
            global_feat = global_feat.view(global_feat.size(0), -1)

            # Bottleneck
            feat = self.bottleneck(global_feat)

            # Feature projection
            feat = self.fc(feat)
            feat = self.bn_feat(feat)

            # L2 normalization
            feat = nn.functional.normalize(feat, p=2, dim=1)

        return feat

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_type": "ResNet50ReID",
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "backbone": "ResNet50",
        }
