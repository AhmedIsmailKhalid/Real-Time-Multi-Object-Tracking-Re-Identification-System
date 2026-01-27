"""
Unit tests for Re-ID module.
"""

import sys
from pathlib import Path

import pytest
import torch

from src.data.market_dataset import Market1501Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.reid.metric_learning import CrossEntropyLabelSmooth, TripletLoss
from src.reid.resnet_reid import ResNet50ReID

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestResNet50ReID:
    """Tests for ResNet50ReID model."""

    def test_model_init(self):
        """Test model initialization."""
        model = ResNet50ReID(num_classes=751, pretrained=False, feature_dim=512)

        assert model is not None
        assert model.num_classes == 751
        assert model.feature_dim == 512

    def test_model_forward_train(self):
        """Test forward pass in training mode."""
        model = ResNet50ReID(num_classes=751, pretrained=False, feature_dim=512)
        model.train()

        batch_size = 4
        images = torch.randn(batch_size, 3, 256, 128)

        output = model(images)

        assert output.shape == (batch_size, 751)

    def test_model_forward_eval(self):
        """Test forward pass in evaluation mode."""
        model = ResNet50ReID(num_classes=751, pretrained=False, feature_dim=512)
        model.eval()

        batch_size = 4
        images = torch.randn(batch_size, 3, 256, 128)

        with torch.no_grad():
            output = model(images)

        assert output.shape == (batch_size, 512)

    def test_extract_features(self):
        """Test feature extraction."""
        model = ResNet50ReID(num_classes=751, pretrained=False, feature_dim=512)

        batch_size = 4
        images = torch.randn(batch_size, 3, 256, 128)

        features = model.extract_features(images)

        assert features.shape == (batch_size, 512)

        # Check L2 normalization
        norms = torch.norm(features, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

    def test_get_model_info(self):
        """Test getting model info."""
        model = ResNet50ReID(num_classes=751, pretrained=False, feature_dim=512)

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert info["model_type"] == "ResNet50ReID"
        assert info["num_classes"] == 751
        assert info["feature_dim"] == 512


class TestTripletLoss:
    """Tests for TripletLoss."""

    def test_triplet_loss_init(self):
        """Test triplet loss initialization."""
        loss_fn = TripletLoss(margin=0.3)

        assert loss_fn is not None
        assert loss_fn.margin == 0.3

    def test_triplet_loss_forward(self):
        """Test triplet loss computation."""
        loss_fn = TripletLoss(margin=0.3)

        # Create dummy features and labels
        batch_size = 8
        feature_dim = 128
        features = torch.randn(batch_size, feature_dim)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = loss_fn(features, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_triplet_loss_zero_for_perfect_embedding(self):
        """Test that loss is zero for perfect embeddings."""
        loss_fn = TripletLoss(margin=0.3)

        # Create perfect embeddings (same class = identical, different class = far)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(features, labels)

        # Loss should be very small (close to zero)
        assert loss.item() < 0.1


class TestCrossEntropyLabelSmooth:
    """Tests for CrossEntropyLabelSmooth."""

    def test_label_smooth_init(self):
        """Test label smoothing loss initialization."""
        loss_fn = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.1)

        assert loss_fn is not None
        assert loss_fn.num_classes == 10
        assert loss_fn.epsilon == 0.1

    def test_label_smooth_forward(self):
        """Test label smoothing loss computation."""
        loss_fn = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.1)

        batch_size = 4
        logits = torch.randn(batch_size, 10)
        labels = torch.tensor([0, 2, 5, 9])

        loss = loss_fn(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestTransforms:
    """Tests for data transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_train_transforms(
            image_size=(256, 128), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        assert transform is not None

    def test_val_transforms(self):
        """Test validation transforms."""
        transform = get_val_transforms(
            image_size=(256, 128), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        assert transform is not None


class TestMarket1501Dataset:
    """Tests for Market1501Dataset."""

    @pytest.fixture
    def dataset_available(self):
        """Check if dataset is available."""
        data_dir = Path("data/processed/market1501")
        return data_dir.exists() and (data_dir / "train").exists()

    def test_dataset_init_train(self, dataset_available):
        """Test dataset initialization for train split."""
        if not dataset_available:
            pytest.skip("Market-1501 dataset not available")

        dataset = Market1501Dataset(
            data_dir=Path("data/processed/market1501"),
            split="train",
            transform=get_train_transforms(),
        )

        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_init_val(self, dataset_available):
        """Test dataset initialization for val split."""
        if not dataset_available:
            pytest.skip("Market-1501 dataset not available")

        dataset = Market1501Dataset(
            data_dir=Path("data/processed/market1501"), split="val", transform=get_val_transforms()
        )

        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_getitem(self, dataset_available):
        """Test getting item from dataset."""
        if not dataset_available:
            pytest.skip("Market-1501 dataset not available")

        dataset = Market1501Dataset(
            data_dir=Path("data/processed/market1501"),
            split="train",
            transform=get_val_transforms(),
        )

        image, label, camera_id, person_id = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 128)
        assert isinstance(label, int)
        assert isinstance(camera_id, int)
        assert isinstance(person_id, int)

    def test_dataset_get_num_classes(self, dataset_available):
        """Test getting number of classes."""
        if not dataset_available:
            pytest.skip("Market-1501 dataset not available")

        dataset = Market1501Dataset(
            data_dir=Path("data/processed/market1501"),
            split="train",
            transform=get_val_transforms(),
        )

        num_classes = dataset.get_num_classes()

        assert num_classes > 0
        assert isinstance(num_classes, int)
