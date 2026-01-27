"""
Unit tests for detection module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from src.detection.utils import clip_boxes, compute_iou, filter_detections_by_class
from src.detection.yolo_detector import YOLODetector

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestYOLODetector:
    """Tests for YOLODetector class."""

    @pytest.fixture
    def detector(self, request):
        """Create detector instance for testing."""
        model_path = Path("models/detection/pretrained/yolov8s.pt")
        if not model_path.exists():
            pytest.skip("YOLOv8s model not found")

        # Get --gpu flag from request
        use_gpu = request.config.getoption("--gpu")

        return YOLODetector(
            model_path=model_path,
            device="cuda" if use_gpu else "cpu",
            conf_threshold=0.5,
            target_classes=[0],  # person class
        )

    def test_detector_init(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert detector.conf_threshold == 0.5
        assert detector.target_classes == [0]

    def test_detector_detect_empty_image(self, detector):
        """Test detection on empty image."""
        empty_image = np.array([])
        detections = detector.detect(empty_image)
        assert detections == []

    def test_detector_detect_black_image(self, detector):
        """Test detection on black image."""
        black_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect(black_image)
        assert isinstance(detections, list)

    def test_detector_detect_sample_image(self, detector):
        """Test detection on sample image with people."""
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        detections = detector.detect(test_image)

        assert isinstance(detections, list)

        if len(detections) > 0:
            det = detections[0]
            assert len(det) == 6
            assert 0 <= det[4] <= 1
            assert det[5] == 0

    def test_set_confidence_threshold(self, detector):
        """Test setting confidence threshold."""
        detector.set_confidence_threshold(0.7)
        assert detector.conf_threshold == 0.7

        with pytest.raises(ValueError):
            detector.set_confidence_threshold(1.5)

    def test_get_model_info(self, detector):
        """Test getting model info."""
        info = detector.get_model_info()

        assert isinstance(info, dict)
        assert "model_path" in info
        assert "device" in info
        assert "conf_threshold" in info
        assert info["model_type"] == "YOLOv8"


class TestDetectionUtils:
    """Tests for detection utility functions."""

    def test_compute_iou_identical_boxes(self):
        """Test IoU of identical boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)

        iou = compute_iou(box1, box2)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)

        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)

        iou = compute_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_filter_detections_by_class(self):
        """Test filtering detections by class."""
        detections = [
            (0, 0, 100, 100, 0.9, 0),
            (100, 100, 200, 200, 0.8, 1),
            (200, 200, 300, 300, 0.95, 0),
        ]

        filtered = filter_detections_by_class(detections, [0])

        assert len(filtered) == 2
        assert all(det[5] == 0 for det in filtered)

    def test_clip_boxes_within_bounds(self):
        """Test clipping boxes that are within image bounds."""
        boxes = [(10, 10, 90, 90)]
        image_shape = (100, 100)

        clipped = clip_boxes(boxes, image_shape)

        assert clipped == boxes

    def test_clip_boxes_outside_bounds(self):
        """Test clipping boxes that exceed image bounds."""
        boxes = [(-10, -10, 110, 110)]
        image_shape = (100, 100)

        clipped = clip_boxes(boxes, image_shape)

        assert clipped[0] == (0, 0, 100, 100)

    def test_clip_boxes_partially_outside(self):
        """Test clipping boxes partially outside bounds."""
        boxes = [(50, 50, 150, 150)]
        image_shape = (100, 100)

        clipped = clip_boxes(boxes, image_shape)

        assert clipped[0] == (50, 50, 100, 100)
