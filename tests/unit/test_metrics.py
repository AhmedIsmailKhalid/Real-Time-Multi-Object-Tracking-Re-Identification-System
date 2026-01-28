"""
Unit tests for evaluation metrics.
"""

import sys
from pathlib import Path

import numpy as np

from src.evaluation.mot_metrics import MOTMetrics
from src.evaluation.reid_metrics import ReIDMetrics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMOTMetrics:
    """Tests for MOT metrics."""

    def test_mot_metrics_init(self):
        """Test MOT metrics initialization."""
        metrics = MOTMetrics()

        assert metrics is not None
        assert metrics.num_frames == 0
        assert metrics.num_matches == 0

    def test_mot_metrics_perfect_tracking(self):
        """Test metrics with perfect tracking."""
        metrics = MOTMetrics()

        # Perfect match: same boxes
        gt_boxes = [(10, 10, 50, 50, 1), (60, 60, 100, 100, 2)]
        pred_boxes = [(10, 10, 50, 50, 1), (60, 60, 100, 100, 2)]

        metrics.update(gt_boxes, pred_boxes, iou_threshold=0.5)

        results = metrics.compute_metrics()

        assert results["MOTA"] == 100.0  # Perfect accuracy
        assert results["FP"] == 0
        assert results["FN"] == 0
        assert results["IDsw"] == 0

    def test_mot_metrics_no_detections(self):
        """Test metrics with no detections."""
        metrics = MOTMetrics()

        gt_boxes = [(10, 10, 50, 50, 1)]
        pred_boxes = []

        metrics.update(gt_boxes, pred_boxes, iou_threshold=0.5)

        results = metrics.compute_metrics()

        assert results["FN"] == 1
        assert results["Recall"] == 0.0

    def test_mot_metrics_false_positives(self):
        """Test metrics with false positives."""
        metrics = MOTMetrics()

        gt_boxes = []
        pred_boxes = [(10, 10, 50, 50, 1)]

        metrics.update(gt_boxes, pred_boxes, iou_threshold=0.5)

        results = metrics.compute_metrics()

        assert results["FP"] == 1

    def test_mot_metrics_id_switch(self):
        """Test ID switch detection."""
        metrics = MOTMetrics()

        # Frame 1: Track 1 matches GT 1
        gt_boxes = [(10, 10, 50, 50, 1)]
        pred_boxes = [(10, 10, 50, 50, 1)]
        metrics.update(gt_boxes, pred_boxes, iou_threshold=0.5)

        # Frame 2: Track 1 now matches GT 2 (ID switch)
        gt_boxes = [(10, 10, 50, 50, 2)]
        pred_boxes = [(10, 10, 50, 50, 1)]
        metrics.update(gt_boxes, pred_boxes, iou_threshold=0.5)

        results = metrics.compute_metrics()

        assert results["IDsw"] == 1

    def test_mot_metrics_reset(self):
        """Test metrics reset."""
        metrics = MOTMetrics()

        gt_boxes = [(10, 10, 50, 50, 1)]
        pred_boxes = [(10, 10, 50, 50, 1)]
        metrics.update(gt_boxes, pred_boxes)

        metrics.reset()

        assert metrics.num_frames == 0
        assert metrics.num_matches == 0


class TestReIDMetrics:
    """Tests for Re-ID metrics."""

    def test_reid_metrics_init(self):
        """Test Re-ID metrics initialization."""
        metrics = ReIDMetrics(distance_metric="euclidean")

        assert metrics is not None
        assert metrics.distance_metric == "euclidean"

    def test_reid_metrics_distance_matrix_euclidean(self):
        """Test Euclidean distance matrix computation."""
        metrics = ReIDMetrics(distance_metric="euclidean")

        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        gallery = np.array([[1.0, 0.0], [0.0, 1.0]])

        dist = metrics.compute_distance_matrix(query, gallery)

        assert dist.shape == (2, 2)
        assert np.isclose(dist[0, 0], 0.0)  # Same vector
        assert np.isclose(dist[1, 1], 0.0)  # Same vector
        assert dist[0, 1] > 0  # Different vectors

    def test_reid_metrics_distance_matrix_cosine(self):
        """Test cosine distance matrix computation."""
        metrics = ReIDMetrics(distance_metric="cosine")

        # L2-normalized vectors
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        gallery = np.array([[1.0, 0.0], [0.0, 1.0]])

        dist = metrics.compute_distance_matrix(query, gallery)

        assert dist.shape == (2, 2)
        assert np.isclose(dist[0, 0], 0.0)  # Same vector
        assert np.isclose(dist[1, 1], 0.0)  # Same vector

    def test_reid_metrics_perfect_ranking(self):
        """Test metrics with perfect ranking."""
        metrics = ReIDMetrics(distance_metric="euclidean")

        # Create features where query matches exactly
        query_features = np.array([[1.0, 0.0]])
        query_ids = np.array([1])
        query_cam_ids = np.array([1])

        gallery_features = np.array([[1.0, 0.0], [0.0, 1.0]])
        gallery_ids = np.array([1, 2])
        gallery_cam_ids = np.array([2, 2])

        results = metrics.evaluate(
            query_features,
            query_ids,
            query_cam_ids,
            gallery_features,
            gallery_ids,
            gallery_cam_ids,
            max_rank=10,
        )

        assert results["Rank-1"] == 100.0  # Perfect match at rank 1
        assert results["mAP"] == 100.0

    def test_reid_metrics_no_matches(self):
        """Test metrics with no matches."""
        metrics = ReIDMetrics(distance_metric="euclidean")

        query_features = np.array([[1.0, 0.0]])
        query_ids = np.array([1])
        query_cam_ids = np.array([1])

        gallery_features = np.array([[0.0, 1.0]])
        gallery_ids = np.array([2])  # Different ID
        gallery_cam_ids = np.array([2])

        results = metrics.evaluate(
            query_features,
            query_ids,
            query_cam_ids,
            gallery_features,
            gallery_ids,
            gallery_cam_ids,
            max_rank=10,
        )

        # No matches should result in 0% metrics
        assert results["Rank-1"] == 0.0
        assert results["mAP"] == 0.0

    def test_reid_metrics_compute_ap(self):
        """Test Average Precision computation."""
        metrics = ReIDMetrics()

        # All matches
        matches = np.array([True, True, True])
        ap = metrics._compute_ap(matches)
        assert ap == 1.0

        # No matches
        matches = np.array([False, False, False])
        ap = metrics._compute_ap(matches)
        assert ap == 0.0

        # Partial matches
        matches = np.array([True, False, True])
        ap = metrics._compute_ap(matches)
        assert 0.0 < ap < 1.0
