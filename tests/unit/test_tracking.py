"""
Unit tests for tracking module.
"""

import sys
from pathlib import Path

import numpy as np

from src.tracking.bytetrack import ByteTracker
from src.tracking.kalman_filter import KalmanFilter, bbox_to_z, z_to_bbox
from src.tracking.matching import iou_batch, iou_distance, linear_assignment
from src.tracking.track import Track, TrackState

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestKalmanFilter:
    """Tests for Kalman filter."""

    def test_kalman_init(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter()
        assert kf.x.shape == (8, 1)
        assert kf.P.shape == (8, 8)
        assert kf.F.shape == (8, 8)

    def test_kalman_predict(self):
        """Test Kalman filter prediction."""
        kf = KalmanFilter()

        # Initialize with a measurement and set some velocity
        measurement = np.array([100, 100, 1.0, 50])
        kf.x[:4] = measurement.reshape((4, 1))
        kf.x[4:6] = np.array([[5], [5]])  # Set some velocity in x and y

        # Predict
        predicted = kf.predict()

        assert predicted.shape == (8, 1)
        # Position should change due to velocity
        assert not np.allclose(predicted[:4], measurement.reshape((4, 1)))

    def test_kalman_update(self):
        """Test Kalman filter update."""
        kf = KalmanFilter()

        # Initialize
        measurement1 = np.array([100, 100, 1.0, 50])
        kf.x[:4] = measurement1.reshape((4, 1))

        # Predict and update
        kf.predict()
        measurement2 = np.array([105, 105, 1.0, 50])
        kf.update(measurement2)

        state = kf.get_state()
        assert state.shape == (8,)

    def test_bbox_to_z(self):
        """Test bounding box to measurement conversion."""
        bbox = (0, 0, 100, 100)
        z = bbox_to_z(bbox)

        assert z.shape == (4,)
        assert z[0] == 50  # x center
        assert z[1] == 50  # y center
        assert z[2] == 1.0  # aspect ratio
        assert z[3] == 100  # height

    def test_z_to_bbox(self):
        """Test measurement to bounding box conversion."""
        z = np.array([50, 50, 1.0, 100])
        bbox = z_to_bbox(z)

        assert len(bbox) == 4
        assert bbox[0] == 0  # x1
        assert bbox[1] == 0  # y1
        assert bbox[2] == 100  # x2
        assert bbox[3] == 100  # y2


class TestMatching:
    """Tests for matching algorithms."""

    def test_iou_batch_identical(self):
        """Test IoU computation for identical boxes."""
        boxes1 = np.array([[0, 0, 100, 100]])
        boxes2 = np.array([[0, 0, 100, 100]])

        iou = iou_batch(boxes1, boxes2)

        assert iou.shape == (1, 1)
        assert np.isclose(iou[0, 0], 1.0)

    def test_iou_batch_no_overlap(self):
        """Test IoU computation for non-overlapping boxes."""
        boxes1 = np.array([[0, 0, 50, 50]])
        boxes2 = np.array([[100, 100, 150, 150]])

        iou = iou_batch(boxes1, boxes2)

        assert iou.shape == (1, 1)
        assert np.isclose(iou[0, 0], 0.0)

    def test_iou_batch_partial_overlap(self):
        """Test IoU computation for partially overlapping boxes."""
        boxes1 = np.array([[0, 0, 100, 100]])
        boxes2 = np.array([[50, 50, 150, 150]])

        iou = iou_batch(boxes1, boxes2)

        assert iou.shape == (1, 1)
        assert 0.0 < iou[0, 0] < 1.0

    def test_iou_distance(self):
        """Test IoU distance computation."""
        tracks = [(0, 0, 100, 100, 0.9, 0)]
        detections = [(50, 50, 150, 150, 0.9, 0)]

        cost = iou_distance(tracks, detections)

        assert cost.shape == (1, 1)
        assert 0.0 < cost[0, 0] < 1.0

    def test_linear_assignment_perfect_match(self):
        """Test linear assignment with perfect matches."""
        cost_matrix = np.array([[0.1, 0.9], [0.9, 0.1]])
        thresh = 0.5

        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, thresh)

        assert len(matches) == 2
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

    def test_linear_assignment_no_match(self):
        """Test linear assignment with no valid matches."""
        cost_matrix = np.array([[0.9, 0.9], [0.9, 0.9]])
        thresh = 0.5

        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, thresh)

        assert len(matches) == 0
        assert len(unmatched_tracks) == 2
        assert len(unmatched_dets) == 2


class TestTrack:
    """Tests for Track class."""

    def test_track_init(self):
        """Test track initialization."""
        bbox = (0, 0, 100, 100)
        track = Track(bbox=bbox, score=0.9, track_id=1, class_id=0)

        assert track.track_id == 1
        assert track.score == 0.9
        assert track.class_id == 0
        assert track.state == TrackState.New

    def test_track_predict(self):
        """Test track prediction."""
        bbox = (0, 0, 100, 100)
        track = Track(bbox=bbox, score=0.9, track_id=1)

        initial_bbox = track.bbox  # noqa: F841
        track.predict()

        predicted_bbox = track.get_current_bbox()
        assert predicted_bbox is not None

    def test_track_activate(self):
        """Test track activation."""
        bbox = (0, 0, 100, 100)
        track = Track(bbox=bbox, score=0.9, track_id=-1)

        track.activate(frame_id=1)

        assert track.is_activated
        assert track.state == TrackState.Tracked
        assert track.frame_id == 1

    def test_track_update(self):
        """Test track update."""
        bbox1 = (0, 0, 100, 100)
        track1 = Track(bbox=bbox1, score=0.9, track_id=1)
        track1.activate(frame_id=1)

        bbox2 = (10, 10, 110, 110)
        track2 = Track(bbox=bbox2, score=0.95, track_id=2)

        track1.update(track2, frame_id=2)

        assert track1.frame_id == 2
        assert track1.score == 0.95
        assert track1.state == TrackState.Tracked


class TestByteTracker:
    """Tests for ByteTracker."""

    def test_bytetrack_init(self):
        """Test ByteTracker initialization."""
        tracker = ByteTracker(
            high_conf_thresh=0.6,
            low_conf_thresh=0.1,
            track_buffer=30,
            match_thresh=0.8,
        )

        assert tracker.high_conf_thresh == 0.6
        assert tracker.low_conf_thresh == 0.1
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.8

    def test_bytetrack_update_empty(self):
        """Test ByteTracker update with no detections."""
        tracker = ByteTracker()

        detections = []
        tracks = tracker.update(detections)

        assert tracks == []

    def test_bytetrack_update_single_detection(self):
        """Test ByteTracker update with single detection."""
        tracker = ByteTracker()

        detections = [(0, 0, 100, 100, 0.9, 0)]
        tracks = tracker.update(detections)

        assert len(tracks) == 1
        assert len(tracks[0]) == 5  # (x1, y1, x2, y2, track_id)

    def test_bytetrack_update_multiple_frames(self):
        """Test ByteTracker across multiple frames."""
        tracker = ByteTracker()

        # Frame 1
        detections1 = [(0, 0, 100, 100, 0.9, 0)]
        tracks1 = tracker.update(detections1)
        track_id_1 = tracks1[0][4]

        # Frame 2 (same object, slightly moved)
        detections2 = [(10, 10, 110, 110, 0.85, 0)]
        tracks2 = tracker.update(detections2)
        track_id_2 = tracks2[0][4]

        # Track ID should be maintained
        assert track_id_1 == track_id_2

    def test_bytetrack_reset(self):
        """Test ByteTracker reset."""
        tracker = ByteTracker()

        detections = [(0, 0, 100, 100, 0.9, 0)]
        tracker.update(detections)

        assert tracker.get_track_count() > 0

        tracker.reset()

        assert tracker.get_track_count() == 0
        assert tracker.frame_id == 0

    def test_bytetrack_get_track_count(self):
        """Test getting track count."""
        tracker = ByteTracker()

        assert tracker.get_track_count() == 0

        detections = [(0, 0, 100, 100, 0.9, 0), (200, 200, 300, 300, 0.85, 0)]
        tracker.update(detections)

        assert tracker.get_track_count() == 2
