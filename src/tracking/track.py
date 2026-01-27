"""
Track class representing a single tracked object.
"""

from src.tracking.kalman_filter import KalmanFilter, bbox_to_z, z_to_bbox


class TrackState:
    """Track state enumeration."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    """Represents a single tracked object."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        score: float,
        track_id: int,
        class_id: int = 0,
    ):
        """
        Initialize track.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            score: Detection confidence score
            track_id: Unique track ID
            class_id: Object class ID
        """
        self.track_id = track_id
        self.class_id = class_id
        self.score = score

        # Kalman filter for motion prediction
        self.kalman_filter = KalmanFilter()
        measurement = bbox_to_z(bbox)
        self.kalman_filter.x[:4] = measurement.reshape((4, 1))

        # Track state
        self.state = TrackState.New
        self.is_activated = False

        # Frame tracking
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        # Bounding box history
        self.bbox = bbox
        self.mean = self.kalman_filter.get_state()

    def predict(self):
        """Predict next position using Kalman filter."""
        mean_state = self.kalman_filter.predict()
        self.mean = mean_state.flatten()

    def update(self, new_track, frame_id: int):
        """
        Update track with new detection.

        Args:
            new_track: New track with updated detection
            frame_id: Current frame ID
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Update Kalman filter with new measurement
        measurement = bbox_to_z(new_track.bbox)
        self.kalman_filter.update(measurement)
        self.mean = self.kalman_filter.get_state()

        # Update track properties
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True

    def activate(self, frame_id: int):
        """
        Activate a new track.

        Args:
            frame_id: Frame ID where track is first activated
        """
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int):
        """
        Re-activate a lost track.

        Args:
            new_track: New detection to re-activate with
            frame_id: Current frame ID
        """
        # Update Kalman filter
        measurement = bbox_to_z(new_track.bbox)
        self.kalman_filter.update(measurement)
        self.mean = self.kalman_filter.get_state()

        # Update track properties
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.Removed

    def get_current_bbox(self) -> tuple[float, float, float, float]:
        """
        Get current bounding box from Kalman filter state.

        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        return z_to_bbox(self.mean)

    @staticmethod
    def next_id():
        """Get next track ID (class-level counter)."""
        Track._count += 1
        return Track._count


# Initialize track ID counter
Track._count = 0
