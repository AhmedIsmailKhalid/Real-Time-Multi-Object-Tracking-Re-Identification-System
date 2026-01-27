"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

Key innovation: Uses both high and low confidence detections.
- High-conf detections: Create new tracks
- Low-conf detections: Recover lost tracks
"""

import numpy as np

from src.tracking.matching import iou_distance, linear_assignment
from src.tracking.track import Track, TrackState
from src.tracking.tracker_base import TrackerBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ByteTracker(TrackerBase):
    """ByteTrack multi-object tracker."""

    def __init__(
        self,
        high_conf_thresh: float = 0.6,
        low_conf_thresh: float = 0.1,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        """
        Initialize ByteTrack.

        Args:
            high_conf_thresh: Threshold for high confidence detections
            low_conf_thresh: Threshold for low confidence detections
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching (as distance, so 1-IoU)
        """
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self.frame_id = 0
        self.track_id_count = 0

        logger.info(
            f"ByteTracker initialized (high_thresh={high_conf_thresh}, "
            f"low_thresh={low_conf_thresh}, buffer={track_buffer})"
        )

    def update(
        self, detections: list[tuple[float, float, float, float, float, int]]
    ) -> list[tuple[float, float, float, float, int]]:
        """
        Update tracks with new detections.

        Two-stage association:
        1. Match high-conf detections to existing tracks
        2. Match low-conf detections to unmatched tracks

        Args:
            detections: List of detections [(x1, y1, x2, y2, conf, class_id), ...]

        Returns:
            List of tracks [(x1, y1, x2, y2, track_id), ...]
        """
        self.frame_id += 1

        # Split detections by confidence
        high_dets = [d for d in detections if d[4] >= self.high_conf_thresh]
        low_dets = [d for d in detections if self.low_conf_thresh <= d[4] < self.high_conf_thresh]

        # Create track objects from high confidence detections
        high_tracks = [Track(bbox=d[:4], score=d[4], track_id=-1, class_id=d[5]) for d in high_dets]

        # Predict current position for all tracks
        for track in self.tracked_tracks:
            track.predict()

        # First association: high confidence detections with tracked tracks
        tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]

        matches, unmatched_tracks, unmatched_dets = self._match(
            tracked_tracks, high_tracks, self.match_thresh
        )

        # Update matched tracks
        for itracked, idet in matches:
            tracked_tracks[itracked].update(high_tracks[idet], self.frame_id)

        # Handle unmatched tracked tracks
        for it in unmatched_tracks:
            track = tracked_tracks[it]
            track.mark_lost()

        # Initialize new tracks from unmatched high confidence detections
        new_tracks = []
        for idet in unmatched_dets:
            track = high_tracks[idet]
            if track.score >= self.high_conf_thresh:
                track.activate(self.frame_id)
                new_tracks.append(track)

        # Second association: low confidence detections with lost tracks
        lost_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Lost]

        low_tracks = [Track(bbox=d[:4], score=d[4], track_id=-1, class_id=d[5]) for d in low_dets]

        matches, unmatched_lost, unmatched_low = self._match(
            lost_tracks, low_tracks, self.match_thresh
        )

        # Re-activate matched lost tracks
        for ilost, idet in matches:
            track = lost_tracks[ilost]
            track.re_activate(low_tracks[idet], self.frame_id)

        # Remove tracks that have been lost for too long
        for track in self.tracked_tracks:
            if track.state == TrackState.Lost:
                if self.frame_id - track.frame_id > self.track_buffer:
                    track.mark_removed()

        # Update track lists
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state != TrackState.Removed]
        self.tracked_tracks.extend(new_tracks)

        # Prepare output
        output_tracks = []
        for track in self.tracked_tracks:
            if track.is_activated and track.state == TrackState.Tracked:
                x1, y1, x2, y2 = track.bbox
                output_tracks.append((x1, y1, x2, y2, track.track_id))

        return output_tracks

    def _match(self, tracks: list[Track], detections: list[Track], thresh: float):
        """
        Match tracks to detections using IoU.

        Args:
            tracks: List of tracks
            detections: List of detections
            thresh: IoU distance threshold

        Returns:
            matches, unmatched_tracks, unmatched_detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        # Compute cost matrix
        track_boxes = [t.bbox for t in tracks]
        det_boxes = [d.bbox for d in detections]
        cost_matrix = iou_distance(track_boxes, det_boxes)

        # Hungarian matching
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, thresh)

        return matches, unmatched_tracks, unmatched_dets

    def reset(self):
        """Reset tracker state (clear all tracks)."""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        Track._count = 0
        logger.info("ByteTracker reset")

    def get_track_count(self) -> int:
        """
        Get number of active tracks.

        Returns:
            Number of active tracks
        """
        return len([t for t in self.tracked_tracks if t.state == TrackState.Tracked])
