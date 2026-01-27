"""
Abstract base class for multi-object trackers.
"""

from abc import ABC, abstractmethod


class TrackerBase(ABC):
    """Base class for object trackers."""

    @abstractmethod
    def update(
        self, detections: list[tuple[float, float, float, float, float, int]]
    ) -> list[tuple[float, float, float, float, int]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detections [(x1, y1, x2, y2, conf, class_id), ...]

        Returns:
            List of tracks [(x1, y1, x2, y2, track_id), ...]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state (clear all tracks)."""
        pass

    @abstractmethod
    def get_track_count(self) -> int:
        """
        Get number of active tracks.

        Returns:
            Number of active tracks
        """
        pass
