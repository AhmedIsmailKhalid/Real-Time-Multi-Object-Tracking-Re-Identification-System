"""
Bounding box drawing and visualization.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BBoxDrawer:
    """Draw bounding boxes and track IDs on frames."""

    def __init__(self, thickness: int = 2, font_scale: float = 0.5, draw_trails: bool = False):
        """
        Initialize drawer.

        Args:
            thickness: Line thickness for boxes
            font_scale: Font scale for text
            draw_trails: Whether to draw track trails
        """
        self.thickness = thickness
        self.font_scale = font_scale
        self.draw_trails = draw_trails

        # Generate distinct colors
        self.colors = self._generate_colors(100)

        # Track history for trails
        self.track_history = {}

    def _generate_colors(self, n: int) -> list[tuple[int, int, int]]:
        """
        Generate distinct colors for track IDs.

        Args:
            n: Number of colors to generate

        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def draw_tracks(
        self, frame: np.ndarray, tracks: list[dict], draw_ids: bool = True, draw_boxes: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and IDs on frame.

        Args:
            frame: Input frame
            tracks: List of track dictionaries with 'bbox' and 'track_id'
            draw_ids: Whether to draw track IDs
            draw_boxes: Whether to draw bounding boxes

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for track in tracks:
            bbox = track["bbox"]
            track_id = track["track_id"]

            x1, y1, x2, y2 = map(int, bbox)

            # Get color for this track
            color = self._get_color(track_id)

            # Draw bounding box
            if draw_boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)

            # Draw track ID
            if draw_ids:
                label = f"ID: {track_id}"

                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
                )

                # Draw background rectangle
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.thickness,
                )

            # Draw trail
            if self.draw_trails:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append(center)

                # Keep only last N points
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                # Draw trail
                points = np.array(self.track_history[track_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(annotated, [points], False, color, 2)

        return annotated

    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        """
        Get consistent color for track ID.

        Args:
            track_id: Track ID

        Returns:
            BGR color tuple
        """
        return self.colors[track_id % len(self.colors)]

    def reset_history(self):
        """Reset track history."""
        self.track_history = {}
