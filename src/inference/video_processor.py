"""
Video input/output handling.
"""

from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Handle video reading and writing."""

    def __init__(self, video_path: Path):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video loaded: {self.width}x{self.height} @ {self.fps} FPS, "
            f"{self.total_frames} frames"
        )

    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields video frames.

        Yields:
            Frame as numpy array (H, W, 3) BGR
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            logger.info("Video capture released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class VideoWriter:
    """Handle video writing."""

    def __init__(self, output_path: Path, fps: float, width: int, height: int, codec: str = "mp4v"):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"Cannot open video writer: {output_path}")

        logger.info(f"Video writer initialized: {output_path}")

    def write(self, frame: np.ndarray):
        """
        Write frame to video.

        Args:
            frame: Frame to write (H, W, 3) BGR
        """
        self.writer.write(frame)

    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
