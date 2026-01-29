"""
Create a test video from MOT17 sequence images.
"""

import sys
from pathlib import Path

import cv2

from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def create_video_from_sequence(
    sequence_path: Path, output_path: Path, fps: int = 30, max_frames: int = 300
):
    """
    Create video from image sequence.

    Args:
        sequence_path: Path to sequence directory
        output_path: Path to output video
        fps: Frames per second
        max_frames: Maximum number of frames to include
    """
    # Try img1 subfolder first (raw data), then sequence folder directly (processed data)
    img_dir = sequence_path / "img1" if (sequence_path / "img1").exists() else sequence_path

    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        return

    # Get image files
    image_files = sorted(img_dir.glob("*.jpg"))[:max_frames]

    if len(image_files) == 0:
        logger.error("No images found")
        return

    logger.info(f"Found {len(image_files)} images")

    # Get image size from first frame
    first_frame = cv2.imread(str(image_files[0]))
    height, width = first_frame.shape[:2]

    # Create video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(f"Creating video: {width}x{height} @ {fps} FPS")

    # Write frames
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is not None:
            writer.write(frame)

    writer.release()

    logger.info(f"Video created: {output_path}")
    logger.info(f"Duration: {len(image_files) / fps:.2f} seconds")


def main():
    # Use MOT17-11-FRCNN validation sequence
    sequence_path = Path("data/processed/mot17/images/val/MOT17-11-FRCNN")
    output_path = Path("data/external/test_videos/mot17_test.mp4")

    if not sequence_path.exists():
        logger.error(f"Sequence not found: {sequence_path}")
        return

    logger.info("=" * 60)
    logger.info("Creating Test Video")
    logger.info("=" * 60)

    create_video_from_sequence(sequence_path, output_path, fps=30, max_frames=300)  # ~10 seconds

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
