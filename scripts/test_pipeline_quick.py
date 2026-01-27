"""
Quick test of inference pipeline on single image.
"""

import sys
from pathlib import Path

import cv2

from src.inference.pipeline import InferencePipeline
from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def main():
    logger.info("Testing inference pipeline...")

    # Initialize pipeline
    config_path = Path("configs/inference.yaml")
    pipeline = InferencePipeline(config_path)

    # Create test image
    test_image = cv2.imread("data/processed/mot17/images/val/MOT17-11-FRCNN/000001.jpg")

    if test_image is None:
        logger.error("Cannot load test image")
        logger.info("Creating dummy image instead...")
        import numpy as np

        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Process frame
    logger.info("Processing frame...")
    tracks = pipeline.process_frame(test_image, frame_id=1)

    logger.info(f"Found {len(tracks)} tracks")

    for track in tracks:
        logger.info(f"Track ID: {track['track_id']}, BBox: {track['bbox']}")

    # Visualize
    vis_frame = pipeline._visualize_frame(test_image, tracks)

    # Save
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_output.jpg"

    cv2.imwrite(str(output_path), vis_frame)
    logger.info(f"Saved visualization: {output_path}")

    logger.info("Test complete!")


if __name__ == "__main__":
    main()
