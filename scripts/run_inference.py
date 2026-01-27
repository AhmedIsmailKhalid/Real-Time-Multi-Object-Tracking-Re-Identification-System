"""
Run inference on video file.
"""

import argparse
import sys
from pathlib import Path

from src.inference.pipeline import InferencePipeline
from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MOT+ReID inference on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video (default: outputs/results/output.mp4)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml", help="Path to inference config"
    )
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MOT + Re-ID Inference")
    logger.info("=" * 60)
    logger.info(f"Video: {args.video}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)

    # Set default output path
    if args.output is None:
        output_dir = Path("outputs/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / "output.mp4"

    # Initialize pipeline
    pipeline = InferencePipeline(Path(args.config))

    # Process video
    results = pipeline.process_video(
        video_path=Path(args.video),
        output_path=Path(args.output) if not args.no_viz else None,
        visualize=not args.no_viz,
    )

    # Print statistics
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total frames: {results['total_frames']}")
    logger.info(f"Unique tracks: {results['unique_tracks']}")
    logger.info(f"Average FPS: {results['avg_fps']:.2f}")
    logger.info(f"Processing time: {results['processing_time']:.2f}s")

    if not args.no_viz:
        logger.info(f"Output saved: {args.output}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
