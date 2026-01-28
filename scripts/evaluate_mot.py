"""
Evaluate MOT tracking performance on MOT17 validation set.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from src.evaluation.mot_metrics import MOTMetrics
from src.inference.pipeline import InferencePipeline
from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))


logger = get_logger(__name__)


def load_mot_gt(sequence_path: Path):
    """
    Load MOT ground truth annotations.

    Args:
        sequence_path: Path to MOT sequence directory

    Returns:
        Dictionary mapping frame_id to list of boxes
    """
    gt_file = sequence_path / "gt" / "gt.txt"

    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    gt_dict = {}

    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])

            # Only use pedestrian class (1) with conf=1
            if class_id != 1 or conf != 1:
                continue

            # Convert to (x1, y1, x2, y2, track_id)
            box = (x, y, x + w, y + h, track_id)

            if frame_id not in gt_dict:
                gt_dict[frame_id] = []

            gt_dict[frame_id].append(box)

    return gt_dict


def evaluate_sequence(pipeline: InferencePipeline, sequence_path: Path) -> dict:
    """
    Evaluate one MOT sequence.

    Args:
        pipeline: Inference pipeline
        sequence_path: Path to sequence directory

    Returns:
        Metrics dictionary
    """
    logger.info(f"Evaluating sequence: {sequence_path.name}")

    # Load ground truth
    gt_dict = load_mot_gt(sequence_path)

    # Get image directory
    img_dir = sequence_path / "img1"

    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        return {}

    # Initialize metrics
    metrics_calc = MOTMetrics()

    # Reset tracker
    pipeline.tracker.reset()

    # Process frames
    frame_files = sorted(img_dir.glob("*.jpg"))

    for frame_id, frame_file in enumerate(tqdm(frame_files, desc=sequence_path.name), start=1):
        # Load frame
        frame = cv2.imread(str(frame_file))

        if frame is None:
            logger.warning(f"Cannot load frame: {frame_file}")
            continue

        # Run inference
        pred_tracks = pipeline.process_frame(frame, frame_id)

        # Convert to format for metrics
        pred_boxes = [
            (t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3], t["track_id"])
            for t in pred_tracks
        ]

        gt_boxes = gt_dict.get(frame_id, [])

        # Update metrics
        metrics_calc.update(gt_boxes, pred_boxes, iou_threshold=0.5)

    # Compute final metrics
    metrics = metrics_calc.compute_metrics()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking")
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml", help="Path to inference config"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequences to evaluate (default: all validation sequences)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MOT Evaluation")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = InferencePipeline(Path(args.config))

    # Get validation sequences
    val_sequences_path = Path("data/processed/mot17/images/val")

    if not val_sequences_path.exists():
        logger.error(f"Validation directory not found: {val_sequences_path}")
        return

    if args.sequences:
        sequences = [val_sequences_path / seq for seq in args.sequences]
    else:
        sequences = sorted(val_sequences_path.iterdir())

    logger.info(f"Evaluating {len(sequences)} sequences")

    # Evaluate each sequence
    all_metrics = []

    for seq_path in sequences:
        if not seq_path.is_dir():
            continue

        # Get original sequence path for GT
        seq_name = seq_path.name
        original_seq_path = Path("data/raw/MOT17/train") / seq_name

        if not original_seq_path.exists():
            logger.warning(f"Original sequence not found: {original_seq_path}")
            continue

        metrics = evaluate_sequence(pipeline, original_seq_path)

        if metrics:
            all_metrics.append(metrics)

            logger.info(f"\n{seq_name} Results:")
            logger.info(f"  MOTA: {metrics['MOTA']:.2f}%")
            logger.info(f"  IDF1: {metrics['IDF1']:.2f}%")
            logger.info(f"  Precision: {metrics['Precision']:.2f}%")
            logger.info(f"  Recall: {metrics['Recall']:.2f}%")
            logger.info(f"  ID Switches: {metrics['IDsw']}")

    # Compute average metrics
    if all_metrics:
        avg_metrics = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in all_metrics[0].keys()
        }

        logger.info("\n" + "=" * 60)
        logger.info("Average Results")
        logger.info("=" * 60)
        logger.info(f"MOTA: {avg_metrics['MOTA']:.2f}%")
        logger.info(f"MOTP: {avg_metrics['MOTP']:.2f}%")
        logger.info(f"IDF1: {avg_metrics['IDF1']:.2f}%")
        logger.info(f"Precision: {avg_metrics['Precision']:.2f}%")
        logger.info(f"Recall: {avg_metrics['Recall']:.2f}%")
        logger.info(f"FP: {avg_metrics['FP']:.0f}")
        logger.info(f"FN: {avg_metrics['FN']:.0f}")
        logger.info(f"ID Switches: {avg_metrics['IDsw']:.0f}")

        # Save results
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "mot_evaluation.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "average": avg_metrics,
                    "per_sequence": {
                        sequences[i].name: all_metrics[i] for i in range(len(all_metrics))
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
