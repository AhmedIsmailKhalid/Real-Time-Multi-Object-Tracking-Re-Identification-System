"""
Evaluate Re-ID model on Market-1501.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.market_dataset import Market1501Dataset
from src.data.transforms import get_val_transforms
from src.evaluation.reid_metrics import ReIDMetrics
from src.reid.resnet_reid import ResNet50ReID
from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))


logger = get_logger(__name__)


def extract_features(model: ResNet50ReID, data_loader: DataLoader, device: torch.device):
    """
    Extract features from dataset.

    Args:
        model: Re-ID model
        data_loader: Data loader
        device: Device

    Returns:
        features, person_ids, camera_ids
    """
    model.eval()

    features_list = []
    person_ids_list = []
    camera_ids_list = []

    with torch.no_grad():
        for images, _labels, camera_ids, person_ids in tqdm(
            data_loader, desc="Extracting features"
        ):
            images = images.to(device)

            # Extract features
            features = model.extract_features(images)

            features_list.append(features.cpu().numpy())
            person_ids_list.append(person_ids.numpy())
            camera_ids_list.append(camera_ids.numpy())

    # Concatenate
    features = np.vstack(features_list)
    person_ids = np.hstack(person_ids_list)
    camera_ids = np.hstack(camera_ids_list)

    return features, person_ids, camera_ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate Re-ID model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/reid/final/resnet50_market_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Re-ID Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Distance metric: {args.distance}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model...")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train the Re-ID model first using scripts/train_reid.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ResNet50ReID(
        num_classes=600, pretrained=False, feature_dim=512  # Not used in evaluation
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Create datasets
    data_dir = Path("data/processed/market1501")

    query_dataset = Market1501Dataset(
        data_dir=data_dir, split="query", transform=get_val_transforms()
    )

    gallery_dataset = Market1501Dataset(
        data_dir=data_dir, split="gallery", transform=get_val_transforms()
    )

    logger.info(f"Query: {len(query_dataset)} images")
    logger.info(f"Gallery: {len(gallery_dataset)} images")

    # Create dataloaders
    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Extract features
    logger.info("Extracting query features...")
    query_features, query_ids, query_cam_ids = extract_features(model, query_loader, device)

    logger.info("Extracting gallery features...")
    gallery_features, gallery_ids, gallery_cam_ids = extract_features(model, gallery_loader, device)

    # Evaluate
    logger.info("Computing metrics...")
    metrics_calc = ReIDMetrics(distance_metric=args.distance)

    metrics = metrics_calc.evaluate(
        query_features,
        query_ids,
        query_cam_ids,
        gallery_features,
        gallery_ids,
        gallery_cam_ids,
        max_rank=10,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Rank-1: {metrics['Rank-1']:.2f}%")
    logger.info(f"Rank-5: {metrics['Rank-5']:.2f}%")
    logger.info(f"Rank-10: {metrics['Rank-10']:.2f}%")
    logger.info(f"mAP: {metrics['mAP']:.2f}%")
    logger.info("=" * 60)

    # Save results
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "reid_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
