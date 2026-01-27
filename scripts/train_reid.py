"""
Standalone script to train Re-ID model.
"""

import argparse
import sys
from pathlib import Path

from src.training.train_reid import ReIDTrainer
from src.utils.config import load_config
from src.utils.logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Re-ID model")
    parser.add_argument(
        "--config", type=str, default="configs/reid.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    logger.info("=" * 60)
    logger.info("Starting Re-ID Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = ReIDTrainer(config)

    # Train
    trainer.train(num_epochs=config["training"]["epochs"])

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
