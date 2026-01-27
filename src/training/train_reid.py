"""
Training pipeline for Re-ID model on Market-1501.
"""

import time
from pathlib import Path

import mlflow
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.market_dataset import Market1501Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.reid.metric_learning import CrossEntropyLabelSmooth, TripletLoss
from src.reid.resnet_reid import ResNet50ReID
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReIDTrainer:
    """Trainer for Re-ID model."""

    def __init__(self, config: dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Create datasets
        data_dir = Path("data/processed/market1501")

        self.train_dataset = Market1501Dataset(
            data_dir=data_dir,
            split="train",
            transform=get_train_transforms(
                image_size=tuple(config["data"]["image_size"]),
                mean=tuple(config["data"]["mean"]),
                std=tuple(config["data"]["std"]),
            ),
        )

        self.val_dataset = Market1501Dataset(
            data_dir=data_dir,
            split="val",
            transform=get_val_transforms(
                image_size=tuple(config["data"]["image_size"]),
                mean=tuple(config["data"]["mean"]),
                std=tuple(config["data"]["std"]),
            ),
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )

        # Initialize model
        num_classes = self.train_dataset.get_num_classes()
        logger.info(f"Training on {num_classes} person identities")

        self.model = ResNet50ReID(
            num_classes=num_classes,
            pretrained=config["model"]["pretrained_imagenet"],
            feature_dim=config["model"]["feature_dim"],
        )
        self.model.to(self.device)

        # Loss functions
        self.criterion_ce = CrossEntropyLabelSmooth(
            num_classes=num_classes, epsilon=config["training"]["label_smooth"]
        )
        self.criterion_triplet = TripletLoss(margin=config["training"]["triplet_margin"])

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["training"]["step_size"],
            gamma=config["training"]["gamma"],
        )

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0

        logger.info("ReIDTrainer initialized")

    def train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for _batch_idx, (images, labels, _camera_ids, _person_ids) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get features (before classification head)
            features = self.model(images)

            # Compute losses
            ce_loss = self.criterion_ce(features, labels)

            # For triplet loss, need to extract features without classification
            self.model.eval()
            with torch.no_grad():
                feat_for_triplet = self.model.extract_features(images)
            self.model.train()

            triplet_loss = self.criterion_triplet(feat_for_triplet, labels)

            # Combined loss
            loss = (
                self.config["training"]["ce_loss_weight"] * ce_loss
                + self.config["training"]["triplet_loss_weight"] * triplet_loss
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()

            _, predicted = features.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "tri": f"{triplet_loss.item():.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_ce_loss = total_ce_loss / len(self.train_loader)
        avg_triplet_loss = total_triplet_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total

        return {
            "train_loss": avg_loss,
            "train_ce_loss": avg_ce_loss,
            "train_triplet_loss": avg_triplet_loss,
            "train_accuracy": train_acc,
        }

    def validate(self) -> dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _camera_ids, _person_ids in tqdm(
                self.val_loader, desc="Validation"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                features = self.model(images)

                # Compute loss
                ce_loss = self.criterion_ce(features, labels)
                total_loss += ce_loss.item()

                # Statistics
                _, predicted = features.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        return {"val_loss": avg_loss, "val_accuracy": val_acc}

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config["checkpoint"]["save_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }

        # Save regular checkpoint
        if self.current_epoch % self.config["checkpoint"]["save_period"] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

            # Also save to final directory
            final_dir = Path("models/reid/final")
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / "resnet50_market_best.pth"
            torch.save(checkpoint, final_path)
            logger.info(f"Saved to final directory: {final_path}")

    def train(self, num_epochs: int):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # MLflow tracking
        mlflow.set_experiment("reid_training")

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(
                {
                    "model": self.config["model"]["name"],
                    "feature_dim": self.config["model"]["feature_dim"],
                    "batch_size": self.config["training"]["batch_size"],
                    "learning_rate": self.config["training"]["learning_rate"],
                    "epochs": num_epochs,
                    "optimizer": "Adam",
                    "ce_loss_weight": self.config["training"]["ce_loss_weight"],
                    "triplet_loss_weight": self.config["training"]["triplet_loss_weight"],
                }
            )

            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch

                start_time = time.time()

                # Train
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Step scheduler
                self.scheduler.step()

                epoch_time = time.time() - start_time

                # Log metrics
                metrics = {**train_metrics, **val_metrics}
                metrics["epoch_time"] = epoch_time
                metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=epoch)

                # Print summary
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Loss: {train_metrics['train_loss']:.4f} - "
                    f"Train Acc: {train_metrics['train_accuracy']:.2f}% - "
                    f"Val Acc: {val_metrics['val_accuracy']:.2f}% - "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save checkpoint
                is_best = val_metrics["val_accuracy"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["val_accuracy"]
                    logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")

                self.save_checkpoint(is_best=is_best)

            logger.info(f"Training complete! Best val accuracy: {self.best_val_acc:.2f}%")
            mlflow.log_metric("best_val_accuracy", self.best_val_acc)
