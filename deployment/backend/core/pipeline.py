"""
Inference pipeline wrapper for API.
"""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from src.inference.pipeline import InferencePipeline
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = get_logger(__name__)


class PipelineWrapper:
    """Wrapper around inference pipeline for API use."""

    def __init__(
        self,
        yolo_model_path: str,
        reid_model_path: str | None = None,
        device: str = "cuda",
        enable_reid: bool = True,
        confidence_threshold: float = 0.5,
        reid_distance_threshold: float = 0.85,
    ):
        """
        Initialize pipeline.

        Args:
            yolo_model_path: Path to YOLO model
            reid_model_path: Path to Re-ID model
            device: Device to use
            enable_reid: Enable Re-ID
            confidence_threshold: Detection confidence threshold
            reid_distance_threshold: Cosine similarity threshold for Re-ID matching
        """
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        config = {
            "device": self.device,
            "detection": {
                "model_path": yolo_model_path,
                "conf_threshold": confidence_threshold,
                "iou_threshold": 0.45,
                "target_classes": [0]
            },
            "tracking": {
                "high_conf_thresh": 0.6,
                "low_conf_thresh": 0.1,
                "track_buffer": 30,
                "match_thresh": 0.8
            },
            "use_reid": enable_reid and reid_model_path is not None
        }

        if enable_reid and reid_model_path:
            config["reid"] = {
                "model_path": reid_model_path,
                "distance_threshold": reid_distance_threshold  # ← use parameter, not hardcoded
            }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        logger.info(
            f"Initializing pipeline (device={self.device}, reid={config['use_reid']}, "
            f"conf_threshold={confidence_threshold}, reid_threshold={reid_distance_threshold})"
        )

        self.pipeline = InferencePipeline(config_path)
        self.use_reid = config['use_reid']

        config_path.unlink()

    def process_video(
        self,
        video_path: str,
        output_path: str,
        enable_reid: bool = True,
        show_trails: bool = True,
        progress_callback: Optional[Callable] = None,  # noqa: UP007
    ) -> dict[str, Any]:
        """
        Process video with tracking.

        Args:
            video_path: Input video path
            output_path: Output video path
            enable_reid: Enable Re-ID
            show_trails: Show movement trails (ignored, for API compatibility)
            progress_callback: Callback for progress updates

        Returns:
            Processing statistics
        """
        self.pipeline.use_reid = enable_reid

        if not enable_reid and hasattr(self.pipeline, 'feature_gallery'):
            self.pipeline.feature_gallery.clear()
            logger.info("Re-ID disabled - cleared feature gallery")

        results = self.pipeline.process_video(
            video_path=Path(video_path),
            output_path=Path(output_path),
            visualize=True,
            save_results=True,
            progress_callback=progress_callback
        )

        # reid_matches tracked directly by inner pipeline - no inference needed
        if "reid_matches" not in results:
            results["reid_matches"] = 0

        return results
