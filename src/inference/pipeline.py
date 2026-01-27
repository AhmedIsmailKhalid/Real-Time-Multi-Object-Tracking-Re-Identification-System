"""
End-to-end inference pipeline.
Integrates detection, tracking, and Re-ID.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.yolo_detector import YOLODetector
from src.reid.resnet_reid import ResNet50ReID
from src.tracking.bytetrack import ByteTracker
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """End-to-end MOT + Re-ID inference pipeline."""

    def __init__(self, config_path: Path):
        """
        Initialize pipeline with all components.

        Args:
            config_path: Path to inference config file
        """
        self.config = load_config(config_path)

        device = self.config.get("device", "cuda")
        self.device = torch.device(device)

        # Initialize detector
        logger.info("Loading detector...")
        self.detector = YOLODetector(
            model_path=Path(self.config["detection"]["model_path"]),
            device=device,
            conf_threshold=self.config["detection"]["conf_threshold"],
            iou_threshold=self.config["detection"]["iou_threshold"],
            target_classes=self.config["detection"]["target_classes"],
        )

        # Initialize tracker
        logger.info("Loading tracker...")
        self.tracker = ByteTracker(
            high_conf_thresh=self.config["tracking"]["high_conf_thresh"],
            low_conf_thresh=self.config["tracking"]["low_conf_thresh"],
            track_buffer=self.config["tracking"]["track_buffer"],
            match_thresh=self.config["tracking"]["match_thresh"],
        )

        # Initialize Re-ID (optional)
        self.use_reid = self.config.get("use_reid", False)
        if self.use_reid:
            logger.info("Loading Re-ID model...")
            self.reid_model = ResNet50ReID(
                num_classes=751, pretrained=False, feature_dim=512  # Not used in inference
            )

            # Load trained weights
            reid_checkpoint = torch.load(self.config["reid"]["model_path"], map_location=device)
            self.reid_model.load_state_dict(reid_checkpoint["model_state_dict"])
            self.reid_model.eval()
            self.reid_model.to(self.device)

            # Feature gallery for Re-ID matching
            self.feature_gallery = {}
            self.reid_distance_thresh = self.config["reid"]["distance_threshold"]
        else:
            logger.info("Re-ID disabled")

        logger.info("Inference pipeline initialized")

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> list[dict]:
        """
        Process single frame.

        Args:
            frame: Input frame (H, W, 3) BGR
            frame_id: Frame number

        Returns:
            List of tracks with Re-ID info
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return []

        # Step 1: Detection
        detections = self.detector.detect(frame)

        # Step 2: Tracking
        tracks = self.tracker.update(detections)

        # Step 3: Re-ID (if enabled)
        track_results = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track

            result = {"bbox": (x1, y1, x2, y2), "track_id": track_id, "frame_id": frame_id}

            # Extract Re-ID features if enabled
            if self.use_reid:
                feature = self._extract_reid_feature(frame, (x1, y1, x2, y2))
                if feature is not None:
                    result["feature"] = feature

                    # Update feature gallery
                    if track_id not in self.feature_gallery:
                        self.feature_gallery[track_id] = []

                    self.feature_gallery[track_id].append(feature)

                    # Keep only last N features per track
                    if len(self.feature_gallery[track_id]) > 10:
                        self.feature_gallery[track_id].pop(0)

            track_results.append(result)

        return track_results

    def _extract_reid_feature(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """
        Extract Re-ID feature from person crop.

        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Feature vector or None if extraction fails
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clip to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Check valid crop
        if x2 <= x1 or y2 <= y1:
            return None

        # Crop person
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        # Resize to Re-ID input size
        person_crop = cv2.resize(person_crop, (128, 256))

        # Convert BGR to RGB
        person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Normalize
        person_crop = person_crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        person_crop = (person_crop - mean) / std

        # To tensor
        person_crop = torch.from_numpy(person_crop).permute(2, 0, 1).unsqueeze(0)
        person_crop = person_crop.to(self.device)

        # Extract features
        with torch.no_grad():
            feature = self.reid_model.extract_features(person_crop)
            feature = feature.cpu().numpy().flatten()

        return feature

    def process_video(
        self,
        video_path: Path,
        output_path: Path | None = None,
        visualize: bool = True,
        save_results: bool = True,
    ) -> dict:
        """
        Process entire video.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            visualize: Draw bounding boxes and IDs
            save_results: Save tracking results to file

        Returns:
            Processing statistics and results
        """
        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Initialize video writer
        writer = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Reset tracker
        self.tracker.reset()

        # Process frames
        frame_id = 0
        all_tracks = []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Process frame
            tracks = self.process_frame(frame, frame_id)
            all_tracks.extend(tracks)

            # Visualize
            if visualize and writer:
                vis_frame = self._visualize_frame(frame, tracks)
                writer.write(vis_frame)

            # Progress
            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id}/{total_frames} frames")

        # Cleanup
        cap.release()
        if writer:
            writer.release()

        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_fps = frame_id / elapsed_time if elapsed_time > 0 else 0

        unique_tracks = len({t["track_id"] for t in all_tracks})

        stats = {
            "total_frames": frame_id,
            "total_tracks": len(all_tracks),
            "unique_tracks": unique_tracks,
            "processing_time": elapsed_time,
            "avg_fps": avg_fps,
        }

        logger.info(
            f"Processing complete: {frame_id} frames in {elapsed_time:.2f}s ({avg_fps:.2f} FPS)"
        )
        logger.info(f"Total tracks: {len(all_tracks)}, Unique IDs: {unique_tracks}")

        return stats

    def _visualize_frame(self, frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and track IDs on frame.

        Args:
            frame: Input frame
            tracks: List of track dictionaries

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]

            # Draw bounding box
            color = self._get_color(track_id)
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(
                vis_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return vis_frame

    def _get_color(self, track_id: int) -> tuple:
        """
        Get consistent color for track ID.

        Args:
            track_id: Track ID

        Returns:
            BGR color tuple
        """
        # Generate color from ID
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
