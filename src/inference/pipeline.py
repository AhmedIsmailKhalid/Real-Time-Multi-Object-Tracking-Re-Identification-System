"""
End-to-end inference pipeline.
Integrates detection, tracking, and Re-ID.
"""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

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
                num_classes=600,
                pretrained=False,
                feature_dim=512
            )

            reid_checkpoint = torch.load(
                self.config["reid"]["model_path"], map_location=device
            )
            self.reid_model.load_state_dict(reid_checkpoint["model_state_dict"])
            self.reid_model.eval()
            self.reid_model.to(self.device)

            self.feature_gallery = {}
            self.reid_distance_thresh = self.config["reid"]["distance_threshold"]
            self.reid_min_gallery_age = 30
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

        if frame_id % 100 == 0:
            logger.info(f"Frame {frame_id}: {len(detections)} detections")

        # Step 2: Tracking
        tracks = self.tracker.update(detections)

        if frame_id % 100 == 0:
            logger.info(f"Frame {frame_id}: {len(tracks)} tracks")

        # Step 3: Re-ID matching and feature extraction
        track_results = []
        reid_matches = 0

        for track in tracks:
            x1, y1, x2, y2, track_id = track

            result = {
                "bbox": (x1, y1, x2, y2),
                "track_id": track_id,
                "frame_id": frame_id
            }

            if self.use_reid:
                feature = self._extract_reid_feature(frame, (x1, y1, x2, y2))

                if feature is not None:
                    result["feature"] = feature

                    matched_id = self._match_with_gallery(feature, track_id)

                    if matched_id is not None and matched_id != track_id:
                        result["track_id"] = matched_id
                        result["reid_matched"] = True
                        reid_matches += 1
                        logger.debug(f"Re-ID: Matched track {track_id} -> {matched_id}")

                    gallery_id = matched_id if matched_id is not None else track_id

                    if gallery_id not in self.feature_gallery:
                        self.feature_gallery[gallery_id] = {"features": [], "age": 0}

                    self.feature_gallery[gallery_id]["features"].append(feature)
                    self.feature_gallery[gallery_id]["age"] += 1

                    if len(self.feature_gallery[gallery_id]["features"]) > 10:
                        self.feature_gallery[gallery_id]["features"].pop(0)

            track_results.append(result)

        if frame_id % 100 == 0 and self.use_reid and reid_matches > 0:
            logger.info(f"Frame {frame_id}: {reid_matches} Re-ID matches")

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
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        person_crop = cv2.resize(person_crop, (128, 256))
        person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        person_crop = person_crop.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        person_crop = (person_crop - mean) / std

        person_crop = torch.from_numpy(person_crop).permute(2, 0, 1).unsqueeze(0)
        person_crop = person_crop.float().to(self.device)

        with torch.no_grad():
            feature = self.reid_model.extract_features(person_crop)
            feature = feature.cpu().numpy().flatten()

        return feature

    def _match_with_gallery(self, feature: np.ndarray, current_track_id: int) -> int | None:
        """
        Match feature with existing gallery using cosine similarity.

        Args:
            feature: Feature vector to match
            current_track_id: Current track ID (to avoid self-matching)

        Returns:
            Matched track ID or None if no match
        """
        if not self.feature_gallery:
            return None

        best_similarity = -1
        best_match_id = None

        for gallery_id, gallery_data in self.feature_gallery.items():
            if gallery_id == current_track_id:
                continue

            # Require minimum observation age before allowing matches
            if gallery_data["age"] < self.reid_min_gallery_age:
                continue

            similarities = [
                np.dot(feature, gf) / (
                    np.linalg.norm(feature) * np.linalg.norm(gf) + 1e-10
                )
                for gf in gallery_data["features"]
            ]

            if similarities:
                max_sim = max(similarities)
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_match_id = gallery_id

        if best_similarity > 0.3:
            logger.debug(
                f"Re-ID: Track {current_track_id} best match {best_match_id} "
                f"similarity={best_similarity:.3f} (threshold={self.reid_distance_thresh})"
            )

        if best_similarity > self.reid_distance_thresh:
            logger.info(
                f"✓ Re-ID MATCH: Track {current_track_id} -> {best_match_id} "
                f"(similarity={best_similarity:.3f})"
            )
            return best_match_id

        return None

    def process_video(
        self,
        video_path: Path,
        output_path: Path | None = None,
        visualize: bool = True,
        save_results: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,  # noqa: UP007
    ) -> dict:
        """
        Process entire video.

        Args:
            video_path: Path to input video
            output_path: Path to save output video
            visualize: Draw bounding boxes and IDs
            save_results: Save tracking results to file
            progress_callback: Callback function(current_frame, total_frames)

        Returns:
            Processing statistics
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Try codecs in order - prioritize browser compatibility
        writer = None
        if output_path and visualize:
            for codec in ["mp4v", "XVID", "MJPG"]:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec for better compatibility
                output_fps = min(fps, 30.0)
                test_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    output_fps,
                    (width, height),
                    isColor=True
                )
                if test_writer.isOpened():
                    writer = test_writer
                    logger.info(f"VideoWriter initialized with codec: {codec}")
                    break
                test_writer.release()

            if writer is None:
                logger.error("All codecs failed - output video will not be saved")

        self.tracker.reset()

        frame_id = 0
        all_tracks = []
        total_reid_matches = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            tracks = self.process_frame(frame, frame_id)
            all_tracks.extend(tracks)

            frame_reid_matches = sum(1 for t in tracks if t.get("reid_matched", False))
            total_reid_matches += frame_reid_matches

            if visualize and writer:
                vis_frame = self._visualize_frame(frame, tracks)
                writer.write(vis_frame)

            if progress_callback and frame_id % 10 == 0:
                progress_callback(frame_id, total_frames)

            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id}/{total_frames} frames")

        cap.release()
        if writer:
            writer.release()

            # Re-encode with FFmpeg for browser compatibility if output exists
            if output_path and output_path.exists():
                logger.info("Re-encoding video for browser compatibility...")
                temp_output = output_path.with_name(output_path.stem + '_temp.mp4')

                try:
                    # Check if FFmpeg is available
                    version_check = subprocess.run(
                        ['ffmpeg', '-version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    logger.info(f"FFmpeg available: {version_check.stdout.split()[0:3]}")

                    # Re-encode with H.264
                    result = subprocess.run([
                        'ffmpeg',
                        '-i', str(output_path),
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',  # Enable streaming
                        '-y',
                        str(temp_output)
                    ], capture_output=True, text=True, timeout=300)

                    if result.returncode == 0:
                        # Replace original with re-encoded version
                        temp_output.replace(output_path)
                        logger.info("✓ Video re-encoded successfully with H.264")
                    else:
                        logger.error(f"FFmpeg encoding failed with code {result.returncode}")
                        logger.error(f"stderr: {result.stderr}")

                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg re-encoding timed out after 5 minutes")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg re-encoding failed: {e.stderr}")
                except FileNotFoundError:
                    logger.warning("FFmpeg not found in PATH - video may not play in browsers")
                    logger.warning("Add FFmpeg to system PATH or reinstall")
                except Exception as e:
                    logger.error(f"Unexpected error during re-encoding: {e}")

        elapsed_time = time.time() - start_time
        avg_fps = frame_id / elapsed_time if elapsed_time > 0 else 0
        unique_tracks = len({t["track_id"] for t in all_tracks})

        stats = {
            "total_frames": frame_id,
            "total_tracks": len(all_tracks),
            "unique_tracks": unique_tracks,
            "reid_matches": total_reid_matches,
            "processing_time": elapsed_time,
            "avg_fps": avg_fps,
        }

        logger.info(
            f"Processing complete: {frame_id} frames in {elapsed_time:.2f}s ({avg_fps:.2f} FPS)"
        )
        logger.info(f"Total tracks: {len(all_tracks)}, Unique IDs: {unique_tracks}")
        logger.info(f"Re-ID matches: {total_reid_matches}")

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

            color = self._get_color(track_id)

            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            cv2.putText(
                vis_frame,
                str(track_id),
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
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
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
