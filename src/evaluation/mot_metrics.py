"""
Multi-Object Tracking evaluation metrics.
Implements CLEAR MOT metrics: MOTA, MOTP, IDF1, etc.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MOTMetrics:
    """Calculate MOT evaluation metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.num_frames = 0
        self.num_matches = 0
        self.num_misses = 0
        self.num_false_positives = 0
        self.num_switches = 0
        self.num_fragmentations = 0

        # For IDF1 calculation
        self.idtp = 0  # ID True Positives
        self.idfn = 0  # ID False Negatives
        self.idfp = 0  # ID False Positives

        # Track history for switch detection
        self.last_match = {}  # track_id -> gt_id mapping from previous frame

        # Distance errors for MOTP
        self.distance_errors = []

    def update(
        self,
        gt_boxes: list[tuple[float, float, float, float, int]],
        pred_boxes: list[tuple[float, float, float, float, int]],
        iou_threshold: float = 0.5,
    ):
        """
        Update metrics with one frame.

        Args:
            gt_boxes: Ground truth boxes [(x1, y1, x2, y2, gt_id), ...]
            pred_boxes: Predicted boxes [(x1, y1, x2, y2, track_id), ...]
            iou_threshold: IoU threshold for matching
        """
        self.num_frames += 1

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)

        # Greedy matching (Hungarian algorithm would be better)
        matches, unmatched_gt, unmatched_pred = self._match_boxes(iou_matrix, iou_threshold)

        # Update counters
        self.num_matches += len(matches)
        self.num_misses += len(unmatched_gt)
        self.num_false_positives += len(unmatched_pred)

        # Track switches
        current_match = {}
        for gt_idx, pred_idx in matches:
            gt_id = gt_boxes[gt_idx][4]
            track_id = pred_boxes[pred_idx][4]

            current_match[track_id] = gt_id

            # Check for ID switch
            if track_id in self.last_match:
                if self.last_match[track_id] != gt_id:
                    self.num_switches += 1

            # Compute distance error for MOTP
            gt_box = gt_boxes[gt_idx][:4]
            pred_box = pred_boxes[pred_idx][:4]
            distance = self._compute_distance(gt_box, pred_box)
            self.distance_errors.append(distance)

        self.last_match = current_match

        # Update IDF1 counters
        self.idtp += len(matches)
        self.idfn += len(unmatched_gt)
        self.idfp += len(unmatched_pred)

    def compute_metrics(self) -> dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with metric values
        """
        # Total ground truth objects
        num_objects = self.num_matches + self.num_misses

        # MOTA (Multi-Object Tracking Accuracy)
        if num_objects > 0:
            mota = 1 - (
                (self.num_misses + self.num_false_positives + self.num_switches) / num_objects
            )
        else:
            mota = 0.0

        # MOTP (Multi-Object Tracking Precision)
        if len(self.distance_errors) > 0:
            motp = np.mean(self.distance_errors)
        else:
            motp = 0.0

        # IDF1 (ID F1 Score)
        if (self.idtp + self.idfn + self.idfp) > 0:
            idf1 = 2 * self.idtp / (2 * self.idtp + self.idfn + self.idfp)
        else:
            idf1 = 0.0

        # Precision and Recall
        if (self.num_matches + self.num_false_positives) > 0:
            precision = self.num_matches / (self.num_matches + self.num_false_positives)
        else:
            precision = 0.0

        if num_objects > 0:
            recall = self.num_matches / num_objects
        else:
            recall = 0.0

        # Mostly Tracked / Mostly Lost / Partially Tracked
        # (Would need track-level analysis, simplified here)

        metrics = {
            "MOTA": mota * 100,  # Convert to percentage
            "MOTP": motp * 100,  # Convert to percentage
            "IDF1": idf1 * 100,  # Convert to percentage
            "Precision": precision * 100,
            "Recall": recall * 100,
            "MT": 0,  # Mostly Tracked (requires track-level analysis)
            "ML": 0,  # Mostly Lost
            "PT": 0,  # Partially Tracked
            "FP": self.num_false_positives,
            "FN": self.num_misses,
            "IDsw": self.num_switches,
            "Frag": self.num_fragmentations,
        }

        return metrics

    def _compute_iou_matrix(self, gt_boxes: list[tuple], pred_boxes: list[tuple]) -> np.ndarray:
        """
        Compute IoU matrix between GT and predicted boxes.

        Args:
            gt_boxes: Ground truth boxes
            pred_boxes: Predicted boxes

        Returns:
            IoU matrix (num_gt, num_pred)
        """
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return np.zeros((len(gt_boxes), len(pred_boxes)))

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou = self._compute_iou(gt_box[:4], pred_box[:4])
                iou_matrix[i, j] = iou

        return iou_matrix

    def _compute_iou(
        self, box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
    ) -> float:
        """
        Compute IoU between two boxes.

        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)

        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _match_boxes(
        self, iou_matrix: np.ndarray, threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Match GT and predicted boxes using greedy matching.

        Args:
            iou_matrix: IoU matrix
            threshold: IoU threshold

        Returns:
            matches, unmatched_gt, unmatched_pred
        """
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        matches = []
        matched_gt = set()
        matched_pred = set()

        # Greedy matching (pick highest IoU first)
        while True:
            max_iou = iou_matrix.max()
            if max_iou < threshold:
                break

            gt_idx, pred_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

            matches.append((gt_idx, pred_idx))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

            # Mark matched boxes
            iou_matrix[gt_idx, :] = 0
            iou_matrix[:, pred_idx] = 0

        unmatched_gt = [i for i in range(iou_matrix.shape[0]) if i not in matched_gt]
        unmatched_pred = [i for i in range(iou_matrix.shape[1]) if i not in matched_pred]

        return matches, unmatched_gt, unmatched_pred

    def _compute_distance(
        self, box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
    ) -> float:
        """
        Compute center distance between two boxes.

        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)

        Returns:
            Euclidean distance between centers
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2

        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
