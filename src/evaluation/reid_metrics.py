"""
Person Re-Identification evaluation metrics.
Implements Rank-1, Rank-5, Rank-10, and mAP.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReIDMetrics:
    """Calculate Re-ID evaluation metrics."""

    def __init__(self, distance_metric: str = "euclidean"):
        """
        Initialize metrics calculator.

        Args:
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        self.distance_metric = distance_metric

    def compute_distance_matrix(
        self, query_features: np.ndarray, gallery_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance matrix between query and gallery features.

        Args:
            query_features: Query feature vectors (N_q, feature_dim)
            gallery_features: Gallery feature vectors (N_g, feature_dim)

        Returns:
            Distance matrix (N_q, N_g)
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance
            m, n = query_features.shape[0], gallery_features.shape[0]
            dist_matrix = (
                np.sum(query_features**2, axis=1, keepdims=True).repeat(n, axis=1)
                + np.sum(gallery_features**2, axis=1).reshape(1, -1).repeat(m, axis=0)
                - 2 * np.dot(query_features, gallery_features.T)
            )
            dist_matrix = np.sqrt(np.clip(dist_matrix, 0, None))

        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            similarity = np.dot(query_features, gallery_features.T)
            dist_matrix = 1 - similarity

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return dist_matrix

    def evaluate(
        self,
        query_features: np.ndarray,
        query_ids: np.ndarray,
        query_cam_ids: np.ndarray,
        gallery_features: np.ndarray,
        gallery_ids: np.ndarray,
        gallery_cam_ids: np.ndarray,
        max_rank: int = 10,
    ) -> dict:
        """
        Evaluate Re-ID performance.

        Args:
            query_features: Query feature vectors
            query_ids: Query person IDs
            query_cam_ids: Query camera IDs
            gallery_features: Gallery feature vectors
            gallery_ids: Gallery person IDs
            gallery_cam_ids: Gallery camera IDs
            max_rank: Maximum rank to compute

        Returns:
            Dictionary with metric values
        """
        logger.info(f"Computing Re-ID metrics (distance={self.distance_metric})...")

        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(query_features, gallery_features)

        # Evaluate
        cmc = np.zeros(max_rank)
        ap_list = []

        num_valid_queries = 0

        for i in range(len(query_ids)):
            query_id = query_ids[i]
            _query_cam = query_cam_ids[i]

            # Get distances for this query
            distances = dist_matrix[i]

            # In Market-1501, images from same person + same camera are allowed
            # We only remove junk IDs (person_id == -1 or 0)
            junk = gallery_ids <= 0
            valid_gallery = ~junk

            # Get matches (same person ID)
            matches = (gallery_ids == query_id) & valid_gallery

            if not np.any(matches):
                continue

            num_valid_queries += 1

            # Sort by distance
            indices = np.argsort(distances)
            matches = matches[indices]

            # CMC (Cumulative Matching Characteristics)
            for rank in range(max_rank):
                if np.any(matches[: rank + 1]):
                    cmc[rank] += 1

            # Average Precision
            ap = self._compute_ap(matches)
            ap_list.append(ap)

        if num_valid_queries == 0:
            logger.warning("No valid queries found!")
            return {
                "Rank-1": 0.0,
                "Rank-5": 0.0,
                "Rank-10": 0.0,
                "mAP": 0.0,
            }

        # Normalize CMC
        cmc = cmc / num_valid_queries * 100

        # Mean Average Precision
        mAP = np.mean(ap_list) * 100

        metrics = {
            "Rank-1": cmc[0],
            "Rank-5": cmc[4] if max_rank >= 5 else 0.0,
            "Rank-10": cmc[9] if max_rank >= 10 else 0.0,
            "mAP": mAP,
        }

        logger.info(f"Rank-1: {metrics['Rank-1']:.2f}%")
        logger.info(f"mAP: {metrics['mAP']:.2f}%")

        return metrics

    def _compute_ap(self, matches: np.ndarray) -> float:
        """
        Compute Average Precision for one query.

        Args:
            matches: Boolean array indicating matches

        Returns:
            Average Precision
        """
        if not np.any(matches):
            return 0.0

        # Compute precision at each position where match occurs
        match_positions = np.where(matches)[0]
        precisions = []

        for i, pos in enumerate(match_positions):
            precision = (i + 1) / (pos + 1)
            precisions.append(precision)

        return np.mean(precisions)
