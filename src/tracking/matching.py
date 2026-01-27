"""
Matching algorithms for data association.
Associates detections to existing tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bboxes1: First set of boxes (N, 4) in format (x1, y1, x2, y2)
        bboxes2: Second set of boxes (M, 4) in format (x1, y1, x2, y2)

    Returns:
        IoU matrix (N, M)
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    # Compute intersection
    x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0])
    y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1])
    x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2])
    y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute union
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2 - intersection

    # Compute IoU
    iou = intersection / (union + 1e-6)

    return iou


def iou_distance(tracks: list, detections: list) -> np.ndarray:
    """
    Compute IoU distance matrix between tracks and detections.

    Args:
        tracks: List of track bounding boxes
        detections: List of detection bounding boxes

    Returns:
        Cost matrix (num_tracks, num_detections)
        Lower cost = better match (1 - IoU)
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    iou_matrix = iou_batch(np.array([t[:4] for t in tracks]), np.array([d[:4] for d in detections]))

    # Convert IoU to distance (1 - IoU)
    cost_matrix = 1 - iou_matrix

    return cost_matrix


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Solve linear assignment problem using Hungarian algorithm.

    Args:
        cost_matrix: Cost matrix (num_tracks, num_detections)
        thresh: Cost threshold for valid matches

    Returns:
        matches: Array of matched indices (N, 2)
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by threshold
    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    for row in range(cost_matrix.shape[0]):
        if row not in row_indices:
            unmatched_tracks.append(row)

    for col in range(cost_matrix.shape[1]):
        if col not in col_indices:
            unmatched_detections.append(col)

    for row, col in zip(row_indices, col_indices, strict=False):
        if cost_matrix[row, col] > thresh:
            unmatched_tracks.append(row)
            unmatched_detections.append(col)
        else:
            matches.append([row, col])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, unmatched_tracks, unmatched_detections
