"""
Utility functions for object detection.
"""


def compute_iou(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)

    Returns:
        IoU score (0.0-1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def filter_detections_by_class(
    detections: list[tuple[float, float, float, float, float, int]], target_classes: list[int]
) -> list[tuple[float, float, float, float, float, int]]:
    """
    Filter detections by class ID.

    Args:
        detections: list of detections [(x1, y1, x2, y2, conf, class_id), ...]
        target_classes: list of class IDs to keep

    Returns:
        Filtered detections
    """
    return [det for det in detections if det[5] in target_classes]


def scale_boxes(
    boxes: list[tuple[float, float, float, float]],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """
    Scale bounding boxes to different image size.

    Args:
        boxes: list of boxes [(x1, y1, x2, y2), ...]
        original_size: Original image size (height, width)
        target_size: Target image size (height, width)

    Returns:
        Scaled boxes
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_boxes = []
    for x1, y1, x2, y2 in boxes:
        scaled_boxes.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))

    return scaled_boxes


def clip_boxes(
    boxes: list[tuple[float, float, float, float]], image_shape: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: list of boxes [(x1, y1, x2, y2), ...]
        image_shape: Image shape (height, width)

    Returns:
        Clipped boxes
    """
    height, width = image_shape

    clipped_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        clipped_boxes.append((x1, y1, x2, y2))

    return clipped_boxes
