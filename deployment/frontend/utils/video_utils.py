"""
Video utilities for frontend.
"""

from pathlib import Path

import cv2


def get_video_info(video_path: str) -> dict:
    """
    Get video information.

    Args:
        video_path: Path to video

    Returns:
        Video info dict
    """
    cap = cv2.VideoCapture(video_path)

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info


def get_video_thumbnail(video_path: str, frame_number: int = 0) -> str:
    """
    Extract thumbnail from video.

    Args:
        video_path: Path to video
        frame_number: Frame to extract

    Returns:
        Path to thumbnail image
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if ret:
        thumbnail_path = Path(video_path).parent / f"{Path(video_path).stem}_thumb.jpg"
        cv2.imwrite(str(thumbnail_path), frame)
        return str(thumbnail_path)

    return None
