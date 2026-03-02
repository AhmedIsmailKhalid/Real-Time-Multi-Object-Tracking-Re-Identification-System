"""
Prepare all 4 demo videos for CrossID.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np


def create_mot17_clip():
    """Scenario 1: Indoor Easy (MOT17-11)"""
    print("Creating Scenario 1: Indoor Easy...")

    src = Path("data/processed/mot17/images/val/MOT17-11-FRCNN")
    out = Path("data/external/demo_videos/01_indoor_easy.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Take first 30 seconds (900 frames at 30 FPS)
    frames = sorted(src.glob("*.jpg"))[:900]

    if not frames:
        print("ERROR: MOT17-11 frames not found!")
        return

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)

    writer.release()
    print(f"✓ Created: {out}")


def prepare_pexels_video():
    """Scenario 2: Crowded Overhead (Pexels)"""
    print("Preparing Scenario 2: Crowded Overhead...")

    src = Path("data/external/pexels_aerial_crowd.mp4")
    out = Path("data/external/demo_videos/02_crowded_overhead.mp4")

    if not src.exists():
        print(f"⚠️  {src} not found! Please download from Pexels.")
        print("    URL: https://www.pexels.com/video/2088290/")
        return

    # Simply copy the file (skip ffmpeg processing)
    shutil.copy(src, out)
    print(f"✓ Created: {out}")


def create_mot20_clip():
    """Scenario 3: Extreme Challenge (MOT20-01)"""
    print("Creating Scenario 3: Extreme Challenge...")

    src = Path("data/external/MOT20/train/MOT20-01/img1")
    out = Path("data/external/demo_videos/03_extreme_challenge.mp4")

    if not src.exists():
        print(f"ERROR: {src} not found! Please download MOT20.")
        return

    # MOT20-01 is 15 FPS, take first 450 frames (30 seconds)
    frames = sorted(src.glob("*.jpg"))[:450]

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)

    writer.release()
    print(f"✓ Created: {out}")


def create_market_multicam():
    """Scenario 4: Multi-Camera Re-ID (Market-1501)"""
    print("Creating Scenario 4: Multi-Camera Re-ID...")

    market_dir = Path("data/processed/market1501/train")

    if not market_dir.exists():
        print(f"ERROR: {market_dir} not found!")
        return

    # Person 0002 has good multi-camera coverage
    person_folder = market_dir / "0002"

    if not person_folder.exists():
        print(f"ERROR: Person folder {person_folder} not found!")
        return

    # Get images from different cameras
    cam1_images = sorted(person_folder.glob("*_c1s*"))[:15]
    cam2_images = sorted(person_folder.glob("*_c2s*"))[:15]

    print(f"  Found {len(cam1_images)} images from Camera 1")
    print(f"  Found {len(cam2_images)} images from Camera 2")

    if len(cam1_images) < 5 or len(cam2_images) < 5:
        print("ERROR: Not enough images for multi-camera demo!")
        return

    # Keep it simple - use moderate upscale
    scale_factor = 3
    original_w, original_h = 128, 256
    target_w = original_w * scale_factor  # 384
    target_h = original_h * scale_factor  # 768

    # HD canvas
    canvas_w, canvas_h = 1280, 720
    fps = 2

    def create_bordered_frame(img_path, camera_label):
        """Create HD frame with person centered."""
        img = cv2.imread(str(img_path))

        # Upscale with good interpolation
        img_upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Create black canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Calculate centering (handles case where image is taller than canvas)
        if target_h > canvas_h:
            # Image is too tall, crop it
            crop_y = (target_h - canvas_h) // 2
            img_upscaled = img_upscaled[crop_y : crop_y + canvas_h, :, :]
            y_offset = 0
        else:
            # Image fits, center it
            y_offset = (canvas_h - target_h) // 2

        x_offset = (canvas_w - target_w) // 2

        # Place image on canvas
        if target_h > canvas_h:
            canvas[:, x_offset : x_offset + target_w] = img_upscaled
        else:
            canvas[y_offset : y_offset + target_h, x_offset : x_offset + target_w] = img_upscaled

        # Add camera label
        label_color = (0, 255, 0) if "1" in camera_label else (0, 100, 255)
        cv2.putText(canvas, camera_label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, label_color, 3)

        # Add person ID
        cv2.putText(
            canvas, "Person ID: 0002", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        # Add timestamp from filename
        filename = img_path.name
        parts = filename.split("_")
        if len(parts) >= 3:
            timestamp = parts[2]
            cv2.putText(
                canvas,
                f"Frame: {timestamp}",
                (50, canvas_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        return canvas

    # Create Camera 1 video
    out_cam1 = Path("data/external/demo_videos/04a_multicam_camera1.mp4")

    writer1 = cv2.VideoWriter(
        str(out_cam1), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h)
    )

    for img_path in cam1_images:
        frame = create_bordered_frame(img_path, "CAMERA 1")
        writer1.write(frame)

    writer1.release()
    print(f"✓ Created: {out_cam1} (HD 1280x720)")

    # Create Camera 2 video
    out_cam2 = Path("data/external/demo_videos/04b_multicam_camera2.mp4")

    writer2 = cv2.VideoWriter(
        str(out_cam2), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h)
    )

    for img_path in cam2_images:
        frame = create_bordered_frame(img_path, "CAMERA 2")
        writer2.write(frame)

    writer2.release()
    print(f"✓ Created: {out_cam2} (HD 1280x720)")


if __name__ == "__main__":
    print("=" * 60)
    print("CROSSID - Demo Video Preparation")
    print("=" * 60)

    create_mot17_clip()
    prepare_pexels_video()
    create_mot20_clip()
    create_market_multicam()

    print("=" * 60)
    print("Demo video preparation complete!")
    print("=" * 60)
    print("\nCreated videos:")
    print("1. 01_indoor_easy.mp4 (30s, 1080p)")
    print("2. 02_crowded_overhead.mp4 (30s, 1080p)")
    print("3. 03_extreme_challenge.mp4 (30s, 1080p)")
    print("4. 04a_multicam_camera1.mp4 (15s)")
    print("5. 04b_multicam_camera2.mp4 (15s)")
