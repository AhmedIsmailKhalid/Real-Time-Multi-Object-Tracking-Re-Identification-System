"""
Preprocess MOT17 dataset:
1. Filter FRCNN sequences only (ignore DPM and SDP)
2. Convert to COCO format
3. Create train/val split
4. Generate metadata
"""

import json
import shutil
from pathlib import Path

from tqdm import tqdm


def filter_frcnn_sequences(raw_dir: Path, processed_dir: Path):
    """Copy only FRCNN sequences to processed directory."""

    train_dir = raw_dir / "train"
    frcnn_sequences = [d for d in train_dir.iterdir() if d.is_dir() and "FRCNN" in d.name]

    print(f"Found {len(frcnn_sequences)} FRCNN sequences")

    for seq in frcnn_sequences:
        dest = processed_dir / "images" / "all" / seq.name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy images
        img_src = seq / "img1"
        img_dest = dest

        print(f"Copying {seq.name}...")
        shutil.copytree(img_src, img_dest, dirs_exist_ok=True)

    return frcnn_sequences


def parse_mot_gt(gt_file: Path) -> list[dict]:
    """Parse MOT ground truth file."""

    annotations = []

    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            # Filter: only pedestrian class (class_id == 1) with conf == 1
            if class_id == 1 and conf == 1:
                annotations.append(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "bbox": [x, y, w, h],
                        "visibility": visibility,
                    }
                )

    return annotations


def create_coco_format(sequences: list[Path], output_file: Path, image_base_dir: Path):
    """Convert MOT annotations to COCO format."""

    coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}

    image_id = 0
    annotation_id = 0

    for seq in tqdm(sequences, desc="Converting to COCO format"):
        gt_file = seq / "gt" / "gt.txt"
        annotations = parse_mot_gt(gt_file)

        # Get sequence info
        seqinfo_file = seq / "seqinfo.ini"
        with open(seqinfo_file) as f:
            lines = f.readlines()

        seq_length = int([line for line in lines if "seqLength" in line][0].split("=")[1])
        im_width = int([line for line in lines if "imWidth" in line][0].split("=")[1])
        im_height = int([line for line in lines if "imHeight" in line][0].split("=")[1])

        # Process each frame
        for frame_num in range(1, seq_length + 1):
            # Add image
            image_filename = f"{seq.name}/{frame_num:06d}.jpg"
            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": image_filename,
                    "width": im_width,
                    "height": im_height,
                    "frame_id": frame_num,
                    "sequence": seq.name,
                }
            )

            # Add annotations for this frame
            frame_annotations = [a for a in annotations if a["frame_id"] == frame_num]

            for ann in frame_annotations:
                x, y, w, h = ann["bbox"]
                coco_data["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": ann["track_id"],
                        "visibility": ann["visibility"],
                    }
                )
                annotation_id += 1

            image_id += 1

    # Save COCO JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(coco_data, f)

    print(f"Saved COCO annotations to {output_file}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")


def create_train_val_split(raw_dir: Path, processed_dir: Path):
    """Create train/val split."""

    train_dir = raw_dir / "train"
    frcnn_sequences = sorted([d for d in train_dir.iterdir() if d.is_dir() and "FRCNN" in d.name])

    # Split: 5 sequences for train, 2 for val
    train_sequences = [
        s
        for s in frcnn_sequences
        if s.name
        in [
            "MOT17-02-FRCNN",
            "MOT17-04-FRCNN",
            "MOT17-05-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-10-FRCNN",
        ]
    ]

    val_sequences = [s for s in frcnn_sequences if s.name in ["MOT17-11-FRCNN", "MOT17-13-FRCNN"]]

    print(f"\nTrain sequences: {[s.name for s in train_sequences]}")
    print(f"Val sequences: {[s.name for s in val_sequences]}")

    # Copy images to train/val directories
    for split_name, sequences in [("train", train_sequences), ("val", val_sequences)]:
        for seq in sequences:
            src = raw_dir / "train" / seq.name / "img1"
            dest = processed_dir / "images" / split_name / seq.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"Copying {seq.name} to {split_name}...")
            shutil.copytree(src, dest, dirs_exist_ok=True)

    # Create COCO annotations for each split
    create_coco_format(
        train_sequences,
        processed_dir / "annotations" / "train.json",
        processed_dir / "images" / "train",
    )

    create_coco_format(
        val_sequences, processed_dir / "annotations" / "val.json", processed_dir / "images" / "val"
    )

    # Generate metadata
    metadata = {
        "train_sequences": [s.name for s in train_sequences],
        "val_sequences": [s.name for s in val_sequences],
        "total_sequences": len(frcnn_sequences),
        "split_ratio": "5:2",
    }

    metadata_file = processed_dir / "annotations" / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_file}")


def main():
    """Main preprocessing pipeline."""

    raw_dir = Path("data/raw/MOT17")
    processed_dir = Path("data/processed/mot17")

    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        print("Please download MOT17 dataset first")
        return

    print("Starting MOT17 preprocessing...")
    print(f"Raw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")

    # Create train/val split and COCO annotations
    create_train_val_split(raw_dir, processed_dir)

    print("\nMOT17 preprocessing complete!")


if __name__ == "__main__":
    main()
