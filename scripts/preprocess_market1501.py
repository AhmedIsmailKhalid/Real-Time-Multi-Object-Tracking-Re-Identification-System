"""
Preprocess Market-1501 dataset:
1. Filter valid images (remove junk IDs: 0000, -1)
2. Create train/val split by person ID
3. Resize images to 256x128
4. Generate metadata
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_filename(filename: str) -> dict:
    """
    Parse Market-1501 filename.

    Format: PPPP_cC_fFFFFFFF.jpg
    - PPPP: Person ID
    - C: Camera ID
    - FFFFFFF: Frame number
    """
    parts = filename.replace(".jpg", "").split("_")

    person_id = int(parts[0])
    camera_id = int(parts[1][1])  # Remove 'c' prefix
    frame_num = int(parts[2][1])  # Remove 'f' prefix

    return {
        "person_id": person_id,
        "camera_id": camera_id,
        "frame_num": frame_num,
        "filename": filename,
    }


def filter_valid_images(raw_dir: Path) -> list[dict]:
    """Filter out junk images (person_id == 0000 or -1)."""

    train_dir = raw_dir / "bounding_box_train"

    valid_images = []

    for img_file in train_dir.glob("*.jpg"):
        info = parse_filename(img_file.name)

        # Filter: Remove junk (0000) and distractors (-1)
        if info["person_id"] > 0:
            info["filepath"] = img_file
            valid_images.append(info)

    print(f"Total images in bounding_box_train: {len(list(train_dir.glob('*.jpg')))}")
    print(f"Valid images after filtering: {len(valid_images)}")

    return valid_images


def create_train_val_split(valid_images: list[dict], split_ratio: float = 0.8):
    """
    Create train/val split by person ID.

    Args:
        valid_images: List of valid image info dicts
        split_ratio: Train split ratio (default: 0.8)
    """
    # Get unique person IDs
    person_ids = sorted({img["person_id"] for img in valid_images})

    # Split person IDs
    num_train = int(len(person_ids) * split_ratio)
    train_ids = set(person_ids[:num_train])
    val_ids = set(person_ids[num_train:])

    # Split images
    train_images = [img for img in valid_images if img["person_id"] in train_ids]
    val_images = [img for img in valid_images if img["person_id"] in val_ids]

    print("\nPerson ID split:")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print("\nImage split:")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    return train_images, val_images, train_ids, val_ids


def process_and_copy_images(images: list[dict], output_dir: Path, target_size: tuple = (256, 128)):
    """
    Resize and copy images to output directory organized by person ID.

    Args:
        images: List of image info dicts
        output_dir: Output directory
        target_size: Target image size (height, width)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_info in tqdm(images, desc=f"Processing {output_dir.name}"):
        person_id = img_info["person_id"]

        # Create person directory
        person_dir = output_dir / f"{person_id:04d}"
        person_dir.mkdir(exist_ok=True)

        # Resize and save image
        src_path = img_info["filepath"]
        dst_path = person_dir / img_info["filename"]

        img = Image.open(src_path)
        img_resized = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        img_resized.save(dst_path)


def create_labels_csv(images: list[dict], output_file: Path):
    """Create CSV file with image labels."""

    data = []
    for img in images:
        data.append(
            {
                "filename": img["filename"],
                "person_id": img["person_id"],
                "camera_id": img["camera_id"],
                "frame_num": img["frame_num"],
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Saved labels to {output_file}")


def compute_dataset_stats(processed_dir: Path):
    """Compute mean and std for normalization."""

    train_dir = processed_dir / "train"

    # Sample 1000 images for stats computation
    all_images = list(train_dir.rglob("*.jpg"))
    sample_images = np.random.choice(all_images, min(1000, len(all_images)), replace=False)

    print("\nComputing dataset statistics...")

    pixels = []
    for img_path in tqdm(sample_images, desc="Sampling images"):
        img = Image.open(img_path)
        img_array = np.array(img).astype(np.float32) / 255.0
        pixels.append(img_array.reshape(-1, 3))

    pixels = np.vstack(pixels)

    mean = pixels.mean(axis=0).tolist()
    std = pixels.std(axis=0).tolist()

    print(f"Mean (RGB): {mean}")
    print(f"Std (RGB): {std}")

    return mean, std


def create_camera_mappings(images: list[dict]) -> dict:
    """Create mapping of person_id to camera_ids."""

    mappings = {}

    for img in images:
        person_id = img["person_id"]
        camera_id = img["camera_id"]

        if person_id not in mappings:
            mappings[person_id] = set()

        mappings[person_id].add(camera_id)

    # Convert sets to lists for JSON serialization
    mappings = {str(k): sorted(v) for k, v in mappings.items()}

    return mappings


def copy_query_gallery(raw_dir: Path, processed_dir: Path):
    """Copy query and gallery sets for evaluation."""

    query_src = raw_dir / "query"
    gallery_src = raw_dir / "bounding_box_test"

    query_dst = processed_dir / "query"
    gallery_dst = processed_dir / "gallery"

    query_dst.mkdir(parents=True, exist_ok=True)
    gallery_dst.mkdir(parents=True, exist_ok=True)

    print("\nCopying query set...")
    for img_file in tqdm(list(query_src.glob("*.jpg"))):
        shutil.copy(img_file, query_dst / img_file.name)

    print("Copying gallery set...")
    for img_file in tqdm(list(gallery_src.glob("*.jpg"))):
        # Filter junk images
        info = parse_filename(img_file.name)
        if info["person_id"] > 0:
            shutil.copy(img_file, gallery_dst / img_file.name)


def main():
    """Main preprocessing pipeline."""

    raw_dir = Path("data/raw/Market-1501-v15.09.15")
    processed_dir = Path("data/processed/market1501")

    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        print("Please download Market-1501 dataset first")
        return

    print("Starting Market-1501 preprocessing...")
    print(f"Raw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")

    # Step 1: Filter valid images
    print("\nStep 1: Filtering valid images...")
    valid_images = filter_valid_images(raw_dir)

    # Step 2: Create train/val split
    print("\nStep 2: Creating train/val split...")
    train_images, val_images, train_ids, val_ids = create_train_val_split(valid_images)

    # Step 3: Process and copy images
    print("\nStep 3: Processing and copying images...")
    process_and_copy_images(train_images, processed_dir / "train")
    process_and_copy_images(val_images, processed_dir / "val")

    # Step 4: Create labels CSV
    print("\nStep 4: Creating labels CSV...")
    create_labels_csv(train_images, processed_dir / "train_labels.csv")
    create_labels_csv(val_images, processed_dir / "val_labels.csv")

    # Step 5: Copy query and gallery sets
    print("\nStep 5: Copying query and gallery sets...")
    copy_query_gallery(raw_dir, processed_dir)

    # Step 6: Compute dataset statistics
    mean, std = compute_dataset_stats(processed_dir)

    # Step 7: Create camera mappings
    print("\nStep 7: Creating camera mappings...")
    camera_mappings = create_camera_mappings(train_images + val_images)

    # Step 8: Save metadata
    print("\nStep 8: Saving metadata...")
    metadata = {
        "num_train_ids": len(train_ids),
        "num_val_ids": len(val_ids),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "image_size": [256, 128],
        "mean": mean,
        "std": std,
        "num_cameras": 6,
    }

    metadata_file = processed_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    camera_mappings_file = processed_dir / "camera_mappings.json"
    with open(camera_mappings_file, "w") as f:
        json.dump(camera_mappings, f, indent=2)

    print(f"\nMetadata saved to {metadata_file}")
    print(f"Camera mappings saved to {camera_mappings_file}")

    print("\nMarket-1501 preprocessing complete!")
    print("\nSummary:")
    print(f"Train: {len(train_images)} images ({len(train_ids)} person IDs)")
    print(f"Val: {len(val_images)} images ({len(val_ids)} person IDs)")


if __name__ == "__main__":
    main()
