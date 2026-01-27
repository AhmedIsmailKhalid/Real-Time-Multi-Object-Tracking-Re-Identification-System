# Data Sources - Multi-Object Tracking & Re-Identification System

## Document Purpose
This document provides comprehensive information about all datasets used in this project, including download instructions, storage requirements, preprocessing steps, and expected directory structure.

---

## Dataset Overview

This project uses two primary datasets totaling approximately 6.7GB of compressed data:

| Dataset | Purpose | Compressed Size | Extracted Size | License | Download Complexity |
|---------|---------|-----------------|----------------|---------|-------------------|
| MOT17 | Multi-Object Tracking Training/Evaluation | 5.5GB | 8GB | CC BY-NC-SA 4.0 | Medium (registration required) |
| Market-1501 | Person Re-Identification Training/Evaluation | 1.2GB | 1.5GB | Academic Use Only | Easy (direct download) |

**Total Storage Required:**
- Raw datasets: ~10GB
- Processed datasets: ~12GB
- Model checkpoints: ~15GB
- Outputs/experiments: ~10GB
- **Minimum free space needed: 75GB**
- **Recommended free space: 100GB**

---

## Dataset 1: MOT17 (Multi-Object Tracking Benchmark)

### Overview
**Full Name:** Multi-Object Tracking 2017 Challenge Dataset  
**Provider:** MOTChallenge  
**Official URL:** https://motchallenge.net/data/MOT17/  
**Compressed Size:** 5.5GB  
**Extracted Size:** ~8GB  
**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (Academic research use)

### Why MOT17 (Not MOT18/19/20)?

| Version | Sequences | Avg People/Frame | Primary Use Case | Status |
|---------|-----------|------------------|------------------|--------|
| MOT17 | 14 (7 train/7 test) | 20-40 | Standard benchmark | **Most cited, industry standard** |
| MOT18 | 8 | 20-40 | Incremental update | Less commonly used |
| MOT19 | 8 | 100-150 | Very crowded scenes | Specialized research |
| MOT20 | 8 | 100-200+ | Extremely crowded | Specialized research |

**Decision Rationale:**
- **MOT17 is the industry benchmark:** 90%+ of tracking papers report MOT17 results
- **Scenario variety:** Indoor/outdoor, sparse/dense crowds (better for learning robust features)
- **Real-world relevance:** Represents 95% of actual tracking applications (retail, security, AV)
- **Achievable results:** Can realistically target 70-75 MOTA (production-grade performance)
- **Demo quality:** Clean results at 30-50 FPS vs. MOT20's choppy 10-15 FPS with worse accuracy
- **Interview recognition:** Saying "72 MOTA on MOT17" is immediately understood by hiring managers

MOT20 would require larger models (YOLOv8m instead of YOLOv8s), achieve 15-20 points lower MOTA, and produce a worse-looking demo while representing edge cases rarely seen in practice.

### Purpose in This Project
- **Primary:** Training and evaluating ByteTrack tracking algorithm
- **Secondary:** Validating YOLOv8 detection performance on pedestrians
- **Metrics:** MOTA (Multiple Object Tracking Accuracy), IDF1 (ID F1 Score), MT (Mostly Tracked), ML (Mostly Lost), ID Switches

### Dataset Statistics
- **Training sequences:** 7 sequences with ground truth
- **Test sequences:** 7 sequences (ground truth available only on evaluation server)
- **Total training frames:** ~5,300 frames
- **Total test frames:** ~5,800 frames
- **Resolution range:** 640x480 to 1920x1080 (variable)
- **Frame rate:** 25-30 FPS
- **Scenarios:** Indoor (shopping mall, metro station), outdoor (street, campus)
- **Crowd density:** Sparse (5-10 people) to dense (50+ people)
- **Annotations:** Bounding boxes, person IDs, visibility ratios, occlusion flags

### Sequence Details

**Training Sequences:**

| Sequence | Frames | Resolution | FPS | Scenario | Avg People | Difficulty |
|----------|--------|------------|-----|----------|------------|------------|
| MOT17-02 | 600 | 1920x1080 | 30 | Outdoor street | 30 | Medium |
| MOT17-04 | 1050 | 1920x1080 | 30 | Crowded indoor | 45 | Hard |
| MOT17-05 | 837 | 640x480 | 14 | Sparse outdoor | 10 | Easy |
| MOT17-09 | 525 | 1920x1080 | 30 | Indoor metro | 25 | Medium |
| MOT17-10 | 654 | 1920x1080 | 30 | Outdoor street | 35 | Medium |
| MOT17-11 | 900 | 1920x1080 | 30 | Indoor mall | 40 | Hard |
| MOT17-13 | 750 | 1920x1080 | 25 | Outdoor campus | 50 | Hard |

### Download Instructions

#### Step 1: Register for MOTChallenge
```
1. Visit: https://motchallenge.net/
2. Click "Register" (top-right)
3. Create free account (academic/research email preferred but not required)
4. Verify email
5. Login
```

#### Step 2: Download MOT17 Dataset
```
1. After login, navigate to: https://motchallenge.net/data/MOT17/
2. Scroll to "Download" section
3. Click "MOT17.zip" link (requires login)
4. Save to: /path/to/project/data/raw/
```

Alternatively, if the direct download link works:
```bash
# Create directory structure
mkdir -p data/raw
cd data/raw

# Manual download required - browser will redirect to login
# After login, download will start automatically
# File: MOT17.zip (5.5GB)
```

#### Step 3: Extract Dataset
```bash
cd data/raw

# Extract (takes 2-3 minutes)
unzip MOT17.zip

# Verify extraction
ls -lh MOT17/
# Should show: train/ and test/ directories
```

#### Step 4: Verify Download Integrity
```bash
# Check file size (should be within ±100MB)
du -sh MOT17.zip
# Expected: 5.5GB

# Check extracted size
du -sh MOT17/
# Expected: ~8GB

# Test archive integrity
unzip -t MOT17.zip
# Should complete without errors

# Verify directory structure
tree -L 3 MOT17/
```

### Expected Directory Structure After Extraction
```
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── det/
│   │   │   └── det.txt           # Public detections (DPM detector)
│   │   ├── gt/
│   │   │   └── gt.txt            # Ground truth annotations
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ... (600 frames)
│   │   └── seqinfo.ini           # Sequence metadata
│   ├── MOT17-02-FRCNN/           # Same sequence, FRCNN detections
│   ├── MOT17-02-SDP/             # Same sequence, SDP detections
│   ├── MOT17-04-DPM/
│   ├── MOT17-04-FRCNN/
│   ├── MOT17-04-SDP/
│   ├── MOT17-05-DPM/
│   ├── MOT17-05-FRCNN/
│   ├── MOT17-05-SDP/
│   ├── MOT17-09-DPM/
│   ├── MOT17-09-FRCNN/
│   ├── MOT17-09-SDP/
│   ├── MOT17-10-DPM/
│   ├── MOT17-10-FRCNN/
│   ├── MOT17-10-SDP/
│   ├── MOT17-11-DPM/
│   ├── MOT17-11-FRCNN/
│   ├── MOT17-11-SDP/
│   ├── MOT17-13-DPM/
│   ├── MOT17-13-FRCNN/
│   └── MOT17-13-SDP/
└── test/
    ├── MOT17-01-DPM/
    ├── MOT17-01-FRCNN/
    ├── MOT17-01-SDP/
    ├── MOT17-03-DPM/
    ├── MOT17-03-FRCNN/
    ├── MOT17-03-SDP/
    ├── MOT17-06-DPM/
    ├── MOT17-06-FRCNN/
    ├── MOT17-06-SDP/
    ├── MOT17-07-DPM/
    ├── MOT17-07-FRCNN/
    ├── MOT17-07-SDP/
    ├── MOT17-08-DPM/
    ├── MOT17-08-FRCNN/
    ├── MOT17-08-SDP/
    ├── MOT17-12-DPM/
    ├── MOT17-12-FRCNN/
    ├── MOT17-12-SDP/
    ├── MOT17-14-DPM/
    ├── MOT17-14-FRCNN/
    └── MOT17-14-SDP/
```

### Important Dataset Notes

**Three Detection Variants:**
Each sequence exists in three versions (DPM, FRCNN, SDP) representing different public detectors used in the benchmark.

**For this project:**
- **We use ONLY the FRCNN variants** (most commonly benchmarked in research)
- FRCNN detections are most similar to modern detectors like YOLOv8
- Ignore DPM and SDP variants (reduces storage and processing)

**Ground Truth Format (gt/gt.txt):**
```
Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

Example:
1, 1, 794.27, 247.59, 71.245, 174.88, 1, 1, 0.86
1, 2, 164.02, 263.51, 58.485, 131.86, 1, 1, 0.73

Fields:
- frame: Frame number (1-indexed)
- id: Track ID (unique per person)
- bb_left, bb_top: Top-left corner of bounding box
- bb_width, bb_height: Bounding box dimensions
- conf: Confidence (always 1 for ground truth)
- class: Object class (1 = pedestrian, we only use this class)
- visibility: Fraction of bounding box visible (0.0-1.0)
```

**Test Set:**
- Ground truth NOT publicly available
- Used only for official MOTChallenge evaluation server submission
- For this project: We use ONLY training sequences, split into train/val

### Preprocessing Strategy

We will implement preprocessing in `scripts/preprocess_mot17.py`:

**Step 1: Filter FRCNN Sequences Only**
```python
# Keep only: MOT17-XX-FRCNN directories
# Delete or ignore: MOT17-XX-DPM, MOT17-XX-SDP
# Reduces storage from 8GB to ~3GB
```

**Step 2: Convert to COCO Format**
```python
# Convert gt.txt to COCO JSON format
# COCO format used by YOLOv8 and most modern frameworks
# Output: train_annotations.json, val_annotations.json
```

**Step 3: Create Train/Val Split**
```python
# Strategy: Use 5 sequences for training, 2 for validation
# Train: MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10 (3,566 frames)
# Val: MOT17-11, MOT17-13 (1,650 frames)
# Split ensures variety in scenarios and difficulty levels
```

**Step 4: Generate Metadata**
```python
# Compute dataset statistics:
# - Total frames per split
# - Average people per frame
# - Min/max people per frame
# - Average visibility per sequence
# - ID distribution
# Output: metadata.json
```

### Post-Processing Directory Structure
```
data/processed/mot17/
├── annotations/
│   ├── train.json              # COCO format: 3,566 images, ~100K annotations
│   ├── val.json                # COCO format: 1,650 images, ~60K annotations
│   └── metadata.json           # Dataset statistics
├── images/
│   ├── train/
│   │   ├── MOT17-02-FRCNN/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   ├── MOT17-04-FRCNN/
│   │   ├── MOT17-05-FRCNN/
│   │   ├── MOT17-09-FRCNN/
│   │   └── MOT17-10-FRCNN/
│   └── val/
│       ├── MOT17-11-FRCNN/
│       └── MOT17-13-FRCNN/
└── README.txt                  # Processing log
```

### Expected Performance Targets

| Metric | Baseline (Simple Tracker) | Our Target (ByteTrack) | SOTA (Research) |
|--------|---------------------------|------------------------|-----------------|
| **MOTA** | 45-50% | 70-75% | 78-80% |
| **IDF1** | 50-55% | 72-76% | 78-82% |
| **MT** | 30-35% | 55-60% | 65-70% |
| **ML** | 25-30% | 10-15% | 5-10% |
| **ID Switches** | 800-1000 | 200-300 | 100-150 |
| **FPS** | 60+ | 30-50 | 20-30 |

**Target Interpretation:**
- **70-75 MOTA:** Very good performance, suitable for production applications
- **30-50 FPS:** Real-time performance on consumer GPU (RTX 3060)
- **200-300 ID switches:** Acceptable for 5K+ frames across varied scenarios

---

## Dataset 2: Market-1501 (Person Re-Identification)

### Overview
**Full Name:** Market-1501 Person Re-Identification Dataset  
**Provider:** Zheng et al., ICCV 2015  
**Official URL:** http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html  
**Google Drive Mirror:** https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view  
**Compressed Size:** 1.2GB  
**Extracted Size:** ~1.5GB  
**License:** Academic research use only

### Purpose in This Project
- **Primary:** Training ResNet50 Re-Identification network for person appearance feature extraction
- **Use case:** Given a person crop from one camera, find the same person in crops from another camera
- **Integration:** ByteTrack uses Re-ID features to re-associate lost tracks and handle occlusions
- **Metrics:** Rank-1 accuracy, Rank-5 accuracy, mAP (mean Average Precision)

### Dataset Statistics
- **Unique identities:** 1,501 persons
- **Total images:** 32,668 bounding box crops
- **Training set:** 12,936 images (751 identities, 6 cameras)
- **Query set:** 3,368 images (750 identities, 6 cameras)
- **Gallery set:** 15,913 images (750 identities, 6 cameras)
- **Distractors:** 500,000+ additional images (NOT used in this project)
- **Cameras:** 6 cameras positioned around a university campus market area
- **Image resolution:** Variable, typically ~128x64 after cropping
- **Format:** JPEG images with encoded metadata in filenames

### Why Market-1501 (The Re-ID Standard)?

| Dataset | Year | Identities | Images | Cameras | Size | Status |
|---------|------|------------|--------|---------|------|--------|
| **Market-1501** | 2015 | 1,501 | 32,668 | 6 | 1.2GB | **Industry benchmark** |
| CUHK03 | 2014 | 1,467 | 28,192 | 2 | 1.8GB | Outdated |
| DukeMTMC-reID | 2017 | 1,404 | 36,411 | 8 | 2.8GB | Good alternative |
| MSMT17 | 2018 | 4,101 | 126,441 | 15 | 11GB | Too large for portfolio |

**Decision Rationale:**
- **Benchmark standard:** 95% of Re-ID papers report Market-1501 results
- **Interview recognition:** "85% Rank-1 on Market-1501" is immediately understood
- **Manageable size:** 1.2GB compressed, trains in 4-6 hours on RTX 3060
- **Real-world relevance:** 6 cameras represents realistic multi-camera retail/office scenario
- **Achievable performance:** Can realistically target 85-88% Rank-1 accuracy
- **Not too easy/hard:** Balanced difficulty for learning without frustration

MSMT17 would require 4x longer training for only marginal portfolio value improvement.

### Filename Convention (Metadata Encoding)
```
Format: PPPP_cC_fFFFFFFF.jpg

Where:
- PPPP: Person ID (0001-1501)
  - 0001-1501: Valid person IDs
  - 0000: Junk images (ignore)
  - -1: Distractors (ignore)
- C: Camera ID (1-6)
- FFFFFFF: Frame number in original video

Examples:
0002_c1_f0044158.jpg  → Person 0002, Camera 1, Frame 0044158
0042_c3_f0026551.jpg  → Person 0042, Camera 3, Frame 0026551
0000_c1_f0000001.jpg  → Junk image (filter out)
-1_c2_f0012345.jpg    → Distractor (filter out)
```

### Download Instructions

#### Step 1: Download via Google Drive

**Option A: Manual Download (Recommended)**
```
1. Visit: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view
2. Click "Download" button (may require Google account)
3. Save as: Market-1501-v15.09.15.zip (1.2GB)
4. Move to: /path/to/project/data/raw/
```

**Option B: Automated Download with gdown**
```bash
# Install gdown
pip install gdown

# Create directory
mkdir -p data/raw
cd data/raw

# Download
gdown 0B8-rUzbwVRk0c054eEozWG9COHM

# Will download as: Market-1501-v15.09.15.zip
```

#### Step 2: Extract Dataset
```bash
cd data/raw

# Extract (takes ~1 minute)
unzip Market-1501-v15.09.15.zip

# Verify extraction
ls -lh Market-1501-v15.09.15/
# Should show: bounding_box_train/, bounding_box_test/, query/, gt_bbox/, gt_query/
```

#### Step 3: Verify Download Integrity
```bash
# Check compressed file size
du -sh Market-1501-v15.09.15.zip
# Expected: 1.2GB

# Check extracted size
du -sh Market-1501-v15.09.15/
# Expected: ~1.5GB

# Test archive integrity
unzip -t Market-1501-v15.09.15.zip
# Should complete without errors

# Count images in training set
ls Market-1501-v15.09.15/bounding_box_train/*.jpg | wc -l
# Expected: 12,936 images
```

### Expected Directory Structure After Extraction
```
Market-1501-v15.09.15/
├── bounding_box_train/           # 12,936 images (751 IDs, 6 cameras)
│   ├── 0002_c1_f0044158.jpg
│   ├── 0002_c1_f0044164.jpg
│   ├── 0002_c1_f0044176.jpg
│   ├── 0002_c2_f0042188.jpg
│   ├── 0003_c1_f0024781.jpg
│   └── ... (12,936 total)
├── bounding_box_test/            # 19,732 images (750 IDs, contains query + gallery)
│   └── ... (DO NOT USE for training)
├── query/                        # 3,368 images (750 IDs, for evaluation)
│   ├── 0001_c1_f0011158.jpg
│   └── ...
├── gt_bbox/                      # Ground truth bounding boxes (NOT USED)
│   └── ... (manual annotations, not needed for this project)
└── gt_query/                     # Ground truth for query (NOT USED)
    └── ... (not needed for this project)
```

### Dataset Usage Strategy

**Training:**
- Use `bounding_box_train/` (12,936 images, 751 identities)
- Split into train (80%) and validation (20%)
- Train: ~10,350 images (601 IDs)
- Val: ~2,586 images (150 IDs)

**Evaluation:**
- Use `query/` (3,368 images) as query set
- Use `bounding_box_test/` (19,732 images) as gallery set
- These are distinct from training (750 different IDs)
- Standard Re-ID evaluation protocol

**Ignore:**
- `gt_bbox/` : Manual bounding box annotations (dataset already cropped)
- `gt_query/` : Additional ground truth files (not needed)
- Images with ID `0000` or `-1` (junk/distractors)

### Preprocessing Strategy

We will implement preprocessing in `scripts/preprocess_market1501.py`:

**Step 1: Filter Valid Images**
```python
# Remove junk images (ID = 0000 or -1)
# From 12,936 images → ~12,000 valid images
# Parse filenames to extract person ID, camera ID, frame number
```

**Step 2: Create Train/Val Split**
```python
# Strategy: Split by person ID (not images)
# Ensures each person appears in only train OR val, not both
# Train: 601 person IDs (~10,350 images)
# Val: 150 person IDs (~2,586 images)
# Maintains ~80/20 split
```

**Step 3: Resize and Normalize**
```python
# Resize all images to 256x128 (standard Re-ID input size)
# Compute dataset mean/std for normalization:
# Mean: [0.485, 0.456, 0.406] (ImageNet standard, likely close)
# Std: [0.229, 0.224, 0.225]
# Save stats to metadata.json
```

**Step 4: Generate Camera Mappings**
```python
# Create ID → Camera mapping
# Used for multi-camera evaluation (cross-camera retrieval)
# Essential for evaluating Re-ID in multi-camera scenarios
```

**Step 5: Create Training Labels CSV**
```python
# Format: image_path, person_id, camera_id, frame_num
# train_labels.csv: ~10,350 rows
# val_labels.csv: ~2,586 rows
# Makes PyTorch DataLoader implementation easier
```

### Post-Processing Directory Structure
```
data/processed/market1501/
├── train/                        # Training images (organized by person ID)
│   ├── 0002/
│   │   ├── 0002_c1_f0044158.jpg
│   │   ├── 0002_c1_f0044164.jpg
│   │   └── ... (all images of person 0002)
│   ├── 0003/
│   │   └── ...
│   └── ... (601 person IDs)
├── val/                          # Validation images (organized by person ID)
│   ├── 1432/
│   ├── 1433/
│   └── ... (150 person IDs)
├── query/                        # Evaluation query set (original)
│   ├── 0001_c1_f0011158.jpg
│   └── ... (3,368 images)
├── gallery/                      # Evaluation gallery set (original)
│   ├── 0001_c2_f0022187.jpg
│   └── ... (19,732 images)
├── train_labels.csv              # person_id, camera_id, filepath
├── val_labels.csv                # person_id, camera_id, filepath
├── camera_mappings.json          # person_id → list of camera_ids
└── metadata.json                 # Dataset stats, mean/std, split info
```

### Expected Performance Targets

| Metric | Baseline (ResNet50 No Pretraining) | Our Target (ResNet50 + ImageNet) | SOTA (ViT/Complex) |
|--------|-------------------------------------|----------------------------------|---------------------|
| **Rank-1** | 70-75% | 85-88% | 92-95% |
| **Rank-5** | 85-90% | 95-97% | 98-99% |
| **mAP** | 55-60% | 70-75% | 85-90% |
| **Training Time** | 6-8 hours | 4-6 hours | 10-20 hours |
| **Inference (1 crop)** | 0.5ms | 0.5ms | 2-5ms |

**Target Interpretation:**
- **85-88% Rank-1:** Strong performance, competitive with published ResNet50 baselines
- **70-75% mAP:** Good overall retrieval quality across all queries
- **0.5ms per crop:** Real-time feature extraction for tracking integration

### Re-ID Metrics Explained

**Rank-1 Accuracy:**
```
Given a query image of person X:
- Search gallery for same person
- Rank-1 = Top result is correct person
- 85% Rank-1 = 85% of queries return correct person as #1 result
```

**Rank-5 Accuracy:**
```
Same as Rank-1, but check if correct person appears in top 5 results
95% Rank-5 = Very usable system (correct answer almost always in top 5)
```

**mAP (mean Average Precision):**
```
Average precision across all queries
Measures overall retrieval quality, not just top-1
70% mAP = Good overall performance across diverse queries
```

---

## Combined Dataset Storage Summary

### Raw Datasets
```
data/raw/
├── MOT17.zip                     5.5GB (delete after extraction to save space)
├── MOT17/                        8.0GB (keep FRCNN variants only → 3.0GB)
├── Market-1501-v15.09.15.zip     1.2GB (delete after extraction to save space)
└── Market-1501-v15.09.15/        1.5GB
-----------------------------------------------------------
Total (before cleanup):           16.2GB
Total (after cleanup):            4.5GB (delete .zip files, filter MOT17 variants)
```

### Processed Datasets
```
data/processed/
├── mot17/                        10.0GB (COCO annotations + organized images)
└── market1501/                   2.0GB (resized images + metadata)
-----------------------------------------------------------
Total processed:                  12.0GB
```

### Additional Project Storage
```
models/
├── detection/
│   └── yolov8s.pt                11MB (pre-trained)
├── reid/
│   └── resnet50-imagenet.pth     100MB (pre-trained)
└── checkpoints/                  15GB (training checkpoints, saved models)

outputs/
├── experiments/                  5GB (MLflow artifacts, training logs)
└── results/                      5GB (inference outputs, visualizations)
-----------------------------------------------------------
Total additional:                 25GB
```

### Total Project Storage Requirement
```
Raw datasets (after cleanup):     4.5GB
Processed datasets:               12GB
Model checkpoints:                15GB
Outputs/experiments:              10GB
-----------------------------------------------------------
Total:                            41.5GB

Recommended free space:           75-100GB (for safety margin)
```

---

## Data Download Strategy

### Phase 1: Immediate Download (Day 1)
**Priority:** CRITICAL (Required to start development)

1. **Download MOT17** (~5.5GB, 30-45 min)
2. **Download Market-1501** (~1.2GB, 5-10 min)

**Total download time:** ~1 hour  
**Total download size:** ~6.7GB

### Phase 2: Preprocessing (Day 1-2)
**Priority:** HIGH (Required before training)

1. Run `scripts/preprocess_mot17.py`
2. Run `scripts/preprocess_market1501.py`
3. Verify processed data integrity
4. Update DVC tracking (optional)

**Total preprocessing time:** ~2-3 hours (mostly I/O operations)

### Phase 3: Cleanup (Day 2)
**Priority:** MEDIUM (Save storage space)

1. Delete `MOT17.zip` and `Market-1501-v15.09.15.zip` (saves 6.7GB)
2. Delete MOT17 DPM and SDP variants (saves 5GB)
3. Keep only processed data + raw FRCNN sequences

**Space saved:** ~11.7GB

---

## Data Versioning with DVC

### Initial DVC Setup
```bash
# Initialize DVC in project
cd /path/to/mot-reid-system
dvc init

# Add raw datasets to DVC tracking
dvc add data/raw/MOT17
dvc add data/raw/Market-1501-v15.09.15

# Add processed datasets to DVC tracking
dvc add data/processed/mot17
dvc add data/processed/market1501

# Commit DVC metadata files to git
git add data/raw/*.dvc data/processed/*.dvc .gitignore
git commit -m "Add datasets to DVC tracking"
```

### Benefits of DVC for This Project
- **Git stays small:** Datasets not stored in git (only metadata .dvc files)
- **Version control for data:** Track changes to preprocessing, splits, augmentation
- **Reproducibility:** Lock exact dataset versions with model versions
- **Collaboration:** Share data with team via remote storage (optional)

### Optional: Remote Storage Setup
```bash
# Example: Configure AWS S3 as remote storage (optional)
# dvc remote add -d storage s3://my-bucket/mot-reid-data
# dvc push

# Or use Google Drive (free)
# dvc remote add -d storage gdrive://folder-id
# dvc push
```

**For this portfolio project:** DVC tracking is valuable even without remote storage (local versioning still helpful).

---

## Data Preprocessing Pipeline

### Automated Preprocessing
A complete preprocessing pipeline will be provided in `scripts/preprocess_all_datasets.py`:
```bash
# After downloading both datasets, run:
python scripts/preprocess_all_datasets.py

# This will:
# 1. Filter MOT17 to FRCNN variants only
# 2. Convert MOT17 to COCO format
# 3. Create MOT17 train/val split
# 4. Filter Market-1501 junk images
# 5. Create Market-1501 train/val split
# 6. Resize Market-1501 images
# 7. Compute dataset statistics
# 8. Generate metadata.json files
# 9. Update DVC tracking

# Expected output:
# - data/processed/mot17/ fully populated
# - data/processed/market1501/ fully populated
# - Preprocessing log: preprocessing_report.txt
```

**Preprocessing time:** ~2-3 hours (mostly I/O operations)

---

## Dataset Licenses & Usage Compliance

### Academic Use Restrictions

| Dataset | License | Commercial Use? | Attribution Required? |
|---------|---------|-----------------|---------------------|
| **MOT17** | CC BY-NC-SA 4.0 | NO | YES |
| **Market-1501** | Academic Only | NO (contact authors) | YES |

**For this portfolio project:**
- Academic research use is compliant
- Suitable for GitHub portfolio, resume, interviews
- NOT suitable for commercial deployment without license

**For commercial deployment, you would need to:**
1. Obtain commercial licenses from dataset providers
2. Train on proprietary company data
3. Use only datasets with commercial licenses (e.g., COCO for detection)

**Citation Requirements:**
```
MOT17:
@article{MOT16,
  title = {MOT16: A Benchmark for Multi-Object Tracking},
  author = {Milan, Anton and Leal-Taix\'{e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal = {arXiv:1603.00831},
  year = {2016}
}

Market-1501:
@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={ICCV},
  year={2015}
}
```

---

## Troubleshooting Common Issues

### Issue 1: MOTChallenge Registration Not Working
**Symptoms:** Email not received, registration fails

**Solutions:**
1. Check spam folder for confirmation email
2. Use institutional email if available (higher success rate)
3. Wait 24 hours, sometimes approval is manual
4. Alternative: Some sequences available on unofficial GitHub mirrors (not recommended)

### Issue 2: Google Drive Download Limit
**Symptoms:** "Download quota exceeded" error

**Solutions:**
```bash
# Method 1: Use gdown with resume capability
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view" --output Market-1501.zip

# Method 2: Download via institutional network (often higher quotas)

# Method 3: Wait 24 hours and retry

# Method 4: Use official website (slower but no quota)
# http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
```

### Issue 3: Slow Download Speeds
**Symptoms:** Downloads taking hours instead of minutes

**Solutions:**
1. Use university/corporate network (often faster)
2. Download overnight
3. Use download managers with resume capability:
```bash
# aria2c (fast, supports resume)
aria2c -x 16 -s 16 <URL>

# wget with resume
wget -c <URL>
```

### Issue 4: Insufficient Disk Space
**Symptoms:** Extraction fails, "No space left on device"

**Solutions:**
1. Delete .zip files immediately after extraction (saves 6.7GB)
2. Use external HDD/SSD for `data/` directory
3. Remove MOT17 non-FRCNN variants after preprocessing (saves 5GB)
4. Clean up old model checkpoints regularly

### Issue 5: Corrupted Download
**Symptoms:** Unzip fails, images won't open

**Solutions:**
```bash
# Test zip integrity before extraction
unzip -t dataset.zip

# If corrupted, re-download
# Verify file size first:
ls -lh dataset.zip

# MOT17.zip should be ~5.5GB ± 100MB
# Market-1501.zip should be ~1.2GB ± 50MB
```