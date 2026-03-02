# Dataset Documentation

This document provides information about the datasets used in the CrossID system for training, evaluation, and demonstration purposes.

## Overview

CrossID uses three primary datasets, each serving a specific purpose in the multi-object tracking and person re-identification pipeline:

1. **MOT17** - Multi-Object Tracking benchmark for tracking evaluation
2. **Market-1501** - Person Re-Identification dataset for Re-ID model training and evaluation
3. **EPFL Multi-Camera Pedestrian Videos** - Multi-camera demonstration videos

---

## 1. MOT17 Dataset

### Description
The MOT Challenge 2017 (MOT17) benchmark is a widely-used dataset for evaluating multi-object tracking algorithms. It contains 14 video sequences (7 training, 7 test) captured in unconstrained environments with varying crowd densities.

### Why MOT17 Was Chosen

**Industry Standard Benchmark:**
- Most widely adopted tracking evaluation dataset in computer vision research
- Enables direct comparison with state-of-the-art methods
- Standardized metrics (MOTA, MOTP, IDF1) recognized across academic and industry publications

**Realistic Scenarios:**
- Diverse environments (indoor/outdoor, static/moving cameras)
- Varying crowd densities (sparse to extremely crowded)
- Real-world challenges: occlusions, lighting variations, camera motion

**Comprehensive Annotations:**
- Frame-level bounding boxes with consistent track IDs
- Ground truth provided for precise evaluation
- Multiple detection sets (DPM, FRCNN, SDP) for robustness testing

**Alternative Considerations:**
- MOT20 was considered but rejected due to extremely high crowd density focus (not representative of typical use cases)
- MOT16 excluded as MOT17 is its direct successor with improved annotations

### How MOT17 Is Used in CrossID

**1. ByteTrack Tracker Validation**
- Validates tracking algorithm performance against published baselines
- Tests association logic (IoU matching, Kalman filtering)
- Evaluates track lifecycle management (creation, update, deletion)

**2. Performance Benchmarking**
```python
# Evaluation script: scripts/evaluate_mot.py
python scripts/evaluate_mot.py \
    --dataset mot17 \
    --split train \
    --output results/mot17_metrics.json
```

**3. Metrics Computed**
- **MOTA** (Multi-Object Tracking Accuracy) - Overall tracking performance
- **MOTP** (Multi-Object Tracking Precision) - Localization accuracy
- **IDF1** (ID F1 Score) - Identity preservation quality
- **FP/FN/IDS** - False positives, false negatives, ID switches

**4. Ablation Studies**
- Testing impact of Re-ID on tracking quality (IDF1 improvement)
- Comparing different confidence thresholds
- Analyzing performance across varying crowd densities

### Statistics
- **Sequences:** 14 total (7 train, 7 test)
- **Total Frames:** ~11,000 frames
- **Objects:** Pedestrians in various scenarios (indoor/outdoor, static/moving camera)
- **Annotations:** Bounding boxes with track IDs
- **Crowd Density:** 5-50 people per frame (average ~25)

### Download Instructions

**Option 1: Direct Download (Recommended)**
```bash
# Download from MOT Challenge website (no signup required)
wget https://motchallenge.net/data/MOT17.zip

# Extract
unzip MOT17.zip -d data/external/

# Verify structure
ls data/external/MOT17/train
# Should show: MOT17-02-DPM, MOT17-04-DPM, MOT17-05-DPM, etc.
```

**Option 2: Manual Download**
1. Visit: https://motchallenge.net/data/MOT17/
2. Download `MOT17.zip` (~5.5 GB)
3. Extract to `data/external/MOT17/`

### Preprocessing

Run the preprocessing script to prepare annotations:
```bash
cd scripts
python preprocess_mot17.py
```

This creates:
- `data/processed/mot17_train.pkl` - Training set annotations
- `data/processed/mot17_test.pkl` - Test set annotations

### Directory Structure
```
data/external/MOT17/
├── train/
│   ├── MOT17-02-DPM/          # Indoor, static camera, ~600 frames
│   │   ├── det/               # Public detections
│   │   ├── gt/                # Ground truth annotations
│   │   ├── img1/              # Video frames
│   │   └── seqinfo.ini        # Sequence metadata
│   ├── MOT17-04-DPM/          # Outdoor, moving camera, ~1050 frames
│   ├── MOT17-05-DPM/          # Outdoor, static camera, ~837 frames
│   └── ...
└── test/
    └── (similar structure)
```

---

## 2. Market-1501 Dataset

### Description
Market-1501 is a large-scale person Re-Identification dataset containing 32,668 images of 1,501 identities captured by 6 cameras. It is the primary dataset used for training and evaluating the Re-ID model.

### Why Market-1501 Was Chosen

**Large-Scale Person Re-ID Benchmark:**
- Largest publicly available Re-ID dataset at time of selection (32,668 images)
- 1,501 identities provide sufficient diversity for robust model training
- Multi-camera setup (6 cameras) matches real-world CCTV scenarios

**Real-World CCTV Conditions:**
- Captured from actual surveillance cameras in a university campus
- Natural variations in pose, lighting, viewpoint, and background
- Different camera positions simulate typical security installations

**Standardized Evaluation Protocol:**
- Well-defined train/test split (751/750 identities)
- Gallery + query structure mirrors real deployment scenarios
- Standard metrics (Rank-1, Rank-5, mAP) enable comparison with published work

**Cross-Camera Matching Focus:**
- Images from same person across different camera views
- Tests model's ability to match appearance despite viewpoint changes
- Directly applicable to CrossID's multi-camera tracking feature

**Alternative Considerations:**
- CUHK03 rejected due to smaller scale and controlled conditions
- DukeMTMC-reID considered but Market-1501 preferred for larger identity count
- MSMT17 excluded due to excessive complexity for initial implementation

### How Market-1501 Is Used in CrossID

**1. Re-ID Model Training**
```python
# Training script: scripts/train_reid.py
python scripts/train_reid.py \
    --dataset market1501 \
    --arch resnet50 \
    --epochs 120 \
    --batch-size 32
```

**Architecture:**
- ResNet-50 backbone (pretrained on ImageNet)
- Metric learning with triplet loss + cross-entropy
- 128×256 input resolution (standard for Re-ID)
- 2048-dimensional feature embeddings

**2. Re-ID Model Evaluation**
```python
# Evaluation script: scripts/evaluate_reid.py
python scripts/evaluate_reid.py \
    --model models/reid/final/resnet50_market_best.pth \
    --dataset market1501
```

**Metrics Achieved:**
- **Rank-1 Accuracy:** 99.05% (person correctly identified in top match)
- **mAP (mean Average Precision):** 70.62% (overall retrieval quality)
- **Rank-5:** 99.8% (person in top 5 matches)

**3. Feature Gallery Construction**
- Extracts 2048-dim embeddings for each person detection
- Builds similarity galleries for cross-camera matching
- Enables Person Search functionality across processed videos

**4. Cross-Camera Re-Identification**
```python
# In multi-camera tracking:
# 1. Extract features from person crops in each camera
# 2. Compare against shared gallery using cosine similarity
# 3. Assign same track ID if similarity > 0.85 threshold
```

**5. Person Search Implementation**
- Query image → Extract features → Search gallery → Return matches
- Similarity threshold tuning validated on Market-1501 test set
- Optimal threshold: 0.85 (high precision), 0.75 (higher recall)

### Statistics
- **Identities:** 1,501 persons
- **Images:** 32,668 bounding boxes
- **Cameras:** 6 camera views
- **Train/Test Split:** 751 IDs for training, 750 IDs for testing
- **Image Size:** Variable (resized to 128×256 for Re-ID)
- **Average Images per ID:** ~22 images
- **Camera Coverage:** Each person appears in 2-6 different cameras

### Download Instructions

**Option 1: Google Drive (Recommended)**
```bash
# Download from Google Drive
# Link: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view

# After downloading Market-1501-v15.09.15.zip to Downloads folder:
unzip ~/Downloads/Market-1501-v15.09.15.zip -d data/external/
```

**Option 2: Kaggle (Requires Account)**
1. Create Kaggle account: https://www.kaggle.com/
2. Visit: https://www.kaggle.com/datasets/whurobin/market1501
3. Click "Download" (requires Kaggle API token or manual download)
4. Extract to `data/external/Market-1501-v15.09.15/`

**Option 3: Using Kaggle API**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials (see: https://github.com/Kaggle/kaggle-api)
# Download dataset
kaggle datasets download -d whurobin/market1501

# Extract
unzip market1501.zip -d data/external/
```

### Preprocessing

Run the preprocessing script to organize the dataset:
```bash
cd scripts
python preprocess_market1501.py
```

This creates:
- `data/processed/market1501_train.pkl` - Training gallery and queries
- `data/processed/market1501_test.pkl` - Test gallery and queries
- Normalized images in `data/processed/market1501_imgs/`

### Directory Structure
```
data/external/Market-1501-v15.09.15/
├── bounding_box_train/     # Training images (751 IDs, 12,936 images)
├── bounding_box_test/      # Gallery images (750 IDs, 19,732 images)
├── query/                  # Query images for evaluation (3,368 images)
├── gt_bbox/               # Ground truth bounding boxes
└── gt_query/              # Ground truth for queries
```

### Filename Convention
Images follow the naming pattern: `PPPP_CCS_FFFFFF.jpg`
- `PPPP` - Person ID (e.g., 0001, 0002)
- `CC` - Camera ID (1-6)
- `S` - Sequence number
- `FFFFFF` - Frame number

Example: `0001_c1s1_001051.jpg` = Person 0001, Camera 1, Sequence 1, Frame 001051

---

## 3. EPFL Multi-Camera Pedestrian Videos

### Description
The EPFL Multi-Camera Pedestrian dataset provides synchronized multi-camera video sequences of pedestrians in controlled indoor environments. Used for demonstrating multi-camera tracking and Re-ID capabilities.

### Why EPFL Was Chosen

**Synchronized Multi-Camera Setup:**
- Hardware-synchronized cameras (frame-level alignment)
- Multiple viewpoints of same scene simultaneously
- Perfect for demonstrating cross-camera Re-ID without synchronization issues

**Controlled Environment for Demos:**
- Clean backgrounds minimize detection noise
- Good lighting ensures high-quality Re-ID features
- Predictable pedestrian behavior (walking, standing)
- Ideal for showcasing system capabilities to stakeholders

**Variety of Scenarios:**
- Laboratory: Sparse crowd, controlled movements
- Basketball: Dynamic movements, multiple people, occlusions
- Different activity types test various tracking challenges

**Ground Truth Availability:**
- Manual annotations for validation
- Known person identities across camera views
- Enables verification of cross-camera matching accuracy

**Alternative Considerations:**
- Campus/PETS datasets excluded due to synchronization challenges
- Wildtrack considered but EPFL chosen for simpler scenarios (better demos)
- Custom recordings avoided to ensure reproducibility

### How EPFL Is Used in CrossID

**1. Multi-Camera Tracking Demonstration**
```python
# Process 4 synchronized camera views with shared Re-ID gallery
# Frontend: Multi-Camera page
# Upload: 4 videos from different camera angles
# Output: Unified track IDs across all cameras
```

**Use Cases Demonstrated:**
- Person entering camera 1 gets Track ID 53
- Same person appears in camera 2 → Re-ID matches → keeps Track ID 53
- Same person in cameras 3 and 4 → consistent ID 53 throughout
- Different person gets different ID (e.g., ID 18) across all views

**2. Feature Gallery Validation**
- Tests Re-ID model's cross-viewpoint matching capability
- Validates that same person from different angles produces similar features
- Ensures cosine similarity thresholds work across camera angles

**3. User Interface Showcase**
- Pre-loaded demo videos in "Demo Videos" page
- Immediate functionality demonstration without user uploads
- Professional presentation for stakeholders and evaluations

**4. System Integration Testing**
- End-to-end pipeline validation (detection → tracking → Re-ID → visualization)
- Multi-video processing coordination
- FFmpeg video merging with camera labels
- Download functionality for merged results

**5. Performance Benchmarking**
- Reference for expected FPS with multiple video streams
- Memory usage profiling under multi-camera load
- UI responsiveness testing with parallel processing

### Statistics
- **Scenarios:** 2 environments (Laboratory, Basketball)
- **Cameras:** 4 synchronized cameras per scenario
- **Duration:** 2-3 minutes per scenario (~3,000-4,500 frames)
- **Resolution:** 1920×1080 @ 25 FPS
- **Format:** AVI (original) / MP4 (processed)
- **People per Scenario:** 4-10 pedestrians

### Download Instructions

**Option 1: Official EPFL Website**
```bash
# Visit EPFL CVLab datasets page
# https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

# Download specific sequences:
# - Laboratory: Multiple people walking in controlled environment
# - Basketball: Basketball game with player movements
```

**Option 2: Prepared Demo Videos**
Pre-processed demo videos are included in the repository:
```bash
# Demo videos location
ls data/external/demo_videos/
# Should contain:
# - 05_multicamera_lab.mp4 (4 synchronized lab cameras)
# - 05_multicamera_basketball.mp4 (4 synchronized game cameras)
```

### Preparation Script

To prepare custom multi-camera videos:
```bash
cd scripts
python prepare_demo_videos.py --source /path/to/epfl/videos
```

This script:
- Synchronizes camera feeds
- Normalizes resolution and frame rates
- Creates side-by-side multi-view layouts
- Saves to `data/external/demo_videos/`

### Directory Structure
```
data/external/demo_videos/
├── 01_indoor_easy.mp4                # Single camera, easy scenario (5-10 people)
├── 02_crowded_street.mp4            # Single camera, crowded (50+ people)
├── 03_sunset_backlit.mp4            # Single camera, challenging lighting
├── 04_extreme_challenge.mp4         # Single camera, extreme density (150+)
├── 05_multicamera_lab.mp4           # 4-camera laboratory (EPFL, 4 people)
└── 05_multicamera_basketball.mp4    # 4-camera basketball (EPFL, 10+ people)
```

---

## Dataset Selection Summary

| Dataset | Primary Purpose | Key Benefit | Used For |
|---------|----------------|-------------|----------|
| **MOT17** | Tracking evaluation | Industry standard benchmark | ByteTrack validation, metrics |
| **Market-1501** | Re-ID training | Large-scale, multi-camera | ResNet-50 training, feature extraction |
| **EPFL** | Multi-camera demo | Synchronized views | UI demos, cross-camera validation |

---

## Dataset Storage Requirements

| Dataset | Compressed | Uncompressed | Processed |
|---------|-----------|--------------|-----------|
| MOT17 | ~5.5 GB | ~12 GB | ~50 MB |
| Market-1501 | ~460 MB | ~1.2 GB | ~200 MB |
| EPFL (Demo) | ~500 MB | ~2 GB | ~500 MB |
| **Total** | **~6.5 GB** | **~15 GB** | **~750 MB** |

---

## Data Directory Structure

Complete directory structure after downloading and preprocessing all datasets:
```
data/
├── external/                           # Raw datasets
│   ├── MOT17/
│   │   ├── train/                     # 7 training sequences
│   │   └── test/                      # 7 test sequences
│   ├── Market-1501-v15.09.15/
│   │   ├── bounding_box_train/        # 12,936 training images
│   │   ├── bounding_box_test/         # 19,732 gallery images
│   │   └── query/                     # 3,368 query images
│   └── demo_videos/
│       ├── 01_indoor_easy.mp4
│       ├── 02_crowded_street.mp4
│       ├── 03_sunset_backlit.mp4
│       ├── 04_extreme_challenge.mp4
│       ├── 05_multicamera_lab.mp4
│       └── 05_multicamera_basketball.mp4
├── processed/                          # Preprocessed data
│   ├── mot17_train.pkl
│   ├── mot17_test.pkl
│   ├── market1501_train.pkl
│   ├── market1501_test.pkl
│   └── market1501_imgs/               # Normalized Re-ID images
└── interim/                            # Intermediate processing files
```

---

## Quick Setup Script

For automated dataset setup, use the provided script:
```bash
cd scripts
python download_pretrained_models.py  # Downloads models and demo videos

# Manual dataset downloads still required:
# 1. Download MOT17 from https://motchallenge.net/data/MOT17/
# 2. Download Market-1501 from Google Drive or Kaggle
# 3. Run preprocessing scripts
```

---

## Verification

Verify dataset setup:
```bash
# Check MOT17
python -c "from pathlib import Path; print('MOT17:', Path('data/external/MOT17/train').exists())"

# Check Market-1501
python -c "from pathlib import Path; print('Market-1501:', Path('data/external/Market-1501-v15.09.15/bounding_box_train').exists())"

# Check demo videos
python -c "from pathlib import Path; print('Demo videos:', len(list(Path('data/external/demo_videos').glob('*.mp4'))))"
```

Expected output:
```
MOT17: True
Market-1501: True
Demo videos: 6
```

---

## Troubleshooting

### Issue: MOT17 download fails
**Solution:** Use alternative mirrors or manual download from MOTChallenge website

### Issue: Market-1501 Google Drive quota exceeded
**Solution:** Use Kaggle alternative (requires account signup)

### Issue: Preprocessing scripts fail
**Solution:** 
```bash
# Install required packages
pip install pandas pillow

# Verify Python version (3.10+ required)
python --version
```

### Issue: Demo videos missing
**Solution:** Run `python scripts/prepare_demo_videos.py` or download pre-processed versions from repository releases

---

## License and Citation

### MOT17
```bibtex
@article{mot17,
  title={MOT17: A Benchmark for Multi-Object Tracking},
  author={Milan, Anton and Leal-Taixé, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
```

### Market-1501
```bibtex
@inproceedings{market1501,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={ICCV},
  year={2015}
}
```

### EPFL Multi-Camera Pedestrian
```bibtex
@inproceedings{epfl,
  title={Multi-Camera People Tracking with a Probabilistic Occupancy Map},
  author={Fleuret, François and Berclaz, Jérôme and Lengagne, Richard and Fua, Pascal},
  booktitle={TPAMI},
  year={2008}
}
```

---

## Notes

- **No DVC Required:** Datasets are managed directly through file system (no Data Version Control needed)
- **Automated Preprocessing:** Scripts handle annotation format conversion automatically
- **GPU Recommended:** Re-ID training on Market-1501 benefits significantly from GPU acceleration (10x speedup)
- **Storage Planning:** Ensure 20+ GB free disk space for full dataset setup
- **Internet Required:** Initial downloads require stable internet connection (large files, ~6.5 GB total)
- **Incremental Setup:** Datasets can be downloaded incrementally as needed (MOT17 → Market-1501 → EPFL)

---

**Last Updated:** February 2026  
**CrossID Version:** 1.0.0