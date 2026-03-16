# CrossID: Real-Time Multi-Camera Object Tracking & Person Re-Identification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](https://github.com/your-repo/crossid)

A system for multi-object tracking and person re-identification across multiple camera views. CrossID combines YOLOv8, ByteTrack, and ResNet-50 Re-ID with a web interface for video processing.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Results & Benchmarks](#results--benchmarks)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

CrossID is a video processing system that:
- Detects and tracks people in video frames
- Maintains consistent track IDs across multiple camera views
- Enables searching for specific individuals using query images
- Provides a web interface for video upload and processing

**Measured Performance:**
- Processes 1080p video at 15-30 FPS on NVIDIA RTX 3060
- Achieves 74% MOTA on MOT17 benchmark
- Achieves 99.05% Rank-1 accuracy on Market-1501 Re-ID benchmark
- Uses <6 GB GPU memory during processing

**Target Applications:**
- Multi-camera surveillance systems
- Retail analytics
- Event monitoring
- Video analysis research

---

## The Problem

### Challenges in Multi-Camera Video Processing

**1. Identity Consistency Across Cameras**
Standard tracking systems assign new track IDs when a person moves between camera views. This prevents tracking individuals across a camera network.

**2. Occlusions in Crowded Scenarios**
When tracking 50+ people simultaneously:
- Track IDs switch when people pass behind obstacles
- Tracks are lost when individuals temporarily disappear
- Performance degrades as crowd density increases

**3. Real-Time Processing Requirements**
Video processing must operate at ≥15 FPS to be practical for surveillance applications. This requires balancing accuracy with computational efficiency.

**4. Cross-Camera Viewpoint Variations**
Matching people across cameras requires handling:
- Different viewing angles
- Varying lighting conditions
- Resolution differences
- Background variations

**5. Deployment Barriers**
Many research systems require:
- Manual configuration of multiple parameters
- Command-line expertise
- Complex installation procedures
- Limited documentation

---

## The Solution

CrossID implements a multi-stage pipeline:

### 1. Object Detection
**Component:** YOLOv8s (Small)

**Measured Performance:**
- Speed: 35 FPS on RTX 3060 @ 1080p
- Precision: 92% on MOT17
- Recall: 88% on MOT17

**Implementation:**
```python
from ultralytics import YOLO
detector = YOLO("models/detection/pretrained/yolov8s.pt")
detections = detector(frame, classes=[0])  # Person class only
```

### 2. Multi-Object Tracking
**Component:** ByteTrack with Kalman Filter

**Measured Performance (MOT17 benchmark):**
- MOTA: 74% (with Re-ID enabled)
- IDF1: 78% (with Re-ID enabled)
- ID Switches: 72 per sequence average

**Features:**
- Kalman filter for motion prediction during occlusions
- Two-stage association (high confidence + low confidence detections)
- Track buffer retains lost tracks for 30 frames

### 3. Person Re-Identification
**Component:** ResNet-50 trained on Market-1501

**Measured Performance (Market-1501 test set):**
- Rank-1 Accuracy: 99.05%
- Rank-5 Accuracy: 99.8%
- mAP: 70.62%
- Feature Extraction Speed: 120 FPS

**Training Configuration:**
- Dataset: Market-1501 (32,668 images, 1,501 identities)
- Loss: Triplet Loss (margin=0.3) + Cross-Entropy
- Epochs: 120
- Batch Size: 32
- Optimizer: Adam (lr=0.00035)

**Re-ID Matching:**
```python
# Extract features from person crop
features = reid_model.extract_features(person_crop)

# Compare against gallery using cosine similarity
similarities = cosine_similarity(features, gallery_features)

# Match if similarity > 0.85 threshold
if max(similarities) > 0.85:
    assign_existing_track_id()
else:
    assign_new_track_id()
```

### 4. Multi-Camera Processing
**Implementation:** Shared Feature Gallery

All cameras use the same Re-ID feature gallery, enabling consistent ID assignment when a person moves between camera views.

**Measured Cross-Camera Match Rate (EPFL dataset):**
- Laboratory scenario (4 cameras, 4 people): 95%
- Basketball scenario (4 cameras, 10 people): 88%
- Average: 91%

### 5. Web Interface
**Components:** FastAPI (backend) + Streamlit (frontend)

**Features:**
- Video upload (MP4, AVI, MOV)
- Real-time progress tracking (0.5s polling interval)
- Processing settings configuration
- Processed video download
- Person search functionality

### 6. Person Search
**Implementation:** Feature Gallery Persistence

Feature galleries are automatically saved after processing. Users can upload a query image to search for specific individuals across all processed videos.

**Search Process:**
1. Upload query image
2. Extract Re-ID features
3. Compare against saved galleries using cosine similarity
4. Return matches above similarity threshold (default: 0.85)

---

## Key Features

### Core Functionality
- ✅ YOLOv8-based person detection
- ✅ ByteTrack multi-object tracking
- ✅ ResNet-50 person re-identification
- ✅ Multi-camera tracking (2-4 videos)
- ✅ Person search across processed videos

### Web Interface Pages
- ✅ Home - Project overview
- ✅ Demo Videos - 6 pre-loaded test scenarios
- ✅ Upload Video - Single video processing
- ✅ Multi-Camera - 2-4 video processing
- ✅ Person Search - Query-based search

### Output Formats
- ✅ Processed videos with tracking visualization
- ✅ Processing statistics (JSON)
- ✅ Search results (CSV, JSON, TXT)
- ✅ Merged multi-camera video with camera labels

---

## System Architecture
<p align="center">
  <img src="assets/Architectural-Diagram.svg" width="1200">
</p>

### Processing Pipeline
```
Input Video → Frame Extraction → YOLOv8 Detection → ByteTrack Tracking →
Re-ID Feature Extraction → Gallery Matching → Track Update → 
Visualization → FFmpeg H.264 Encoding → Output Video
```

**Measured Per-Frame Latency (1080p):**
- Detection: 10ms
- Tracking: 5ms
- Re-ID: 8ms
- Visualization: 2ms
- **Total:** 25ms (40 FPS theoretical maximum)

---

## Technology Stack

### Deep Learning

| Component | Technology | Version |
|-----------|-----------|---------|
| Detection | YOLOv8s (Ultralytics) | 8.0+ |
| Tracking | ByteTrack | Custom implementation |
| Re-ID | ResNet-50 | PyTorch implementation |
| Framework | PyTorch | 2.0+ |
| GPU Acceleration | CUDA + cuDNN | 11.8+ |

### Web Application

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend API | FastAPI | 0.104+ |
| Frontend | Streamlit | 1.28+ |
| Video I/O | OpenCV | 4.8+ |
| Video Encoding | FFmpeg | 6.0+ |

### Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| MOT17 | Tracking evaluation | 14 sequences, ~11K frames |
| Market-1501 | Re-ID training/eval | 32,668 images, 1,501 IDs |
| EPFL Multi-Camera | Multi-camera demos | 6 videos, 4 cameras |

### Development

| Tool | Purpose |
|------|---------|
| Poetry | Dependency management |
| Pytest | Testing (51 tests, 82% coverage) |
| Ruff | Code linting |

---

## Performance

### Detection Performance (MOT17)
- Precision: 92%
- Recall: 88%
- F1-Score: 90%
- Speed: 35 FPS @ 1080p (RTX 3060)
- GPU Memory: 2.8 GB

### Tracking Performance (MOT17)

| Metric | Without Re-ID | With Re-ID |
|--------|---------------|------------|
| MOTA | 72% | 74% |
| IDF1 | 65% | 78% |
| MOTP | 78% | 79% |
| ID Switches/seq | 95 | 72 |

### Re-ID Performance (Market-1501)
- Rank-1 Accuracy: 99.05%
- Rank-5 Accuracy: 99.8%
- mAP: 70.62%
- Feature Extraction: 120 FPS

### End-to-End Processing Speed

| Scenario | People Count | Resolution | FPS | GPU Memory |
|----------|--------------|-----------|-----|------------|
| Indoor Easy | 5-10 | 1080p | 30 | 4.2 GB |
| Crowded Street | 50+ | 1080p | 20 | 5.0 GB |
| Extreme Challenge | 150+ | 1080p | 12 | 5.8 GB |
| Multi-Camera (4 views) | Varied | 1080p | 15 | 5.2 GB |

**Test Hardware:** NVIDIA RTX 3060 12GB, Intel i7, 32GB RAM

### Hardware Requirements

**Minimum (for testing):**
- GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
- CPU: 4 cores
- RAM: 16 GB
- Storage: 20 GB

**Recommended (for production):**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- CPU: 8 cores
- RAM: 32 GB
- Storage: 50 GB

---

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+
- Poetry package manager

### Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/crossid.git
cd crossid

# 2. Install dependencies
poetry install

# 3. Download models and demo videos
poetry run python scripts/download_pretrained_models.py

# 4. Verify installation
poetry run python scripts/test_installation.py
```

### Running the System

**Terminal 1 - Backend:**
```bash
cd deployment/backend
poetry run python app.py
# Serves at http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd deployment/frontend
poetry run streamlit run app.py
# Opens browser at http://localhost:8501
```

### Processing Your First Video

1. Navigate to "Demo Videos" page
2. Select "Indoor Easy" scenario
3. Click "Process Video"
4. Wait for completion (progress bar updates every 0.5s)
5. Download processed result

---

## Installation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed installation instructions.

### Dataset Setup (Optional)

**MOT17 (for tracking evaluation):**
```bash
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d data/external/
python scripts/preprocess_mot17.py
```

**Market-1501 (for Re-ID training/evaluation):**
```bash
# Download from Google Drive or Kaggle (see docs/DATASET.md)
# Extract to data/external/Market-1501-v15.09.15/
python scripts/preprocess_market1501.py
```

---

## Usage

### Web Interface

**Single Video Processing:**
```
Home → Upload Video → Upload File → Configure Settings → 
Process → View Results → Download
```

**Multi-Camera Processing:**
```
Home → Multi-Camera → Upload 2-4 Videos → Process → 
Download Individual Videos + Merged Output
```

**Person Search:**
```
Home → Person Search → Upload Query Image → Set Threshold → 
Search → View Matches → Export Results
```

### Command-Line Interface

**Process Video:**
```bash
poetry run python scripts/run_inference.py \
    --input path/to/video.mp4 \
    --output path/to/output.mp4 \
    --enable-reid \
    --confidence 0.5
```

**Evaluate Tracking:**
```bash
poetry run python scripts/evaluate_mot.py --split train
```

**Evaluate Re-ID:**
```bash
poetry run python scripts/evaluate_reid.py \
    --model models/reid/final/resnet50_market_best.pth
```

### API Usage

**Upload Video:**
```bash
curl -X POST "http://localhost:8000/api/v1/track/video" \
     -F "video=@video.mp4" \
     -F "enable_reid=true" \
     -F "confidence_threshold=0.5"
```

**Check Status:**
```bash
curl "http://localhost:8000/api/v1/track/status/{job_id}"
```

**Download Result:**
```bash
curl "http://localhost:8000/api/v1/track/result/{job_id}" \
     --output result.mp4
```

See [docs/API.md](docs/API.md) for complete API reference.

---

## Documentation

### User Documentation
- [README.md](README.md) - Project overview (this file)
- [INSTALLATION.md](docs/INSTALLATION.md) - Setup guide
- [USAGE.md](docs/USAGE.md) - User workflows
- [API.md](docs/API.md) - API reference

### Technical Documentation
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [TECHNOLOGY_STACK.md](docs/TECHNOLOGY_STACK.md) - Technology choices
- [DATASET.md](docs/DATASET.md) - Dataset information
- [DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md) - File organization

### Development Documentation
- [IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) - Development timeline
- [SUCCESS_CRITERIA.md](docs/SUCCESS_CRITERIA.md) - Performance benchmarks

---

## Results & Benchmarks

### Tracking Performance (MOT17 Train Set)

**Comparison: ByteTrack vs ByteTrack + Re-ID**

| Metric | ByteTrack Only | + Re-ID | Change |
|--------|----------------|---------|--------|
| MOTA | 72.0% | 74.0% | +2.0pp |
| IDF1 | 65.0% | 78.0% | +13.0pp |
| MOTP | 78.2% | 79.1% | +0.9pp |
| ID Switches | 95/seq | 72/seq | -24.2% |
| FP | 3.8% | 3.2% | -0.6pp |
| FN | 21.4% | 20.8% | -0.6pp |

**Evaluation Method:**
- Dataset: MOT17 training set (7 sequences)
- Metrics computed using py-motmetrics library
- Both configurations use identical detection results
- Re-ID similarity threshold: 0.85

### Re-ID Performance (Market-1501 Test Set)

| Metric | Score |
|--------|-------|
| Rank-1 Accuracy | 99.05% |
| Rank-5 Accuracy | 99.8% |
| Rank-10 Accuracy | 99.9% |
| mAP | 70.62% |

**Training Configuration:**
- Training set: 751 identities, 12,936 images
- Test set: 750 identities, 19,732 gallery + 3,368 query
- Input size: 128×256
- Architecture: ResNet-50 + 2048-dim embedding

### Cross-Camera Matching (EPFL Dataset)

**Validation Method:**
- Manual ground truth annotation
- Cosine similarity threshold: 0.85
- Match considered correct if same person assigned same ID

| Scenario | Cameras | People | Correct Matches | Total Opportunities | Match Rate |
|----------|---------|--------|-----------------|---------------------|------------|
| Laboratory | 4 | 4 | 38 | 40 | 95% |
| Basketball | 4 | 10 | 88 | 100 | 88% |

### Processing Speed Analysis

**Component Latency (measured on RTX 3060, 1080p frame):**

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| YOLOv8 Detection | 10.2 | 40.8% |
| ByteTrack Tracking | 5.1 | 20.4% |
| ResNet-50 Re-ID | 8.3 | 33.2% |
| Visualization | 1.4 | 5.6% |
| **Total** | **25.0** | **100%** |

**Theoretical Maximum:** 40 FPS (1000ms / 25ms)  
**Actual Measured Range:** 15-30 FPS (varies by scenario + I/O overhead)

---

## Project Roadmap

### ✅ Completed (v1.0.0)

**Phase 1-2: Core Development**
- [x] YOLOv8 detection integration
- [x] ByteTrack tracking implementation
- [x] ResNet-50 Re-ID model training
- [x] Unified processing pipeline

**Phase 3: Deployment**
- [x] FastAPI backend with async processing
- [x] Streamlit web interface (5 pages)
- [x] API documentation (Swagger UI)

**Phase 4: Advanced Features**
- [x] Multi-camera tracking (2-4 videos)
- [x] Person search functionality
- [x] Feature gallery persistence
- [x] Export capabilities (CSV, JSON, TXT)

### 🔄 Planned

**v1.1 (Estimated Q2 2026)**
- [ ] Auto-crop person detection in query images
- [ ] Display matched frame thumbnails
- [ ] Video playback at match timestamps
- [ ] TensorRT optimization

**v2.0 (Estimated Q4 2026)**
- [ ] Docker containerization
- [ ] Cloud deployment support
- [ ] Authentication system
- [ ] Analytics dashboard

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`poetry run pytest`)
5. Update documentation
6. Submit a pull request

### Development Setup
```bash
git clone https://github.com/your-username/crossid.git
cd crossid
poetry install --with dev
poetry run pytest  # Run tests
poetry run ruff check src/  # Lint code
```

### Code Standards
- Follow PEP 8
- Add type hints
- Write docstrings (Google style)
- Maintain >80% test coverage

---

## License

MIT License - see [LICENSE](LICENSE) file.

### Third-Party Components

- YOLOv8: AGPL-3.0 (Ultralytics)
- PyTorch: BSD-style
- OpenCV: Apache 2.0
- FastAPI: MIT
- Streamlit: Apache 2.0

---

## Citation
```bibtex
@software{crossid2026,
  title={CrossID: Real-Time Multi-Camera Object Tracking and Person Re-Identification},
  author={Your Name},
  year={2026},
  url={https://github.com/your-username/crossid},
  version={1.0.0}
}
```

### References

**Detection:**
```bibtex
@software{yolov8_ultralytics,
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title={Ultralytics YOLOv8},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}
```

**Tracking:**
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and others},
  journal={ECCV},
  year={2022}
}
```

**Re-ID Dataset:**
```bibtex
@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and others},
  booktitle={ICCV},
  year={2015}
}
```

---

## FAQ

**Q: What hardware is required?**  
A: Minimum NVIDIA GTX 1660 Ti (6GB). Recommended RTX 3060 (12GB). CPU-only mode available but slow (3 FPS).

**Q: What video formats are supported?**  
A: MP4, AVI, MOV. Output is H.264 MP4.

**Q: How many cameras can be processed?**  
A: Currently 2-4 cameras simultaneously.

**Q: Can I use custom Re-ID models?**  
A: Yes. Place `.pth` file in `models/reid/final/` and update config.

**Q: Why is GPU memory usage high?**  
A: 4K video or 150+ people increase memory. Reduce resolution or adjust batch size in config.

**Q: How do I troubleshoot slow processing?**  
A: Check GPU utilization with `nvidia-smi`. Verify CUDA with `torch.cuda.is_available()`.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for more troubleshooting.

---

**Version:** 1.0.0  
**Last Updated:** February 2026  
**Status:** Production Ready
