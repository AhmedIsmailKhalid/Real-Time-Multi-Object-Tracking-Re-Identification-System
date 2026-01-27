# Implementation Roadmap - Multi-Object Tracking & Re-Identification System

## Document Purpose
This document provides a detailed, week-by-week implementation plan for building the complete MOT + Re-ID system. This is the execution plan that turns architecture and design decisions into working code.

---

## Project Timeline Overview

**Total Duration:** 4 weeks (part-time: 15-20 hours/week) OR 2 weeks (full-time: 40+ hours/week)

**Phases:**
- **Week 1:** Environment Setup + Data Preparation + Detection Module
- **Week 2:** Tracking Module + Re-ID Module
- **Week 3:** Integration + Evaluation + Optimization
- **Week 4:** Deployment + Demo + Documentation

**Milestone-Driven:** Each week has clear deliverables and success criteria

**Flexible:** Can adjust pace based on available time and progress

---

## Week 1: Foundation (Environment + Data + Detection)

### Objectives
- Set up complete development environment
- Download and preprocess all datasets
- Implement and validate detection module
- Establish MLOps infrastructure

### Day 1: Environment Setup

**Tasks:**

1. **Repository Initialization**
```bash
   mkdir mot-reid-system
   cd mot-reid-system
   git init
   dvc init
```

2. **Create Directory Structure**
```bash
   # Use the exact structure from DIRECTORY_STRUCTURE.md
   mkdir -p data/{raw,processed,external}
   mkdir -p models/{detection,reid,tracking,exported}/{pretrained,checkpoints,final}
   mkdir -p src/{detection,tracking,reid,data,training,evaluation,inference,visualization,utils}
   mkdir -p api/{routes,schemas,middleware}
   mkdir -p tests/{unit,integration,fixtures}
   mkdir -p deployment/docker
   mkdir -p configs scripts notebooks docs outputs
```

3. **Python Environment**
```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install core dependencies
   pip install torch==2.0.1 torchvision==0.15.2
   pip install ultralytics==8.0.200
   pip install opencv-python==4.8.1.78
   pip install fastapi==0.104.1 uvicorn
   pip install mlflow==2.8.0
   pip install dvc==3.0.0
   pip install pytest==7.4.0 black==23.0.0 flake8 mypy
   
   # Save dependencies
   pip freeze > requirements.txt
```

4. **Setup Files**
   
   Create `setup.py`:
```python
   from setuptools import setup, find_packages
   
   setup(
       name="mot-reid-system",
       version="1.0.0",
       packages=find_packages(),
       install_requires=[
           "torch>=2.0.1",
           "torchvision>=0.15.2",
           "ultralytics>=8.0.200",
           "opencv-python>=4.8.1",
           "fastapi>=0.104.1",
           "mlflow>=2.8.0",
       ],
   )
```
   
   Install as editable package:
```bash
   pip install -e .
```

5. **Git Configuration**
   
   Create `.gitignore`:
```
   # Python
   __pycache__/
   *.pyc
   *.pyo
   venv/
   .env
   
   # Data and models (tracked by DVC)
   data/
   models/
   outputs/
   mlruns/
   
   # IDEs
   .vscode/
   .idea/
   
   # OS
   .DS_Store
   Thumbs.db
```

6. **Initial Commit**
```bash
   git add .
   git commit -m "Initial project structure and environment setup"
```

**Deliverable:** Working Python environment with project structure

**Success Criteria:**
- Can run: `python -c "from src.utils.logger import get_logger"`
- Directory structure matches DIRECTORY_STRUCTURE.md
- Git repository initialized with proper .gitignore

**Time Estimate:** 2-3 hours

---

### Day 2-3: Data Preparation

**Tasks:**

1. **Download MOT17**
```bash
   # Manual download from https://motchallenge.net/data/MOT17/
   # Save to: data/raw/MOT17.zip
   
   cd data/raw
   unzip MOT17.zip
   
   # Verify extraction
   ls MOT17/train  # Should show 21 sequences (7 × 3 detectors)
```

2. **Download Market-1501**
```bash
   # Install gdown
   pip install gdown
   
   # Download from Google Drive
   cd data/raw
   gdown 0B8-rUzbwVRk0c054eEozWG9COHM
   unzip Market-1501-v15.09.15.zip
   
   # Verify
   ls Market-1501-v15.09.15/bounding_box_train | wc -l  # Should be 12936
```

3. **Implement MOT17 Preprocessing**
   
   Create `scripts/preprocess_mot17.py`:
```python
   """
   Preprocess MOT17 dataset:
   1. Filter FRCNN sequences only
   2. Convert to COCO format
   3. Create train/val split
   4. Generate metadata
   """
   import json
   import shutil
   from pathlib import Path
   from typing import Dict, List, Tuple
   
   # Implementation will be provided in code delivery phase
```

4. **Implement Market-1501 Preprocessing**
   
   Create `scripts/preprocess_market1501.py`:
```python
   """
   Preprocess Market-1501 dataset:
   1. Filter valid images (remove junk)
   2. Create train/val split by person ID
   3. Resize images to 256x128
   4. Generate metadata (mean/std, camera mappings)
   """
   import os
   import shutil
   from pathlib import Path
   from PIL import Image
   import pandas as pd
   
   # Implementation will be provided in code delivery phase
```

5. **Run Preprocessing**
```bash
   python scripts/preprocess_mot17.py
   python scripts/preprocess_market1501.py
```

6. **Verify Processed Data**
```bash
   # Check MOT17 processed structure
   ls data/processed/mot17/annotations/
   # Should show: train.json, val.json, metadata.json
   
   # Check Market-1501 processed structure
   ls data/processed/market1501/
   # Should show: train/, val/, query/, gallery/, metadata.json
```

7. **DVC Tracking**
```bash
   # Track raw data
   dvc add data/raw/MOT17
   dvc add data/raw/Market-1501-v15.09.15
   
   # Track processed data
   dvc add data/processed/mot17
   dvc add data/processed/market1501
   
   # Commit DVC metadata
   git add data/raw/*.dvc data/processed/*.dvc .gitignore
   git commit -m "Add datasets to DVC tracking"
```

**Deliverable:** Preprocessed datasets ready for training

**Success Criteria:**
- MOT17 train.json has ~3,566 images, val.json has ~1,650 images
- Market-1501 train/ has ~10,350 images, val/ has ~2,586 images
- metadata.json files contain dataset statistics
- DVC tracking configured

**Time Estimate:** 4-6 hours

---

### Day 4-5: Detection Module Implementation

**Tasks:**

1. **Create Base Detector Interface**
   
   Create `src/detection/detector_base.py`:
```python
   """
   Abstract base class for object detectors.
   All detectors must implement this interface.
   """
   from abc import ABC, abstractmethod
   from typing import List, Tuple
   import numpy as np
   
   class DetectorBase(ABC):
       """Base class for object detectors."""
       
       @abstractmethod
       def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
           """
           Run object detection on input image.
           
           Args:
               image: Input image (H, W, 3) in BGR format
               
           Returns:
               List of detections: [(x1, y1, x2, y2, confidence, class_id), ...]
           """
           pass
```

2. **Implement YOLOv8 Detector**
   
   Create `src/detection/yolo_detector.py`:
```python
   """
   YOLOv8 object detector wrapper.
   Provides clean interface to Ultralytics YOLOv8.
   """
   from typing import List, Tuple
   from pathlib import Path
   import numpy as np
   import torch
   from ultralytics import YOLO
   import logging
   
   from src.detection.detector_base import DetectorBase
   
   logger = logging.getLogger(__name__)
   
   class YOLODetector(DetectorBase):
       """YOLOv8 object detector."""
       
       def __init__(
           self,
           model_path: Path,
           device: str = "cuda",
           conf_threshold: float = 0.5,
           iou_threshold: float = 0.45,
           target_classes: List[int] = None
       ):
           """
           Initialize YOLOv8 detector.
           
           Args:
               model_path: Path to YOLOv8 model (.pt file)
               device: Device to run inference on ("cuda" or "cpu")
               conf_threshold: Confidence threshold for detections
               iou_threshold: IoU threshold for NMS
               target_classes: List of class IDs to keep (None = all classes)
           """
           self.model_path = model_path
           self.device = device
           self.conf_threshold = conf_threshold
           self.iou_threshold = iou_threshold
           self.target_classes = target_classes or []
           
           # Load model
           self.model = self._load_model()
           logger.info(f"YOLOv8 detector initialized on {device}")
       
       def _load_model(self) -> YOLO:
           """Load YOLOv8 model from checkpoint."""
           # Implementation details
           pass
       
       def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
           """Run detection on image."""
           # Implementation details
           pass
```

3. **Create Detection Utilities**
   
   Create `src/detection/utils.py`:
```python
   """
   Utility functions for object detection.
   """
   import numpy as np
   from typing import List, Tuple
   
   def compute_iou(box1: Tuple, box2: Tuple) -> float:
       """Compute IoU between two bounding boxes."""
       pass
   
   def nms(boxes: List, scores: List, iou_threshold: float) -> List[int]:
       """Non-maximum suppression."""
       pass
   
   def scale_boxes(boxes: List, scale_factor: float) -> List:
       """Scale bounding box coordinates."""
       pass
```

4. **Download Pre-trained YOLOv8**
```bash
   # YOLOv8s (small) pre-trained on COCO
   mkdir -p models/detection/pretrained
   cd models/detection/pretrained
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

5. **Create Detection Config**
   
   Create `configs/detection.yaml`:
```yaml
   model:
     name: "yolov8s"
     pretrained_path: "models/detection/pretrained/yolov8s.pt"
     checkpoint_path: "models/detection/checkpoints"
     final_path: "models/detection/final"
   
   inference:
     device: "cuda"  # or "cpu"
     conf_threshold: 0.5
     iou_threshold: 0.45
     target_classes: [0]  # 0 = person class in COCO
     image_size: 640
   
   training:
     epochs: 30
     batch_size: 16
     learning_rate: 0.001
     image_size: 640
     device: "cuda"
     num_workers: 4
```

6. **Unit Tests for Detection**
   
   Create `tests/unit/test_detection.py`:
```python
   """
   Unit tests for detection module.
   """
   import pytest
   import numpy as np
   from pathlib import Path
   from src.detection.yolo_detector import YOLODetector
   
   def test_yolo_detector_init():
       """Test YOLODetector initialization."""
       # Test implementation
       pass
   
   def test_yolo_detector_detect():
       """Test detection on sample image."""
       # Test implementation
       pass
   
   def test_detection_output_format():
       """Test detection output format is correct."""
       # Test implementation
       pass
```

7. **Run Tests**
```bash
   pytest tests/unit/test_detection.py -v
```

**Deliverable:** Working detection module with YOLOv8

**Success Criteria:**
- YOLODetector can load pre-trained YOLOv8s model
- Detection on test image returns bounding boxes in correct format
- All unit tests pass
- Detection config loads correctly

**Time Estimate:** 6-8 hours

---

### Day 6-7: Detection Training (Optional Fine-tuning)

**Tasks:**

1. **Create Detection Trainer**
   
   Create `src/training/train_detection.py`:
```python
   """
   YOLOv8 training pipeline for MOT17.
   """
   import mlflow
   from ultralytics import YOLO
   from pathlib import Path
   import yaml
   
   class DetectionTrainer:
       """Trainer for YOLOv8 on MOT17 dataset."""
       
       def __init__(self, config_path: Path):
           """Initialize trainer with config."""
           pass
       
       def train(self):
           """Run training loop."""
           pass
       
       def validate(self):
           """Run validation."""
           pass
```

2. **Create Training Script**
   
   Create `scripts/train_detection.py`:
```python
   """
   Standalone script to train YOLOv8 detector.
   """
   import argparse
   from pathlib import Path
   from src.training.train_detection import DetectionTrainer
   from src.utils.config import load_config
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", type=str, default="configs/detection.yaml")
       args = parser.parse_args()
       
       # Load config
       config = load_config(args.config)
       
       # Initialize trainer
       trainer = DetectionTrainer(config)
       
       # Train
       trainer.train()
   
   if __name__ == "__main__":
       main()
```

3. **Optional: Fine-tune YOLOv8 on MOT17**
```bash
   # This is optional - pre-trained YOLOv8 works well on MOT17
   # Only run if you want to improve detection accuracy
   python scripts/train_detection.py --config configs/detection.yaml
```

4. **MLflow Tracking**
```bash
   # Start MLflow server
   mlflow server --host 0.0.0.0 --port 5000
   
   # Access UI at http://localhost:5000
```

**Deliverable:** (Optional) Fine-tuned YOLOv8 model

**Success Criteria:**
- If training: Model achieves >0.70 mAP on MOT17 validation
- MLflow logs experiments correctly
- Training can be resumed from checkpoints

**Time Estimate:** 4-6 hours (if training), 1 hour (if skipping)

**Note:** Pre-trained YOLOv8 already achieves good performance on pedestrian detection. Fine-tuning is optional for this project.

---

### Week 1 Deliverables Summary

**Completed:**
- Development environment fully configured
- MOT17 and Market-1501 datasets downloaded and preprocessed
- Detection module implemented and tested
- MLflow experiment tracking set up
- Git + DVC version control configured

**Success Metrics:**
- All unit tests pass
- Detection module can process images and return bounding boxes
- Preprocessed data validated (correct counts, formats)
- Can import from src package: `from src.detection.yolo_detector import YOLODetector`

**Ready for Week 2:** Tracking and Re-ID implementation

---

## Week 2: Core Modules (Tracking + Re-ID)

### Objectives
- Implement ByteTrack tracking algorithm
- Implement ResNet50 Re-ID network
- Train Re-ID model on Market-1501
- Integrate tracking with detection

### Day 8-9: Tracking Module Implementation

**Tasks:**

1. **Create Base Tracker Interface**
   
   Create `src/tracking/tracker_base.py`:
```python
   """
   Abstract base class for multi-object trackers.
   """
   from abc import ABC, abstractmethod
   from typing import List, Tuple
   import numpy as np
   
   class TrackerBase(ABC):
       """Base class for object trackers."""
       
       @abstractmethod
       def update(self, detections: List[Tuple]) -> List[Tuple]:
           """
           Update tracks with new detections.
           
           Args:
               detections: List of detections [(x1,y1,x2,y2,conf,class), ...]
               
           Returns:
               List of tracks [(x1,y1,x2,y2,track_id), ...]
           """
           pass
       
       @abstractmethod
       def reset(self):
           """Reset tracker state."""
           pass
```

2. **Implement Kalman Filter**
   
   Create `src/tracking/kalman_filter.py`:
```python
   """
   Kalman filter for object motion prediction.
   Predicts object position in next frame based on motion history.
   """
   import numpy as np
   
   class KalmanFilter:
       """
       Kalman filter for bounding box tracking.
       State: [x, y, w, h, dx, dy, dw, dh]
       """
       
       def __init__(self):
           """Initialize Kalman filter."""
           # State transition matrix, measurement matrix, etc.
           pass
       
       def predict(self):
           """Predict next state."""
           pass
       
       def update(self, measurement):
           """Update state with measurement."""
           pass
```

3. **Implement Matching Logic**
   
   Create `src/tracking/matching.py`:
```python
   """
   Matching algorithms for data association.
   """
   import numpy as np
   from typing import List, Tuple
   from scipy.optimize import linear_sum_assignment
   
   def iou_distance(tracks: List, detections: List) -> np.ndarray:
       """Compute IoU distance matrix between tracks and detections."""
       pass
   
   def hungarian_matching(cost_matrix: np.ndarray, threshold: float) -> Tuple[List, List, List]:
       """
       Hungarian algorithm for optimal assignment.
       
       Returns:
           matched_indices, unmatched_tracks, unmatched_detections
       """
       pass
```

4. **Implement ByteTrack**
   
   Create `src/tracking/bytetrack.py`:
```python
   """
   ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
   
   Key innovation: Uses both high and low confidence detections.
   - High-conf detections: Create new tracks
   - Low-conf detections: Recover lost tracks
   """
   from typing import List, Tuple
   import numpy as np
   
   from src.tracking.tracker_base import TrackerBase
   from src.tracking.kalman_filter import KalmanFilter
   from src.tracking.matching import iou_distance, hungarian_matching
   
   class ByteTracker(TrackerBase):
       """ByteTrack multi-object tracker."""
       
       def __init__(
           self,
           high_conf_thresh: float = 0.6,
           low_conf_thresh: float = 0.1,
           track_buffer: int = 30,
           match_thresh: float = 0.8
       ):
           """
           Initialize ByteTrack.
           
           Args:
               high_conf_thresh: Threshold for high confidence detections
               low_conf_thresh: Threshold for low confidence detections
               track_buffer: Number of frames to keep lost tracks
               match_thresh: IoU threshold for matching
           """
           self.high_conf_thresh = high_conf_thresh
           self.low_conf_thresh = low_conf_thresh
           self.track_buffer = track_buffer
           self.match_thresh = match_thresh
           
           self.tracks = []
           self.track_id_count = 0
           self.frame_id = 0
       
       def update(self, detections: List[Tuple]) -> List[Tuple]:
           """
           Update tracks with new detections.
           
           Two-stage association:
           1. Match high-conf detections to existing tracks
           2. Match low-conf detections to unmatched tracks
           """
           # Implementation details
           pass
       
       def reset(self):
           """Reset tracker state."""
           self.tracks = []
           self.track_id_count = 0
           self.frame_id = 0
```

5. **Create Tracking Config**
   
   Create `configs/tracking.yaml`:
```yaml
   bytetrack:
     high_conf_thresh: 0.6
     low_conf_thresh: 0.1
     track_buffer: 30
     match_thresh: 0.8
     
   evaluation:
     min_track_length: 3  # Minimum frames for valid track
     min_confidence: 0.3  # Minimum confidence for evaluation
```

6. **Unit Tests for Tracking**
   
   Create `tests/unit/test_tracking.py`:
```python
   """
   Unit tests for tracking module.
   """
   import pytest
   import numpy as np
   from src.tracking.bytetrack import ByteTracker
   from src.tracking.kalman_filter import KalmanFilter
   from src.tracking.matching import iou_distance, hungarian_matching
   
   def test_bytetrack_init():
       """Test ByteTracker initialization."""
       pass
   
   def test_kalman_filter_prediction():
       """Test Kalman filter prediction."""
       pass
   
   def test_iou_matching():
       """Test IoU-based matching."""
       pass
```

**Deliverable:** Working ByteTrack implementation

**Success Criteria:**
- ByteTracker can update tracks with new detections
- Kalman filter predicts object motion
- Matching algorithm correctly associates detections to tracks
- Unit tests pass

**Time Estimate:** 8-10 hours

---

### Day 10-11: Re-ID Module Implementation

**Tasks:**

1. **Create Base Re-ID Interface**
   
   Create `src/reid/reid_base.py`:
```python
   """
   Abstract base class for Re-ID models.
   """
   from abc import ABC, abstractmethod
   import torch
   import numpy as np
   
   class ReIDBase(ABC):
       """Base class for person re-identification models."""
       
       @abstractmethod
       def extract_features(self, images: torch.Tensor) -> torch.Tensor:
           """
           Extract appearance features from person crops.
           
           Args:
               images: Batch of person crops (N, 3, H, W)
               
           Returns:
               Feature vectors (N, feature_dim)
           """
           pass
```

2. **Implement ResNet50 Re-ID Network**
   
   Create `src/reid/resnet_reid.py`:
```python
   """
   ResNet50-based person re-identification network.
   """
   import torch
   import torch.nn as nn
   import torchvision.models as models
   from typing import Optional
   from pathlib import Path
   
   from src.reid.reid_base import ReIDBase
   
   class ResNet50ReID(ReIDBase):
       """ResNet50 backbone for person Re-ID."""
       
       def __init__(
           self,
           num_classes: int,
           pretrained: bool = True,
           feature_dim: int = 512
       ):
           """
           Initialize ResNet50 Re-ID model.
           
           Args:
               num_classes: Number of person identities (for training)
               pretrained: Use ImageNet pre-trained weights
               feature_dim: Dimension of output feature vector
           """
           super().__init__()
           
           # Load ResNet50 backbone
           self.backbone = models.resnet50(pretrained=pretrained)
           
           # Remove final FC layer
           self.backbone.fc = nn.Identity()
           
           # Add Re-ID head
           self.feature_dim = feature_dim
           self.bn = nn.BatchNorm1d(2048)
           self.fc = nn.Linear(2048, feature_dim)
           self.bn_feat = nn.BatchNorm1d(feature_dim)
           
           # Classification head (for training only)
           self.classifier = nn.Linear(feature_dim, num_classes)
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass."""
           # Implementation
           pass
       
       def extract_features(self, images: torch.Tensor) -> torch.Tensor:
           """Extract normalized feature vectors."""
           # Implementation
           pass
```

3. **Implement Triplet Loss**
   
   Create `src/reid/metric_learning.py`:
```python
   """
   Metric learning losses for Re-ID training.
   """
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   
   class TripletLoss(nn.Module):
       """
       Triplet loss for metric learning.
       
       Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
       """
       
       def __init__(self, margin: float = 0.3):
           """Initialize triplet loss."""
           super().__init__()
           self.margin = margin
       
       def forward(self, anchor, positive, negative):
           """Compute triplet loss."""
           # Implementation
           pass
```

4. **Create Market-1501 Dataset Loader**
   
   Create `src/data/market_dataset.py`:
```python
   """
   PyTorch Dataset for Market-1501.
   """
   import torch
   from torch.utils.data import Dataset
   from pathlib import Path
   from PIL import Image
   import pandas as pd
   from typing import Tuple
   
   class Market1501Dataset(Dataset):
       """Market-1501 person re-identification dataset."""
       
       def __init__(
           self,
           data_dir: Path,
           split: str = "train",
           transform=None
       ):
           """
           Initialize Market-1501 dataset.
           
           Args:
               data_dir: Path to processed Market-1501 data
               split: "train", "val", "query", or "gallery"
               transform: Image transformations
           """
           self.data_dir = data_dir
           self.split = split
           self.transform = transform
           
           # Load labels
           if split in ["train", "val"]:
               self.labels_df = pd.read_csv(data_dir / f"{split}_labels.csv")
           
       def __len__(self):
           """Dataset length."""
           pass
       
       def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
           """Get item by index."""
           # Implementation
           pass
```

5. **Create Re-ID Config**
   
   Create `configs/reid.yaml`:
```yaml
   model:
     name: "resnet50_reid"
     pretrained_imagenet: true
     feature_dim: 512
     num_classes: 751  # Market-1501 training identities
   
   training:
     epochs: 100
     batch_size: 32
     learning_rate: 0.00035
     weight_decay: 0.0005
     step_size: 40
     gamma: 0.1
     margin: 0.3  # Triplet loss margin
     device: "cuda"
     num_workers: 4
     
   evaluation:
     batch_size: 64
     distance_metric: "cosine"  # or "euclidean"
```

**Deliverable:** ResNet50 Re-ID model architecture

**Success Criteria:**
- ResNet50ReID model can be instantiated
- Model can perform forward pass on batch of images
- TripletLoss computes correctly
- Market1501Dataset loads images and labels

**Time Estimate:** 6-8 hours

---

### Day 12-14: Re-ID Training

**Tasks:**

1. **Create Re-ID Trainer**
   
   Create `src/training/train_reid.py`:
```python
   """
   Training pipeline for Re-ID model on Market-1501.
   """
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   import mlflow
   from pathlib import Path
   from tqdm import tqdm
   
   from src.reid.resnet_reid import ResNet50ReID
   from src.reid.metric_learning import TripletLoss
   from src.data.market_dataset import Market1501Dataset
   from src.utils.config import load_config
   
   class ReIDTrainer:
       """Trainer for Re-ID model."""
       
       def __init__(self, config: dict):
           """Initialize trainer."""
           self.config = config
           
           # Initialize model
           self.model = ResNet50ReID(
               num_classes=config['model']['num_classes'],
               pretrained=config['model']['pretrained_imagenet'],
               feature_dim=config['model']['feature_dim']
           )
           
           # Loss functions
           self.criterion_ce = nn.CrossEntropyLoss()
           self.criterion_triplet = TripletLoss(margin=config['training']['margin'])
           
           # Optimizer
           self.optimizer = optim.Adam(
               self.model.parameters(),
               lr=config['training']['learning_rate'],
               weight_decay=config['training']['weight_decay']
           )
           
           # Learning rate scheduler
           self.scheduler = optim.lr_scheduler.StepLR(
               self.optimizer,
               step_size=config['training']['step_size'],
               gamma=config['training']['gamma']
           )
       
       def train_epoch(self, dataloader: DataLoader) -> float:
           """Train for one epoch."""
           # Implementation
           pass
       
       def validate(self, dataloader: DataLoader) -> float:
           """Run validation."""
           # Implementation
           pass
       
       def train(self, num_epochs: int):
           """Full training loop."""
           # Implementation
           pass
```

2. **Create Training Script**
   
   Create `scripts/train_reid.py`:
```python
   """
   Standalone script to train Re-ID model.
   """
   import argparse
   from pathlib import Path
   from src.training.train_reid import ReIDTrainer
   from src.utils.config import load_config
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", type=str, default="configs/reid.yaml")
       args = parser.parse_args()
       
       # Load config
       config = load_config(args.config)
       
       # Initialize trainer
       trainer = ReIDTrainer(config)
       
       # Train
       trainer.train(num_epochs=config['training']['epochs'])
   
   if __name__ == "__main__":
       main()
```

3. **Run Re-ID Training**
```bash
   # Start MLflow (if not already running)
   mlflow server --host 0.0.0.0 --port 5000 &
   
   # Train Re-ID model
   python scripts/train_reid.py --config configs/reid.yaml
```

4. **Monitor Training**
```bash
   # View training progress in MLflow UI
   # http://localhost:5000
   
   # Training should take 4-6 hours on RTX 3060
   # Target metrics:
   # - Training loss: < 0.5 after 100 epochs
   # - Validation accuracy: > 85%
```

5. **Save Best Model**
```bash
   # Best model automatically saved to models/reid/checkpoints/
   # Copy best checkpoint to final/
   cp models/reid/checkpoints/resnet50_market_best.pth models/reid/final/
```

**Deliverable:** Trained Re-ID model

**Success Criteria:**
- Model trains without errors
- Training loss decreases over epochs
- Validation Rank-1 accuracy > 85%
- MLflow logs all metrics and artifacts
- Best checkpoint saved

**Time Estimate:** 8-10 hours (including training time)

---

### Week 2 Deliverables Summary

**Completed:**
- ByteTrack tracking algorithm implemented and tested
- ResNet50 Re-ID network implemented
- Re-ID model trained on Market-1501
- Both modules tested independently

**Success Metrics:**
- ByteTracker successfully tracks objects across frames
- Re-ID model achieves >85% Rank-1 on Market-1501
- All unit tests pass
- MLflow tracks all experiments

**Ready for Week 3:** Integration and evaluation

---

## Week 3: Integration + Evaluation + Optimization

### Objectives
- Integrate detection + tracking + Re-ID into end-to-end pipeline
- Evaluate on MOT17 and Market-1501 benchmarks
- Optimize performance (speed and accuracy)
- Create visualizations and demo videos

### Day 15-16: Pipeline Integration

**Tasks:**

1. **Create Inference Pipeline**
   
   Create `src/inference/pipeline.py`:
```python
   """
   End-to-end inference pipeline.
   Integrates detection, tracking, and Re-ID.
   """
   from typing import List, Dict, Optional
   from pathlib import Path
   import cv2
   import numpy as np
   import torch
   
   from src.detection.yolo_detector import YOLODetector
   from src.tracking.bytetrack import ByteTracker
   from src.reid.resnet_reid import ResNet50ReID
   from src.utils.config import load_config
   from src.utils.logger import get_logger
   
   logger = get_logger(__name__)
   
   class InferencePipeline:
       """End-to-end MOT + Re-ID inference pipeline."""
       
       def __init__(self, config_path: Path):
           """
           Initialize pipeline with all components.
           
           Args:
               config_path: Path to inference config file
           """
           self.config = load_config(config_path)
           
           # Initialize detector
           self.detector = YOLODetector(
               model_path=Path(self.config['detection']['model_path']),
               device=self.config['device'],
               conf_threshold=self.config['detection']['conf_threshold']
           )
           
           # Initialize tracker
           self.tracker = ByteTracker(
               high_conf_thresh=self.config['tracking']['high_conf_thresh'],
               low_conf_thresh=self.config['tracking']['low_conf_thresh'],
               track_buffer=self.config['tracking']['track_buffer'],
               match_thresh=self.config['tracking']['match_thresh']
           )
           
           # Initialize Re-ID (optional)
           if self.config.get('use_reid', True):
               self.reid_model = ResNet50ReID(
                   num_classes=751,  # Not used in inference
                   pretrained=False,
                   feature_dim=512
               )
               # Load trained weights
               checkpoint = torch.load(self.config['reid']['model_path'])
               self.reid_model.load_state_dict(checkpoint)
               self.reid_model.eval()
               self.reid_model.to(self.config['device'])
           
           # Feature gallery for Re-ID
           self.feature_gallery = {}
           
           logger.info("Inference pipeline initialized")
       
       def process_frame(self, frame: np.ndarray) -> List[Dict]:
           """
           Process single frame.
           
           Args:
               frame: Input frame (H, W, 3) BGR
               
           Returns:
               List of tracks with Re-ID info
           """
           # Step 1: Detection
           detections = self.detector.detect(frame)
           
           # Step 2: Tracking
           tracks = self.tracker.update(detections)
           
           # Step 3: Re-ID (if enabled)
           if hasattr(self, 'reid_model'):
               tracks = self._add_reid_features(frame, tracks)
           
           return tracks
       
       def _add_reid_features(self, frame: np.ndarray, tracks: List) -> List:
           """Extract Re-ID features for each track."""
           # Implementation
           pass
       
       def process_video(
           self,
           video_path: Path,
           output_path: Optional[Path] = None,
           visualize: bool = True
       ) -> Dict:
           """
           Process entire video.
           
           Args:
               video_path: Path to input video
               output_path: Path to save output video
               visualize: Draw bounding boxes and IDs
               
           Returns:
               Processing statistics and results
           """
           # Implementation
           pass
```

2. **Create Video Processor**
   
   Create `src/inference/video_processor.py`:
```python
   """
   Video input/output handling.
   """
   import cv2
   from pathlib import Path
   from typing import Generator
   import numpy as np
   
   class VideoProcessor:
       """Handle video reading and writing."""
       
       def __init__(self, video_path: Path):
           """Initialize video processor."""
           self.video_path = video_path
           self.cap = cv2.VideoCapture(str(video_path))
           
           # Get video properties
           self.fps = self.cap.get(cv2.CAP_PROP_FPS)
           self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
           self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
       def read_frames(self) -> Generator[np.ndarray, None, None]:
           """Generator that yields video frames."""
           # Implementation
           pass
       
       def write_video(self, frames: List[np.ndarray], output_path: Path):
           """Write frames to video file."""
           # Implementation
           pass
```

3. **Create Visualization Module**
   
   Create `src/visualization/bbox_drawer.py`:
```python
   """
   Bounding box drawing and visualization.
   """
   import cv2
   import numpy as np
   from typing import List, Dict, Tuple
   
   class BBoxDrawer:
       """Draw bounding boxes and track IDs on frames."""
       
       def __init__(self, thickness: int = 2, font_scale: float = 0.5):
           """Initialize drawer."""
           self.thickness = thickness
           self.font_scale = font_scale
           self.colors = self._generate_colors(100)
       
       def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
           """Generate distinct colors for track IDs."""
           # Implementation
           pass
       
       def draw_tracks(
           self,
           frame: np.ndarray,
           tracks: List[Dict]
       ) -> np.ndarray:
           """
           Draw bounding boxes and IDs on frame.
           
           Args:
               frame: Input frame
               tracks: List of tracks with (x1,y1,x2,y2,track_id)
               
           Returns:
               Annotated frame
           """
           # Implementation
           pass
```

4. **Create Inference Config**
   
   Create `configs/inference.yaml`:
```yaml
   device: "cuda"  # or "cpu"
   use_reid: true
   
   detection:
     model_path: "models/detection/pretrained/yolov8s.pt"
     conf_threshold: 0.5
     iou_threshold: 0.45
   
   tracking:
     high_conf_thresh: 0.6
     low_conf_thresh: 0.1
     track_buffer: 30
     match_thresh: 0.8
   
   reid:
     model_path: "models/reid/final/resnet50_market_final.pth"
     distance_threshold: 0.3
     update_gallery: true
   
   visualization:
     draw_boxes: true
     draw_ids: true
     draw_trails: false
     thickness: 2
     font_scale: 0.5
```

5. **Create Standalone Inference Script**
   
   Create `scripts/run_inference.py`:
```python
   """
   Run inference on video file.
   """
   import argparse
   from pathlib import Path
   from src.inference.pipeline import InferencePipeline
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--video", type=str, required=True, help="Path to input video")
       parser.add_argument("--output", type=str, help="Path to output video")
       parser.add_argument("--config", type=str, default="configs/inference.yaml")
       parser.add_argument("--no-viz", action="store_true", help="Don't visualize")
       args = parser.parse_args()
       
       # Initialize pipeline
       pipeline = InferencePipeline(Path(args.config))
       
       # Process video
       results = pipeline.process_video(
           video_path=Path(args.video),
           output_path=Path(args.output) if args.output else None,
           visualize=not args.no_viz
       )
       
       # Print statistics
       print(f"Processed {results['total_frames']} frames")
       print(f"Average FPS: {results['avg_fps']:.2f}")
       print(f"Total tracks: {results['total_tracks']}")
   
   if __name__ == "__main__":
       main()
```

6. **Test Integration**
```bash
   # Download a test video or use MOT17 sequence
   python scripts/run_inference.py \
       --video data/processed/mot17/images/val/MOT17-11-FRCNN/000001.jpg \
       --output outputs/results/test_output.mp4 \
       --config configs/inference.yaml
```

**Deliverable:** End-to-end inference pipeline

**Success Criteria:**
- Pipeline processes video without errors
- Output video has bounding boxes and track IDs
- Re-ID features are extracted and used for matching
- Processing achieves >25 FPS

**Time Estimate:** 8-10 hours

---

### Day 17-18: Evaluation

**Tasks:**

1. **Create MOT Metrics**
   
   Create `src/evaluation/mot_metrics.py`:
```python
   """
   MOT evaluation metrics: MOTA, IDF1, MT, ML, etc.
   Uses py-motmetrics library.
   """
   import motmetrics as mm
   import pandas as pd
   from pathlib import Path
   from typing import Dict
   
   class MOTEvaluator:
       """Evaluate tracking performance on MOT benchmarks."""
       
       def __init__(self):
           """Initialize evaluator."""
           self.accumulator = mm.MOTAccumulator(auto_id=True)
       
       def update(self, gt_boxes: List, pred_boxes: List):
           """Update with ground truth and predictions for one frame."""
           # Implementation
           pass
       
       def compute_metrics(self) -> Dict[str, float]:
           """Compute final metrics."""
           # Implementation
           # Returns: MOTA, IDF1, MT, ML, FP, FN, ID switches, etc.
           pass
```

2. **Create Re-ID Metrics**
   
   Create `src/evaluation/reid_metrics.py`:
```python
   """
   Re-ID evaluation metrics: Rank-1, Rank-5, mAP.
   """
   import torch
   import numpy as np
   from typing import Dict, Tuple
   
   class ReIDEvaluator:
       """Evaluate Re-ID performance on Market-1501."""
       
       def __init__(self, distance_metric: str = "cosine"):
           """Initialize evaluator."""
           self.distance_metric = distance_metric
       
       def compute_distance_matrix(
           self,
           query_features: torch.Tensor,
           gallery_features: torch.Tensor
       ) -> np.ndarray:
           """Compute pairwise distances."""
           # Implementation
           pass
       
       def evaluate(
           self,
           query_features: torch.Tensor,
           query_labels: np.ndarray,
           gallery_features: torch.Tensor,
           gallery_labels: np.ndarray
       ) -> Dict[str, float]:
           """
           Compute Rank-1, Rank-5, Rank-10, mAP.
           
           Returns:
               Dictionary with evaluation metrics
           """
           # Implementation
           pass
```

3. **Create Evaluation Scripts**
   
   Create `scripts/evaluate_mot.py`:
```python
   """
   Evaluate tracking on MOT17 validation set.
   """
   import argparse
   from pathlib import Path
   from src.inference.pipeline import InferencePipeline
   from src.evaluation.mot_metrics import MOTEvaluator
   from src.utils.config import load_config
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", type=str, default="configs/inference.yaml")
       parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
       args = parser.parse_args()
       
       # Initialize pipeline and evaluator
       pipeline = InferencePipeline(Path(args.config))
       evaluator = MOTEvaluator()
       
       # Process each sequence in validation set
       # Compare predictions to ground truth
       # Accumulate metrics
       
       # Compute final metrics
       metrics = evaluator.compute_metrics()
       
       # Print results
       print(f"MOTA: {metrics['mota']:.2%}")
       print(f"IDF1: {metrics['idf1']:.2%}")
       print(f"MT: {metrics['mt']:.2%}")
       print(f"ML: {metrics['ml']:.2%}")
       print(f"ID Switches: {metrics['num_switches']}")
   
   if __name__ == "__main__":
       main()
```
   
   Create `scripts/evaluate_reid.py`:
```python
   """
   Evaluate Re-ID on Market-1501.
   """
   import argparse
   import torch
   from pathlib import Path
   from src.reid.resnet_reid import ResNet50ReID
   from src.data.market_dataset import Market1501Dataset
   from src.evaluation.reid_metrics import ReIDEvaluator
   from src.utils.config import load_config
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", type=str, default="configs/reid.yaml")
       args = parser.parse_args()
       
       config = load_config(args.config)
       
       # Load model
       model = ResNet50ReID(num_classes=751, feature_dim=512)
       checkpoint = torch.load("models/reid/final/resnet50_market_final.pth")
       model.load_state_dict(checkpoint)
       model.eval()
       
       # Load query and gallery datasets
       # Extract features for all images
       # Compute metrics
       
       evaluator = ReIDEvaluator()
       metrics = evaluator.evaluate(query_features, query_labels, gallery_features, gallery_labels)
       
       # Print results
       print(f"Rank-1: {metrics['rank1']:.2%}")
       print(f"Rank-5: {metrics['rank5']:.2%}")
       print(f"mAP: {metrics['map']:.2%}")
   
   if __name__ == "__main__":
       main()
```

4. **Run Evaluations**
```bash
   # Evaluate tracking
   python scripts/evaluate_mot.py --config configs/inference.yaml --split val
   
   # Expected results:
   # MOTA: 70-75%
   # IDF1: 72-76%
   # ID Switches: 200-300
   
   # Evaluate Re-ID
   python scripts/evaluate_reid.py --config configs/reid.yaml
   
   # Expected results:
   # Rank-1: 85-88%
   # Rank-5: 95-97%
   # mAP: 70-75%
```

5. **Log Results to MLflow**
```python
   # In evaluation scripts, log final metrics
   import mlflow
   
   mlflow.set_experiment("final_evaluation")
   with mlflow.start_run():
       mlflow.log_metrics(metrics)
       mlflow.log_artifact("evaluation_report.txt")
```

**Deliverable:** Complete evaluation results

**Success Criteria:**
- MOT17 evaluation: MOTA >70%, IDF1 >72%
- Market-1501 evaluation: Rank-1 >85%, mAP >70%
- Results logged to MLflow
- Evaluation scripts run without errors

**Time Estimate:** 6-8 hours

---

### Day 19-21: Optimization & Demo Creation

**Tasks:**

1. **Performance Profiling**
```python
   # Create profiling script
   # scripts/profile_inference.py
   import time
   import torch
   from src.inference.pipeline import InferencePipeline
   
   # Profile each component
   # Identify bottlenecks
   # Report timing breakdown
```

2. **ONNX Export (Optional)**
   
   Create `scripts/export_models.py`:
```python
   """
   Export models to ONNX for faster inference.
   """
   import torch
   from pathlib import Path
   from src.detection.yolo_detector import YOLODetector
   from src.reid.resnet_reid import ResNet50ReID
   
   def export_yolo_onnx():
       """Export YOLOv8 to ONNX."""
       # YOLOv8 has built-in export
       from ultralytics import YOLO
       model = YOLO("models/detection/pretrained/yolov8s.pt")
       model.export(format="onnx", opset=12)
   
   def export_reid_onnx():
       """Export Re-ID model to ONNX."""
       model = ResNet50ReID(num_classes=751, feature_dim=512)
       checkpoint = torch.load("models/reid/final/resnet50_market_final.pth")
       model.load_state_dict(checkpoint)
       model.eval()
       
       dummy_input = torch.randn(1, 3, 256, 128)
       torch.onnx.export(
           model,
           dummy_input,
           "models/exported/resnet50_reid.onnx",
           opset_version=12,
           input_names=["input"],
           output_names=["features"]
       )
   
   if __name__ == "__main__":
       export_yolo_onnx()
       export_reid_onnx()
```

3. **Create Demo Videos**
```bash
   # Select best MOT17 sequences for demo
   # Process with visualization enabled
   python scripts/run_inference.py \
       --video data/processed/mot17/images/val/MOT17-11-FRCNN \
       --output outputs/results/demo_mot17_11.mp4 \
       --config configs/inference.yaml
   
   # Create montage of different scenarios
   # - Sparse crowds (MOT17-05)
   # - Dense crowds (MOT17-11)
   # - Occlusions (MOT17-13)
```

4. **Create Visualization Plots**
   
   Create `scripts/visualize_results.py`:
```python
   """
   Create visualization plots for results.
   """
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path
   
   # Plot training curves
   # Plot evaluation metrics comparison
   # Plot tracking statistics (track lengths, ID switches over time)
   # Save to outputs/visualizations/
```

**Deliverable:** Optimized system with demo videos

**Success Criteria:**
- Inference achieves >30 FPS on RTX 3060
- Demo videos showcase clean tracking
- Visualization plots generated
- ONNX models exported (optional)

**Time Estimate:** 8-10 hours

---

### Week 3 Deliverables Summary

**Completed:**
- End-to-end inference pipeline integrated
- Comprehensive evaluation on both benchmarks
- Performance optimization
- Demo videos and visualizations

**Success Metrics:**
- MOT17: MOTA 70-75%, IDF1 72-76%
- Market-1501: Rank-1 85-88%, mAP 70-75%
- Inference: 30-50 FPS
- Clean demo videos

**Ready for Week 4:** Deployment and documentation

---

## Week 4: Deployment + Demo + Documentation

### Objectives
- Deploy API with Docker
- Create web demo interface
- Complete all documentation
- Prepare portfolio presentation
- Final testing and polish

### Day 22-23: API Development

**Tasks:**

1. **Implement FastAPI Routes**
   
   Create `api/routes/inference.py`:
```python
   """
   Inference API endpoints.
   """
   from fastapi import APIRouter, UploadFile, File, HTTPException
   from pathlib import Path
   import shutil
   
   from src.inference.pipeline import InferencePipeline
   from api.schemas.response import InferenceResponse
   
   router = APIRouter()
   
   # Initialize pipeline once at startup (loaded via app lifespan)
   pipeline = None
   
   @router.post("/inference/video", response_model=InferenceResponse)
   async def inference_video(
       video: UploadFile = File(...),
       visualize: bool = True
   ):
       """
       Run inference on uploaded video.
       
       Args:
           video: Video file to process
           visualize: Draw bounding boxes and track IDs
           
       Returns:
           Inference results and output video path
       """
       # Save uploaded video
       video_path = Path(f"outputs/uploads/{video.filename}")
       video_path.parent.mkdir(parents=True, exist_ok=True)
       
       with video_path.open("wb") as buffer:
           shutil.copyfileobj(video.file, buffer)
       
       # Process video
       try:
           results = pipeline.process_video(
               video_path=video_path,
               output_path=Path(f"outputs/results/{video.filename}"),
               visualize=visualize
           )
           
           return InferenceResponse(**results)
       
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
```
   
   Create `api/routes/health.py`:
```python
   """
   Health check endpoints.
   """
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.get("/health")
   async def health_check():
       """Health check endpoint."""
       return {"status": "healthy"}
```

2. **Create Pydantic Schemas**
   
   Create `api/schemas/response.py`:
```python
   """
   Response models for API.
   """
   from pydantic import BaseModel
   from typing import List, Dict
   
   class InferenceResponse(BaseModel):
       """Response for inference endpoint."""
       total_frames: int
       total_tracks: int
       avg_fps: float
       output_video_path: str
       statistics: Dict[str, float]
```

3. **Create Main FastAPI App**
   
   Create `api/main.py`:
```python
   """
   FastAPI application for MOT + Re-ID inference.
   """
   from fastapi import FastAPI
   from fastapi.middleware.cors import CORSMiddleware
   from fastapi.staticfiles import StaticFiles
   from contextlib import asynccontextmanager
   from pathlib import Path
   
   from api.routes import inference, health
   from src.inference.pipeline import InferencePipeline
   from src.utils.logger import get_logger
   
   logger = get_logger(__name__)
   
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       """Load models on startup, cleanup on shutdown."""
       # Startup: Load models
       logger.info("Loading models...")
       inference.pipeline = InferencePipeline(Path("configs/inference.yaml"))
       logger.info("Models loaded successfully")
       
       yield
       
       # Shutdown: Cleanup
       logger.info("Shutting down...")
   
   # Initialize FastAPI app
   app = FastAPI(
       title="MOT + Re-ID API",
       description="Multi-Object Tracking with Person Re-Identification",
       version="1.0.0",
       lifespan=lifespan
   )
   
   # CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   # Mount static files (for serving output videos)
   app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
   
   # Include routers
   app.include_router(health.router, tags=["health"])
   app.include_router(inference.router, prefix="/api", tags=["inference"])
   
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)
```

4. **Test API Locally**
```bash
   # Run API server
   python api/main.py
   
   # Test health endpoint
   curl http://localhost:8000/health
   
   # Test inference endpoint (upload video via Swagger UI)
   # Open http://localhost:8000/docs
```

**Deliverable:** Working FastAPI server

**Success Criteria:**
- API starts without errors
- Health endpoint returns 200
- Inference endpoint processes videos
- Swagger docs accessible at /docs

**Time Estimate:** 6-8 hours

---

### Day 24-25: Docker Deployment

**Tasks:**

1. **Create Dockerfile**
   
   Create `deployment/docker/Dockerfile`:
```dockerfile
   # Multi-stage build for optimized image
   
   # Stage 1: Base image with system dependencies
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       python3.9 \
       python3-pip \
       ffmpeg \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1 \
       && rm -rf /var/lib/apt/lists/*
   
   # Stage 2: Python dependencies
   FROM base AS python-deps
   
   WORKDIR /app
   
   # Copy requirements and install
   COPY requirements.txt .
   RUN pip3 install --no-cache-dir -r requirements.txt
   
   # Stage 3: Application
   FROM python-deps AS app
   
   # Copy source code
   COPY src/ /app/src/
   COPY api/ /app/api/
   COPY configs/ /app/configs/
   COPY setup.py /app/
   
   # Install package
   RUN pip3 install -e .
   
   # Create directories
   RUN mkdir -p /app/outputs /app/models /app/data
   
   # Expose port
   EXPOSE 8000
   
   # Set entrypoint
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Create Docker Compose**
   
   Create `deployment/docker/docker-compose.yml`:
```yaml
   version: '3.8'
   
   services:
     api:
       build:
         context: ../..
         dockerfile: deployment/docker/Dockerfile
       runtime: nvidia
       ports:
         - "8000:8000"
       volumes:
         - ../../data:/app/data:ro
         - ../../models:/app/models:ro
         - ../../outputs:/app/outputs:rw
         - ../../configs:/app/configs:ro
       environment:
         - CUDA_VISIBLE_DEVICES=0
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
     
     mlflow:
       image: ghcr.io/mlflow/mlflow:latest
       ports:
         - "5000:5000"
       volumes:
         - ../../mlruns:/mlruns
       command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlruns
     
     # Optional: Grafana for monitoring
     grafana:
       image: grafana/grafana:latest
       ports:
         - "3000:3000"
       volumes:
         - grafana-storage:/var/lib/grafana
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin
   
   volumes:
     grafana-storage:
```

3. **Create .dockerignore**
   
   Create `deployment/docker/.dockerignore`:
```
   __pycache__
   *.pyc
   *.pyo
   .git
   .gitignore
   .dvc
   venv
   .vscode
   .idea
   *.md
   data/
   models/
   outputs/
   mlruns/
   notebooks/
   tests/
```

4. **Build and Run Docker**
```bash
   cd deployment/docker
   
   # Build image
   docker-compose build
   
   # Run services
   docker-compose up -d
   
   # Check logs
   docker-compose logs -f api
   
   # Test API
   curl http://localhost:8000/health
```

5. **Test GPU Access in Container**
```bash
   # Exec into container
   docker-compose exec api bash
   
   # Check GPU
   python3 -c "import torch; print(torch.cuda.is_available())"
   # Should print: True
```

**Deliverable:** Dockerized application

**Success Criteria:**
- Docker image builds successfully
- Container starts and API responds
- GPU accessible in container
- All services (API, MLflow) running

**Time Estimate:** 4-6 hours

---

### Day 26-27: Web Demo Interface

**Tasks:**

1. **Create Gradio Interface**
   
   Create `demo/gradio_app.py`:
```python
   """
   Gradio web interface for MOT + Re-ID demo.
   """
   import gradio as gr
   from pathlib import Path
   import tempfile
   
   from src.inference.pipeline import InferencePipeline
   
   # Initialize pipeline
   pipeline = InferencePipeline(Path("configs/inference.yaml"))
   
   def process_video(video_file, visualize):
       """Process uploaded video and return result."""
       # Save to temp file
       temp_path = Path(tempfile.mktemp(suffix=".mp4"))
       
       # Process video
       results = pipeline.process_video(
           video_path=Path(video_file.name),
           output_path=temp_path,
           visualize=visualize
       )
       
       # Return output video and statistics
       stats_text = f"""
       Total Frames: {results['total_frames']}
       Total Tracks: {results['total_tracks']}
       Average FPS: {results['avg_fps']:.2f}
       """
       
       return str(temp_path), stats_text
   
   # Create Gradio interface
   demo = gr.Interface(
       fn=process_video,
       inputs=[
           gr.Video(label="Upload Video"),
           gr.Checkbox(label="Visualize Tracking", value=True)
       ],
       outputs=[
           gr.Video(label="Output Video"),
           gr.Textbox(label="Statistics", lines=5)
       ],
       title="Multi-Object Tracking + Re-Identification",
       description="Upload a video to track people across frames with persistent IDs.",
       examples=[
           ["examples/mall_demo.mp4", True],
           ["examples/street_demo.mp4", True]
       ]
   )
   
   if __name__ == "__main__":
       demo.launch(server_name="0.0.0.0", server_port=7860)
```

2. **Test Gradio Locally**
```bash
   python demo/gradio_app.py
   
   # Access at http://localhost:7860
```

3. **Deploy to Hugging Face Spaces (Optional)**
```bash
   # Create Hugging Face Space
   # Upload code and models
   # Configure GPU (if available on free tier)
```

**Deliverable:** Interactive web demo

**Success Criteria:**
- Gradio interface loads
- Can upload video and get results
- Output video displays with tracking
- Statistics shown correctly

**Time Estimate:** 4-6 hours

---

### Day 28: Documentation Completion

**Tasks:**

1. **Complete README.md**
   
   Create comprehensive `README.md`:
```markdown
   # Multi-Object Tracking + Person Re-Identification System
   
   Production-ready MOT system with Re-ID for identity preservation.
   
   ## Features
   - Real-time tracking (30-50 FPS on RTX 3060)
   - Person re-identification across occlusions
   - RESTful API for deployment
   - Docker containerization
   - Comprehensive evaluation metrics
   
   ## Quick Start
   (Installation, usage, examples)
   
   ## Architecture
   (Link to SYSTEM_ARCHITECTURE.md)
   
   ## Results
   - MOT17: MOTA 72%, IDF1 75%
   - Market-1501: Rank-1 86%, mAP 72%
   
   ## Demo
   (Link to demo video/Hugging Face Space)
```

2. **Write Remaining Documentation**
   
   Create `docs/TRAINING_GUIDE.md`:
```markdown
   # Training Guide
   
   Step-by-step instructions to:
   - Train YOLOv8 detector on custom data
   - Train Re-ID model on custom dataset
   - Fine-tune existing models
```
   
   Create `docs/DEPLOYMENT_GUIDE.md`:
```markdown
   # Deployment Guide
   
   How to deploy to:
   - Local Docker
   - Cloud platforms (AWS, GCP, Render)
   - Kubernetes (optional)
```
   
   Create `docs/TROUBLESHOOTING.md`:
```markdown
   # Troubleshooting
   
   Common issues and solutions:
   - GPU not detected
   - Out of memory errors
   - Slow inference
   - Model loading errors
```

3. **Create CHANGELOG.md**
```markdown
   # Changelog
   
   ## [1.0.0] - 2026-01-22
   
   ### Added
   - Initial release
   - YOLOv8 detection
   - ByteTrack tracking
   - ResNet50 Re-ID
   - FastAPI deployment
   - Docker containerization
```

**Deliverable:** Complete documentation

**Success Criteria:**
- README is comprehensive and professional
- All documentation files complete
- Installation instructions tested
- API documentation auto-generated

**Time Estimate:** 4-6 hours

---

### Week 4 Deliverables Summary

**Completed:**
- FastAPI deployed with Docker
- Web demo interface (Gradio)
- Complete documentation
- Portfolio-ready presentation

**Success Metrics:**
- API accessible and functional
- Docker deployment works
- Documentation comprehensive
- Demo showcases system capabilities

**Project Complete!**

---

## Post-Completion Checklist

### Code Quality
- [ ] All code follows PEP8 (run `black` and `flake8`)
- [ ] Type hints on all functions (run `mypy`)
- [ ] Docstrings on all classes and functions
- [ ] Unit tests pass (`pytest tests/`)
- [ ] No hardcoded paths or values

### Documentation
- [ ] README.md complete with setup instructions
- [ ] All markdown docs in `docs/` complete
- [ ] API documentation (Swagger) accessible
- [ ] Comments explain complex logic

### Git & Version Control
- [ ] Clean commit history
- [ ] `.gitignore` properly configured
- [ ] DVC tracks data and models
- [ ] No sensitive data in git

### Deployment
- [ ] Docker image builds successfully
- [ ] docker-compose.yml tested locally
- [ ] GPU support configured
- [ ] API responds to requests

### Demo & Results
- [ ] Demo videos created
- [ ] Evaluation results documented
- [ ] Performance metrics logged
- [ ] Portfolio-ready presentation

### Portfolio Presentation
- [ ] GitHub repository public
- [ ] README showcases results
- [ ] Demo link included (if deployed)
- [ ] Architecture diagrams clear
- [ ] Interview talking points documented

---

## Contingency Plans

### If Behind Schedule

**Week 1 Behind:**
- Skip detection fine-tuning (use pre-trained YOLOv8)
- Reduces Week 1 from 7 days to 5 days

**Week 2 Behind:**
- Train Re-ID for fewer epochs (50 instead of 100)
- Skip ViT Re-ID bonus feature
- Reduces Week 2 from 7 days to 5 days

**Week 3 Behind:**
- Skip ONNX export
- Create demo from single sequence (not multiple)
- Reduces Week 3 from 7 days to 5 days

**Week 4 Behind:**
- Skip Gradio demo (API only)
- Skip Hugging Face deployment
- Reduces Week 4 from 7 days to 4 days

### If Ahead of Schedule

**Extra Time Available:**
- Implement ViT Re-ID as alternative model
- Add TensorRT optimization
- Deploy to Hugging Face Spaces
- Create comprehensive Jupyter notebooks
- Add action recognition module
- Implement multi-camera fusion

---

## Final Timeline Summary

| Week | Focus | Key Deliverables | Time |
|------|-------|------------------|------|
| **Week 1** | Environment + Data + Detection | Working detection module, preprocessed data | 20-25 hours |
| **Week 2** | Tracking + Re-ID | ByteTrack implementation, trained Re-ID model | 20-25 hours |
| **Week 3** | Integration + Evaluation | End-to-end pipeline, benchmark results | 20-25 hours |
| **Week 4** | Deployment + Demo + Docs | Docker deployment, web demo, complete docs | 15-20 hours |
| **Total** | 4 weeks part-time | Production-ready MOT + Re-ID system | 75-95 hours |

**OR**

| Week | Focus | Time |
|------|-------|------|
| **Week 1** | Weeks 1+2 from above | 40-50 hours |
| **Week 2** | Weeks 3+4 from above | 35-45 hours |
| **Total** | 2 weeks full-time | 75-95 hours |

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-22  
**Owner:** [Your Name]