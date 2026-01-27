# Directory Structure - Multi-Object Tracking & Re-Identification System

## Document Purpose
This document defines the complete project directory structure. This structure is **IMMUTABLE** once finalized. All code, imports, configurations, and documentation will reference this structure. Changes to this structure mid-project will break the system.

---

## Design Principles

**1. Separation of Concerns**
- Data, models, code, configs, outputs are clearly separated
- Each directory has a single, well-defined responsibility

**2. Scalability**
- Structure supports growth (more models, more datasets, more features)
- Modular organization allows independent component development

**3. Industry Standards**
- Follows Python package conventions
- Compatible with Docker, DVC, MLflow, pytest
- Mimics production ML system organization

**4. Self-Documenting**
- Directory names are clear and unambiguous
- Structure immediately reveals system architecture
- README files in key directories explain purpose

**5. Portfolio-Ready**
- Clean, professional organization
- Easy for reviewers to navigate
- Demonstrates senior-level software engineering

---

## Complete Directory Tree

```
mot-reid-system/
│
├── data/                                   # All datasets (tracked by DVC, NOT git)
│   ├── raw/                                # Original downloaded datasets (read-only)
│   │   ├── MOT17/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── Market-1501-v15.09.15/
│   │       ├── bounding_box_train/
│   │       ├── bounding_box_test/
│   │       ├── query/
│   │       └── gt_bbox/
│   │
│   ├── processed/                          # Preprocessed datasets (DVC tracked)
│   │   ├── mot17/
│   │   │   ├── annotations/
│   │   │   │   ├── train.json
│   │   │   │   ├── val.json
│   │   │   │   └── metadata.json
│   │   │   ├── images/
│   │   │   │   ├── train/
│   │   │   │   │   ├── MOT17-02-FRCNN/
│   │   │   │   │   ├── MOT17-04-FRCNN/
│   │   │   │   │   ├── MOT17-05-FRCNN/
│   │   │   │   │   ├── MOT17-09-FRCNN/
│   │   │   │   │   └── MOT17-10-FRCNN/
│   │   │   │   └── val/
│   │   │   │       ├── MOT17-11-FRCNN/
│   │   │   │       └── MOT17-13-FRCNN/
│   │   │   └── README.txt
│   │   │
│   │   └── market1501/
│   │       ├── train/                      # Organized by person ID
│   │       │   ├── 0002/
│   │       │   ├── 0003/
│   │       │   └── ...
│   │       ├── val/
│   │       │   └── ...
│   │       ├── query/
│   │       ├── gallery/
│   │       ├── train_labels.csv
│   │       ├── val_labels.csv
│   │       ├── camera_mappings.json
│   │       └── metadata.json
│   │
│   └── external/                           # User-uploaded test videos/images
│       ├── test_videos/
│       │   ├── mall_demo.mp4
│       │   └── street_demo.mp4
│       └── test_images/
│
├── models/                                 # Trained models and checkpoints
│   ├── detection/                          # Object detection models
│   │   ├── pretrained/
│   │   │   └── yolov8s.pt                  # Pre-trained YOLOv8
│   │   ├── checkpoints/
│   │   │   ├── yolov8_mot17_epoch_10.pt
│   │   │   ├── yolov8_mot17_epoch_20.pt
│   │   │   └── yolov8_mot17_best.pt
│   │   └── final/
│   │       └── yolov8_mot17_final.pt       # Final production model
│   │
│   ├── reid/                               # Re-ID models
│   │   ├── pretrained/
│   │   │   └── resnet50_imagenet.pth       # ImageNet pre-trained ResNet50
│   │   ├── checkpoints/
│   │   │   ├── resnet50_market_epoch_50.pth
│   │   │   ├── resnet50_market_epoch_100.pth
│   │   │   └── resnet50_market_best.pth
│   │   └── final/
│   │       └── resnet50_market_final.pth   # Final production model
│   │
│   ├── tracking/                           # Tracking-specific saved states
│   │   └── bytetrack_configs/
│   │       ├── mot17_tuned.yaml
│   │       └── mot17_best.yaml
│   │
│   └── exported/                           # Optimized models for deployment
│       ├── yolov8_onnx.onnx
│       ├── resnet50_reid_onnx.onnx
│       └── tensorrt/                       # TensorRT optimized (optional)
│           └── yolov8_trt.engine
│
├── src/                                    # Source code (Python package)
│   ├── __init__.py
│   │
│   ├── detection/                          # Object detection module
│   │   ├── __init__.py
│   │   ├── yolo_detector.py                # YOLOv8 wrapper
│   │   ├── detector_base.py                # Abstract detector interface
│   │   └── utils.py                        # Detection utilities (NMS, etc.)
│   │
│   ├── tracking/                           # Multi-object tracking module
│   │   ├── __init__.py
│   │   ├── bytetrack.py                    # ByteTrack implementation
│   │   ├── kalman_filter.py                # Kalman filter for motion prediction
│   │   ├── matching.py                     # IoU/appearance matching logic
│   │   ├── tracker_base.py                 # Abstract tracker interface
│   │   └── utils.py                        # Tracking utilities
│   │
│   ├── reid/                               # Person Re-Identification module
│   │   ├── __init__.py
│   │   ├── resnet_reid.py                  # ResNet50 Re-ID model
│   │   ├── feature_extractor.py            # Feature extraction wrapper
│   │   ├── metric_learning.py              # Triplet loss, distance metrics
│   │   ├── reid_base.py                    # Abstract Re-ID interface
│   │   └── utils.py                        # Re-ID utilities
│   │
│   ├── data/                               # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── mot_dataset.py                  # MOT17 PyTorch Dataset
│   │   ├── market_dataset.py               # Market-1501 PyTorch Dataset
│   │   ├── transforms.py                   # Data augmentation transforms
│   │   ├── preprocessing.py                # Preprocessing utilities
│   │   └── utils.py                        # Data utilities
│   │
│   ├── training/                           # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_detection.py              # YOLOv8 training script
│   │   ├── train_reid.py                   # Re-ID training script
│   │   ├── trainer_base.py                 # Abstract trainer class
│   │   ├── callbacks.py                    # Training callbacks (checkpoint, early stop)
│   │   └── utils.py                        # Training utilities
│   │
│   ├── evaluation/                         # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── mot_metrics.py                  # MOTA, IDF1, MT, ML calculation
│   │   ├── reid_metrics.py                 # Rank-1, mAP calculation
│   │   ├── evaluator.py                    # Main evaluation orchestrator
│   │   └── utils.py                        # Evaluation utilities
│   │
│   ├── inference/                          # Inference pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                     # End-to-end inference pipeline
│   │   ├── video_processor.py              # Video input/output handling
│   │   ├── batch_processor.py              # Batch inference for efficiency
│   │   └── utils.py                        # Inference utilities
│   │
│   ├── visualization/                      # Visualization and rendering
│   │   ├── __init__.py
│   │   ├── bbox_drawer.py                  # Draw bounding boxes with IDs
│   │   ├── track_visualizer.py             # Visualize tracks over time
│   │   ├── metrics_plotter.py              # Plot training/eval metrics
│   │   └── utils.py                        # Visualization utilities
│   │
│   └── utils/                              # Global utilities
│       ├── __init__.py
│       ├── logger.py                       # Logging configuration
│       ├── config.py                       # Configuration management
│       ├── checkpoint.py                   # Model checkpoint utilities
│       ├── metrics_tracker.py              # MLflow integration
│       └── general.py                      # General helper functions
│
├── api/                                    # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                             # FastAPI app entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py                       # Health check endpoint
│   │   ├── inference.py                    # Inference endpoints
│   │   ├── upload.py                       # File upload endpoints
│   │   └── metrics.py                      # Metrics endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py                      # Request models (Pydantic)
│   │   └── response.py                     # Response models (Pydantic)
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                         # Authentication (if needed)
│   │   └── logging.py                      # Request/response logging
│   └── utils.py                            # API utilities
│
├── tests/                                  # All tests (pytest)
│   ├── __init__.py
│   ├── conftest.py                         # Pytest fixtures and configuration
│   │
│   ├── unit/                               # Unit tests (isolated components)
│   │   ├── __init__.py
│   │   ├── test_detection.py
│   │   ├── test_tracking.py
│   │   ├── test_reid.py
│   │   ├── test_data_loading.py
│   │   └── test_utils.py
│   │
│   ├── integration/                        # Integration tests (multiple components)
│   │   ├── __init__.py
│   │   ├── test_inference_pipeline.py
│   │   ├── test_training_pipeline.py
│   │   └── test_api.py
│   │
│   └── fixtures/                           # Test data and fixtures
│       ├── sample_images/
│       ├── sample_videos/
│       └── mock_data.py
│
├── deployment/                             # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile                      # Main Docker image
│   │   ├── Dockerfile.gpu                  # GPU-enabled Docker image
│   │   ├── Dockerfile.api                  # API-only Docker image
│   │   ├── docker-compose.yml              # Multi-service orchestration
│   │   ├── docker-compose.dev.yml          # Development override
│   │   └── .dockerignore
│   │
│   ├── kubernetes/                         # Kubernetes manifests (optional)
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   └── configmap.yaml
│   │
│   └── cloud/                              # Cloud deployment configs
│       ├── aws/                            # AWS-specific configs (optional)
│       ├── gcp/                            # GCP-specific configs (optional)
│       └── render.yaml                     # Render.com config
│
├── configs/                                # Configuration files
│   ├── default.yaml                        # Default configuration
│   ├── detection.yaml                      # Detection model config
│   ├── tracking.yaml                       # Tracking algorithm config
│   ├── reid.yaml                           # Re-ID model config
│   ├── training.yaml                       # Training hyperparameters
│   ├── inference.yaml                      # Inference settings
│   └── deployment.yaml                     # Deployment settings
│
├── scripts/                                # Standalone scripts
│   ├── download_datasets.sh                # Download MOT17 + Market-1501
│   ├── preprocess_all_datasets.py          # Run all preprocessing
│   ├── preprocess_mot17.py                 # MOT17-specific preprocessing
│   ├── preprocess_market1501.py            # Market-1501 preprocessing
│   ├── train_detection.py                  # Standalone detection training
│   ├── train_reid.py                       # Standalone Re-ID training
│   ├── evaluate_mot.py                     # Evaluate on MOT17
│   ├── evaluate_reid.py                    # Evaluate on Market-1501
│   ├── export_models.py                    # Export to ONNX/TensorRT
│   ├── run_inference.py                    # Run inference on video
│   ├── visualize_results.py                # Visualize tracking results
│   └── setup_environment.sh                # Environment setup script
│
├── notebooks/                              # Jupyter notebooks (exploratory)
│   ├── 01_data_exploration.ipynb           # Dataset statistics and visualization
│   ├── 02_model_debugging.ipynb            # Debug model outputs
│   ├── 03_hyperparameter_tuning.ipynb      # Hyperparameter experiments
│   ├── 04_results_analysis.ipynb           # Analyze evaluation results
│   └── 05_demo_creation.ipynb              # Create demo visualizations
│
├── docs/                                   # Documentation
│   ├── TECH_STACK.md                       # Technology stack (COMPLETED)
│   ├── DATA_SOURCES.md                     # Dataset information (COMPLETED)
│   ├── DIRECTORY_STRUCTURE.md              # This file
│   ├── SYSTEM_ARCHITECTURE.md              # High-level system design (TODO)
│   ├── SYSTEM_DESIGN_DECISIONS.md          # Design rationale (TODO)
│   ├── IMPLEMENTATION_ROADMAP.md           # Phased plan (TODO)
│   ├── SUCCESS_CRITERIA.md                 # Metrics and goals (TODO)
│   ├── API_DOCUMENTATION.md                # API endpoint docs (auto-gen)
│   ├── TRAINING_GUIDE.md                   # How to train models
│   ├── DEPLOYMENT_GUIDE.md                 # How to deploy
│   └── TROUBLESHOOTING.md                  # Common issues and solutions
│
├── outputs/                                # All outputs (NOT in git)
│   ├── experiments/                        # MLflow experiment artifacts
│   │   ├── detection_experiments/
│   │   └── reid_experiments/
│   │
│   ├── results/                            # Inference results
│   │   ├── videos/                         # Output videos with tracking
│   │   ├── images/                         # Output images with annotations
│   │   └── metrics/                        # Saved metric reports
│   │
│   ├── visualizations/                     # Plots and charts
│   │   ├── training_curves/
│   │   ├── evaluation_plots/
│   │   └── demo_outputs/
│   │
│   └── logs/                               # Application logs
│       ├── training/
│       ├── inference/
│       └── api/
│
├── mlruns/                                 # MLflow tracking directory (NOT in git)
│   └── ...                                 # Auto-generated by MLflow
│
├── .dvc/                                   # DVC configuration (auto-generated)
│   └── ...
│
├── .github/                                # GitHub-specific files
│   ├── workflows/
│   │   ├── ci.yml                          # Continuous Integration
│   │   └── docker-build.yml                # Docker image build
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── .gitignore                              # Git ignore rules
├── .dockerignore                           # Docker ignore rules
├── .dvcignore                              # DVC ignore rules
├── .env.example                            # Example environment variables
├── .env                                    # Actual environment variables (NOT in git)
│
├── requirements.txt                        # Production dependencies (pinned)
├── requirements-dev.txt                    # Development dependencies
├── requirements-api.txt                    # API-only dependencies (minimal)
│
├── setup.py                                # Package setup (for pip install -e .)
├── pyproject.toml                          # Modern Python project config
│
├── pytest.ini                              # Pytest configuration
├── .flake8                                 # Flake8 linting configuration
├── .black                                  # Black formatting configuration
├── mypy.ini                                # Mypy type checking configuration
│
├── dvc.yaml                                # DVC pipeline definition
├── dvc.lock                                # DVC pipeline lock file
│
├── README.md                               # Project README (portfolio-ready)
├── LICENSE                                 # Project license
└── CHANGELOG.md                            # Version history and changes
```

---

## Directory Descriptions

### Root Level

**`data/`**
- **Purpose:** All datasets (raw, processed, external test data)
- **DVC Tracked:** Yes (raw/ and processed/)
- **Git Tracked:** No (too large, tracked by DVC instead)
- **Size:** ~40-50GB total

**`models/`**
- **Purpose:** All model files (pre-trained, checkpoints, final, exported)
- **DVC Tracked:** Selected files (final models, best checkpoints)
- **Git Tracked:** No (too large)
- **Size:** ~15GB

**`src/`**
- **Purpose:** Main Python package (all source code)
- **Git Tracked:** Yes (this is the code)
- **Structure:** Modular, each subdirectory is a component
- **Import style:** `from src.detection.yolo_detector import YOLODetector`

**`api/`**
- **Purpose:** FastAPI REST API for model serving
- **Git Tracked:** Yes
- **Deployment:** Runs as separate service in Docker Compose

**`tests/`**
- **Purpose:** All tests (unit, integration, fixtures)
- **Git Tracked:** Yes
- **Run with:** `pytest tests/`

**`deployment/`**
- **Purpose:** Deployment configurations (Docker, K8s, cloud)
- **Git Tracked:** Yes
- **Usage:** `docker-compose up` for local deployment

**`configs/`**
- **Purpose:** YAML configuration files
- **Git Tracked:** Yes
- **Usage:** Loaded by `src/utils/config.py`

**`scripts/`**
- **Purpose:** Standalone executable scripts
- **Git Tracked:** Yes
- **Usage:** `python scripts/train_detection.py --config configs/detection.yaml`

**`notebooks/`**
- **Purpose:** Jupyter notebooks for exploration and analysis
- **Git Tracked:** Yes (but outputs cleared before commit)
- **Usage:** Experimentation, not production code

**`docs/`**
- **Purpose:** All documentation (markdown files)
- **Git Tracked:** Yes
- **Structure:** Seven key markdown files covering architecture, design, roadmap, etc.

**`outputs/`**
- **Purpose:** All outputs (experiments, results, visualizations, logs)
- **Git Tracked:** No (regenerable)
- **MLflow Tracked:** experiments/ subdirectory

**`mlruns/`**
- **Purpose:** MLflow tracking server database
- **Git Tracked:** No
- **Generated by:** MLflow automatically

---

## Key Design Decisions

### 1. Data Organization

**Three-tier data structure:**
```
data/raw/          → Original downloads (read-only, never modify)
data/processed/    → Preprocessed, ready for training (DVC tracked)
data/external/     → User test videos/images (for demo)
```

**Rationale:** Clear separation prevents accidental raw data corruption. DVC tracks both raw and processed to lock versions.

### 2. Model Organization

**Four-tier model structure:**
```
models/*/pretrained/   → Downloaded pre-trained models
models/*/checkpoints/  → Training checkpoints (every N epochs)
models/*/final/        → Final trained model (best checkpoint)
models/exported/       → ONNX/TensorRT optimized for deployment
```

**Rationale:** Clear progression from pre-training → training → finalization → deployment optimization.

### 3. Source Code Modularity

**src/ follows component-based architecture:**
- Each subdirectory is an independent module
- Modules communicate through well-defined interfaces (base classes)
- Enables parallel development and easy testing
- Mimics production ML system organization

**Module hierarchy:**
```
detection/   → Finds objects (YOLOv8)
tracking/    → Tracks objects across frames (ByteTrack)
reid/        → Extracts appearance features (ResNet50)
data/        → Loads and preprocesses datasets
training/    → Training pipelines
evaluation/  → Metrics and evaluation
inference/   → End-to-end inference
visualization/ → Rendering and plotting
utils/       → Shared utilities
```

### 4. API Separation

**api/ is separate from src/ because:**
- API is a deployment artifact, not core logic
- Can be deployed independently (microservice architecture)
- Has different dependencies (FastAPI, uvicorn)
- Imports from src/ as a library: `from src.inference.pipeline import InferencePipeline`

### 5. Configuration Management

**All configs in configs/ as YAML:**
- No hardcoded values in code
- Easy to experiment (change config, not code)
- Version controlled (track config changes)
- Environment-specific configs (dev, prod)

### 6. Testing Strategy

**Three-tier testing:**
```
tests/unit/        → Test individual functions/classes
tests/integration/ → Test component interactions
tests/fixtures/    → Shared test data
```

**Coverage goal:** >70% for src/, >90% for critical paths (inference pipeline)

### 7. Deployment Flexibility

**Multiple deployment targets:**
```
deployment/docker/       → Local development (Docker Compose)
deployment/kubernetes/   → Cloud production (K8s)
deployment/cloud/        → Platform-specific (Render, AWS, GCP)
```

**All use same Docker image, different orchestration.**

### 8. Output Organization

**outputs/ mirrors project phases:**
```
experiments/     → Training experiments (MLflow artifacts)
results/         → Inference outputs (videos, images, metrics)
visualizations/  → Plots and charts for analysis
logs/            → Application logs (structured logging)
```

**None tracked by git (too large, regenerable).**

---

## Import Path Convention

### Python Package Installation

Install project as editable package:
```bash
pip install -e .
```

This enables clean imports:
```python
# Instead of messy relative imports:
from ../../src/detection/yolo_detector import YOLODetector  # BAD

# Use clean absolute imports:
from src.detection.yolo_detector import YOLODetector  # GOOD
```

### Import Examples

**From scripts:**
```python
# scripts/train_detection.py
from src.detection.yolo_detector import YOLODetector
from src.data.mot_dataset import MOTDataset
from src.training.train_detection import DetectionTrainer
from src.utils.config import load_config
```

**From API:**
```python
# api/routes/inference.py
from src.inference.pipeline import InferencePipeline
from src.utils.logger import get_logger
```

**From tests:**
```python
# tests/unit/test_detection.py
from src.detection.yolo_detector import YOLODetector
from tests.fixtures.mock_data import create_mock_image
```

---

## File Naming Conventions

### Python Files

**Modules (lowercase with underscores):**
```
yolo_detector.py        # Good
YOLODetector.py         # Bad (not PEP8)
yolodetector.py         # Bad (hard to read)
```

**Classes (PascalCase):**
```python
class YOLODetector:      # Good
class Yolo_Detector:     # Bad
class yoloDetector:      # Bad
```

**Functions/variables (lowercase with underscores):**
```python
def load_checkpoint():   # Good
def loadCheckpoint():    # Bad
def LoadCheckpoint():    # Bad
```

### Configuration Files

**YAML files (lowercase, descriptive):**
```
detection.yaml           # Good
config.yaml             # Too generic
Detection.yaml          # Bad (not lowercase)
```

### Model Files

**Format: {model}_{dataset}_{descriptor}.{ext}**
```
yolov8_mot17_best.pt              # Good
resnet50_market_epoch_100.pth     # Good
model.pt                          # Bad (not descriptive)
```

### Script Files

**Action-oriented names:**
```
train_detection.py                # Good (verb + noun)
detection_train.py                # Less clear
trainer.py                        # Too generic
```

---

## Git and DVC Tracking Strategy

### Git Tracked (Code and Docs)
```
src/                    ✓ Track
api/                    ✓ Track
tests/                  ✓ Track
scripts/                ✓ Track
configs/                ✓ Track
docs/                   ✓ Track
deployment/             ✓ Track
requirements*.txt       ✓ Track
setup.py                ✓ Track
README.md               ✓ Track
.gitignore              ✓ Track
```

### DVC Tracked (Data and Models)
```
data/raw/               ✓ DVC track
data/processed/         ✓ DVC track
models/*/final/         ✓ DVC track (final models only)
models/exported/        ✓ DVC track
```

### Not Tracked (Generated/Temporary)
```
data/external/          ✗ User-specific test files
models/*/checkpoints/   ✗ Too many, too large (keep locally)
outputs/                ✗ Regenerable
mlruns/                 ✗ MLflow manages this
__pycache__/            ✗ Python cache
*.pyc                   ✗ Compiled Python
.env                    ✗ Secrets (use .env.example instead)
```

---

## Docker Volume Mounts

When running in Docker, mount these directories:

```yaml
# docker-compose.yml
volumes:
  - ./data:/app/data:ro                    # Read-only (don't modify in container)
  - ./models:/app/models:rw                # Read-write (save checkpoints)
  - ./outputs:/app/outputs:rw              # Read-write (save results)
  - ./configs:/app/configs:ro              # Read-only
  - ./mlruns:/app/mlruns:rw                # Read-write (MLflow)
```

**Rationale:**
- Code baked into Docker image (not mounted)
- Data and models mounted (too large for image)
- Outputs mounted (need to access from host)

---

## Environment-Specific Configurations

### Development
```
configs/default.yaml          # Base config
configs/detection.yaml        # Detection overrides
.env                          # Local secrets (GPU IDs, API keys)
```

### Production (Docker)
```
configs/deployment.yaml       # Production settings (batch size, workers)
deployment/docker/.env        # Production secrets (via Docker secrets)
```

### CI/CD (GitHub Actions)
```
configs/ci.yaml               # CI-specific settings (CPU-only, small batch)
.github/workflows/ci.yml      # Test pipeline
```

---

## README Files in Key Directories

Each major directory should have a README.txt explaining its purpose:

**data/raw/README.txt:**
```
This directory contains original downloaded datasets.
DO NOT MODIFY FILES IN THIS DIRECTORY.
Use scripts/preprocess_*.py to create processed versions.
```

**models/detection/README.txt:**
```
Detection model storage.
- pretrained/: Downloaded pre-trained models
- checkpoints/: Training checkpoints (auto-saved every N epochs)
- final/: Best model selected for production
```

**outputs/README.txt:**
```
All outputs generated by training and inference.
This directory is NOT tracked by git (regenerable).
Clean periodically to save disk space.
```

---

## Scalability Considerations

### Adding New Datasets
```
1. Download to: data/raw/{new_dataset}/
2. Create preprocessing script: scripts/preprocess_{new_dataset}.py
3. Output to: data/processed/{new_dataset}/
4. Add DVC tracking: dvc add data/raw/{new_dataset}
5. Update DATA_SOURCES.md
```

### Adding New Models
```
1. Create module: src/{new_component}/{new_model}.py
2. Inherit from base class: src/{new_component}/{component}_base.py
3. Add config: configs/{new_model}.yaml
4. Add tests: tests/unit/test_{new_model}.py
```

### Adding New API Endpoints
```
1. Create route: api/routes/{new_endpoint}.py
2. Define schemas: api/schemas/request.py, response.py
3. Add tests: tests/integration/test_api.py
4. Update API_DOCUMENTATION.md
```

---

## Verification Checklist

After setting up directory structure, verify:

**1. Data directories exist:**
```bash
ls data/raw data/processed data/external
```

**2. Model directories exist:**
```bash
ls models/detection models/reid models/tracking models/exported
```

**3. Source code structure:**
```bash
ls src/detection src/tracking src/reid src/data src/training
```

**4. Configuration files:**
```bash
ls configs/*.yaml
```

**5. Python package installable:**
```bash
pip install -e .
python -c "from src.detection.yolo_detector import YOLODetector"
```

**6. Tests runnable:**
```bash
pytest tests/ --collect-only  # Should discover all tests
```

**7. Git ignores correct files:**
```bash
git status  # Should NOT show data/, models/, outputs/, mlruns/
```

**8. DVC tracks correct files:**
```bash
dvc status  # Should show data/ and models/ tracked
```

---

## Common Mistakes to Avoid

**1. DO NOT put code in scripts/ that should be in src/**
```
Bad:  scripts/bytetrack_implementation.py
Good: src/tracking/bytetrack.py
      scripts/run_tracking.py  (calls src.tracking.bytetrack)
```

**2. DO NOT hardcode paths**
```python
# Bad
data_path = "/home/user/mot-reid-system/data/processed/mot17"

# Good
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / "data" / "processed" / "mot17"
```

**3. DO NOT commit large files to git**
```
Use DVC for:
- Datasets (data/)
- Models (models/)
- Large outputs (outputs/ if needed)

Use git for:
- Code (src/, api/, tests/)
- Configs (configs/)
- Documentation (docs/)
```

**4. DO NOT mix concerns**
```
Each module has ONE responsibility:
- detection/: ONLY detection
- tracking/: ONLY tracking
- reid/: ONLY re-identification

If you need detection + tracking, create in inference/pipeline.py
```

**5. DO NOT use relative imports across packages**
```python
# Bad (fragile)
from ..detection.yolo_detector import YOLODetector

# Good (absolute, clear)
from src.detection.yolo_detector import YOLODetector
```

---

## Migration from Other Structures

If you have existing code in a different structure:

**Step 1: Create new structure**
```bash
mkdir -p data/{raw,processed,external}
mkdir -p models/{detection,reid,tracking,exported}/{pretrained,checkpoints,final}
mkdir -p src/{detection,tracking,reid,data,training,evaluation,inference,visualization,utils}
# ... (create all directories)
```

**Step 2: Move files systematically**
```bash
# Move detection code
mv old_project/yolo*.py src/detection/

# Move data scripts
mv old_project/dataset.py src/data/

# Move configs
mv old_project/*.yaml configs/
```

**Step 3: Update imports**
```bash
# Use sed or manual find-replace
sed -i 's/from yolo_detector/from src.detection.yolo_detector/g' **/*.py
```

**Step 4: Install as package**
```bash
pip install -e .
```

**Step 5: Test imports**
```bash
python -c "from src.detection.yolo_detector import YOLODetector"