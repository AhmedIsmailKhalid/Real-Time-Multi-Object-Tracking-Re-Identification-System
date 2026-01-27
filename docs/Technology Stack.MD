# Technology Stack - Multi-Object Tracking & Re-Identification System

## Document Purpose
This document defines the complete technology stack, version requirements, and rationale for all major technical decisions. This serves as the foundation for all implementation work.

---

## Core Language & Runtime

### Python 3.9+
**Selected Version:** Python 3.9.x or 3.10.x (NOT 3.11+ yet)

**Rationale:**
- PyTorch 2.0+ compatibility: Full support, no edge case bugs
- Mature ecosystem: All CV libraries have stable builds
- Type hinting: Advanced type hints for better code quality
- Production stability: Battle-tested in enterprise environments
- Why not 3.11+: Some ML libraries still have compatibility issues (2026 adoption lag)

**Alternative Considered:** Python 3.8
- Rejected: Missing some type hinting features, approaching EOL (Oct 2024)

---

## Deep Learning Framework

### PyTorch 2.0+
**Selected Version:** PyTorch 2.0.1+ with CUDA 11.8 or 12.1

**Rationale:**
- Industry standard (2026): 70%+ of CV research and production uses PyTorch
- torch.compile(): 2x inference speedup with minimal code changes
- ONNX export: Seamless conversion for deployment optimization
- Dynamic computation graphs: Critical for tracking (variable number of objects per frame)
- Ecosystem: Ultralytics (YOLOv8), torchvision, timm all PyTorch-native
- Hiring advantage: FAANG/top companies strongly prefer PyTorch experience

**Alternatives Considered:**
- **TensorFlow/Keras**: 
  - Rejected: Declining in CV research community
  - Rejected: Less flexible for custom tracking algorithms
  - Advantage: Better for mobile (TFLite), but not our deployment target
- **JAX**:
  - Rejected: Steep learning curve, smaller ecosystem
  - Advantage: Faster for some workloads, but overkill for this project

**Decision:** PyTorch - aligns with market demand and technical requirements

---

## Computer Vision Libraries

### 1. OpenCV 4.8+
**Purpose:** Image I/O, preprocessing, visualization

**Rationale:**
- Industry standard for CV operations
- Highly optimized C++ backend
- Video codec support (read/write videos)
- Drawing utilities for bounding boxes

**Alternative:** Pillow
- Rejected: Slower, less feature-complete for video processing

---

### 2. Ultralytics YOLOv8
**Purpose:** Object detection backbone

**Rationale:**
- State-of-the-art accuracy/speed trade-off (2024-2026 benchmark leader)
- Easy fine-tuning API: Minimal code to train on custom data
- Multiple model sizes: YOLOv8n (nano) to YOLOv8x (extra-large) for different deployment targets
- Active maintenance: Ultralytics releases frequent updates
- ONNX/TensorRT export: Built-in optimization paths

**Alternatives Considered:**
- **Faster R-CNN (Detectron2)**:
  - Rejected: Slower inference (~5 FPS vs. YOLO's 30+ FPS)
  - Advantage: Slightly better accuracy on small objects
  - Rejected: Speed critical for real-time tracking
  
- **DETR (DEtection TRansformer)**:
  - Rejected: Even slower than Faster R-CNN
  - Advantage: Interesting architecture, but not production-ready for real-time
  - Rejected: Not suitable for >20 FPS requirement

**Decision:** YOLOv8 - best speed/accuracy balance for real-time tracking

---

### 3. torchvision 0.15+
**Purpose:** Image transformations, pre-trained models

**Rationale:**
- Standard transforms for training (resize, normalize, augment)
- Pre-trained ResNet50 for Re-ID backbone
- Tight PyTorch integration

---

## Tracking Algorithm

### ByteTrack
**Implementation:** Custom implementation based on official paper

**Rationale:**
- SOTA performance on MOT benchmarks (2024-2026)
- Simple yet effective: Uses both high and low confidence detections
- Robust to occlusions: Better than classic SORT/DeepSORT
- No heavy dependencies: Can implement in ~300 lines
- Fast: Real-time performance even on crowded scenes

**Alternatives Considered:**
- **DeepSORT**:
  - Rejected: Older (2017), lower benchmark scores
  - Advantage: More widely known (good for interviews)
  - Rejected: ByteTrack is strictly better in performance
  
- **OC-SORT**:
  - Advantage: Newer, slightly better on some benchmarks
  - Rejected: More complex, harder to explain/debug
  - Rejected: Marginal gains not worth complexity

- **StrongSORT**:
  - Advantage: Ensemble approach, very high accuracy
  - Rejected: Much slower, requires multiple models
  - Rejected: Overkill for this project

**Decision:** ByteTrack - best performance/simplicity trade-off

---

## Re-Identification Network

### ResNet50 Backbone
**Pre-trained on:** ImageNet, then fine-tuned on Market-1501

**Rationale:**
- Proven architecture: Industry standard for Re-ID tasks
- Good speed/accuracy: 512-dim embeddings, fast extraction
- Transfer learning: ImageNet pre-training provides strong features
- Well-documented: Easy to explain in interviews
- Moderate size: ~25M parameters (fits in GPU memory with detection model)

**Alternatives Considered:**
- **Vision Transformer (ViT)**:
  - Advantage: Better accuracy on some Re-ID benchmarks
  - Rejected: Slower inference, larger model
  - Rejected: More complex to train/tune
  - Decision: Include as optional bonus feature, not core

- **MobileNetV3**:
  - Advantage: Very fast, small
  - Rejected: Lower accuracy
  - Rejected: Accuracy more critical than speed for Re-ID component

- **EfficientNet**:
  - Advantage: Good balance
  - Rejected: ResNet50 still standard in industry, better for portfolio

**Decision:** ResNet50 core, ViT as optional advanced feature

---

## MLOps & Experiment Tracking

### 1. MLflow 2.8+
**Purpose:** Experiment tracking, model registry, versioning

**Rationale:**
- Industry standard: Used at Uber, Netflix, Microsoft
- Complete lifecycle management: Track experiments, register models, deploy
- Easy setup: Works locally, scales to cloud
- Model versioning: Critical for production ML systems
- Portfolio value: Shows MLOps maturity

**Alternative:** Weights & Biases (W&B)
- Advantage: Better UI, more features
- Rejected: Requires cloud account, not fully free for portfolio projects
- Decision: MLflow for local/free deployment, mention W&B as alternative in docs

---

### 2. DVC (Data Version Control) 3.0+
**Purpose:** Dataset versioning, pipeline management

**Rationale:**
- Git for data: Track 50GB datasets without bloating git repo
- Reproducibility: Lock dataset versions with model versions
- Pipeline tracking: DAG of data processing steps
- Free: Works with local storage or free cloud buckets

**Alternative:** Manual versioning
- Rejected: Error-prone, not reproducible
- Decision: DVC essential for professional project

---

### 3. Prometheus + Grafana (Optional)
**Purpose:** Production monitoring, metrics dashboards

**Rationale:**
- Production-grade monitoring: What real companies use
- Real-time metrics: FPS, latency, memory usage
- Docker Compose deployment: Easy local setup
- Optional: Only if time permits, but adds strong portfolio value

---

## Deployment Stack

### 1. FastAPI 0.104+
**Purpose:** REST API for model serving

**Rationale:**
- Fastest Python web framework: Critical for low-latency inference
- Auto-generated docs: Swagger UI out of the box
- Type hints to validation: Catches errors at API boundary
- Async support: Handle multiple concurrent requests
- Industry adoption: Used at Netflix, Microsoft, Uber

**Alternatives Considered:**
- **Flask**:
  - Rejected: Slower, no native async, manual docs
  - Rejected: FastAPI strictly better for ML serving
  
- **Django**:
  - Rejected: Overkill (full web framework for API)
  - Rejected: Too heavy

**Decision:** FastAPI - modern standard for ML APIs

---

### 2. Docker + Docker Compose
**Purpose:** Containerization, local deployment orchestration

**Rationale:**
- Reproducibility: "Works on my machine" becomes works everywhere
- Dependency isolation: No version conflicts
- Production standard: All cloud platforms support Docker
- Multi-service orchestration: API + MLflow + Grafana in one compose file

---

### 3. ONNX Runtime 1.16+
**Purpose:** Model optimization and cross-platform deployment

**Rationale:**
- Inference speedup: 1.5-3x faster than PyTorch for production
- Cross-platform: Works on CPU, GPU, edge devices
- Standard format: Deploy same model to cloud/edge/mobile
- Easy export: PyTorch to ONNX in 5 lines of code

**Alternative:** TensorRT
- Advantage: Faster (NVIDIA-optimized)
- Rejected: NVIDIA GPUs only, more complex setup
- Decision: ONNX for core deployment, TensorRT as optional optimization

---

### 4. Streamlit 1.28+ / Gradio 4.0+
**Purpose:** Demo web interface

**Rationale:**
- 5-minute setup: Upload video, see results
- Free hosting: Hugging Face Spaces, Streamlit Cloud
- No frontend coding: Pure Python
- Portfolio-friendly: Shareable demo link

**Decision:** Start with Gradio (simpler), optionally add Streamlit if custom UI needed

---

## Testing & Quality Assurance

### 1. pytest 7.4+
**Purpose:** Unit and integration testing

**Rationale:**
- Industry standard for Python testing
- Rich plugin ecosystem (coverage, mocking)
- Fixtures for test data management

---

### 2. black + isort + flake8
**Purpose:** Code formatting and linting

**Rationale:**
- black: Deterministic formatting (no debates)
- isort: Sorts imports
- flake8: Catches common errors
- Portfolio quality: Shows professional code hygiene

---

### 3. mypy
**Purpose:** Static type checking

**Rationale:**
- Catches type errors before runtime
- Enforces type hint discipline
- Shows senior-level code quality

---

## Data Processing

### 1. pandas 2.0+
**Purpose:** Dataset metadata management, result analysis

**Rationale:**
- Standard for tabular data
- Easy CSV/JSON export
- Integration with visualization libraries

---

### 2. NumPy 1.24+
**Purpose:** Array operations, numerical processing

**Rationale:**
- Foundation of Python scientific computing
- Fast array operations
- Interop with OpenCV and PyTorch

---

## Visualization

### 1. matplotlib 3.7+
**Purpose:** Training plots, metric visualization

**Rationale:**
- Standard for scientific plotting
- Publication-quality figures

---

### 2. seaborn 0.12+
**Purpose:** Statistical visualization

**Rationale:**
- Beautiful default styles
- High-level interface for complex plots

---

## Configuration Management

### Hydra 1.3+ or PyYAML 6.0+
**Purpose:** Manage experiment configs, hyperparameters

**Rationale:**
- Hydra: Powerful, supports config overrides from CLI
- PyYAML: Simpler, sufficient for this project
- Decision: Start with PyYAML, migrate to Hydra if needed

---

## Hardware Requirements

### Minimum (Local Development)
- CPU: 4 cores (Intel i5/AMD Ryzen 5 or better)
- RAM: 16GB
- GPU: NVIDIA GTX 1060 (6GB VRAM) or better
- Storage: 100GB free (50GB datasets + 50GB models/checkpoints)
- OS: Ubuntu 20.04+, macOS 12+, Windows 11 (with WSL2)

### Recommended (Training)
- CPU: 8+ cores
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB VRAM) or better
- Storage: 200GB SSD

### Cloud Alternative (if no local GPU)
- Google Colab Pro: $10/month, T4/V100 GPUs
- Kaggle Notebooks: Free P100 GPUs (30h/week)
- AWS/GCP: On-demand instances (~$1/hour for training)

---

## Security & Credentials

### python-dotenv 1.0+
**Purpose:** Manage API keys, secrets

**Rationale:**
- Keep secrets out of git
- Standard pattern for credential management

---

## Dependency Management

### pip + requirements.txt + Docker
**Structure:**
```
requirements.txt          # Pinned versions for reproducibility
requirements-dev.txt      # Development tools (pytest, black, etc.)
Dockerfile               # Production dependencies only
```

**Rationale:**
- Simple: pip is universal
- Reproducible: Pinned versions prevent breakage
- Not using Poetry/Conda: Adds complexity, Docker handles isolation

---

## Version Pinning Strategy

### Production Dependencies
```
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.200
opencv-python==4.8.1.78
fastapi==0.104.1
```
**Pinned exactly** to prevent breaking changes

### Development Dependencies
```
pytest>=7.4.0
black>=23.0.0
```
**Flexible** (latest compatible versions OK)

---

## Rationale Summary Table

| Component | Choice | Why Not Alternative | Interview Talking Point |
|-----------|--------|---------------------|------------------------|
| **Framework** | PyTorch | TensorFlow declining in CV | "I chose PyTorch for its dynamic graphs, essential for variable object counts in tracking" |
| **Detector** | YOLOv8 | Faster R-CNN too slow | "YOLOv8 gives 30+ FPS vs. 5 FPS for two-stage detectors, critical for real-time" |
| **Tracker** | ByteTrack | DeepSORT outdated | "ByteTrack's dual-threshold approach reduces ID switches by 40% on MOT benchmarks" |
| **Re-ID** | ResNet50 | ViT slower | "ResNet50 balances accuracy and speed; ViT would bottleneck real-time inference" |
| **API** | FastAPI | Flask slower | "FastAPI's async support handles 3x more concurrent requests than Flask" |
| **MLOps** | MLflow | W&B paid | "MLflow provides enterprise-grade tracking while remaining fully self-hosted" |

---

## Tech Stack Summary

**Core Stack (Must-Have):**
- Python 3.9 + PyTorch 2.0 + YOLOv8 + ByteTrack + ResNet50
- FastAPI + Docker + ONNX
- MLflow + DVC
- pytest + black

**Optional (Portfolio Boosters):**
- Prometheus + Grafana (monitoring)
- TensorRT (extreme optimization)
- ViT Re-ID (advanced feature)
- Kubernetes manifests (cloud-ready)

**Total Estimated Setup Time:** 2-4 hours (mostly downloading datasets)