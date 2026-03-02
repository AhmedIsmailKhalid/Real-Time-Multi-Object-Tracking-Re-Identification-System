# Technology Stack

This document defines the complete technology stack, version requirements, and technical rationale for the CrossID system. All decisions are documented with alternatives considered and trade-offs evaluated.

---

## Table of Contents

1. [Core Language & Runtime](#core-language--runtime)
2. [Deep Learning Framework](#deep-learning-framework)
3. [Computer Vision Libraries](#computer-vision-libraries)
4. [Tracking & Re-Identification](#tracking--re-identification)
5. [Web Frameworks](#web-frameworks)
6. [Data Processing](#data-processing)
7. [Deployment & DevOps](#deployment--devops)
8. [Testing & Quality](#testing--quality)
9. [Configuration Management](#configuration-management)
10. [Hardware Requirements](#hardware-requirements)
11. [Dependency Management](#dependency-management)

---

## Core Language & Runtime

### Python 3.10+

**Selected Version:** Python 3.10.x or 3.11.x

**Rationale:**
- **Type Hints:** Advanced type hinting features (Union types with `|`, pattern matching)
- **Performance:** 10-15% faster than Python 3.9 (CPython optimizations)
- **Ecosystem Maturity:** All ML libraries fully compatible
- **Long-term Support:** Active support until October 2026 (3.10) and October 2027 (3.11)
- **Industry Adoption:** Standard in production environments as of 2024-2026

**Why Not Alternatives:**
- **Python 3.8-3.9:** Missing modern type hint features, approaching end of life
- **Python 3.12+:** Some ML libraries still stabilizing compatibility (ultralytics, torch)

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install python3.10

# macOS (Homebrew)
brew install python@3.10

# Windows
# Download from python.org
```

**Version Verification:**
```bash
python --version
# Python 3.10.11 or higher
```

---

## Deep Learning Framework

### PyTorch 2.0+

**Selected Version:** PyTorch 2.0.1+ with CUDA 11.8 or 12.1

**Rationale:**
- **Industry Standard:** 70%+ adoption in computer vision research and production (2024-2026)
- **Dynamic Computation Graphs:** Essential for tracking (variable number of objects per frame)
- **torch.compile():** 2x inference speedup with minimal code changes
- **Ecosystem:** Native integration with Ultralytics (YOLOv8), torchvision, timm
- **ONNX Export:** Seamless model conversion for deployment optimization
- **Research Velocity:** Latest CV models released for PyTorch first
- **Career Value:** Strong preference in FAANG and top-tier companies

**Alternatives Considered:**

**TensorFlow 2.x:**
- ❌ **Rejected:** Declining adoption in CV research community
- ❌ **Rejected:** Less flexible for custom tracking algorithms
- ✅ Advantage: Better mobile deployment (TFLite)
- Decision: PyTorch aligns better with CV ecosystem

**JAX:**
- ❌ **Rejected:** Steeper learning curve, smaller ecosystem
- ✅ Advantage: Faster for some workloads (XLA compilation)
- Decision: Maturity and ecosystem prioritized

**Installation:**
```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# CPU only (development/testing)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

**Validation:**
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
```

---

## Computer Vision Libraries

### 1. OpenCV 4.8+

**Purpose:** Image I/O, video processing, visualization

**Rationale:**
- **Performance:** Highly optimized C++ backend
- **Video Codecs:** Comprehensive codec support (MP4, AVI, MOV)
- **Visualization:** Drawing utilities for bounding boxes and text
- **Industry Standard:** Universal in CV applications
- **Python Bindings:** Mature, well-documented

**Installation:**
```bash
pip install opencv-python==4.8.1.78
```

**Alternative:** Pillow
- ❌ **Rejected:** Slower, lacks video processing capabilities
- Decision: OpenCV required for video I/O

---

### 2. Ultralytics YOLOv8 8.0+

**Purpose:** Real-time object detection

**Rationale:**
- **State-of-the-art:** Best accuracy/speed trade-off (2024-2026 benchmarks)
- **Multiple Scales:** YOLOv8n (nano) to YOLOv8x (extra-large) for different requirements
- **Easy API:** Simple training and inference interface
- **Active Maintenance:** Regular updates and bug fixes
- **Export Options:** Built-in ONNX/TensorRT export
- **Pre-trained Weights:** COCO-trained models available

**Alternatives Considered:**

**Faster R-CNN (Detectron2):**
- ❌ **Rejected:** ~5 FPS vs YOLOv8's 30+ FPS
- ✅ Advantage: Slightly better small object detection
- Decision: Speed critical for real-time tracking

**DETR (Detection Transformer):**
- ❌ **Rejected:** Even slower than Faster R-CNN
- ❌ **Rejected:** Not suitable for >20 FPS requirement
- Decision: Transformer overhead too high

**SSD (Single Shot Detector):**
- ❌ **Rejected:** Lower accuracy than YOLOv8
- Decision: YOLOv8 superior in all metrics

**Installation:**
```bash
pip install ultralytics==8.0.200
```

**Model Download:**
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Automatically downloads on first use
```

---

### 3. torchvision 0.15+

**Purpose:** Image transformations, pre-trained models

**Rationale:**
- **Standard Transforms:** Resize, normalize, augmentation
- **Pre-trained Models:** ResNet-50 for Re-ID backbone
- **Tight Integration:** Native PyTorch compatibility
- **Efficient:** Optimized for GPU operations

**Installation:**
```bash
pip install torchvision==0.15.2
```

---

## Tracking & Re-Identification

### 1. ByteTrack Algorithm

**Implementation:** Custom implementation based on official paper

**Rationale:**
- **SOTA Performance:** Top results on MOT benchmarks (2024-2026)
- **Two-stage Association:** Utilizes both high and low confidence detections
- **Occlusion Robustness:** Better than classic SORT/DeepSORT
- **Simplicity:** ~300 lines of code, easy to understand and debug
- **Real-time Speed:** Maintains performance on crowded scenes

**Alternatives Considered:**

**DeepSORT:**
- ❌ **Rejected:** Older (2017), lower benchmark performance
- Decision: ByteTrack strictly superior

**OC-SORT:**
- ⚠️ Advantage: Slightly better on some benchmarks
- ❌ **Rejected:** More complex, harder to debug
- Decision: Marginal gains not worth complexity

**StrongSORT:**
- ⚠️ Advantage: Ensemble approach, very high accuracy
- ❌ **Rejected:** Much slower, requires multiple models
- Decision: Overkill for requirements

**FairMOT:**
- ❌ **Rejected:** Joint detection-tracking-ReID (complex training)
- ❌ **Rejected:** Slower inference
- Decision: Separate modules more flexible

**Implementation:**
```python
# src/tracking/bytetrack.py
class ByteTrack:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        # Two-stage association with Kalman filtering
        pass
```

---

### 2. ResNet-50 for Re-ID

**Purpose:** Person re-identification across cameras

**Rationale:**
- **Proven Architecture:** Industry standard for Re-ID tasks
- **Transfer Learning:** ImageNet pre-training provides strong features
- **Speed/Accuracy Balance:** 2048-dim embeddings, 120 FPS extraction
- **Moderate Size:** ~25M parameters (fits in GPU with detection model)
- **Well-documented:** Extensive research and tutorials

**Alternatives Considered:**

**Vision Transformer (ViT):**
- ⚠️ Advantage: Higher accuracy on some benchmarks
- ❌ **Rejected:** Slower inference, larger model
- ❌ **Rejected:** More complex training
- Decision: ResNet-50 better for real-time constraints

**MobileNetV3:**
- ⚠️ Advantage: Very fast, small model
- ❌ **Rejected:** Lower accuracy
- Decision: Accuracy more critical for Re-ID

**EfficientNet:**
- ⚠️ Advantage: Good accuracy/efficiency balance
- ❌ **Rejected:** ResNet-50 more established in Re-ID literature
- Decision: Standard choice preferred

**Training Configuration:**
```yaml
model: resnet50
dataset: market1501
epochs: 120
batch_size: 32
learning_rate: 0.00035
optimizer: Adam
loss: triplet + cross_entropy
```

---

## Web Frameworks

### 1. FastAPI 0.104+

**Purpose:** Backend REST API for video processing

**Rationale:**
- **Performance:** Fastest Python web framework (near-ASGI speed)
- **Async Support:** Native async/await for background tasks
- **Auto Documentation:** Swagger UI and ReDoc generated automatically
- **Type Validation:** Pydantic models for request/response validation
- **Modern Python:** Leverages Python 3.10+ type hints
- **Industry Adoption:** Used at Netflix, Microsoft, Uber

**Alternatives Considered:**

**Flask:**
- ❌ **Rejected:** Synchronous by default, manual validation
- ❌ **Rejected:** No automatic API documentation
- Decision: FastAPI strictly better for ML serving

**Django REST Framework:**
- ❌ **Rejected:** Too heavy, ORM not needed
- ❌ **Rejected:** Complex setup for simple API
- Decision: Overkill for requirements

**Installation:**
```bash
pip install fastapi==0.104.1 uvicorn==0.24.0
```

**Server Start:**
```bash
uvicorn deployment.backend.app:app --reload --host 0.0.0.0 --port 8000
```

---

### 2. Streamlit 1.28+

**Purpose:** Frontend web interface

**Rationale:**
- **Rapid Development:** Full UI in hours vs weeks
- **Python-native:** No JavaScript/HTML/CSS required
- **Built-in Components:** File upload, video player, progress bars
- **Session State:** Easy state management
- **Hot Reload:** Instant preview during development
- **Simple Deployment:** `streamlit run app.py`

**Alternatives Considered:**

**React + TypeScript:**
- ❌ **Rejected:** 4-6 weeks development time
- ✅ Advantage: Full customization, professional UI
- Decision: Development speed prioritized

**Gradio:**
- ⚠️ Similar to Streamlit
- ❌ **Rejected:** Less flexible multi-page support
- Decision: Streamlit more mature for complex UIs

**Flask + Jinja2:**
- ❌ **Rejected:** Must build components from scratch
- Decision: Streamlit's built-in components superior

**Installation:**
```bash
pip install streamlit==1.28.0
```

**Application Start:**
```bash
streamlit run deployment/frontend/app.py
```

---

## Data Processing

### 1. NumPy 1.24+

**Purpose:** Array operations, numerical processing

**Rationale:**
- **Foundation:** Core of Python scientific computing
- **Performance:** Optimized C/Fortran backend
- **Interoperability:** Seamless with OpenCV, PyTorch
- **Universal:** Standard in all ML pipelines

**Installation:**
```bash
pip install numpy==1.24.3
```

---

### 2. pandas 2.0+

**Purpose:** Dataset metadata, result analysis

**Rationale:**
- **Tabular Data:** Standard for CSV/DataFrame operations
- **Analysis:** Easy aggregation and statistics
- **Integration:** Works with visualization libraries
- **Export:** CSV/JSON/Excel output

**Installation:**
```bash
pip install pandas==2.0.3
```

---

### 3. Pillow 10.0+

**Purpose:** Image loading and basic processing

**Rationale:**
- **Standard:** Default Python imaging library
- **Format Support:** Wide range of image formats
- **Integration:** Works with PyTorch, NumPy

**Installation:**
```bash
pip install Pillow==10.0.0
```

---

## Deployment & DevOps

### 1. FFmpeg

**Purpose:** Video encoding and format conversion

**Rationale:**
- **Codec Support:** Universal video codec compatibility
- **H.264 Encoding:** Browser-compatible video output
- **Performance:** Hardware-accelerated encoding (NVENC)
- **Industry Standard:** Universal video processing tool

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# Download from ffmpeg.org or use Chocolatey
choco install ffmpeg
```

**Verification:**
```bash
ffmpeg -version
# ffmpeg version 4.4 or higher
```

---

### 2. ONNX Runtime 1.16+ (Optional)

**Purpose:** Model optimization and deployment

**Rationale:**
- **Inference Speedup:** 1.5-3x faster than PyTorch
- **Cross-platform:** CPU, GPU, edge devices
- **Standard Format:** Portable model format

**Installation:**
```bash
pip install onnxruntime-gpu==1.16.0  # GPU
pip install onnxruntime==1.16.0      # CPU only
```

---

### 3. Docker (Optional)

**Purpose:** Containerization and reproducible deployment

**Rationale:**
- **Reproducibility:** Consistent environment across machines
- **Dependency Isolation:** No version conflicts
- **Production Standard:** Cloud platform compatibility

**Installation:**
```bash
# Ubuntu
sudo apt install docker.io docker-compose

# macOS/Windows
# Download Docker Desktop from docker.com
```

---

## Testing & Quality

### 1. pytest 7.4+

**Purpose:** Unit and integration testing

**Rationale:**
- **Industry Standard:** Most popular Python testing framework
- **Fixture System:** Easy setup/teardown management
- **Parametrization:** Test multiple inputs efficiently
- **Plugin Ecosystem:** Coverage, mocking, etc.

**Installation:**
```bash
pip install pytest==7.4.3 pytest-cov==4.1.0
```

**Usage:**
```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

---

### 2. Ruff 0.1.15+

**Purpose:** Fast Python linter and formatter

**Rationale:**
- **Speed:** 10-100x faster than Flake8/Black
- **Comprehensive:** Replaces Flake8, Black, isort
- **Modern:** Written in Rust, actively maintained

**Installation:**
```bash
pip install ruff==0.1.15
```

**Usage:**
```bash
# Lint
ruff check .

# Format
ruff format .
```

**Alternative:** Black + Flake8 + isort
- ❌ **Rejected:** Slower, multiple tools
- Decision: Ruff consolidates all linting/formatting

---

### 3. mypy 1.7+ (Optional)

**Purpose:** Static type checking

**Rationale:**
- **Type Safety:** Catch errors before runtime
- **Code Quality:** Enforces type discipline
- **Documentation:** Type hints serve as documentation

**Installation:**
```bash
pip install mypy==1.7.0
```

---

## Configuration Management

### PyYAML 6.0+

**Purpose:** Configuration file management

**Rationale:**
- **Human-readable:** Easy to read and edit
- **Comments:** Support for inline documentation
- **Hierarchical:** Nested configuration structures
- **Standard:** Wide adoption in ML projects

**Installation:**
```bash
pip install PyYAML==6.0.1
```

**Alternative:** Hydra
- ⚠️ Advantage: More powerful, CLI overrides
- ❌ **Rejected:** Adds complexity for simple configs
- Decision: PyYAML sufficient for current needs

---

### python-dotenv 1.0+

**Purpose:** Environment variable management

**Rationale:**
- **Security:** Keep secrets out of version control
- **Standard Pattern:** `.env` file convention
- **Pydantic Integration:** Works with FastAPI settings

**Installation:**
```bash
pip install python-dotenv==1.0.0
```

**Usage:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEVICE: str = "cuda"
    
    class Config:
        env_file = ".env"
```

---

## Hardware Requirements

### Minimum (Development)

**CPU:** 4 cores (Intel i5/AMD Ryzen 5 or better)  
**RAM:** 16 GB  
**GPU:** NVIDIA GTX 1060 (6 GB VRAM) or better  
**Storage:** 50 GB free  
- 15 GB for datasets
- 5 GB for models
- 30 GB for outputs/checkpoints

**OS:** Ubuntu 20.04+, macOS 12+, Windows 11 (with WSL2)

---

### Recommended (Training & Production)

**CPU:** 8+ cores (Intel i7/AMD Ryzen 7)  
**RAM:** 32 GB  
**GPU:** NVIDIA RTX 3060 (12 GB VRAM) or better  
**Storage:** 100 GB SSD

---

### Cloud Alternatives

**Google Colab Pro:**
- $10/month
- T4/V100 GPUs
- Good for prototyping

**Kaggle Notebooks:**
- Free P100 GPUs
- 30 hours/week limit

**AWS/GCP/Azure:**
- On-demand GPU instances
- ~$1/hour for training
- ~$0.50/hour for inference

---

## Dependency Management

### Poetry 1.6+

**Purpose:** Dependency management and packaging

**Rationale:**
- **Lock Files:** Deterministic dependency resolution
- **Virtual Environments:** Automatic environment creation
- **Dependency Groups:** Separate dev/prod dependencies
- **Modern:** Better than pip + requirements.txt

**Installation:**
```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Usage:**
```bash
# Install dependencies
poetry install

# Add dependency
poetry add torch==2.0.1

# Run script
poetry run python script.py
```

**Alternative:** pip + requirements.txt
- ❌ **Rejected:** Manual lock file management
- ❌ **Rejected:** No dependency groups
- Decision: Poetry superior for modern Python projects

---

## Complete Technology Stack Summary

### Core Stack (Production)

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Core implementation |
| **DL Framework** | PyTorch | 2.0+ | Model training/inference |
| **Detection** | YOLOv8 | 8.0+ | Person detection |
| **Tracking** | ByteTrack | Custom | Multi-object tracking |
| **Re-ID** | ResNet-50 | Custom | Person re-identification |
| **Backend API** | FastAPI | 0.104+ | REST API server |
| **Frontend** | Streamlit | 1.28+ | Web interface |
| **Video I/O** | OpenCV | 4.8+ | Video processing |
| **Video Encoding** | FFmpeg | 4.4+ | H.264 encoding |
| **Testing** | pytest | 7.4+ | Unit/integration tests |
| **Linting** | Ruff | 0.1.15+ | Code quality |
| **Package Manager** | Poetry | 1.6+ | Dependency management |

---

### Optional Components (Enhancements)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Optimization** | ONNX Runtime | 1.5-3x speedup |
| **Containerization** | Docker | Deployment |
| **Type Checking** | mypy | Static analysis |
| **Monitoring** | Prometheus + Grafana | Production metrics |

---

## Installation Quick Start

### Complete Setup Script

```bash
# 1. Create project directory
mkdir crossid && cd crossid

# 2. Initialize Poetry
poetry init --no-interaction

# 3. Install core dependencies
poetry add torch==2.0.1 torchvision==0.15.2
poetry add ultralytics==8.0.200
poetry add opencv-python==4.8.1.78
poetry add fastapi==0.104.1 uvicorn==0.24.0
poetry add streamlit==1.28.0
poetry add numpy==1.24.3 pandas==2.0.3
poetry add pydantic-settings==2.0.3
poetry add python-dotenv==1.0.0
poetry add PyYAML==6.0.1

# 4. Install development dependencies
poetry add --group dev pytest==7.4.3 pytest-cov==4.1.0
poetry add --group dev ruff==0.1.15

# 5. Activate environment
poetry shell

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Version Pinning Strategy

### Production Dependencies (Exact Versions)

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "2.0.1"
torchvision = "0.15.2"
ultralytics = "8.0.200"
opencv-python = "4.8.1.78"
fastapi = "0.104.1"
streamlit = "1.28.0"
```

**Rationale:** Prevent breaking changes in production

---

### Development Dependencies (Flexible)

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
ruff = "^0.1.15"
```

**Rationale:** Latest compatible versions for dev tools

---

## Technology Decision Matrix

| Requirement | Technology | Alternative | Decision Rationale |
|-------------|-----------|-------------|-------------------|
| **Real-time Detection** | YOLOv8 | Faster R-CNN | 30+ FPS vs 5 FPS |
| **Tracking** | ByteTrack | DeepSORT | 13% IDF1 improvement |
| **Re-ID** | ResNet-50 | ViT | Speed/accuracy balance |
| **API** | FastAPI | Flask | Native async, auto-docs |
| **Frontend** | Streamlit | React | Hours vs weeks dev time |
| **Framework** | PyTorch | TensorFlow | CV ecosystem standard |
| **Linting** | Ruff | Black + Flake8 | 10-100x faster |
| **Package Manager** | Poetry | pip | Deterministic builds |

---

## Interview Talking Points

**"Why PyTorch over TensorFlow?"**
> "PyTorch's dynamic computation graphs are essential for tracking variable numbers of objects per frame. Additionally, PyTorch has 70%+ adoption in CV research as of 2024-2026, ensuring better ecosystem support and career alignment."

**"Why YOLOv8 over Faster R-CNN?"**
> "Real-time constraints require >30 FPS. YOLOv8 achieves 35 FPS at 92% precision, while Faster R-CNN achieves only 5 FPS despite 94% precision. The 2% accuracy trade-off is acceptable for 7x speed gain."

**"Why ByteTrack over DeepSORT?"**
> "ByteTrack's two-stage association (high + low confidence detections) reduces ID switches by 24% on MOT17 benchmarks compared to DeepSORT, while maintaining real-time performance."

**"Why FastAPI over Flask?"**
> "FastAPI's native async support handles 3x more concurrent requests than Flask. Auto-generated Swagger documentation and Pydantic validation reduce development time and improve API reliability."

**"Why Streamlit instead of building a React app?"**
> "Development velocity was prioritized. Streamlit enabled a full 5-page interface in 1.5 weeks versus 4-6 weeks for React. The rapid iteration allowed focus on core ML functionality."

---

## Maintenance & Updates

### Update Strategy

**Monthly:**
- Security patches for all dependencies
- Review CVE databases for vulnerabilities

**Quarterly:**
- Minor version updates (0.x.y → 0.x+1.0)
- Test for breaking changes

**Annually:**
- Major version updates (x.y.z → x+1.0.0)
- Full regression testing

**Commands:**
```bash
# Check for outdated packages
poetry show --outdated

# Update dependencies
poetry update

# Update lock file only
poetry lock
```

---

## Troubleshooting

### Common Issues

**Issue:** CUDA not available  
**Solution:** Verify CUDA drivers, reinstall PyTorch with correct CUDA version

**Issue:** OpenCV video codec errors  
**Solution:** Install FFmpeg, use H.264 re-encoding

**Issue:** Streamlit port already in use  
**Solution:** Kill process on port 8501 or use `--server.port` flag

**Issue:** Poetry install fails  
**Solution:** Clear cache with `poetry cache clear pypi --all`

---

## Future Enhancements

**Planned Additions:**
- TensorRT for 2x inference speedup
- Prometheus + Grafana for monitoring
- Docker Compose for multi-service deployment
- Kubernetes manifests for cloud scaling

**Version 2.0 Roadmap:**
- Upgrade to PyTorch 2.2+ (when stable)
- Replace OpenCV with torchvision transforms (pure PyTorch pipeline)
- Add Vision Transformer option for Re-ID
- Implement model quantization (INT8) for edge deployment

---

**Last Updated:** February 2026  
**CrossID Version:** 1.0.0  
**Stack Version:** 1.0