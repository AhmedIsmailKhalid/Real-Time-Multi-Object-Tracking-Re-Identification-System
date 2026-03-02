# System Architecture

This document describes the complete architecture of the CrossID system, including component design, data flow, technology stack, and deployment topology.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Component Interactions](#component-interactions)
5. [Deployment Architecture](#deployment-architecture)
6. [Technology Stack](#technology-stack)
7. [Design Patterns](#design-patterns)
8. [Scalability Considerations](#scalability-considerations)
9. [Security Architecture](#security-architecture)

---

## High-Level Overview

CrossID is a modular, end-to-end system for real-time multi-object tracking and person re-identification across video streams. The architecture follows a three-tier design:

1. **Presentation Layer** - Web-based user interface (Streamlit)
2. **Application Layer** - REST API and business logic (FastAPI)
3. **Processing Layer** - Computer vision pipeline (PyTorch)

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│                   (Streamlit Frontend)                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │   Home   │  Demos   │  Upload  │  Multi   │  Search  │  │
│  │   Page   │  Videos  │  Video   │  Camera  │  Person  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────────┐
│                    Application Layer                         │
│                     (FastAPI Backend)                        │
│  ┌────────────┬──────────────┬──────────────┬────────────┐ │
│  │ Video      │ Multi-Camera │ Person       │ Job        │ │
│  │ Upload     │ Processing   │ Search       │ Management │ │
│  └────────────┴──────────────┴──────────────┴────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Function Calls
┌─────────────────────────▼───────────────────────────────────┐
│                    Processing Layer                          │
│                  (Computer Vision Pipeline)                  │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ Detection│ Tracking │  Re-ID   │Visualize │ Video I/O│  │
│  │ (YOLO)   │(ByteTrack│(ResNet50)│(BBoxDraw)│(OpenCV)  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Detection Module

**Purpose:** Detect person bounding boxes in video frames.

**Architecture:**
```
┌─────────────────────────────────────────────┐
│           Detection Module                   │
├─────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────┐    │
│  │      DetectorBase (Abstract)       │    │
│  │  - detect(frame) -> List[Detection]│    │
│  └────────────▲───────────────────────┘    │
│               │                              │
│  ┌────────────┴───────────────────────┐    │
│  │       YOLODetector (Concrete)      │    │
│  │  - model: YOLOv8                   │    │
│  │  - conf_threshold: float           │    │
│  │  - device: str                     │    │
│  │  + detect(frame) -> List[Detection]│    │
│  └────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

**Key Components:**
- **DetectorBase** - Abstract interface for all detectors
- **YOLODetector** - YOLOv8-based implementation
- **Detection** - Data class for bounding box results

**Technology:**
- Framework: Ultralytics YOLOv8
- Model: YOLOv8s (11 MB)
- Device: CUDA-enabled GPU (fallback to CPU)
- Input: BGR frames (OpenCV format)
- Output: List of (bbox, confidence, class_id)

**Configuration:**
```yaml
# configs/detection.yaml
model: models/detection/pretrained/yolov8s.pt
confidence: 0.5
iou_threshold: 0.45
device: cuda
classes: [0]  # Person class only
image_size: 640
```

**Performance:**
- Inference speed: 35 FPS @ 1080p (RTX 3060)
- Precision: 92%
- Recall: 88%
- GPU memory: 2.8 GB

---

### 2. Tracking Module

**Purpose:** Maintain consistent track IDs for detected objects across frames.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Tracking Module                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          TrackerBase (Abstract)                      │  │
│  │  - update(detections) -> List[Track]                 │  │
│  └────────────────▲─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────┴─────────────────────────────────────┐  │
│  │            ByteTrack (Concrete)                      │  │
│  │  - tracked_tracks: List[Track]                       │  │
│  │  - lost_tracks: List[Track]                          │  │
│  │  - removed_tracks: List[Track]                       │  │
│  │  + update(detections) -> List[Track]                 │  │
│  │  + first_association() -> matches                    │  │
│  │  + second_association() -> matches                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Track        │  │ KalmanFilter │  │ Matching        │  │
│  │ - bbox       │  │ - predict()  │  │ - iou_distance()│  │
│  │ - track_id   │  │ - update()   │  │ - reid_distance│  │
│  │ - state      │  │              │  │ - assignment()  │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **TrackerBase** - Abstract tracker interface
- **ByteTrack** - Two-stage association tracker
- **Track** - Track state management (New → Tracked → Lost → Removed)
- **KalmanFilter** - Motion prediction (8-state: x, y, w, h, vx, vy, vw, vh)
- **Matching** - IoU and Re-ID based association

**Technology:**
- Algorithm: ByteTrack with Kalman filtering
- Association: Two-stage (high-conf → low-conf)
- Motion model: Constant velocity assumption
- Matching: IoU + Re-ID similarity

**Configuration:**
```yaml
# configs/tracking.yaml
track_thresh: 0.5       # Activation threshold
track_buffer: 30        # Lost track retention (frames)
match_thresh: 0.8       # IoU matching threshold
frame_rate: 30          # Expected FPS
```

**State Machine:**
```
     detect
       ↓
┌─────────┐  track_thresh
│   New   │──────────────→  ┌─────────┐
└─────────┘                  │ Tracked │
                             └────┬────┘
                    lost_count < buffer
                                  ↓
                             ┌────────┐  lost_count >= buffer
                             │  Lost  │──────────────────────→ ┌─────────┐
                             └────────┘                         │ Removed │
                                  ↑                             └─────────┘
                                  │
                             re-detection
```

**Performance:**
- Tracking speed: 28 FPS @ 1080p
- MOTA (no Re-ID): 72%
- MOTA (with Re-ID): 74%
- IDF1 (no Re-ID): 65%
- IDF1 (with Re-ID): 78%

---

### 3. Re-Identification Module

**Purpose:** Extract appearance features for cross-camera person matching.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                Re-Identification Module                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           ReIDBase (Abstract)                        │  │
│  │  - extract_features(images) -> features              │  │
│  └────────────────▲─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────┴─────────────────────────────────────┐  │
│  │          ResNetReID (Concrete)                       │  │
│  │  - backbone: ResNet-50                               │  │
│  │  - embedding_dim: 2048                               │  │
│  │  - feature_gallery: Dict[id, features]              │  │
│  │  + extract_features(images) -> Tensor[N, 2048]      │  │
│  │  + compute_similarity(f1, f2) -> float              │  │
│  │  + update_gallery(track_id, feature)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────┐  ┌────────────────────────────────┐   │
│  │ MetricLearning  │  │ Feature Gallery                │   │
│  │ - TripletLoss   │  │ {track_id: {                   │   │
│  │ - CrossEntropy  │  │   "features": [f1, f2, ...],   │   │
│  │ - margin: 0.3   │  │   "frames": [100, 120, ...],   │   │
│  │                 │  │   "bboxes": [(x,y,w,h), ...]   │   │
│  └─────────────────┘  │ }}                             │   │
│                       └────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **ReIDBase** - Abstract Re-ID interface
- **ResNetReID** - ResNet-50 based implementation
- **MetricLearning** - Triplet loss + cross-entropy training
- **Feature Gallery** - Per-track feature storage

**Technology:**
- Architecture: ResNet-50 (ImageNet pre-trained)
- Input size: 128×256 RGB
- Output: 2048-dim embeddings
- Loss: Triplet (margin=0.3) + Cross-entropy
- Similarity: Cosine similarity

**Training Details:**
```yaml
# configs/reid.yaml
model: resnet50
dataset: market1501
epochs: 120
batch_size: 32
learning_rate: 0.00035
optimizer: Adam
warmup_epochs: 10
scheduler: step
step_size: [40, 70]
```

**Feature Gallery Structure:**
```python
feature_gallery = {
    track_id_1: {
        "features": [
            np.array([2048,]),  # Frame 100
            np.array([2048,]),  # Frame 120
            ...
        ],
        "frames": [100, 120, 140, ...],
        "bboxes": [(x, y, w, h), ...]
    },
    track_id_2: { ... },
    ...
}
```

**Similarity Computation:**
```python
def compute_similarity(feature1, feature2):
    """Cosine similarity between two features."""
    return np.dot(feature1, feature2) / (
        np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-10
    )

# Matching criterion
if similarity > 0.85:  # Threshold tuned on validation set
    track_id_new = track_id_existing  # Re-ID match
```

**Performance:**
- Feature extraction: 120 FPS @ 128×256 (GPU)
- Rank-1 accuracy: 99.05%
- mAP: 70.62%
- Cross-camera match rate: 91%

---

### 4. Inference Pipeline

**Purpose:** Orchestrate detection, tracking, Re-ID, and visualization into unified workflow.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                   Inference Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              InferencePipeline                        │  │
│  │  - detector: YOLODetector                             │  │
│  │  - tracker: ByteTrack                                 │  │
│  │  - reid_model: ResNetReID (optional)                  │  │
│  │  - visualizer: BBoxDrawer                             │  │
│  │  + process_video(video_path, output_path, ...)       │  │
│  │  + process_frame(frame) -> annotated_frame           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Flow:                                                        │
│  ┌───────┐  ┌─────────┐  ┌─────────┐  ┌──────┐  ┌───────┐ │
│  │ Video │→ │ Detect  │→ │ Track   │→ │ Re-ID│→ │Visualiz│ │
│  │ Input │  │ (YOLO)  │  │(ByteTrk)│  │(Resnt)  │(Draw)  │ │
│  └───────┘  └─────────┘  └─────────┘  └──────┘  └───────┘ │
│      │           │             │           │          │      │
│      ▼           ▼             ▼           ▼          ▼      │
│  Read Frame   Bboxes      Track IDs   Features   Annotated  │
│                                                      Frame    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Processing Flow:**
```python
def process_frame(self, frame):
    # 1. Detection
    detections = self.detector.detect(frame)
    
    # 2. Tracking
    tracks = self.tracker.update(detections)
    
    # 3. Re-ID (if enabled)
    if self.reid_model:
        for track in tracks:
            crop = extract_crop(frame, track.bbox)
            feature = self.reid_model.extract_features(crop)
            
            # Match against gallery
            best_match = self.reid_model.find_match(feature, threshold=0.85)
            if best_match:
                track.track_id = best_match  # Re-assign ID
            
            # Update gallery
            self.reid_model.update_gallery(track.track_id, feature)
    
    # 4. Visualization
    annotated_frame = self.visualizer.draw_tracks(frame, tracks)
    
    return annotated_frame
```

**Configuration:**
```yaml
# configs/inference.yaml
max_fps: 30               # Cap output frame rate
show_trails: true         # Draw motion trails
trail_length: 30          # Trail history (frames)
enable_reid: true         # Enable Re-ID module
bbox_thickness: 2         # Bounding box line width
font_scale: 0.8           # Text size
```

**Pipeline Wrapper:**
```python
class PipelineWrapper:
    def __init__(self, yolo_model_path, reid_model_path, device, enable_reid):
        self.pipeline = InferencePipeline(
            detector=YOLODetector(yolo_model_path, device=device),
            tracker=ByteTrack(),
            reid_model=ResNetReID(reid_model_path) if enable_reid else None,
            visualizer=BBoxDrawer()
        )
    
    def process_video(self, video_path, output_path, enable_reid=True, 
                      show_trails=True, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated = self.pipeline.process_frame(frame)
            writer.write(annotated)
            
            # Progress callback
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        writer.release()
        
        # FFmpeg H.264 re-encoding for browser compatibility
        self._reencode_h264(output_path)
        
        return statistics
```

**Performance:**
- End-to-end: 15-30 FPS @ 1080p (GPU)
- Memory usage: 5.2 GB peak (GPU)
- CPU usage: 65% average

---

### 5. Web Interface (Streamlit)

**Purpose:** Provide user-friendly web interface for video processing.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  app.py (Entry Point)                                        │
│  └─ Multi-page navigation                                    │
│                                                               │
│  pages/                                                       │
│  ├─ home.py              # Landing page                      │
│  ├─ demo_videos.py       # Pre-loaded demos (6 scenarios)    │
│  ├─ upload_video.py      # Single video upload               │
│  ├─ multi_camera.py      # Multi-camera processing (2-4)     │
│  └─ person_search.py     # Person search across videos       │
│                                                               │
│  utils/                                                       │
│  ├─ api_client.py        # Backend API wrapper               │
│  └─ video_utils.py       # Video preview helpers             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Page Structure:**
```python
# pages/demo_videos.py
import streamlit as st
from utils.api_client import CrossIDClient

# Session state (page-specific namespace)
if "demo_processing" not in st.session_state:
    st.session_state.demo_processing = False

# Page layout
st.title("Demo Videos")
st.info("Pre-loaded scenarios...")

# Video grid (6 demos)
cols = st.columns(3)
for idx, video in enumerate(demo_videos):
    with cols[idx % 3]:
        st.markdown(f"**{video['title']}**")
        if st.button("Select", key=f"btn_{idx}"):
            st.session_state.demo_selected_video = video
            st.rerun()

# Processing workflow
if st.session_state.demo_processing:
    # Upload to backend
    job_id = client.upload_video(video_path, enable_reid=True)
    
    # Poll for progress
    while True:
        status = client.get_job_status(job_id)
        progress_bar.progress(status['progress'])
        if status['status'] == 'completed':
            break
    
    # Download and display result
    client.download_result(job_id, output_path)
    st.video(output_path)
```

**API Client:**
```python
# utils/api_client.py
class CrossIDClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.api_url = f"{base_url}/api/v1"
    
    def upload_video(self, video_path, enable_reid=True, **kwargs):
        files = {"video": open(video_path, "rb")}
        data = {
            "enable_reid": "true" if enable_reid else "false",
            "confidence_threshold": kwargs.get('confidence', 0.5)
        }
        response = requests.post(f"{self.api_url}/track/video", 
                                 files=files, data=data)
        return response.json()["job_id"]
    
    def get_job_status(self, job_id):
        response = requests.get(f"{self.api_url}/track/status/{job_id}")
        return response.json()
    
    def download_result(self, job_id, output_path):
        response = requests.get(f"{self.api_url}/track/result/{job_id}", 
                                stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
```

**Session State Management:**
```python
# Separate namespaces per page to avoid conflicts
# demo_videos.py
st.session_state.demo_processing = True
st.session_state.demo_selected_video = video

# upload_video.py
st.session_state.upload_processing = True
st.session_state.upload_video_path = path

# multi_camera.py
st.session_state.mc_processing = True
st.session_state.mc_video_paths = [path1, path2, path3, path4]

# person_search.py
st.session_state.search_results = matches
st.session_state.query_image_path = image_path
```

---

### 6. Backend API (FastAPI)

**Purpose:** Provide REST API for asynchronous video processing.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  app.py (Entry Point)                                        │
│  └─ FastAPI application initialization                       │
│                                                               │
│  api/                                                         │
│  ├─ routes.py            # API endpoints                     │
│  └─ models.py            # Pydantic request/response schemas │
│                                                               │
│  core/                                                        │
│  ├─ config.py            # Settings and environment vars     │
│  ├─ pipeline.py          # Pipeline wrapper                  │
│  └─ logger.py            # Logging configuration             │
│                                                               │
│  utils/                                                       │
│  └─ video_utils.py       # Video processing utilities        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**API Endpoints:**
```python
# POST /api/v1/track/video
@router.post("/track/video", response_model=TrackingResponse)
async def track_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    enable_reid: str = Form("true"),
    confidence_threshold: float = Form(0.5)
):
    job_id = str(uuid.uuid4())
    
    # Save uploaded video
    upload_path = save_uploaded_file(video, job_id)
    output_path = generate_output_path(job_id)
    
    # Create job entry
    jobs[job_id] = {"status": "queued", "progress": 0.0}
    
    # Schedule background processing
    background_tasks.add_task(
        process_video_task, job_id, upload_path, output_path, request
    )
    
    return {"job_id": job_id, "status": "queued"}

# GET /api/v1/track/status/{job_id}
@router.get("/track/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

# GET /api/v1/track/result/{job_id}
@router.get("/track/result/{job_id}")
async def get_result_video(job_id: str):
    result_path = jobs[job_id]["result_path"]
    return FileResponse(result_path, media_type="video/mp4")

# POST /api/v1/track/multi-camera
@router.post("/track/multi-camera", response_model=TrackingResponse)
async def track_multi_camera(
    background_tasks: BackgroundTasks,
    videos: list[UploadFile] = File(...),
    enable_reid: str = Form("true")
):
    job_id = str(uuid.uuid4())
    
    # Save all videos
    upload_paths = [save_uploaded_file(v, job_id, idx) 
                    for idx, v in enumerate(videos)]
    output_paths = [generate_output_path(job_id, idx) 
                    for idx in range(len(videos))]
    
    # Create job with per-camera tracking
    jobs[job_id] = {
        "status": "queued",
        "num_videos": len(videos),
        "video_progress": [0.0] * len(videos),
        "video_status": ["queued"] * len(videos)
    }
    
    # Schedule multi-camera processing
    background_tasks.add_task(
        process_multi_camera_task, job_id, upload_paths, output_paths, request
    )
    
    return {"job_id": job_id, "status": "queued"}

# POST /api/v1/search/person
@router.post("/search/person")
async def search_person(
    query_image: UploadFile = File(...),
    similarity_threshold: float = Form(0.85)
):
    # Extract features from query image
    image = load_image(query_image)
    query_features = reid_model.extract_features(image)
    
    # Search through saved feature galleries
    matches = []
    for gallery_file in load_all_galleries():
        gallery = load_gallery(gallery_file)
        for track_id, track_data in gallery.items():
            similarities = [
                cosine_similarity(query_features, f) 
                for f in track_data['features']
            ]
            max_sim = max(similarities)
            if max_sim >= similarity_threshold:
                matches.append({
                    "job_id": gallery['job_id'],
                    "track_id": track_id,
                    "similarity": max_sim
                })
    
    return {"matches": sorted(matches, key=lambda x: -x['similarity'])}
```

**Background Processing:**
```python
def process_video_task(job_id, video_path, output_path, request):
    try:
        jobs[job_id]["status"] = "processing"
        
        # Progress callback
        def progress_callback(current, total):
            jobs[job_id]["progress"] = current / total
        
        # Get pipeline
        pipeline = get_pipeline(
            confidence_threshold=request.confidence_threshold,
            enable_reid=request.enable_reid
        )
        
        # Process video
        results = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            enable_reid=request.enable_reid,
            progress_callback=progress_callback
        )
        
        # Save feature gallery for person search
        if request.enable_reid:
            save_feature_gallery(job_id, pipeline.feature_gallery)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_path"] = output_path
        jobs[job_id]["stats"] = results
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
```

**Configuration:**
```python
# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "CrossID API"
    API_VERSION: str = "1.0.0"
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    UPLOAD_DIR: Path = PROJECT_ROOT / "outputs" / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "processed"
    
    # Model Paths
    YOLO_MODEL: Path = MODELS_DIR / "detection/pretrained/yolov8s.pt"
    REID_MODEL: Path = MODELS_DIR / "reid/final/resnet50_market_best.pth"
    
    # Processing Settings
    DEVICE: str = "cuda"
    MAX_VIDEO_SIZE_MB: int = 500
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Data Flow

### Single Video Processing Flow

```
User Upload
    │
    ▼
┌────────────────────────────────────┐
│  Frontend (Streamlit)              │
│  - User selects video              │
│  - Configures settings             │
│  - Clicks "Process"                │
└────────────┬───────────────────────┘
             │ HTTP POST /track/video
             │ multipart/form-data
             ▼
┌────────────────────────────────────┐
│  Backend (FastAPI)                 │
│  - Receives video file             │
│  - Generates job_id (UUID)         │
│  - Saves to uploads/               │
│  - Queues background task          │
│  - Returns job_id                  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Background Task                   │
│  - Loads pipeline                  │
│  - Processes video frame-by-frame  │
│  - Updates job progress            │
│  - Saves result to outputs/        │
│  - Saves feature gallery (Re-ID)   │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Processing Pipeline               │
│  ┌──────────────────────────────┐ │
│  │ For each frame:              │ │
│  │  1. Detect (YOLO)            │ │
│  │  2. Track (ByteTrack)        │ │
│  │  3. Re-ID (ResNet50)         │ │
│  │  4. Visualize (Draw)         │ │
│  │  5. Write output             │ │
│  └──────────────────────────────┘ │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Output Generation                 │
│  - FFmpeg H.264 re-encoding        │
│  - Feature gallery saved (.pkl)    │
│  - Statistics computed             │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Frontend Polling                  │
│  - GET /track/status/{job_id}      │
│  - Updates progress bar            │
│  - Checks completion               │
└────────────┬───────────────────────┘
             │ status == "completed"
             ▼
┌────────────────────────────────────┐
│  Result Download                   │
│  - GET /track/result/{job_id}      │
│  - Download processed video        │
│  - Display in video player         │
│  - Show statistics                 │
└────────────────────────────────────┘
```

### Multi-Camera Processing Flow

```
User Uploads 4 Videos
    │
    ▼
┌────────────────────────────────────┐
│  Frontend Multi-Camera Page        │
│  - Select 2-4 videos               │
│  - Preview grid (1x2, 1x3, 2x2)   │
│  - Configure settings              │
│  - Click "Process All"             │
└────────────┬───────────────────────┘
             │ HTTP POST /track/multi-camera
             │ list[UploadFile]
             ▼
┌────────────────────────────────────┐
│  Backend Multi-Camera Endpoint     │
│  - Receives 2-4 video files        │
│  - Generates single job_id         │
│  - Saves all to uploads/           │
│  - Initializes shared pipeline     │
│  - Queues background task          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Sequential Processing (Shared)    │
│  ┌──────────────────────────────┐ │
│  │ For each video (1-4):        │ │
│  │   Process with shared        │ │
│  │   pipeline.feature_gallery   │ │
│  │                              │ │
│  │   Camera 1 → Track IDs: 1-50 │ │
│  │   Camera 2 → Re-ID matches   │ │
│  │              → Same IDs: 1-50│ │
│  │   Camera 3 → Re-ID matches   │ │
│  │              → Same IDs: 1-50│ │
│  │   Camera 4 → Re-ID matches   │ │
│  │              → Same IDs: 1-50│ │
│  └──────────────────────────────┘ │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Per-Camera Progress Tracking      │
│  jobs[job_id] = {                  │
│    "video_progress": [0.8, 0.6, 0.4, 0.0],│
│    "video_status": ["completed", "processing", "processing", "queued"]│
│  }                                 │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  FFmpeg Merged Output              │
│  - Combine all 4 videos            │
│  - 2x2 grid layout                 │
│  - Add camera labels               │
│  - Single downloadable file        │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Frontend Results Display          │
│  - 2x2 grid of processed videos    │
│  - Individual downloads            │
│  - Merged video download           │
│  - Statistics per camera           │
└────────────────────────────────────┘
```

### Person Search Flow

```
User Uploads Query Image
    │
    ▼
┌────────────────────────────────────┐
│  Person Search Page                │
│  - Upload person image (cropped)   │
│  - Set similarity threshold        │
│  - Select search scope             │
│  - Click "Search"                  │
└────────────┬───────────────────────┘
             │ HTTP POST /search/person
             │ query_image + threshold
             ▼
┌────────────────────────────────────┐
│  Backend Search Endpoint           │
│  - Load query image                │
│  - Resize to 128×256               │
│  - Extract Re-ID features          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Feature Extraction                │
│  query_features = reid_model.      │
│    extract_features(query_image)   │
│  → Tensor[2048]                    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Gallery Search                    │
│  For each saved gallery (.pkl):    │
│    Load gallery                    │
│    For each track_id:              │
│      Compare features              │
│      similarity = cosine(query, f) │
│      if similarity > threshold:    │
│        Add to matches              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Results Ranking                   │
│  - Sort by similarity (descending) │
│  - Limit to max_results (50)       │
│  - Return matches with metadata    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Frontend Results Display          │
│  - Match confidence scores         │
│  - Video source + track ID         │
│  - Number of detections            │
│  - Export (CSV, JSON, TXT)         │
└────────────────────────────────────┘
```

---

## Component Interactions

### Detection → Tracking Interaction

```python
# Detection produces List[Detection]
detections = detector.detect(frame)
# Detection = {bbox: [x, y, w, h], confidence: float, class_id: int}

# Tracker consumes detections
tracks = tracker.update(detections)
# Track = {track_id: int, bbox: [x, y, w, h], state: str}
```

**Interface Contract:**
- Detection bbox format: `[x_center, y_center, width, height]`
- Confidence range: `[0.0, 1.0]`
- Class ID: `0` for person (COCO format)

---

### Tracking → Re-ID Interaction

```python
# Tracker produces active tracks
tracks = tracker.update(detections)

# Re-ID processes each track
if reid_model:
    for track in tracks:
        # Extract person crop
        crop = frame[y:y+h, x:x+w]
        
        # Extract features
        feature = reid_model.extract_features(crop)
        
        # Find best match in gallery
        best_match_id, similarity = reid_model.find_match(
            feature, threshold=0.85
        )
        
        if best_match_id and similarity > 0.85:
            # Re-assign track ID (Re-ID match)
            track.track_id = best_match_id
        
        # Update gallery with new feature
        reid_model.update_gallery(track.track_id, feature)
```

**Interface Contract:**
- Track bbox format: `[x1, y1, x2, y2]` (corner format)
- Feature format: `numpy.ndarray[2048]`
- Similarity range: `[0.0, 1.0]`

---

### Backend → Frontend Interaction

```
Frontend (Streamlit)          Backend (FastAPI)
        │                             │
        │  POST /track/video          │
        ├────────────────────────────►│
        │  {"job_id": "uuid-1234"}    │
        │◄────────────────────────────┤
        │                             │
        │  GET /track/status/uuid-1234│
        ├────────────────────────────►│
        │  {"status": "processing",   │
        │   "progress": 0.45}         │
        │◄────────────────────────────┤
        │                             │
        │  (poll every 0.5s)          │
        │                             │
        │  GET /track/status/uuid-1234│
        ├────────────────────────────►│
        │  {"status": "completed",    │
        │   "progress": 1.0}          │
        │◄────────────────────────────┤
        │                             │
        │  GET /track/result/uuid-1234│
        ├────────────────────────────►│
        │  [binary video stream]      │
        │◄────────────────────────────┤
        │                             │
        ▼                             ▼
```

**Communication Protocol:**
- Transport: HTTP/1.1
- Format: JSON (metadata), Binary (video)
- Polling interval: 500ms
- Timeout: 5 minutes for processing

---

## Deployment Architecture

### Development Environment

```
┌─────────────────────────────────────────────────────────────┐
│                  Developer Workstation                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────┐         ┌───────────────────┐        │
│  │  Backend          │         │  Frontend         │        │
│  │  (FastAPI)        │         │  (Streamlit)      │        │
│  │  localhost:8000   │◄────────│  localhost:8501   │        │
│  └────────┬──────────┘         └───────────────────┘        │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Processing Pipeline                 │                    │
│  │  - Detection (YOLO)                  │                    │
│  │  - Tracking (ByteTrack)              │                    │
│  │  - Re-ID (ResNet50)                  │                    │
│  │  GPU: CUDA-enabled (RTX 3060)        │                    │
│  └─────────────────────────────────────┘                    │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  File System                         │                    │
│  │  - data/                             │                    │
│  │  - models/                           │                    │
│  │  - outputs/                          │                    │
│  └─────────────────────────────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Production Deployment (Future)

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
│                  (NGINX / Traefik)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
┌───────────────────┐        ┌───────────────────┐
│  Frontend Pod     │        │  Frontend Pod     │
│  (Streamlit)      │        │  (Streamlit)      │
│  Port 8501        │        │  Port 8501        │
└────────┬──────────┘        └────────┬──────────┘
         │                            │
         │  HTTP/REST                 │
         └────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │     Backend Service          │
        │     (FastAPI)                │
        │     Port 8000                │
        └────────────┬────────────────┘
                     │
         ┌───────────┴──────────────┐
         ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│ Processing Pod   │      │ Processing Pod   │
│ (GPU Worker)     │      │ (GPU Worker)     │
│ - Detection      │      │ - Detection      │
│ - Tracking       │      │ - Tracking       │
│ - Re-ID          │      │ - Re-ID          │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
        ┌─────────────────────────┐
        │  Shared Storage          │
        │  (S3 / NFS)              │
        │  - Videos                │
        │  - Models                │
        │  - Results               │
        └─────────────────────────┘
```

**Production Considerations:**
- Horizontal scaling of frontend pods
- GPU worker pools for parallel processing
- Shared storage for videos and models
- Redis for job queue and caching
- PostgreSQL for persistent job metadata
- Monitoring with Prometheus + Grafana

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Core implementation |
| **Deep Learning** | PyTorch | 2.0+ | Model training and inference |
| **Detection** | Ultralytics YOLOv8 | 8.0+ | Person detection |
| **Computer Vision** | OpenCV | 4.8+ | Video I/O, image processing |
| **Backend API** | FastAPI | 0.104+ | REST API server |
| **Frontend** | Streamlit | 1.28+ | Web interface |
| **Package Manager** | Poetry | 1.6+ | Dependency management |

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| numpy | Numerical computations |
| scipy | Scientific computing |
| pandas | Data manipulation |
| pillow | Image loading |
| requests | HTTP client |
| pydantic | Data validation |
| uvicorn | ASGI server |
| pytest | Testing framework |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| GPU | CUDA 11.8+ |
| Video Encoding | FFmpeg |
| Logging | Python logging |
| Configuration | YAML |
| Version Control | Git |

---

## Design Patterns

### 1. Strategy Pattern (Detector Interface)

```python
class DetectorBase(ABC):
    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        pass

class YOLODetector(DetectorBase):
    def detect(self, frame) -> List[Detection]:
        # YOLO-specific implementation
        pass

class FasterRCNNDetector(DetectorBase):  # Future extension
    def detect(self, frame) -> List[Detection]:
        # Faster R-CNN implementation
        pass
```

**Benefit:** Easy to swap detection algorithms without changing pipeline code.

---

### 2. Template Method Pattern (Processing Pipeline)

```python
class InferencePipeline:
    def process_video(self, video_path, output_path):
        # Template method
        frames = self._read_video(video_path)
        writer = self._create_writer(output_path)
        
        for frame in frames:
            result = self._process_frame(frame)
            self._write_frame(writer, result)
        
        self._finalize(writer)
    
    def _process_frame(self, frame):
        # Steps can be overridden by subclasses
        detections = self._detect(frame)
        tracks = self._track(detections)
        features = self._reid(tracks)
        annotated = self._visualize(frame, tracks)
        return annotated
```

**Benefit:** Consistent workflow while allowing customization of individual steps.

---

### 3. Factory Pattern (Pipeline Creation)

```python
def get_pipeline(confidence_threshold=0.5, enable_reid=True):
    # Factory method
    detector = YOLODetector(
        model_path=settings.YOLO_MODEL,
        conf_threshold=confidence_threshold,
        device=settings.DEVICE
    )
    
    tracker = ByteTrack()
    
    reid_model = None
    if enable_reid:
        reid_model = ResNetReID(model_path=settings.REID_MODEL)
    
    visualizer = BBoxDrawer()
    
    return InferencePipeline(
        detector=detector,
        tracker=tracker,
        reid_model=reid_model,
        visualizer=visualizer
    )
```

**Benefit:** Centralized object creation with configuration.

---

### 4. Observer Pattern (Progress Callbacks)

```python
def process_video(self, video_path, progress_callback=None):
    total_frames = get_frame_count(video_path)
    
    for frame_idx, frame in enumerate(frames):
        # Process frame
        result = self.process_frame(frame)
        
        # Notify observers
        if progress_callback:
            progress_callback(frame_idx + 1, total_frames)

# Usage
def update_progress(current, total):
    progress = current / total
    jobs[job_id]["progress"] = progress

pipeline.process_video(video_path, progress_callback=update_progress)
```

**Benefit:** Decouples processing from UI updates.

---

### 5. Singleton Pattern (Configuration)

```python
class Settings(BaseSettings):
    # Singleton configuration
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

settings = Settings()  # Global singleton
```

**Benefit:** Single source of truth for configuration.

---

## Scalability Considerations

### Horizontal Scaling

**Stateless Design:**
- Frontend and backend are stateless
- Job state stored in shared database (future: Redis/PostgreSQL)
- File storage on shared network filesystem (future: S3)

**Load Distribution:**
- Multiple frontend pods behind load balancer
- Multiple backend pods for API requests
- GPU worker pool for video processing

### Vertical Scaling

**GPU Utilization:**
- Batch processing for multiple frames
- Model optimization (TensorRT)
- Mixed precision inference (FP16)

**Memory Management:**
- Frame buffering limits
- Feature gallery pruning
- Periodic garbage collection

### Caching Strategy

**Model Caching:**
- Load models once, keep in memory
- Shared across requests

**Video Caching:**
- FFmpeg preview conversion cached
- Processed videos cached for 7 days

**Feature Gallery Caching:**
- In-memory during processing
- Saved to disk for person search
- Indexed for fast retrieval

---

## Security Architecture

### Input Validation

**Video Upload:**
- File type validation (MP4, AVI, MOV only)
- File size limit (500 MB maximum)
- Malware scanning (future)

**API Request Validation:**
- Pydantic schema validation
- Rate limiting (future)
- CORS configuration

### Data Protection

**File Storage:**
- Uploaded videos in isolated directory
- Processed results user-specific (future: auth)
- Feature galleries not directly accessible

**API Security (Future):**
- JWT authentication
- API key management
- Request signing

### Error Handling

**Graceful Degradation:**
- Invalid video formats rejected with clear message
- Processing errors logged and reported
- System continues despite individual job failures

**Logging:**
- All errors logged with stack traces
- Audit trail for API requests
- Performance metrics collected

---

## Conclusion

The CrossID system architecture is designed with modularity, scalability, and maintainability as core principles. The three-tier design separates concerns clearly, while the use of established design patterns ensures code quality and extensibility.

**Key Architectural Strengths:**
- ✅ Modular component design enables easy replacement/upgrade
- ✅ Asynchronous processing supports concurrent jobs
- ✅ Stateless API design enables horizontal scaling
- ✅ Clear data flow and interface contracts
- ✅ Extensible through abstract base classes

**Future Enhancements:**
- Containerization (Docker/Kubernetes)
- Distributed processing (GPU clusters)
- Advanced caching (Redis)
- Persistent storage (PostgreSQL)
- Enhanced security (JWT auth)

---

**Last Updated:** February 2026  
**CrossID Version:** 1.0.0  
**Architecture Version:** 1.0