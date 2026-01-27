# System Architecture - Multi-Object Tracking & Re-Identification System

## Document Purpose
This document provides a high-level overview of the system architecture, component interactions, data flow, and integration points. This is NOT implementation-level detail (that lives in code). This is the 30,000-foot view of how the system works.

---

## System Overview

### What This System Does

**Input:** Video stream or video file  
**Output:** Video with tracked objects, annotated bounding boxes, persistent IDs, re-identification across occlusions

**Core Capabilities:**
1. Detect people in each frame (YOLOv8)
2. Track detected people across frames with persistent IDs (ByteTrack)
3. Re-identify people after occlusion or camera changes (ResNet50 Re-ID)
4. Visualize results with bounding boxes and track IDs
5. Serve predictions via REST API
6. Monitor and log system performance

---

## High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐              │
│  │   Gradio     │  │   FastAPI     │  │    CLI       │              │
│  │   Web UI     │  │   REST API    │  │   Scripts    │              │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Video Processor                             │ │
│  │  (Frame extraction, preprocessing, result aggregation)         │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  Detection Module (YOLOv8)                     │ │
│  │  Input: Frame (HxWx3)                                          │ │
│  │  Output: Bounding boxes [(x,y,w,h), confidence, class]         │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  Tracking Module (ByteTrack)                   │ │
│  │  Input: Detections + Previous tracks                           │ │
│  │  Output: Tracked objects with IDs [(x,y,w,h), track_id]        │ │
│  │  Uses: Kalman filter (motion) + IoU matching                   │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Re-ID Module (ResNet50)                           │ │
│  │  Input: Person crops from bounding boxes                       │ │
│  │  Output: 512-dim feature vectors per person                    │ │
│  │  Uses: Cosine similarity for identity matching                 │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Association & Re-ID Integration                   │ │
│  │  - Match lost tracks using Re-ID features                      │ │
│  │  - Handle occlusions and re-appearances                        │ │
│  │  - Multi-camera identity linking                               │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  Visualization Module                          │ │
│  │  - Draw bounding boxes with track IDs                          │ │
│  │  - Color-code tracks for visual distinction                    │ │
│  │  - Render trajectory trails (optional)                         │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Output Video   │
                  │  with Tracking  │
                  └─────────────────┘
```

---

## Detailed Component Architecture

### 1. Detection Component
```
YOLODetector
├── Input: RGB frame (numpy array, HxWx3)
├── Processing:
│   ├── Resize to model input size (640x640)
│   ├── Normalize pixel values (0-1)
│   ├── Run YOLOv8 inference
│   └── Apply NMS (Non-Maximum Suppression)
├── Output: List of detections
│   └── Each detection: (x1, y1, x2, y2, confidence, class_id)
└── Configuration:
    ├── Confidence threshold: 0.5
    ├── NMS IoU threshold: 0.45
    └── Model variant: YOLOv8s (11M params)
```

**Key Responsibilities:**
- Object detection in single frame (stateless)
- Bounding box regression and classification
- Confidence scoring
- Class filtering (person class only for this project)

**Integration Points:**
- **Input:** Video frames from video processor
- **Output:** Detections to tracking module

---

### 2. Tracking Component
```
ByteTracker
├── Input: Current frame detections + Previous frame tracks
├── Processing:
│   ├── Motion Prediction (Kalman Filter)
│   │   ├── Predict track positions in current frame
│   │   └── Update with matched detections
│   ├── First Association (High Confidence)
│   │   ├── Match high-conf detections to existing tracks (IoU)
│   │   └── Create new tracks for unmatched high-conf detections
│   ├── Second Association (Low Confidence)
│   │   ├── Match low-conf detections to unmatched tracks
│   │   └── Recover lost tracks using low-conf detections
│   └── Track Management
│       ├── Remove tracks lost for > N frames
│       ├── Assign new track IDs
│       └── Update track state (active, lost, removed)
├── Output: List of tracked objects
│   └── Each track: (x1, y1, x2, y2, track_id, state)
└── Configuration:
    ├── High confidence threshold: 0.6
    ├── Low confidence threshold: 0.1
    ├── Track buffer: 30 frames
    └── Match IoU threshold: 0.8
```

**Key Responsibilities:**
- Maintain persistent track IDs across frames
- Handle occlusions and re-appearances (without Re-ID)
- Motion prediction using Kalman filtering
- Data association (matching detections to tracks)

**Integration Points:**
- **Input:** Detections from detection module
- **Output:** Tracked objects to Re-ID module (for feature extraction)
- **Output:** Tracked objects to visualization module

---

### 3. Re-Identification Component
```
ReIDNetwork (ResNet50)
├── Input: Person crops (bounding boxes from tracks)
├── Processing:
│   ├── Resize crops to 256x128
│   ├── Normalize (ImageNet mean/std)
│   ├── Extract features via ResNet50 backbone
│   └── L2 normalize feature vectors
├── Output: 512-dim feature vector per person
└── Configuration:
    ├── Backbone: ResNet50 pre-trained on ImageNet
    ├── Fine-tuned on: Market-1501
    ├── Feature dimension: 512
    └── Distance metric: Cosine similarity
```

**Feature Matching:**
```
ReIDMatcher
├── Input: Current frame features + Track feature gallery
├── Processing:
│   ├── Compute pairwise cosine distances
│   ├── Find closest match below distance threshold
│   └── Re-associate lost tracks based on appearance
├── Output: Re-associated track IDs
└── Configuration:
    ├── Distance threshold: 0.3
    └── Match confidence: 0.8
```

**Key Responsibilities:**
- Extract appearance features from person crops
- Match people based on visual similarity
- Handle identity preservation across occlusions
- Enable multi-camera re-identification

**Integration Points:**
- **Input:** Person crops from tracking module
- **Output:** Feature vectors to association logic
- **Output:** Re-association suggestions to tracking module

---

### 4. Data Flow Through System
```
Frame N-1          Frame N          Frame N+1
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│Detection│      │Detection│      │Detection│
└────┬────┘      └────┬────┘      └────┬────┘
     │                │                 │
     │  Detections    │  Detections     │  Detections
     │  [(x,y,w,h)]   │  [(x,y,w,h)]    │  [(x,y,w,h)]
     ▼                ▼                 ▼
┌─────────────────────────────────────────────┐
│            Tracking (ByteTrack)              │
│  - Match detections to existing tracks       │
│  - Predict motion (Kalman filter)            │
│  - Assign IDs                                │
│  - Manage track lifecycle                    │
└────┬────────────────────────────────────┬───┘
     │                                    │
     │ Active Tracks                      │ Lost Tracks
     │ [(x,y,w,h, id)]                    │ [(x,y,w,h, id)]
     ▼                                    ▼
┌─────────┐                          ┌─────────┐
│ Re-ID   │                          │ Re-ID   │
│ Feature │                          │ Matcher │
│Extract  │                          │         │
└────┬────┘                          └────┬────┘
     │                                    │
     │ Features                           │ Re-association
     │ [512-dim vectors]                  │ [(old_id, new_id)]
     ▼                                    │
┌─────────────────┐                       │
│ Feature Gallery │◄──────────────────────┘
│ (Track history) │
└─────────────────┘
     │
     │ Updated Tracks
     │ [(x,y,w,h, id, features)]
     ▼
┌──────────────┐
│Visualization │
└──────────────┘
     │
     ▼
  Output Frame
```

---

### 5. Training Architecture
```
Training Pipeline (Separate from Inference)

┌─────────────────────────────────────────────────────────────┐
│                    Detection Training                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │   MOT17    │───▶│  YOLOv8    │───▶│ Checkpoints│        │
│  │  Dataset   │    │  Trainer   │    │   (*.pt)   │        │
│  └────────────┘    └────────────┘    └────────────┘        │
│       │                  │                    │              │
│       │                  ▼                    │              │
│       │           ┌────────────┐              │              │
│       │           │  MLflow    │◄─────────────┘              │
│       │           │  Tracking  │                             │
│       │           └────────────┘                             │
└───────┼──────────────────────────────────────────────────────┘
        │
        │
┌───────┼──────────────────────────────────────────────────────┐
│       │           Re-ID Training                              │
│       ▼                                                       │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │ Market-1501│───▶│  ResNet50  │───▶│ Checkpoints│        │
│  │  Dataset   │    │  Trainer   │    │   (*.pth)  │        │
│  └────────────┘    └────────────┘    └────────────┘        │
│                          │                    │              │
│                          ▼                    │              │
│                   ┌────────────┐              │              │
│                   │  MLflow    │◄─────────────┘              │
│                   │  Tracking  │                             │
│                   └────────────┘                             │
└─────────────────────────────────────────────────────────────┘

Models exported to: models/detection/final/ and models/reid/final/
Then loaded by inference pipeline
```

**Training Components:**
- **DetectionTrainer:** Handles YOLOv8 fine-tuning on MOT17
- **ReIDTrainer:** Handles ResNet50 training on Market-1501
- **MLflow Integration:** Logs metrics, hyperparameters, artifacts
- **Checkpoint Management:** Saves best models, handles resuming

**Key Separation:**
- Training and inference are SEPARATE pipelines
- Training produces models, inference loads models
- No training code runs during inference

---

### 6. API Architecture (Deployment)
```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Web UI   │  │  Mobile  │  │  CLI     │                  │
│  │ (Gradio) │  │   App    │  │  Tool    │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼─────────────┼────────────────────────┘
        │             │             │
        │   HTTP/REST │             │
        └─────────────┼─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                     Routes                            │   │
│  │  /health          - Health check                      │   │
│  │  /inference       - Single frame/video inference      │   │
│  │  /upload          - Upload video file                 │   │
│  │  /metrics         - System metrics                    │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Inference Pipeline                       │   │
│  │  (Loads models, processes video, returns results)     │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Model Manager                            │   │
│  │  - Load models on startup                             │   │
│  │  - Cache models in memory                             │   │
│  │  - Handle model versioning                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**API Design Principles:**
- **Stateless:** Each request is independent
- **Async:** FastAPI async/await for concurrent requests
- **Model caching:** Load models once at startup, reuse
- **Error handling:** Proper HTTP status codes and error messages
- **Logging:** Request/response logging for debugging
- **Monitoring:** Prometheus metrics endpoint

---

### 7. Component Interaction Sequence

**Typical Inference Flow:**
```
1. Video Frame Arrives
   └─▶ VideoProcessor extracts frame

2. Detection Phase
   └─▶ YOLODetector processes frame
       └─▶ Returns: [(x1,y1,x2,y2, conf, class), ...]

3. Tracking Phase
   └─▶ ByteTrack receives detections
       ├─▶ Kalman filter predicts track positions
       ├─▶ IoU matching associates detections to tracks
       ├─▶ Assigns/updates track IDs
       └─▶ Returns: [(x1,y1,x2,y2, track_id), ...]

4. Re-ID Phase (for each tracked person)
   └─▶ Crop person from frame using bbox
       └─▶ ReIDNetwork extracts features
           └─▶ Returns: [512-dim vector]

5. Re-Association Phase
   └─▶ ReIDMatcher compares features
       ├─▶ If lost track matches current detection
       │   └─▶ Re-assign old track ID
       └─▶ Update feature gallery

6. Visualization Phase
   └─▶ BBoxDrawer renders bounding boxes
       ├─▶ Color-code by track ID
       ├─▶ Draw track ID text
       └─▶ Returns: annotated frame

7. Output
   └─▶ Write frame to output video
```

---

## Module Dependency Graph
```
                    ┌─────────────┐
                    │   utils/    │
                    │  (logging,  │
                    │   config)   │
                    └──────┬──────┘
                           │
                           │ (used by all)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  detection/  │  │   tracking/  │  │     reid/    │
│  (YOLOv8)    │  │  (ByteTrack) │  │  (ResNet50)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       │                 │                  │
       └────────┬────────┴────────┬─────────┘
                │                 │
                ▼                 ▼
         ┌──────────────┐  ┌──────────────┐
         │  inference/  │  │ evaluation/  │
         │  (Pipeline)  │  │  (Metrics)   │
         └──────┬───────┘  └──────────────┘
                │
                ▼
         ┌──────────────┐
         │visualization/│
         │  (Rendering) │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │     api/     │
         │  (FastAPI)   │
         └──────────────┘
```

**Dependency Rules:**
- Lower layers (utils) have NO dependencies on upper layers
- detection, tracking, reid are INDEPENDENT (can be developed in parallel)
- inference depends on detection, tracking, reid
- api depends on inference (NOT on detection/tracking/reid directly)
- evaluation is INDEPENDENT (only for metrics calculation)

**Import Convention:**
```python
# Good (respects dependency graph)
from src.detection.yolo_detector import YOLODetector
from src.tracking.bytetrack import ByteTracker
from src.inference.pipeline import InferencePipeline

# Bad (violates dependency - api should NOT import detection directly)
# In api/routes/inference.py:
from src.detection.yolo_detector import YOLODetector  # BAD
# Instead use:
from src.inference.pipeline import InferencePipeline   # GOOD
```

---

## Deployment Architecture

### Local Development
```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose                         │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   FastAPI   │  │   MLflow    │  │  Grafana    │     │
│  │   Service   │  │   Server    │  │  (Optional) │     │
│  │   (GPU)     │  │   (CPU)     │  │   (CPU)     │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                 │             │
│         │ Port 8000      │ Port 5000       │ Port 3000   │
│         └────────────────┼─────────────────┘             │
│                          │                               │
│  ┌──────────────────────┴─────────────────────────┐     │
│  │         Shared Volumes                          │     │
│  │  - data/                                        │     │
│  │  - models/                                      │     │
│  │  - outputs/                                     │     │
│  │  - mlruns/                                      │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

**Access:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000

---

### Cloud Deployment (Production)
```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                            │
│                   (Nginx/CloudFlare)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Container Orchestration (K8s/Docker)            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        API Pods (Horizontal Scaling)                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │ API Pod 1│  │ API Pod 2│  │ API Pod N│           │  │
│  │  │  (GPU)   │  │  (GPU)   │  │  (GPU)   │           │  │
│  │  └──────────┘  └──────────┘  └──────────┘           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Persistent Storage (S3/GCS)                 │  │
│  │  - Models (versioned)                                 │  │
│  │  - DVC remote storage                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Monitoring & Logging                          │  │
│  │  - Prometheus (metrics)                               │  │
│  │  - Grafana (dashboards)                               │  │
│  │  - CloudWatch/Stackdriver (logs)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Scaling Strategy:**
- Horizontal: Add more API pods for increased throughput
- Vertical: Increase GPU memory for larger batch sizes
- Caching: Redis for repeated inference requests
- CDN: CloudFlare for static asset delivery

---

## Performance Considerations

### Bottleneck Analysis

**Expected bottlenecks (in order):**

1. **Re-ID Feature Extraction (60-70% of time)**
   - 30 people/frame × 0.5ms = 15ms
   - Mitigation: Batch processing, ONNX optimization

2. **Detection (20-25% of time)**
   - YOLOv8s: ~20ms per frame on RTX 3060
   - Mitigation: Use YOLOv8n for faster inference, ONNX export

3. **Tracking Association (5-10% of time)**
   - ByteTrack: O(n²) matching for n objects
   - 30 objects: ~0.5ms
   - 150 objects (MOT20): ~8ms
   - Mitigation: Limit max objects, optimize matching algorithm

4. **Visualization (5% of time)**
   - Drawing boxes: ~2ms per frame
   - Mitigation: Disable for production (only return coordinates)

**Optimization Priority:**
1. Export models to ONNX (1.5-3x speedup)
2. Batch Re-ID feature extraction
3. Use smaller YOLO variant (YOLOv8n) if real-time critical
4. Disable visualization in production API

---

### Throughput Targets

| Hardware | Expected FPS | Batch Size | Avg People/Frame |
|----------|--------------|------------|------------------|
| RTX 3060 (12GB) | 35-50 | 1 | 20-30 |
| RTX 4090 (24GB) | 80-120 | 2-4 | 20-30 |
| CPU (16 cores) | 3-5 | 1 | 20-30 |
| T4 (Cloud) | 25-35 | 1 | 20-30 |

**Real-time definition:** 25-30 FPS (matches video frame rate)

---

## Failure Modes & Handling

### Detection Failures
**Symptom:** No detections in frame  
**Cause:** Poor lighting, extreme viewing angle, occlusion  
**Handling:** 
- Tracker continues with motion prediction (Kalman filter)
- Track marked as "lost" but maintained for N frames
- Re-ID attempts recovery when person reappears

### Tracking Failures
**Symptom:** ID switches, lost tracks  
**Cause:** Fast motion, crowded scenes, long occlusions  
**Handling:**
- ByteTrack's two-stage association reduces ID switches
- Re-ID module attempts re-association for lost tracks
- Track pruning after N frames prevents zombie tracks

### Re-ID Failures
**Symptom:** Incorrect identity matches  
**Cause:** Similar appearance, viewpoint changes, clothing changes  
**Handling:**
- Distance threshold prevents low-confidence matches
- Temporal consistency check (reject if contradicts motion)
- Fall back to motion-based tracking if Re-ID unreliable

### System Failures
**Symptom:** Out of memory, GPU errors  
**Cause:** Too many simultaneous requests, memory leak  
**Handling:**
- Request queuing with max queue size
- Graceful degradation (reduce batch size)
- Health check endpoint for monitoring
- Automatic restart on crash (Docker/K8s)

---

## Security Considerations

### Input Validation
- Video format validation (only mp4, avi, mov)
- File size limits (max 500MB per upload)
- Frame dimension limits (max 4K resolution)
- Malicious file detection (magic number check)

### API Security
- Rate limiting (100 requests/min per IP)
- Authentication (API key for production)
- CORS configuration (whitelist origins)
- Input sanitization (prevent injection attacks)

### Data Privacy
- No storage of uploaded videos (process and delete)
- No facial recognition (only appearance features)
- Anonymized track IDs (no PII)
- GDPR-compliant data handling

---

## Monitoring & Observability

### Metrics to Track

**System Metrics:**
- API request rate (requests/second)
- API latency (P50, P95, P99)
- GPU utilization (%)
- Memory usage (GB)
- Disk I/O (MB/s)

**ML Metrics:**
- Inference FPS (frames/second)
- Detection count per frame
- Average track duration (frames)
- Re-ID match rate (%)

**Business Metrics:**
- Videos processed (count)
- Total processing time (hours)
- Error rate (%)
- User retention (if applicable)

### Logging Strategy

**Log Levels:**
- DEBUG: Detailed component states (development only)
- INFO: Normal operations (requests, processing start/end)
- WARNING: Recoverable issues (low FPS, queue full)
- ERROR: Component failures (model load error, GPU error)
- CRITICAL: System failures (out of memory, crash)

**Structured Logging:**
```json
{
  "timestamp": "2026-01-22T10:15:30Z",
  "level": "INFO",
  "component": "inference.pipeline",
  "message": "Processing video completed",
  "metadata": {
    "video_id": "abc123",
    "frames": 1500,
    "fps": 42.3,
    "avg_people": 28
  }
}
```

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies (e.g., mock detector in tracker tests)
- Fast execution (< 1 second per test)
- Coverage target: 70%+

### Integration Tests
- Test component interactions
- Use real models (small versions)
- Slower execution (acceptable)
- Coverage: Critical paths (inference pipeline)

### End-to-End Tests
- Test full system (video in → annotated video out)
- Run on sample videos
- Verify output quality (manual review)
- Regression testing (compare to baseline)

---

## Future Extensibility

### Easy to Add:
- New detector (implement DetectorBase interface)
- New tracker (implement TrackerBase interface)
- New Re-ID model (implement ReIDBase interface)
- New API endpoints (add to api/routes/)

### Architecture Supports:
- Multi-camera fusion (extend association logic)
- Online learning (update Re-ID features on-the-fly)
- Action recognition (add new module, integrate with tracking)
- Pose estimation (add new module, use tracked boxes)

---

## Summary

**System Type:** Multi-stage computer vision pipeline  
**Architecture Pattern:** Modular, component-based  
**Deployment Model:** Containerized microservices  
**Scalability:** Horizontal (multiple API pods) and vertical (GPU upgrade)  
**Maintainability:** High (clear separation, well-defined interfaces)  
**Testability:** High (mockable components, isolated testing)

**Key Design Principles Followed:**
1. Separation of concerns (each module has one job)
2. Dependency injection (components don't hardcode dependencies)
3. Interface-based design (swap implementations easily)
4. Configuration-driven (behavior controlled via configs)
5. Observable (comprehensive logging and metrics)