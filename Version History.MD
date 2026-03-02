# Version 1.0
- Implemented `scripts/create_directory_structure.py` to create directory structures
- Implemented `pyproject.toml` to create pyproject.toml
- Implemented `scripts/create_init_files.py` to create init.py files in all Python package directories
- Implemented `.gitignore`
- Implemented `src/utils/logger.py` for logging configuration for the project
- Implemented `src/utils/config.py`
- Implemented `scripts/test_installation.py` to test dependencies installed correctly
- Implemented `scripts/download_market1501.py` to download market1501 dataset
- Implemented `scripts/preprocess_mot17.py` to process and save the mot17 dataset
- Implemented `scripts/preprocess_market1501.py` to process and save the market1501 dataset
- **Notes : Completed Data Preprocessing**
- **Version 1.0 zip file created**

---

# Version 1.1
- Implemented `src/detection/detector_base.py` for base detector interface
- Implemented `src/detection/yolo_detector.py` for yolov8 detector
- Implemented `src/detection/utils.py` detection utils
- Implemented `configs/detection.yaml` config for downloading pretrained models
- Implemented `scripts/download_pretrained_models.py` to download pretrained yolov8
- Implemented `tests/unit/test_detection.py` unit tests for detection models
- Implemented `tests/conftest.py` to for pytest configuration and fixtures
- **Notes : Completed Detection Module**
- **Version 1.1 zip file created**

---

# Version 1.2
- Implemented `src/tracking/tracker_base.py` for base tracker interface
- Implemented `src/tracking/kalman_filter.py` kalman filter for motion tracking
- Implemented `src/tracking/matching.py`for matching logic using hungarian algorithm
- Implemented `src/tracking/track.py` for track class to represent single object tracked
- Implemented `src/tracking/bytetrack.py` for byte tracking algorithm
- Implemented `configs/tracking.yaml` for tracking models configs
- Implemented `tests/unit/test_tracking.py` for tracking unit tests
- **Notes : Completed Tracking Module**
- **Version 1.1 zip file created**

---

# Version 1.3
- Implemented `src/reid/reid_base.py` for base re-id interface
- Implemented `src/reid/resnet_reid.py`
- Implemented `src/reid/metric_learning.py` for triplet loss
- Implemented `src/data/market_dataset.py` for market-1501 dataset loader
- Implemented `src/data/transforms.py` for dataset transformation
- Implemented `configs/reid.yaml` for re-id models configs
- Implemented `src/training/train_reid.py` for re-id training
- Implemented `scripts/train_reid.py` the script to train the re-id model
- Implemented `tests/unit/test_reid.py` for unit tests for re-id models
- **Notes : Implement Re-ID module with ResNet50**
- **Version 1.3 zip file created**

---

# Version 1.4
- Implemented `src/inference/pipeline.py` for e2e inference pipeline
- Implemented `configs/inference.yaml` for inference config
- Implemented `src/inference/video_processor.py` for video processing and visualization
- Implemented `src/visualization/bbox_drawer.py` for creating bounding box drawing and visualization
- Implemented `scripts/run_inference.py` for running inference of video
- Implemented `tests/integration/test_inference_pipeline.py` to test inference pipeline
- Implemented `scripts/test_pipeline_quick.py` for quick test of inference pipeline on single image
- **Notes : Implement End-to-End Inference Pipeline**
- **Version 1.4 zip file created**

---

# Version 1.5
- Implemented `src/evaluation/mot_metrics.py` for MOT dataset metrics
- Implemented `src/evaluation/reid_metrics.py` for re-id evaluation metrics
- Implemented `scripts/evaluate_mot.py` to evaluate mot17 validation set
- Implemented `scripts/evaluate_reid.py` to evaluate re-id
- Implemented `tests/unit/test_metrics.py` to test evaluation metrics
- Implemented `scripts/create_test_video.py` for mot17 dataset
- **Notes : Implemented Evaluation Metrics & Ran Inference**
- **Version 1.5 zip file created**

---

# Version 1.6
- Updated `configs/reid.yml` for more aggressive training for better results
- Implemented `scripts/debug_reid_eval.py` to find the bug in evaluation poor results
- Updated `src/evaluation/reid_metrics.py` to fix the evaluation bug to fix the format `{person_id}_c{cam}s{sequence}_{frame}_{track}.jpg`
- Implemented `scripts/prepare_demo_videos.py` to prepare the videos for end product
- **Notes : Fixed the Training Evaluation Bug Filtering All Same-Camera Matches and Created Demo Videos**
- **Version 1.6 zip file created**

---

# Version 1.7
- Implemented `deployment/backend/core/config.py` for cross-id backend configuration
- Implemented `deployment/backend/api/models.py` for API request/response models
- Implemented `deployment/backend/core/pipeline.py` for inference pipeline wrapper for API
- Implemented `deployment/backend/api/routes.py` for API routes for cross-id
- Implemented `deployment/backend/app.py` for FastAPI backend application
- Implemented `deployment/backend/__init__.py` for init
- Implemented `deployment/backend/api/__init__.py` for init
- Implemented `deployment/backend/core/__init__.py` for init
- Implemented `deployment/backend/utils/__init__.py` for init
- Implemented `deployment/backend/utils/logger.py` for logging backend
- Implemented `deployment/backend/.env.example` for backend .env example
- Implemented `deployment/frontend/utils/api_client.py` for React API client for cross-id backend
- Implemented `deployment/frontend/utils/video_utils.py`  for frontend video utilities
- Implemented `deployment/frontend/utils/__init__.py` for init 
- Implemented `deployment/frontend/app.py` for streamlit app
- Implemented `deployment/frontend/pages/demo_videos.py` for streamlit demo videos page
- Implemented `deployment/frontend/pages/upload_video.py` for streamlit upload video page
- Implemented `deployment/frontend/pages/multi_camera.py` for streamlit multi-camera page 
- Implemented `deployment/frontend/pages/person_search.py` for streamlit person_search page
- Implemented `deployment/frontend/pages/home.py` for streamlit homepage
- Updated `deployment/frontend/app.py` to remove `nav.run()` and add conditional logic with complete re-write
- Updated `deployment/frontend/pages/demo_videos.py` to add button to go back to homepage
- Updated `deployment/frontend/pages/upload_video.py` to add button to go back to homepage
- Updated `deployment/frontend/pages/multi_camera.py` to add button to go back to homepage
- Updated `deployment/frontend/pages/person_search.py` to add button to go back to homepage
- Updated `deployment/backend/core/pipeline.py` to update pipeline error to fix processing issue
- Updated `src/inference/pipeline.py` to fix size mismatch for ResNet50 + updated tensor conversion
- Updated `deployment/frontend/pages/demo_videos.py` to fix the progress bar
- Updated `src/inference/pipeline.py` to fixplaying embedded processed video + add re-id gallery matching
- Updated `src/detection/yolo_detector.py` to use faster inference
- Updated `src/reid/resnet_reid.py` to add half precision support
- Updated `deployment/api/routes.py` to use callback for progress bar
- **Notes : Created Streamlit for frontend and FastAPI for backend + other small fixes**
- **Version 1.7 zip file created**

---

# Version 1.8
- Updated `deployment/backend/core/pipeline.py` to ensure confidence threshold is being passed from streamlit
- Updated `deployment/api/routes.py` to ensure sidebar values is being passed from streamlit
- Updated `deployment/backend/app.py` to update startup to not preload
- Updated `deployment/frontend/pages/upload_video.py` to ensure embedded video plays
- Updated `src/inference/pipeline.py` to ensure values are being used from streamlit
- Updated `deployment/api/routes.py` to ensure reid is being set from streamlit
- Updated `deployment/frontend/utils/api_client.py` to fix reid being enabled/disabled from streamlit
- Updated `deployment/frontend/pages/demo_videos.py` to fix widgets properly
- Updated `deployment/frontend/pages/demo_videos.py` to prevent session state loss
- Updated `deployment/frontend/utils/api_client.py` to convert bools to lowercase strings for form serialization
- Updated `deployment/api/routes.py` for correct parsing
- Updated `deployment/backend/core/pipeline.py` to prevent false matches in single-camera videos
- Updated `src/inference/pipeline.py` to add 30-frame min gallery age + fix Windows VideoWriter codec fallback
- Updated `src/reid/resnet_reid.py` to fix deprecated autocast syntax warning
- Updated `deployment/frontend/pages/upload_video.py` to fix video preview after file uploaded
- **Notes : Fixed small bugs to refine app and proper passing of values from streamlit to FastAPI** 
- **Version 1.8 zip file created**

---

# Version 1.9
- Updated `deployment/frontend/pages/upload_video.py` to remove Process Another Video button post-processing
- Updated `deployment/frontend/pages/demo_videosd.py` to remove Process Another Video button post-processing
- Updated `deployment/api/routes.py` to add new endpoint for multi-camera tracking
- Updated `deployment/api/models.py` to update job status to support multi-camera results + add video download
- Updated `deployment/frontend/utils/api_client.py` to add multi-camera video upload processing
- Updated `deployment/api/routes.py` to fix job status for multi-camera support
- Updated `deployment/frontend/pages/multi_camera.py` to fix progress bars for each camera
- Updated `.dvcignore` to ignore datasets tracking as they are public datasets that will not change
- Updated `deployment/api/routes.py` to save feature gallery
- Updated `deployment/frontend/utils/api_client.py` to add search person method
- Updated `deployment/frontend/pages/person_search.py` to make minor additions 
- Updated `deployment/api/routes.py` to save feature gallery correctly
- Updated `deployment/frontend/pages/person_search.py` to adjust matching and downloading results
- **Notes : Fixed small bugs to refine app and added multi-camera tracking functionality + Added Person Search Functionality** 
- **Version 1.9 zip file created**