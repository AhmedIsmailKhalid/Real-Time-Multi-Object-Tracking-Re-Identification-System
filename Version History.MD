# Version 1.0
- Implemented `scripts\create_directory_structure.py` to create directory structures
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
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``
- Implemented ``

---