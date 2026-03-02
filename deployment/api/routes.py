"""
API routes for CrossID.
"""

import io
import pickle
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from deployment.backend.core.config import settings
from deployment.backend.core.pipeline import PipelineWrapper
from deployment.backend.utils.logger import get_logger

from .models import (
    JobStatus,
    TrackingRequest,
    TrackingResponse,
)

logger = get_logger(__name__)

router = APIRouter()

pipeline: PipelineWrapper | None = None

jobs: dict[str, dict] = {}


def _parse_bool(value: str) -> bool:
    """Parse boolean from form string value."""
    return str(value).lower() in ("true", "1", "yes")


def get_pipeline(
    confidence_threshold: float = 0.5,
    enable_reid: bool = True
) -> PipelineWrapper:
    """Get or create pipeline instance with custom settings."""
    global pipeline

    reid_model_path = str(settings.REID_MODEL) if enable_reid else None

    pipeline = PipelineWrapper(
        yolo_model_path=str(settings.YOLO_MODEL),
        reid_model_path=reid_model_path,
        device=settings.DEVICE,
        enable_reid=enable_reid,
        confidence_threshold=confidence_threshold
    )

    return pipeline


def save_feature_gallery(job_id: str, gallery: dict, video_filename: str):
    """Save feature gallery to disk for person search."""
    try:
        logger.info(f"=== ATTEMPTING to save gallery for {job_id} ===")
        logger.info(f"Video filename: {video_filename}")
        logger.info(f"Gallery has {len(gallery)} tracks")

        features_dir = settings.OUTPUT_DIR / "features"
        logger.info(f"Features directory path: {features_dir}")

        features_dir.mkdir(exist_ok=True)
        logger.info(f"Features directory exists: {features_dir.exists()}")

        gallery_data = {
            "job_id": job_id,
            "video_filename": video_filename,
            "timestamp": time.time(),
            "gallery": gallery
        }

        gallery_path = features_dir / f"{job_id}_features.pkl"
        logger.info(f"Saving to: {gallery_path}")

        with open(gallery_path, "wb") as f:
            pickle.dump(gallery_data, f)

        file_size = gallery_path.stat().st_size
        logger.info(f"✓ SUCCESS: Saved {file_size} bytes to {gallery_path}")

    except Exception as e:
        logger.error(f"✗ FAILED to save feature gallery for {job_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


def process_video_task(job_id: str, video_path: str, output_path: str, request: TrackingRequest):
    """Background task for video processing."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.0

        def progress_callback(current: int, total: int):
            jobs[job_id]["progress"] = current / total

        pipeline = get_pipeline(
            confidence_threshold=request.confidence_threshold,
            enable_reid=request.enable_reid
        )

        results = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            enable_reid=request.enable_reid,
            show_trails=request.show_trails,
            progress_callback=progress_callback
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result_path"] = output_path
        jobs[job_id]["stats"] = results

        # === DEBUG: Check gallery saving conditions ===
        logger.info("=== DEBUG: Checking if we should save gallery ===")
        logger.info(f"request.enable_reid = {request.enable_reid}")
        logger.info(f"pipeline type = {type(pipeline)}")
        logger.info(f"hasattr(pipeline, 'pipeline') = {hasattr(pipeline, 'pipeline')}")

        if hasattr(pipeline, 'pipeline'):
            logger.info(f"pipeline.pipeline type = {type(pipeline.pipeline)}")
            logger.info(f"hasattr(pipeline.pipeline, 'feature_gallery') = {hasattr(pipeline.pipeline, 'feature_gallery')}")

            if hasattr(pipeline.pipeline, 'feature_gallery'):
                gallery = pipeline.pipeline.feature_gallery
                logger.info(f"feature_gallery type = {type(gallery)}")
                logger.info(f"feature_gallery length = {len(gallery)}")
                logger.info(f"feature_gallery keys sample = {list(gallery.keys())[:5] if gallery else 'empty'}")

        # Save feature gallery for person search (only if Re-ID was enabled)
        if request.enable_reid and hasattr(pipeline.pipeline, 'feature_gallery'):
            video_filename = Path(video_path).name
            logger.info("=== CONDITION MET: Calling save_feature_gallery ===")
            save_feature_gallery(
                job_id,
                pipeline.pipeline.feature_gallery,
                video_filename
            )
        else:
            logger.warning("=== CONDITION NOT MET: Gallery NOT saved ===")

    except Exception as e:
        logger.error(f"Error processing video {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)


def process_multi_camera_task(
    job_id: str,
    video_paths: list[str],
    output_paths: list[str],
    request: TrackingRequest
):
    """Background task for multi-camera video processing with shared Re-ID gallery."""
    try:
        jobs[job_id]["status"] = "processing"

        pipeline = get_pipeline(
            confidence_threshold=request.confidence_threshold,
            enable_reid=request.enable_reid
        )

        all_results = []
        total_videos = len(video_paths)

        for idx, (video_path, output_path) in enumerate(zip(video_paths, output_paths, strict=False)):
            logger.info(f"Multi-camera job {job_id}: Processing camera {idx+1}/{total_videos}")

            jobs[job_id]["video_status"][idx] = "processing"

            def progress_callback(current: int, total: int, _idx: int = idx):
                video_progress = current / total if total > 0 else 0
                jobs[job_id]["video_progress"][_idx] = video_progress
                overall_progress = sum(jobs[job_id]["video_progress"]) / total_videos
                jobs[job_id]["progress"] = overall_progress

            results = pipeline.process_video(
                video_path=video_path,
                output_path=output_path,
                enable_reid=request.enable_reid,
                show_trails=request.show_trails,
                progress_callback=progress_callback
            )

            results["camera_id"] = idx + 1
            results["output_path"] = str(output_path)
            all_results.append(results)

            jobs[job_id]["video_status"][idx] = "completed"
            jobs[job_id]["video_progress"][idx] = 1.0

            logger.info(
                f"Multi-camera job {job_id}: Camera {idx+1} complete - "
                f"{results.get('unique_tracks', 0)} tracks, "
                f"{results.get('reid_matches', 0)} Re-ID matches"
            )

        total_unique_tracks = max(r.get("unique_tracks", 0) for r in all_results)
        total_reid_matches = sum(r.get("reid_matches", 0) for r in all_results)
        avg_fps = sum(r.get("avg_fps", 0) for r in all_results) / total_videos
        total_time = sum(r.get("processing_time", 0) for r in all_results)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result_paths"] = [str(p) for p in output_paths]
        jobs[job_id]["stats"] = {
            "num_cameras": total_videos,
            "unique_tracks": total_unique_tracks,
            "total_reid_matches": total_reid_matches,
            "avg_fps": avg_fps,
            "total_processing_time": total_time,
            "per_camera_stats": all_results
        }

        logger.info(
            f"Multi-camera job {job_id} complete: "
            f"{total_unique_tracks} unique tracks across {total_videos} cameras, "
            f"{total_reid_matches} total Re-ID matches"
        )

        # Save feature gallery for person search (multi-camera shared gallery)
        if request.enable_reid and hasattr(pipeline, 'pipeline') and hasattr(pipeline.pipeline, 'feature_gallery'):
            # Save once with the main job_id since gallery is shared across cameras
            video_filename = f"multi_camera_{len(video_paths)}_views"
            save_feature_gallery(
                job_id,
                pipeline.pipeline.feature_gallery,
                video_filename
            )

        for idx, p in enumerate(jobs[job_id]["result_paths"]):
            exists = Path(p).exists()
            logger.info(f"Camera {idx+1} output: {p} (exists={exists})")

    except Exception as e:
        logger.error(f"Error processing multi-camera job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)


@router.post("/track/video", response_model=TrackingResponse)
async def track_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    enable_reid: str = Form("true"),
    show_trails: str = Form("true"),
    confidence_threshold: float = Form(0.5)
):
    """Upload and process video for tracking."""
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Invalid video format. Use MP4, AVI, or MOV")

    enable_reid_bool = _parse_bool(enable_reid)
    show_trails_bool = _parse_bool(show_trails)

    logger.info(
        f"Received job: enable_reid={enable_reid!r} -> {enable_reid_bool}, "
        f"confidence={confidence_threshold}"
    )

    job_id = str(uuid.uuid4())

    upload_path = settings.UPLOAD_DIR / f"{job_id}_{video.filename}"
    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    output_path = settings.OUTPUT_DIR / f"{job_id}_tracked.mp4"

    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Video uploaded successfully"
    }

    logger.info(
        f"Job {job_id}: enable_reid={enable_reid_bool}, confidence={confidence_threshold}"
    )

    request = TrackingRequest(
        enable_reid=enable_reid_bool,
        confidence_threshold=confidence_threshold,
        show_trails=show_trails_bool
    )

    background_tasks.add_task(
        process_video_task,
        job_id=job_id,
        video_path=str(upload_path),
        output_path=str(output_path),
        request=request
    )

    return TrackingResponse(
        job_id=job_id,
        status="queued",
        message="Video uploaded and queued for processing"
    )


@router.post("/track/multi-camera", response_model=TrackingResponse)
async def track_multi_camera(
    background_tasks: BackgroundTasks,
    videos: list[UploadFile] = File(...),
    enable_reid: str = Form("true"),
    show_trails: str = Form("true"),
    confidence_threshold: float = Form(0.5)
):
    """Upload and process multiple videos for multi-camera tracking."""
    if len(videos) < 2:
        raise HTTPException(400, "Multi-camera tracking requires at least 2 videos")
    if len(videos) > 4:
        raise HTTPException(400, "Multi-camera tracking supports maximum 4 videos")

    for video in videos:
        if not video.filename.endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(400, f"Invalid video format for {video.filename}. Use MP4, AVI, or MOV")

    enable_reid_bool = _parse_bool(enable_reid)
    show_trails_bool = _parse_bool(show_trails)

    logger.info(
        f"Received multi-camera job: {len(videos)} videos, "
        f"enable_reid={enable_reid!r} -> {enable_reid_bool}, confidence={confidence_threshold}"
    )

    job_id = str(uuid.uuid4())

    upload_paths = []
    for idx, video in enumerate(videos):
        upload_path = settings.UPLOAD_DIR / f"{job_id}_cam{idx+1}_{video.filename}"
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        upload_paths.append(upload_path)

    output_paths = [
        settings.OUTPUT_DIR / f"{job_id}_cam{idx+1}_tracked.mp4"
        for idx in range(len(videos))
    ]

    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Videos uploaded successfully",
        "num_videos": len(videos),
        "video_progress": [0.0] * len(videos),
        "video_status": ["queued"] * len(videos)
    }

    logger.info(f"Multi-camera job {job_id}: {len(videos)} videos queued")

    request = TrackingRequest(
        enable_reid=enable_reid_bool,
        confidence_threshold=confidence_threshold,
        show_trails=show_trails_bool
    )

    background_tasks.add_task(
        process_multi_camera_task,
        job_id=job_id,
        video_paths=[str(p) for p in upload_paths],
        output_paths=[str(p) for p in output_paths],
        request=request
    )

    return TrackingResponse(
        job_id=job_id,
        status="queued",
        message=f"{len(videos)} videos uploaded and queued for multi-camera processing"
    )


@router.get("/track/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of tracking job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job.get("message"),
        result_path=job.get("result_path"),
        result_paths=job.get("result_paths"),
        stats=job.get("stats"),
        num_videos=job.get("num_videos"),
        video_progress=job.get("video_progress"),
        video_status=job.get("video_status")
    )


@router.get("/track/result/{job_id}")
async def get_result_video(job_id: str):
    """Download processed video."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed. Status: {job['status']}")

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        path=result_path,
        media_type="video/mp4",
        filename=f"tracked_{job_id}.mp4"
    )


@router.get("/track/multi-camera-result/{job_id}/{camera_id}")
async def get_multi_camera_result(job_id: str, camera_id: int):
    """Download processed video for specific camera."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    logger.info(f"Multi-camera result request: job={job_id}, camera={camera_id}")
    logger.info(f"Job status: {job['status']}")
    logger.info(f"Result paths: {job.get('result_paths')}")

    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed. Status: {job['status']}")

    result_paths = job.get("result_paths") or []

    logger.info(f"Number of result paths: {len(result_paths)}, requested camera_id: {camera_id}")

    if camera_id < 1 or camera_id > len(result_paths):
        raise HTTPException(
            400,
            f"Invalid camera_id={camera_id}. Available cameras: 1-{len(result_paths)}"
        )

    result_path = Path(result_paths[camera_id - 1])

    logger.info(f"Looking for file: {result_path}, exists: {result_path.exists()}")

    if not result_path.exists():
        raise HTTPException(404, f"Result file not found: {result_path}")

    return FileResponse(
        path=result_path,
        media_type="video/mp4",
        filename=f"tracked_cam{camera_id}_{job_id}.mp4"
    )


@router.post("/search/person")
async def search_person(
    query_image: UploadFile = File(...),
    similarity_threshold: float = Form(0.85),
    max_results: int = Form(50)
):
    """
    Search for a person across all processed videos.
    Requires pre-cropped image of the target person.
    """
    try:
        # Read and validate image
        image_bytes = await query_image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to Re-ID input size
        image = image.resize((128, 256))
        image_array = np.array(image).astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()

        # Ensure Re-ID model is loaded
        global pipeline
        if pipeline is None or not hasattr(pipeline, 'pipeline') or pipeline.pipeline.reid_model is None:
            logger.info("Loading Re-ID model for person search...")
            pipeline = get_pipeline(enable_reid=True)

        if pipeline.pipeline.reid_model is None:
            raise HTTPException(400, "Re-ID model failed to load")

        device = torch.device(settings.DEVICE)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            query_features = pipeline.pipeline.reid_model.extract_features(image_tensor)
            query_features = query_features.cpu().numpy().flatten()

        # Search through all saved feature galleries
        features_dir = settings.OUTPUT_DIR / "features"
        if not features_dir.exists():
            return {"matches": [], "message": "No feature galleries found. Process some videos first."}

        matches = []

        for gallery_file in features_dir.glob("*_features.pkl"):
            try:
                with open(gallery_file, "rb") as f:
                    gallery_data = pickle.load(f)

                job_id = gallery_data["job_id"]
                video_filename = gallery_data.get("video_filename", "unknown")
                gallery = gallery_data["gallery"]

                # Compare query against each track in the gallery
                for track_id, track_data in gallery.items():
                    track_features = track_data.get("features", [])

                    if not track_features:
                        continue

                    # Compute max similarity across all features for this track
                    similarities = []
                    for feature in track_features:
                        similarity = np.dot(query_features, feature) / (
                            np.linalg.norm(query_features) * np.linalg.norm(feature) + 1e-10
                        )
                        similarities.append(float(similarity))

                    max_similarity = max(similarities)

                    if max_similarity >= similarity_threshold:
                        matches.append({
                            "job_id": job_id,
                            "video_filename": video_filename,
                            "track_id": track_id,
                            "similarity": round(max_similarity, 3),
                            "num_detections": len(track_features)
                        })

            except Exception as e:
                logger.error(f"Error loading gallery {gallery_file}: {e}")
                continue

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit results
        matches = matches[:max_results]

        logger.info(f"Person search complete: {len(matches)} matches found (threshold={similarity_threshold})")

        return {
            "matches": matches,
            "query_processed": True,
            "total_matches": len(matches),
            "threshold": similarity_threshold
        }

    except Exception as e:
        logger.error(f"Person search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")  # noqa: B904


@router.get("/debug/features-location")
async def debug_features_location():
    """Debug: Show where features are being saved."""
    features_dir = settings.OUTPUT_DIR / "features"

    return {
        "PROJECT_ROOT": str(settings.PROJECT_ROOT),
        "OUTPUT_DIR": str(settings.OUTPUT_DIR),
        "features_dir": str(features_dir),
        "features_dir_exists": features_dir.exists(),
        "features_dir_absolute": str(features_dir.absolute()),
        "files": [str(f.name) for f in features_dir.glob("*.pkl")] if features_dir.exists() else []
    }


@router.get("/debug/galleries")
async def debug_galleries():
    """Debug endpoint to see saved feature galleries."""
    features_dir = settings.OUTPUT_DIR / "features"

    if not features_dir.exists():
        return {"error": "No features directory found"}

    galleries = []
    for gallery_file in features_dir.glob("*_features.pkl"):
        try:
            with open(gallery_file, "rb") as f:
                gallery_data = pickle.load(f)

            galleries.append({
                "file": gallery_file.name,
                "job_id": gallery_data["job_id"],
                "video": gallery_data.get("video_filename"),
                "tracks": len(gallery_data["gallery"]),
                "timestamp": gallery_data.get("timestamp")
            })
        except Exception as e:
            galleries.append({"file": gallery_file.name, "error": str(e)})

    return {"total_galleries": len(galleries), "galleries": galleries}



@router.get("/demo/videos")
async def list_demo_videos():
    """List available demo videos."""
    demo_videos = []
    if settings.DEMO_VIDEOS_DIR.exists():
        for video in settings.DEMO_VIDEOS_DIR.glob("*.mp4"):
            demo_videos.append({
                "filename": video.name,
                "path": str(video),
                "size_mb": round(video.stat().st_size / (1024 * 1024), 2)
            })
    return {"videos": demo_videos}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": pipeline is not None
    }
