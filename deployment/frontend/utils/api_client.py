"""
API client for CrossID backend.
"""

import time
from pathlib import Path
from typing import Any

import requests


class CrossIDClient:
    """Client for CrossID API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            base_url: Backend API URL
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"

    def health_check(self) -> dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()

    def list_demo_videos(self) -> list[dict[str, Any]]:
        """Get list of demo videos."""
        response = requests.get(f"{self.api_url}/demo/videos")
        response.raise_for_status()
        return response.json()["videos"]

    def upload_video(
        self,
        video_path: str,
        enable_reid: bool = True,
        show_trails: bool = True,
        confidence_threshold: float = 0.5
    ) -> str:
        """
        Upload video for processing.

        Args:
            video_path: Path to video file
            enable_reid: Enable Re-ID
            show_trails: Show movement trails
            confidence_threshold: Detection confidence threshold

        Returns:
            Job ID
        """
        # Explicitly convert to lowercase string - Python bool serialization
        # sends "True"/"False" (capital) which can cause parsing issues.
        # Sending "true"/"false" is unambiguous and matches _parse_bool() in routes.py.
        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {
                "enable_reid": "true" if enable_reid else "false",
                "show_trails": "true" if show_trails else "false",
                "confidence_threshold": confidence_threshold
            }

            response = requests.post(
                f"{self.api_url}/track/video",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()["job_id"]

    def upload_multi_camera_videos(
        self,
        video_paths: list[str],
        enable_reid: bool = True,
        show_trails: bool = True,
        confidence_threshold: float = 0.5
    ) -> str:
        """Upload multiple videos for multi-camera processing."""
        # Open files and create tuple list for requests
        files = []
        file_handles = []

        try:
            for path in video_paths:
                f = open(path, "rb")
                file_handles.append(f)
                files.append(("videos", (Path(path).name, f, "video/mp4")))

            data = {
                "enable_reid": "true" if enable_reid else "false",
                "show_trails": "true" if show_trails else "false",
                "confidence_threshold": confidence_threshold
            }

            response = requests.post(
                f"{self.api_url}/track/multi-camera",
                files=files,
                data=data
            )
            response.raise_for_status()

            result = response.json()
            return result["job_id"]

        finally:
            # Close all file handles
            for f in file_handles:
                try:
                    f.close()
                except Exception:
                    pass

    def download_multi_camera_result(self, job_id: str, camera_id: int, output_path: str):
        """Download result video for specific camera."""
        response = requests.get(
            f"{self.api_url}/track/multi-camera-result/{job_id}/{camera_id}",
            stream=True
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get job status."""
        response = requests.get(f"{self.api_url}/track/status/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_id: str, poll_interval: float = 1.0) -> dict[str, Any]:
        """
        Wait for job completion.

        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds

        Returns:
            Final job status
        """
        while True:
            status = self.get_job_status(job_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(poll_interval)

    def download_result(self, job_id: str, output_path: str):
        """Download result video."""
        response = requests.get(
            f"{self.api_url}/track/result/{job_id}",
            stream=True
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def search_person(
        self,
        query_image_path: str,
        similarity_threshold: float = 0.85,
        max_results: int = 50
    ) -> dict:
        """Search for a person across processed videos."""
        with open(query_image_path, "rb") as f:
            files = {"query_image": f}
            data = {
                "similarity_threshold": similarity_threshold,
                "max_results": max_results
            }

            response = requests.post(
                f"{self.api_url}/search/person",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
