"""
API request/response models.
"""


from pydantic import BaseModel, Field


class TrackingRequest(BaseModel):
    """Request for video tracking."""
    enable_reid: bool = Field(default=True, description="Enable Re-ID matching")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    show_trails: bool = Field(default=True, description="Show movement trails")


class TrackingResponse(BaseModel):
    """Response from video tracking."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    message: str | None = None
    result_path: str | None = None
    result_paths: list[str] | None = None  # ← Add this for multi-camera
    stats: dict | None = None
    num_videos: int | None = None  # ← Add this
    video_progress: list[float] | None = None  # ← Add this
    video_status: list[str] | None = None  # ← Add this


class TrackingStats(BaseModel):
    """Tracking statistics."""
    total_frames: int
    unique_tracks: int
    reid_matches: int
    processing_time: float
    avg_fps: float
    video_duration: float


class SearchRequest(BaseModel):
    """Person search request."""
    query_image: str  # Base64 encoded or path
    video_path: str
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Person search result."""
    track_id: int
    confidence: float
    timestamps: list[float]
    frame_numbers: list[int]
