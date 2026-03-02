"""
CrossID Backend Configuration.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_TITLE: str = "CrossID API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Multi-Camera Object Tracking with Person Re-Identification"

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DEMO_VIDEOS_DIR: Path = PROJECT_ROOT / "data" / "external" / "demo_videos"
    UPLOAD_DIR: Path = PROJECT_ROOT / "outputs" / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "processed"

    # Model Paths
    YOLO_MODEL: Path = MODELS_DIR / "detection" / "pretrained" / "yolov8s.pt"
    REID_MODEL: Path = MODELS_DIR / "reid" / "final" / "resnet50_market_best.pth"

    # Processing Settings
    DEVICE: str = "cuda"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    ENABLE_REID: bool = True
    MAX_VIDEO_SIZE_MB: int = 500

    # CORS
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create directories
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
