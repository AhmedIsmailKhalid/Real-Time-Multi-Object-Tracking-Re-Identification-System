"""
CrossID FastAPI Backend Application.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from deployment.api.routes import router
from deployment.backend.core.config import settings
from deployment.backend.utils.logger import get_logger

# Add project root to path FIRST
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import backend modules using relative imports


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("="*60)
    logger.info(f"{settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("="*60)
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"YOLO Model: {settings.YOLO_MODEL}")
    logger.info(f"Re-ID Model: {settings.REID_MODEL}")
    logger.info("="*60)

    # DON'T pre-load models anymore (we create custom pipelines per request)
    logger.info("✓ Models will be loaded on-demand with custom settings")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

# Serve static files (outputs)
if settings.OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
