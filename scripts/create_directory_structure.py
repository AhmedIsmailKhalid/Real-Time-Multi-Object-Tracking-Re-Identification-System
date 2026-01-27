"""
Create complete project directory structure.
Run this script from project root.
"""

import os
from pathlib import Path


def create_directories():
    """Create all project directories."""

    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    directories = [
        # Data directories
        "data/raw",
        "data/processed",
        "data/external/test_videos",
        "data/external/test_images",
        # Model directories
        "models/detection/pretrained",
        "models/detection/checkpoints",
        "models/detection/final",
        "models/reid/pretrained",
        "models/reid/checkpoints",
        "models/reid/final",
        "models/tracking/bytetrack_configs",
        "models/exported/tensorrt",
        # Source code directories
        "src/detection",
        "src/tracking",
        "src/reid",
        "src/data",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/visualization",
        "src/utils",
        # API directories
        "api/routes",
        "api/schemas",
        "api/middleware",
        # Test directories
        "tests/unit",
        "tests/integration",
        "tests/fixtures/sample_images",
        "tests/fixtures/sample_videos",
        # Deployment directories
        "deployment/docker",
        "deployment/kubernetes",
        "deployment/cloud/aws",
        "deployment/cloud/gcp",
        # Config, scripts, notebooks, docs
        "configs",
        "scripts",
        "notebooks",
        "docs",
        # Output directories
        "outputs/experiments/detection_experiments",
        "outputs/experiments/reid_experiments",
        "outputs/results/videos",
        "outputs/results/images",
        "outputs/results/metrics",
        "outputs/visualizations/training_curves",
        "outputs/visualizations/evaluation_plots",
        "outputs/visualizations/demo_outputs",
        "outputs/logs/training",
        "outputs/logs/inference",
        "outputs/logs/api",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

    print(f"\nDirectory structure created in: {project_root}")
    print("Total directories created: " + str(len(directories)))


if __name__ == "__main__":
    create_directories()
