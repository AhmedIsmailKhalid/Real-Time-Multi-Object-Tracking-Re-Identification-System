"""
Create __init__.py files in all Python package directories.
"""

import os
from pathlib import Path


def create_init_files():
    """Create __init__.py in all src/ subdirectories."""

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    directories = [
        "src",
        "src/detection",
        "src/tracking",
        "src/reid",
        "src/data",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/visualization",
        "src/utils",
        "api",
        "api/routes",
        "api/schemas",
        "api/middleware",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
    ]

    for directory in directories:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        print(f"Created: {init_file}")

    print(f"\n__init__.py files created in: {project_root}")


if __name__ == "__main__":
    create_init_files()
