"""
Configuration management utilities.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict[str, Any], config_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
