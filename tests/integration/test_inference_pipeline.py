"""
Integration tests for inference pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from src.inference.pipeline import InferencePipeline
from src.visualization.bbox_drawer import BBoxDrawer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestInferencePipeline:
    """Tests for InferencePipeline."""

    @pytest.fixture
    def config_path(self):
        """Get config path."""
        return Path("configs/inference.yaml")

    def test_pipeline_init(self, config_path):
        """Test pipeline initialization."""
        if not config_path.exists():
            pytest.skip("Config file not found")

        pipeline = InferencePipeline(config_path)

        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.tracker is not None

    def test_pipeline_process_frame(self, config_path):
        """Test processing single frame."""
        if not config_path.exists():
            pytest.skip("Config file not found")

        pipeline = InferencePipeline(config_path)

        # Create dummy frame
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        tracks = pipeline.process_frame(frame, frame_id=1)

        assert isinstance(tracks, list)

    def test_pipeline_process_frame_empty(self, config_path):
        """Test processing empty frame."""
        if not config_path.exists():
            pytest.skip("Config file not found")

        pipeline = InferencePipeline(config_path)

        # Empty frame
        frame = np.array([])

        tracks = pipeline.process_frame(frame, frame_id=1)

        assert tracks == []


class TestBBoxDrawer:
    """Tests for BBoxDrawer."""

    def test_drawer_init(self):
        """Test drawer initialization."""
        drawer = BBoxDrawer(thickness=2, font_scale=0.5, draw_trails=False)

        assert drawer is not None
        assert drawer.thickness == 2
        assert drawer.font_scale == 0.5
        assert len(drawer.colors) > 0

    def test_draw_tracks(self):
        """Test drawing tracks on frame."""
        drawer = BBoxDrawer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [
            {"bbox": (100, 100, 200, 200), "track_id": 1},
            {"bbox": (300, 300, 400, 400), "track_id": 2},
        ]

        annotated = drawer.draw_tracks(frame, tracks)

        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, frame)  # Frame should be modified

    def test_draw_tracks_empty(self):
        """Test drawing with no tracks."""
        drawer = BBoxDrawer()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = []

        annotated = drawer.draw_tracks(frame, tracks)

        assert annotated.shape == frame.shape
        assert np.array_equal(annotated, frame)  # Frame should be unchanged

    def test_get_color_consistency(self):
        """Test that same track ID gets same color."""
        drawer = BBoxDrawer()

        color1 = drawer._get_color(5)
        color2 = drawer._get_color(5)

        assert color1 == color2
