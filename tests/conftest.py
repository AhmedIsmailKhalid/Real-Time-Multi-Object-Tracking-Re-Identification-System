"""
Pytest configuration and fixtures.
"""

import pytest  # noqa: F401


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption("--gpu", action="store_true", default=False, help="Run tests on GPU")
