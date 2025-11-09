"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationConfig
from core import MACGrid2D, MACGrid3D


@pytest.fixture
def sim_2d_small():
    """Small 2D simulator for fast tests."""
    config = SimulationConfig(nx=16, ny=24, dt=0.01, max_iterations=10)
    return config.create_simulator()


@pytest.fixture
def sim_2d_medium():
    """Medium 2D simulator for accuracy tests."""
    config = SimulationConfig(nx=64, ny=96, dt=0.01, max_iterations=100)
    return config.create_simulator()


@pytest.fixture
def sim_3d_small():
    """Small 3D simulator for fast tests."""
    config = SimulationConfig(nx=8, ny=12, nz=8, dt=0.01, max_iterations=10)
    return config.create_simulator()


@pytest.fixture
def sim_3d_medium():
    """Medium 3D simulator for accuracy tests."""
    config = SimulationConfig(nx=32, ny=48, nz=32, dt=0.01, max_iterations=100)
    return config.create_simulator()


@pytest.fixture
def macgrid_2d():
    """2D MAC grid."""
    return MACGrid2D(nx=16, ny=24, dx=1.0 / 16)


@pytest.fixture
def macgrid_3d():
    """3D MAC grid."""
    return MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)


@pytest.fixture
def random_velocity_field_2d():
    """Random 2D velocity field for testing."""
    np.random.seed(42)
    u = np.random.randn(24, 17).astype(np.float32) * 0.1
    v = np.random.randn(25, 16).astype(np.float32) * 0.1
    return u, v


@pytest.fixture
def random_velocity_field_3d():
    """Random 3D velocity field for testing."""
    np.random.seed(42)
    u = np.random.randn(8, 12, 9).astype(np.float32) * 0.1
    v = np.random.randn(8, 13, 8).astype(np.float32) * 0.1
    w = np.random.randn(9, 12, 8).astype(np.float32) * 0.1
    return u, v, w
