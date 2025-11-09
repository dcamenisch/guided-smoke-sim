"""Tests for NPZ export functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestNPZExport:
    """Tests for NPZ file export."""

    def test_export_2d(self, sim_2d_small, tmp_path):
        """Test exporting 2D simulation state."""
        filepath = tmp_path / "test_2d.npz"

        # Run a few steps
        for _ in range(3):
            sim_2d_small.step()

        # Export
        sim_2d_small.export_to_npz(filepath, timestep=3)

        # Check file exists
        assert filepath.exists()

        # Load and verify
        data = np.load(filepath)
        assert data["ndim"] == 2
        assert data["nx"] == 16
        assert data["ny"] == 24
        assert "density" in data
        assert "pressure" in data
        assert "u_velocity" in data
        assert "v_velocity" in data
        assert "simulation_time" in data
        assert data["timestep"] == 3

    def test_export_3d(self, sim_3d_small, tmp_path):
        """Test exporting 3D simulation state."""
        filepath = tmp_path / "test_3d.npz"

        # Run a few steps
        for _ in range(3):
            sim_3d_small.step()

        # Export
        sim_3d_small.export_to_npz(filepath, timestep=3)

        # Check file exists
        assert filepath.exists()

        # Load and verify
        data = np.load(filepath)
        assert data["ndim"] == 3
        assert data["nx"] == 8
        assert data["ny"] == 12
        assert data["nz"] == 8
        assert "density" in data
        assert "w_velocity" in data
        assert "vorticity" in data
        assert data["timestep"] == 3

    def test_simulation_time_tracking(self, sim_2d_small, tmp_path):
        """Test that simulation time is properly tracked."""
        filepath = tmp_path / "test_time.npz"

        # Run multiple steps
        for _ in range(5):
            sim_2d_small.step()

        sim_time_before = sim_2d_small.simulation_time

        # Export
        sim_2d_small.export_to_npz(filepath, timestep=5)

        # Load and check
        data = np.load(filepath)
        assert data["simulation_time"] == sim_time_before
        assert data["simulation_time"] > 0.0

    def test_export_preserves_data_types(self, sim_2d_small, tmp_path):
        """Test that data types are preserved in export."""
        filepath = tmp_path / "test_dtypes.npz"

        sim_2d_small.step()
        sim_2d_small.export_to_npz(filepath)

        data = np.load(filepath)
        assert data["density"].dtype == np.float32
        assert data["pressure"].dtype == np.float32
        assert data["u_velocity"].dtype == np.float32


@pytest.fixture
def tmp_path():
    """Create temporary directory for test files."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir)
