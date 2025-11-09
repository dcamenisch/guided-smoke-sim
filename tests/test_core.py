"""Tests for core data structures (MAC grids)."""

import pytest
import numpy as np
from core import MACGrid2D, MACGrid3D


class TestMACGrid2D:
    """Tests for 2D MAC grid."""

    def test_initialization(self):
        """Test MAC grid initialization."""
        grid = MACGrid2D(nx=16, ny=24, dx=0.1)

        assert grid.nx == 16
        assert grid.ny == 24
        assert grid.dx == 0.1

        # Check u velocity shape (ny, nx+1)
        assert grid.u_data.shape == (24, 17)

        # Check v velocity shape (ny+1, nx)
        assert grid.v_data.shape == (25, 16)

        # Check initialized to zero
        assert np.allclose(grid.u_data, 0.0)
        assert np.allclose(grid.v_data, 0.0)

    def test_reset(self):
        """Test reset functionality."""
        grid = MACGrid2D(nx=16, ny=24, dx=0.1)

        # Set some values
        grid.u_data[:] = 1.0
        grid.v_data[:] = 2.0

        # Reset
        grid.reset()

        # Check reset to zero
        assert np.allclose(grid.u_data, 0.0)
        assert np.allclose(grid.v_data, 0.0)

    def test_dtype(self):
        """Test data types are float32."""
        grid = MACGrid2D(nx=16, ny=24, dx=0.1)

        assert grid.u_data.dtype == np.float32
        assert grid.v_data.dtype == np.float32


class TestMACGrid3D:
    """Tests for 3D MAC grid."""

    def test_initialization(self):
        """Test MAC grid initialization."""
        grid = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)

        assert grid.nx == 8
        assert grid.ny == 12
        assert grid.nz == 8
        assert grid.dx == 0.1

        # Check u velocity shape (nz, ny, nx+1)
        assert grid.u_data.shape == (8, 12, 9)

        # Check v velocity shape (nz, ny+1, nx)
        assert grid.v_data.shape == (8, 13, 8)

        # Check w velocity shape (nz+1, ny, nx)
        assert grid.w_data.shape == (9, 12, 8)

        # Check initialized to zero
        assert np.allclose(grid.u_data, 0.0)
        assert np.allclose(grid.v_data, 0.0)
        assert np.allclose(grid.w_data, 0.0)

    def test_reset(self):
        """Test reset functionality."""
        grid = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)

        # Set some values
        grid.u_data[:] = 1.0
        grid.v_data[:] = 2.0
        grid.w_data[:] = 3.0

        # Reset
        grid.reset()

        # Check reset to zero
        assert np.allclose(grid.u_data, 0.0)
        assert np.allclose(grid.v_data, 0.0)
        assert np.allclose(grid.w_data, 0.0)

    def test_dtype(self):
        """Test data types are float32."""
        grid = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)

        assert grid.u_data.dtype == np.float32
        assert grid.v_data.dtype == np.float32
        assert grid.w_data.dtype == np.float32
