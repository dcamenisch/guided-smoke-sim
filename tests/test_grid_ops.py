"""Tests for grid operations."""

import pytest
import numpy as np
from kernels import grid_ops
from core import MACGrid2D, MACGrid3D


class TestVelocityInterpolation2D:
    """Tests for 2D velocity interpolation to cell centers."""

    def test_interpolate_velocity_uniform(self):
        """Test interpolation of uniform velocity field."""
        ny, nx = 16, 16
        u = np.ones((ny, nx + 1), dtype=np.float32) * 2.0
        v = np.ones((ny + 1, nx), dtype=np.float32) * 3.0

        # Interpolate at interior cell
        vel_x, vel_y = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, 5, 5)

        # Uniform field should interpolate exactly
        assert abs(vel_x - 2.0) < 1e-6
        assert abs(vel_y - 3.0) < 1e-6

    def test_interpolate_velocity_varying(self):
        """Test interpolation with varying velocity."""
        ny, nx = 16, 16
        u = np.zeros((ny, nx + 1), dtype=np.float32)
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        # Set specific values for averaging test
        u[5, 5] = 1.0
        u[5, 6] = 3.0
        v[5, 5] = 2.0
        v[6, 5] = 4.0

        vel_x, vel_y = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, 5, 5)

        # Should be average of adjacent faces
        assert abs(vel_x - 2.0) < 1e-6  # (1.0 + 3.0) / 2
        assert abs(vel_y - 3.0) < 1e-6  # (2.0 + 4.0) / 2


class TestVelocityInterpolation3D:
    """Tests for 3D velocity interpolation to cell centers."""

    def test_interpolate_velocity_3d_uniform(self):
        """Test 3D interpolation of uniform velocity field."""
        nz, ny, nx = 8, 8, 8
        u = np.ones((nz, ny, nx + 1), dtype=np.float32) * 2.0
        v = np.ones((nz, ny + 1, nx), dtype=np.float32) * 3.0
        w = np.ones((nz + 1, ny, nx), dtype=np.float32) * 4.0

        vel_x, vel_y, vel_z = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, 3, 3, 3)

        assert abs(vel_x - 2.0) < 1e-6
        assert abs(vel_y - 3.0) < 1e-6
        assert abs(vel_z - 4.0) < 1e-6

    def test_interpolate_velocity_3d_varying(self):
        """Test 3D interpolation with varying velocity."""
        nz, ny, nx = 8, 8, 8
        u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        u[3, 3, 3] = 1.0
        u[3, 3, 4] = 5.0
        v[3, 3, 3] = 2.0
        v[3, 4, 3] = 6.0
        w[3, 3, 3] = 3.0
        w[4, 3, 3] = 7.0

        vel_x, vel_y, vel_z = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, 3, 3, 3)

        assert abs(vel_x - 3.0) < 1e-6  # (1 + 5) / 2
        assert abs(vel_y - 4.0) < 1e-6  # (2 + 6) / 2
        assert abs(vel_z - 5.0) < 1e-6  # (3 + 7) / 2


class TestFaceInterpolation2D:
    """Tests for interpolation between MAC grid faces."""

    def test_interpolate_v_to_u_face(self):
        """Test interpolation of v-velocity to u-face."""
        ny, nx = 16, 16
        v = np.ones((ny + 1, nx), dtype=np.float32) * 2.0

        # Should average 4 surrounding values
        result = grid_ops.interpolate_v_to_u_face_2d(v, 5, 5)
        assert abs(result - 2.0) < 1e-6

        # Test with varying values
        v.fill(0.0)
        v[5, 5] = 1.0
        v[5, 4] = 2.0
        v[6, 4] = 3.0
        v[6, 5] = 4.0

        result = grid_ops.interpolate_v_to_u_face_2d(v, 5, 5)
        expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0
        assert abs(result - expected) < 1e-6

    def test_interpolate_u_to_v_face(self):
        """Test interpolation of u-velocity to v-face."""
        ny, nx = 16, 16
        u = np.ones((ny, nx + 1), dtype=np.float32) * 3.0

        result = grid_ops.interpolate_u_to_v_face_2d(u, 5, 5)
        assert abs(result - 3.0) < 1e-6


class TestClamping2D:
    """Tests for boundary clamping operations."""

    def test_clamp_to_cell_center_interior(self):
        """Test that interior points are not clamped."""
        nx, ny = 32, 32

        # Interior point should not change
        x, y = grid_ops.clamp_to_cell_center_2d(15.5, 15.5, nx, ny)
        assert abs(x - 15.5) < 1e-6
        assert abs(y - 15.5) < 1e-6

    def test_clamp_to_cell_center_boundaries(self):
        """Test that boundary points are clamped correctly."""
        nx, ny = 32, 32

        # Test clamping at boundaries
        x, y = grid_ops.clamp_to_cell_center_2d(-1.0, 15.0, nx, ny)
        assert x >= 0.5  # Should be clamped to valid range
        assert abs(y - 15.0) < 1e-6

        x, y = grid_ops.clamp_to_cell_center_2d(35.0, 15.0, nx, ny)
        assert x <= nx - 1.5  # Should be clamped to valid range

        x, y = grid_ops.clamp_to_cell_center_2d(15.0, -1.0, nx, ny)
        assert y >= 0.5

        x, y = grid_ops.clamp_to_cell_center_2d(15.0, 35.0, nx, ny)
        assert y <= ny - 1.5

    def test_clamp_to_u_face(self):
        """Test clamping to u-face boundaries."""
        nx, ny = 32, 32

        # Interior should not change
        x, y = grid_ops.clamp_to_u_face_2d(15.5, 15.5, nx, ny)
        assert abs(x - 15.5) < 1e-6
        assert abs(y - 15.5) < 1e-6

        # Out of bounds should be clamped
        x, y = grid_ops.clamp_to_u_face_2d(-1.0, 15.0, nx, ny)
        assert x >= 0.5

        x, y = grid_ops.clamp_to_u_face_2d(40.0, 15.0, nx, ny)
        assert x <= nx - 0.5  # u goes to nx

    def test_clamp_to_v_face(self):
        """Test clamping to v-face boundaries."""
        nx, ny = 32, 32

        x, y = grid_ops.clamp_to_v_face_2d(15.5, 15.5, nx, ny)
        assert abs(x - 15.5) < 1e-6
        assert abs(y - 15.5) < 1e-6

        x, y = grid_ops.clamp_to_v_face_2d(15.0, 40.0, nx, ny)
        assert y <= ny - 0.5  # v goes to ny


class TestClamping3D:
    """Tests for 3D boundary clamping operations."""

    def test_clamp_to_cell_center_3d_interior(self):
        """Test 3D interior clamping."""
        nx, ny, nz = 16, 16, 16

        x, y, z = grid_ops.clamp_to_cell_center_3d(8.0, 8.0, 8.0, nx, ny, nz)
        assert abs(x - 8.0) < 1e-6
        assert abs(y - 8.0) < 1e-6
        assert abs(z - 8.0) < 1e-6

    def test_clamp_to_cell_center_3d_boundaries(self):
        """Test 3D boundary clamping."""
        nx, ny, nz = 16, 16, 16

        # Test all boundaries
        x, y, z = grid_ops.clamp_to_cell_center_3d(-1.0, 8.0, 8.0, nx, ny, nz)
        assert x >= 0.5

        x, y, z = grid_ops.clamp_to_cell_center_3d(20.0, 8.0, 8.0, nx, ny, nz)
        assert x <= nx - 1.5

        x, y, z = grid_ops.clamp_to_cell_center_3d(8.0, -1.0, 8.0, nx, ny, nz)
        assert y >= 0.5

        x, y, z = grid_ops.clamp_to_cell_center_3d(8.0, 20.0, 8.0, nx, ny, nz)
        assert y <= ny - 1.5

        x, y, z = grid_ops.clamp_to_cell_center_3d(8.0, 8.0, -1.0, nx, ny, nz)
        assert z >= 0.5

        x, y, z = grid_ops.clamp_to_cell_center_3d(8.0, 8.0, 20.0, nx, ny, nz)
        assert z <= nz - 1.5


class TestForceOperations:
    """Tests for force application and reset."""

    def test_reset_forces_2d(self):
        """Test resetting 2D forces to zero."""
        force = MACGrid2D(nx=16, ny=24, dx=0.1)

        # Set some non-zero values
        force.u_data.fill(5.0)
        force.v_data.fill(3.0)

        grid_ops.reset_forces_2d(force)

        assert np.allclose(force.u_data, 0.0)
        assert np.allclose(force.v_data, 0.0)

    def test_reset_forces_3d(self):
        """Test resetting 3D forces to zero."""
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)

        force.u_data.fill(5.0)
        force.v_data.fill(3.0)
        force.w_data.fill(2.0)

        grid_ops.reset_forces_3d(force)

        assert np.allclose(force.u_data, 0.0)
        assert np.allclose(force.v_data, 0.0)
        assert np.allclose(force.w_data, 0.0)

    def test_apply_force_to_velocity_2d(self):
        """Test applying force to 2D velocity."""
        velocity = MACGrid2D(nx=16, ny=24, dx=0.1)
        force = MACGrid2D(nx=16, ny=24, dx=0.1)

        # Set initial velocity
        velocity.u_data.fill(1.0)
        velocity.v_data.fill(2.0)

        # Set forces
        force.u_data.fill(10.0)
        force.v_data.fill(20.0)

        dt = 0.1
        grid_ops.apply_force_to_velocity_2d(velocity, force, dt)

        # Velocity should increase by force * dt
        # Expected: v_new = v_old + f * dt
        expected_u = 1.0 + 10.0 * 0.1
        expected_v = 2.0 + 20.0 * 0.1

        assert np.allclose(velocity.u_data, expected_u)
        assert np.allclose(velocity.v_data, expected_v)

    def test_apply_force_to_velocity_3d(self):
        """Test applying force to 3D velocity."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=0.1)

        velocity.u_data.fill(1.0)
        velocity.v_data.fill(2.0)
        velocity.w_data.fill(3.0)

        force.u_data.fill(10.0)
        force.v_data.fill(20.0)
        force.w_data.fill(30.0)

        dt = 0.1
        grid_ops.apply_force_to_velocity_3d(velocity, force, dt)

        expected_u = 1.0 + 10.0 * 0.1
        expected_v = 2.0 + 20.0 * 0.1
        expected_w = 3.0 + 30.0 * 0.1

        assert np.allclose(velocity.u_data, expected_u)
        assert np.allclose(velocity.v_data, expected_v)
        assert np.allclose(velocity.w_data, expected_w)


class TestAveragingOperations:
    """Tests for averaging operations."""

    def test_average_density_to_faces_2d(self):
        """Test averaging density from cell centers to faces."""
        ny, nx = 16, 16
        density = np.ones((ny, nx), dtype=np.float32) * 5.0

        u_avg = np.zeros((ny, nx + 1), dtype=np.float32)
        v_avg = np.zeros((ny + 1, nx), dtype=np.float32)

        grid_ops.average_center_to_u_faces_2d(density, u_avg, ny, nx)
        grid_ops.average_center_to_v_faces_2d(density, v_avg, ny, nx)

        # Uniform density should give uniform average
        # Interior faces should be 5.0 (boundaries may be 0)
        assert np.allclose(u_avg[1:-1, 1:-1], 5.0)
        assert np.allclose(v_avg[1:-1, 1:-1], 5.0)

    def test_average_density_varying_2d(self):
        """Test averaging with varying density field."""
        ny, nx = 16, 16
        density = np.zeros((ny, nx), dtype=np.float32)

        # Set specific values
        density[5, 5] = 2.0
        density[5, 6] = 4.0

        u_avg = np.zeros((ny, nx + 1), dtype=np.float32)
        grid_ops.average_center_to_u_faces_2d(density, u_avg, ny, nx)

        # u-face between cells (5,5) and (5,6) should be average
        expected = (2.0 + 4.0) / 2.0
        assert abs(u_avg[5, 6] - expected) < 1e-5
