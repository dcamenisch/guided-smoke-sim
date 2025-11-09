"""Tests for velocity correction kernels."""

import pytest
import numpy as np
from kernels.velocity import correct_velocity_kernel_2d, correct_velocity_kernel_3d


class TestVelocityCorrection2D:
    """Tests for 2D velocity correction."""

    def test_correct_velocity_zero_pressure(self):
        """Test that zero pressure produces no correction."""
        ny, nx = 32, 32
        u = np.random.rand(ny, nx + 1).astype(np.float32)
        v = np.random.rand(ny + 1, nx).astype(np.float32)
        u_orig = u.copy()
        v_orig = v.copy()

        pressure = np.zeros((ny, nx), dtype=np.float32)

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # With zero pressure, velocity should not change
        np.testing.assert_allclose(u, u_orig, rtol=1e-6)
        np.testing.assert_allclose(v, v_orig, rtol=1e-6)

    def test_correct_velocity_uniform_pressure(self):
        """Test that uniform pressure produces no correction."""
        ny, nx = 32, 32
        u = np.random.rand(ny, nx + 1).astype(np.float32)
        v = np.random.rand(ny + 1, nx).astype(np.float32)
        u_orig = u.copy()
        v_orig = v.copy()

        # Uniform pressure (constant everywhere)
        pressure = np.ones((ny, nx), dtype=np.float32) * 10.0

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # Uniform pressure has zero gradient, no correction
        # Interior should be unchanged (boundaries may change)
        np.testing.assert_allclose(u[1:-1, :], u_orig[1:-1, :], rtol=1e-6)
        np.testing.assert_allclose(v[:, 1:-1], v_orig[:, 1:-1], rtol=1e-6)

    def test_correct_velocity_linear_pressure_x(self):
        """Test velocity correction with linear pressure gradient in x."""
        ny, nx = 32, 32
        u = np.ones((ny, nx + 1), dtype=np.float32) * 5.0  # Initial u velocity
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        # Linear pressure field in x: p = x
        pressure = np.zeros((ny, nx), dtype=np.float32)
        for x in range(nx):
            pressure[:, x] = float(x)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # Pressure gradient: dp/dx = 1.0 everywhere
        # Velocity correction: u_new = u_old - dt * dp/dx = 5.0 - 0.1 * 1.0 = 4.9
        expected_u = 5.0 - dt * 1.0

        # Check interior u values (boundaries not corrected)
        interior_u = u[1:-1, 1:-1]
        np.testing.assert_allclose(interior_u, expected_u, rtol=1e-5)

        # v should not change (no y-gradient)
        np.testing.assert_allclose(v, 0.0, atol=1e-6)

    def test_correct_velocity_linear_pressure_y(self):
        """Test velocity correction with linear pressure gradient in y."""
        ny, nx = 32, 32
        u = np.zeros((ny, nx + 1), dtype=np.float32)
        v = np.ones((ny + 1, nx), dtype=np.float32) * 5.0

        # Linear pressure field in y: p = y
        pressure = np.zeros((ny, nx), dtype=np.float32)
        for y in range(ny):
            pressure[y, :] = float(y)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # Pressure gradient: dp/dy = 1.0 everywhere
        # Velocity correction: v_new = v_old - dt * dp/dy = 5.0 - 0.1 * 1.0 = 4.9
        expected_v = 5.0 - dt * 1.0

        # u should not change
        np.testing.assert_allclose(u, 0.0, atol=1e-6)

        # Check interior v values
        interior_v = v[1:-1, 1:-1]
        np.testing.assert_allclose(interior_v, expected_v, rtol=1e-5)

    def test_correct_velocity_with_pressure_gradient(self):
        """Test that velocity correction responds to pressure gradients."""
        ny, nx = 32, 32

        # Start with uniform velocity
        u = np.ones((ny, nx + 1), dtype=np.float32) * 5.0
        v = np.ones((ny + 1, nx), dtype=np.float32) * 3.0

        # Create pressure gradient (high on left, low on right)
        pressure = np.zeros((ny, nx), dtype=np.float32)
        for x in range(nx):
            pressure[:, x] = 10.0 * (1.0 - x / nx)  # Decreasing left to right

        dx = 1.0 / nx
        dt = 0.1

        # Store initial velocities
        u_initial = u.copy()
        v_initial = v.copy()

        # Apply correction
        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # u-velocity should decrease (negative pressure gradient pushes left)
        # Check that interior u values changed
        u_change = np.abs(u[1:-1, 1:-1] - u_initial[1:-1, 1:-1])
        assert np.mean(u_change) > 0.01, "u-velocity should change with pressure gradient"

        # v should not change much (no y-gradient)
        v_change = np.abs(v[1:-1, 1:-1] - v_initial[1:-1, 1:-1])
        assert np.mean(v_change) < 0.01, "v-velocity should not change without y-gradient"


class TestVelocityCorrection3D:
    """Tests for 3D velocity correction."""

    def test_correct_velocity_3d_zero_pressure(self):
        """Test that zero pressure produces no correction in 3D."""
        nz, ny, nx = 16, 16, 16
        u = np.random.rand(nz, ny, nx + 1).astype(np.float32)
        v = np.random.rand(nz, ny + 1, nx).astype(np.float32)
        w = np.random.rand(nz + 1, ny, nx).astype(np.float32)
        u_orig, v_orig, w_orig = u.copy(), v.copy(), w.copy()

        pressure = np.zeros((nz, ny, nx), dtype=np.float32)

        correct_velocity_kernel_3d(u, v, w, pressure, dx=1.0/16, dt=0.1, nz=nz, ny=ny, nx=nx)

        np.testing.assert_allclose(u, u_orig, rtol=1e-6)
        np.testing.assert_allclose(v, v_orig, rtol=1e-6)
        np.testing.assert_allclose(w, w_orig, rtol=1e-6)

    def test_correct_velocity_3d_uniform_pressure(self):
        """Test that uniform pressure produces no correction in 3D."""
        nz, ny, nx = 16, 16, 16
        u = np.random.rand(nz, ny, nx + 1).astype(np.float32)
        v = np.random.rand(nz, ny + 1, nx).astype(np.float32)
        w = np.random.rand(nz + 1, ny, nx).astype(np.float32)
        u_orig, v_orig, w_orig = u.copy(), v.copy(), w.copy()

        pressure = np.ones((nz, ny, nx), dtype=np.float32) * 10.0

        correct_velocity_kernel_3d(u, v, w, pressure, dx=1.0/16, dt=0.1, nz=nz, ny=ny, nx=nx)

        # Uniform pressure = zero gradient
        np.testing.assert_allclose(u[1:-1, 1:-1, :], u_orig[1:-1, 1:-1, :], rtol=1e-6)
        np.testing.assert_allclose(v[1:-1, :, 1:-1], v_orig[1:-1, :, 1:-1], rtol=1e-6)
        np.testing.assert_allclose(w[:, 1:-1, 1:-1], w_orig[:, 1:-1, 1:-1], rtol=1e-6)

    def test_correct_velocity_3d_linear_pressure(self):
        """Test 3D velocity correction with linear pressure gradients."""
        nz, ny, nx = 16, 16, 16

        # Test x-gradient
        u = np.ones((nz, ny, nx + 1), dtype=np.float32) * 5.0
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        pressure = np.zeros((nz, ny, nx), dtype=np.float32)
        for x in range(nx):
            pressure[:, :, x] = float(x)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_3d(u, v, w, pressure, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx)

        expected_u = 5.0 - dt * 1.0
        interior_u = u[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(interior_u, expected_u, rtol=1e-5)

    def test_correct_velocity_3d_with_pressure_gradient(self):
        """Test that 3D velocity correction responds to pressure gradients."""
        nz, ny, nx = 16, 16, 16

        # Start with uniform velocity
        u = np.ones((nz, ny, nx + 1), dtype=np.float32) * 5.0
        v = np.ones((nz, ny + 1, nx), dtype=np.float32) * 3.0
        w = np.ones((nz + 1, ny, nx), dtype=np.float32) * 2.0

        # Create pressure gradient in x-direction
        pressure = np.zeros((nz, ny, nx), dtype=np.float32)
        for x in range(nx):
            pressure[:, :, x] = 10.0 * (1.0 - x / nx)

        dx = 1.0 / nx
        dt = 0.1

        u_initial = u.copy()
        v_initial = v.copy()
        w_initial = w.copy()

        # Apply correction
        correct_velocity_kernel_3d(u, v, w, pressure, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx)

        # u should change (x-gradient exists)
        u_change = np.abs(u[1:-1, 1:-1, 1:-1] - u_initial[1:-1, 1:-1, 1:-1])
        assert np.mean(u_change) > 0.01, "u-velocity should change with pressure gradient"

        # v and w should not change much (no y or z gradients)
        v_change = np.abs(v[1:-1, 1:-1, 1:-1] - v_initial[1:-1, 1:-1, 1:-1])
        w_change = np.abs(w[1:-1, 1:-1, 1:-1] - w_initial[1:-1, 1:-1, 1:-1])
        assert np.mean(v_change) < 0.01, "v-velocity should not change"
        assert np.mean(w_change) < 0.01, "w-velocity should not change"


class TestVelocityCorrectionEdgeCases:
    """Edge case tests for velocity correction."""

    def test_correct_velocity_large_pressure(self):
        """Test that large pressure values don't cause instability."""
        ny, nx = 32, 32
        u = np.ones((ny, nx + 1), dtype=np.float32)
        v = np.ones((ny + 1, nx), dtype=np.float32)

        # Very large pressure
        pressure = np.random.rand(ny, nx).astype(np.float32) * 1000.0

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0/32, dt=0.01, ny=ny, nx=nx)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(u))
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(u))
        assert not np.any(np.isinf(v))

    def test_correct_velocity_negative_pressure(self):
        """Test correction with negative pressure values."""
        ny, nx = 32, 32
        u = np.ones((ny, nx + 1), dtype=np.float32)
        v = np.ones((ny + 1, nx), dtype=np.float32)

        # Negative pressure
        pressure = -np.abs(np.random.rand(ny, nx).astype(np.float32))

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # Should complete without errors
        assert not np.any(np.isnan(u))
        assert not np.any(np.isnan(v))

    def test_correct_velocity_small_dt(self):
        """Test correction with very small timestep."""
        ny, nx = 32, 32
        u = np.random.rand(ny, nx + 1).astype(np.float32)
        v = np.random.rand(ny + 1, nx).astype(np.float32)
        u_orig, v_orig = u.copy(), v.copy()

        pressure = np.random.rand(ny, nx).astype(np.float32)

        # Very small timestep
        dt = 1e-8
        correct_velocity_kernel_2d(u, v, pressure, dx=1.0/32, dt=dt, ny=ny, nx=nx)

        # With tiny dt, velocity should barely change
        np.testing.assert_allclose(u, u_orig, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(v, v_orig, rtol=1e-4, atol=1e-6)
