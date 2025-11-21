"""Tests for velocity correction kernels."""

import torch
from kernels.velocity import correct_velocity_kernel_2d, correct_velocity_kernel_3d


class TestVelocityCorrection2D:
    """Tests for 2D velocity correction."""

    def test_correct_velocity_zero_pressure(self):
        """Test that zero pressure produces no correction."""
        ny, nx = 32, 32
        u = torch.rand((ny, nx + 1), dtype=torch.float32)
        v = torch.rand((ny + 1, nx), dtype=torch.float32)
        u_orig = u.clone()
        v_orig = v.clone()

        pressure = torch.zeros((ny, nx), dtype=torch.float32)

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx)

        torch.testing.assert_close(u, u_orig, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v, v_orig, rtol=1e-6, atol=1e-6)

    def test_correct_velocity_uniform_pressure(self):
        """Test that uniform pressure produces no correction."""
        ny, nx = 32, 32
        u = torch.rand((ny, nx + 1), dtype=torch.float32)
        v = torch.rand((ny + 1, nx), dtype=torch.float32)
        u_orig = u.clone()
        v_orig = v.clone()

        pressure = torch.full((ny, nx), 10.0, dtype=torch.float32)

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx)

        torch.testing.assert_close(u[1:-1, :], u_orig[1:-1, :], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v[:, 1:-1], v_orig[:, 1:-1], rtol=1e-6, atol=1e-6)

    def test_correct_velocity_linear_pressure_x(self):
        """Test velocity correction with linear pressure gradient in x."""
        ny, nx = 32, 32
        u = torch.full((ny, nx + 1), 5.0, dtype=torch.float32)
        v = torch.zeros((ny + 1, nx), dtype=torch.float32)

        x_vals = torch.arange(nx, dtype=torch.float32)
        pressure = x_vals.unsqueeze(0).repeat(ny, 1)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # Pressure gradient: dp/dx = 1.0 everywhere
        # Velocity correction: u_new = u_old - dt * dp/dx = 5.0 - 0.1 * 1.0 = 4.9
        expected_u = 5.0 - dt * 1.0

        # Check interior u values (boundaries not corrected)
        interior_u = u[1:-1, 1:-1]
        torch.testing.assert_close(
            interior_u, torch.full_like(interior_u, expected_u), rtol=1e-5, atol=1e-5
        )

        # v should not change (no y-gradient)
        torch.testing.assert_close(v, torch.zeros_like(v), atol=1e-6, rtol=0)

    def test_correct_velocity_linear_pressure_y(self):
        """Test velocity correction with linear pressure gradient in y."""
        ny, nx = 32, 32
        u = torch.zeros((ny, nx + 1), dtype=torch.float32)
        v = torch.full((ny + 1, nx), 5.0, dtype=torch.float32)

        y_vals = torch.arange(ny, dtype=torch.float32)
        pressure = y_vals.unsqueeze(1).repeat(1, nx)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # Pressure gradient: dp/dy = 1.0 everywhere
        # Velocity correction: v_new = v_old - dt * dp/dy = 5.0 - 0.1 * 1.0 = 4.9
        expected_v = 5.0 - dt * 1.0

        torch.testing.assert_close(u, torch.zeros_like(u), atol=1e-6, rtol=0)

        # Check interior v values
        interior_v = v[1:-1, 1:-1]
        torch.testing.assert_close(
            interior_v, torch.full_like(interior_v, expected_v), rtol=1e-5, atol=1e-5
        )

    def test_correct_velocity_with_pressure_gradient(self):
        """Test that velocity correction responds to pressure gradients."""
        ny, nx = 32, 32

        # Start with uniform velocity
        u = torch.full((ny, nx + 1), 5.0, dtype=torch.float32)
        v = torch.full((ny + 1, nx), 3.0, dtype=torch.float32)

        x_vals = torch.arange(nx, dtype=torch.float32) / nx
        pressure = 10.0 * (1.0 - x_vals).unsqueeze(0).repeat(ny, 1)

        dx = 1.0 / nx
        dt = 0.1

        # Store initial velocities
        u_initial = u.clone()
        v_initial = v.clone()

        # Apply correction
        correct_velocity_kernel_2d(u, v, pressure, dx=dx, dt=dt, ny=ny, nx=nx)

        # u-velocity should decrease (negative pressure gradient pushes left)
        # Check that interior u values changed
        u_change = torch.abs(u[1:-1, 1:-1] - u_initial[1:-1, 1:-1])
        assert (
            torch.mean(u_change).item() > 0.01
        ), "u-velocity should change with pressure gradient"

        v_change = torch.abs(v[1:-1, 1:-1] - v_initial[1:-1, 1:-1])
        assert (
            torch.mean(v_change).item() < 0.01
        ), "v-velocity should not change without y-gradient"


class TestVelocityCorrection3D:
    """Tests for 3D velocity correction."""

    def test_correct_velocity_3d_zero_pressure(self):
        """Test that zero pressure produces no correction in 3D."""
        nz, ny, nx = 16, 16, 16
        u = torch.rand((nz, ny, nx + 1), dtype=torch.float32)
        v = torch.rand((nz, ny + 1, nx), dtype=torch.float32)
        w = torch.rand((nz + 1, ny, nx), dtype=torch.float32)
        u_orig, v_orig, w_orig = u.clone(), v.clone(), w.clone()

        pressure = torch.zeros((nz, ny, nx), dtype=torch.float32)

        correct_velocity_kernel_3d(
            u, v, w, pressure, dx=1.0 / 16, dt=0.1, nz=nz, ny=ny, nx=nx
        )

        torch.testing.assert_close(u, u_orig, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v, v_orig, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(w, w_orig, rtol=1e-6, atol=1e-6)

    def test_correct_velocity_3d_uniform_pressure(self):
        """Test that uniform pressure produces no correction in 3D."""
        nz, ny, nx = 16, 16, 16
        u = torch.rand((nz, ny, nx + 1), dtype=torch.float32)
        v = torch.rand((nz, ny + 1, nx), dtype=torch.float32)
        w = torch.rand((nz + 1, ny, nx), dtype=torch.float32)
        u_orig, v_orig, w_orig = u.clone(), v.clone(), w.clone()

        pressure = torch.full((nz, ny, nx), 10.0, dtype=torch.float32)

        correct_velocity_kernel_3d(
            u, v, w, pressure, dx=1.0 / 16, dt=0.1, nz=nz, ny=ny, nx=nx
        )

        # Uniform pressure = zero gradient
        torch.testing.assert_close(
            u[1:-1, 1:-1, :], u_orig[1:-1, 1:-1, :], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            v[1:-1, :, 1:-1], v_orig[1:-1, :, 1:-1], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            w[:, 1:-1, 1:-1], w_orig[:, 1:-1, 1:-1], rtol=1e-6, atol=1e-6
        )

    def test_correct_velocity_3d_linear_pressure(self):
        """Test 3D velocity correction with linear pressure gradients."""
        nz, ny, nx = 16, 16, 16

        # Test x-gradient
        u = torch.full((nz, ny, nx + 1), 5.0, dtype=torch.float32)
        v = torch.zeros((nz, ny + 1, nx), dtype=torch.float32)
        w = torch.zeros((nz + 1, ny, nx), dtype=torch.float32)

        x_vals = torch.arange(nx, dtype=torch.float32)
        pressure = x_vals.view(1, 1, nx).repeat(nz, ny, 1)

        dx = 1.0
        dt = 0.1

        correct_velocity_kernel_3d(u, v, w, pressure, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx)

        expected_u = 5.0 - dt * 1.0
        interior_u = u[1:-1, 1:-1, 1:-1]
        torch.testing.assert_close(
            interior_u, torch.full_like(interior_u, expected_u), rtol=1e-5, atol=1e-5
        )

    def test_correct_velocity_3d_with_pressure_gradient(self):
        """Test that 3D velocity correction responds to pressure gradients."""
        nz, ny, nx = 16, 16, 16

        # Start with uniform velocity
        u = torch.full((nz, ny, nx + 1), 5.0, dtype=torch.float32)
        v = torch.full((nz, ny + 1, nx), 3.0, dtype=torch.float32)
        w = torch.full((nz + 1, ny, nx), 2.0, dtype=torch.float32)

        x_vals = torch.arange(nx, dtype=torch.float32) / nx
        pressure = 10.0 * (1.0 - x_vals).view(1, 1, nx).repeat(nz, ny, 1)

        dx = 1.0 / nx
        dt = 0.1

        u_initial = u.clone()
        v_initial = v.clone()
        w_initial = w.clone()

        # Apply correction
        correct_velocity_kernel_3d(u, v, w, pressure, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx)

        # u should change (x-gradient exists)
        u_change = torch.abs(u[1:-1, 1:-1, 1:-1] - u_initial[1:-1, 1:-1, 1:-1])
        assert (
            torch.mean(u_change).item() > 0.01
        ), "u-velocity should change with pressure gradient"

        v_change = torch.abs(v[1:-1, 1:-1, 1:-1] - v_initial[1:-1, 1:-1, 1:-1])
        w_change = torch.abs(w[1:-1, 1:-1, 1:-1] - w_initial[1:-1, 1:-1, 1:-1])
        assert torch.mean(v_change).item() < 0.01, "v-velocity should not change"
        assert torch.mean(w_change).item() < 0.01, "w-velocity should not change"


class TestVelocityCorrectionEdgeCases:
    """Edge case tests for velocity correction."""

    def test_correct_velocity_large_pressure(self):
        """Test that large pressure values don't cause instability."""
        ny, nx = 32, 32
        u = torch.ones((ny, nx + 1), dtype=torch.float32)
        v = torch.ones((ny + 1, nx), dtype=torch.float32)

        pressure = torch.rand((ny, nx), dtype=torch.float32) * 1000.0

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0 / 32, dt=0.01, ny=ny, nx=nx)

        assert not torch.isnan(u).any().item()
        assert not torch.isnan(v).any().item()
        assert not torch.isinf(u).any().item()
        assert not torch.isinf(v).any().item()

    def test_correct_velocity_negative_pressure(self):
        """Test correction with negative pressure values."""
        ny, nx = 32, 32
        u = torch.ones((ny, nx + 1), dtype=torch.float32)
        v = torch.ones((ny + 1, nx), dtype=torch.float32)

        pressure = -torch.rand((ny, nx), dtype=torch.float32)

        correct_velocity_kernel_2d(u, v, pressure, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx)

        assert not torch.isnan(u).any().item()
        assert not torch.isnan(v).any().item()

    def test_correct_velocity_small_dt(self):
        """Test correction with very small timestep."""
        ny, nx = 32, 32
        u = torch.rand((ny, nx + 1), dtype=torch.float32)
        v = torch.rand((ny + 1, nx), dtype=torch.float32)
        u_orig, v_orig = u.clone(), v.clone()

        pressure = torch.rand((ny, nx), dtype=torch.float32)

        # Very small timestep
        dt = 1e-8
        correct_velocity_kernel_2d(u, v, pressure, dx=1.0 / 32, dt=dt, ny=ny, nx=nx)

        # With tiny dt, velocity should barely change
        torch.testing.assert_close(u, u_orig, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(v, v_orig, rtol=1e-4, atol=1e-6)
