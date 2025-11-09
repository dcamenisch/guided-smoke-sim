"""Tests for advection kernels."""

import pytest
import numpy as np
from kernels.advection import (
    advect_density_kernel_2d,
    advect_u_velocity_kernel_2d,
    advect_v_velocity_kernel_2d,
    advect_density_kernel_3d,
    advect_u_velocity_kernel_3d,
    advect_v_velocity_kernel_3d,
    advect_w_velocity_kernel_3d,
)


class TestAdvection2D:
    """Tests for 2D advection kernels."""

    def test_advect_density_no_velocity(self):
        """Test that density doesn't change with zero velocity."""
        ny, nx = 32, 32
        density = np.random.rand(ny, nx).astype(np.float32)
        density_new = np.zeros((ny, nx), dtype=np.float32)
        u = np.zeros((ny, nx + 1), dtype=np.float32)
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # With zero velocity, density should remain unchanged (except boundaries)
        interior = density_new[1:-1, 1:-1]
        interior_orig = density[1:-1, 1:-1]
        np.testing.assert_allclose(interior, interior_orig, rtol=1e-5)

    def test_advect_density_translation(self):
        """Test density advection with uniform translation.

        A blob should translate without changing shape significantly.
        """
        ny, nx = 64, 64
        density = np.zeros((ny, nx), dtype=np.float32)
        density_new = np.zeros((ny, nx), dtype=np.float32)

        # Create a Gaussian blob at center
        cy, cx = ny // 2, nx // 2
        for y in range(ny):
            for x in range(nx):
                dist2 = (y - cy)**2 + (x - cx)**2
                density[y, x] = np.exp(-dist2 / 20.0)

        # Uniform velocity to the right (u = 1.0)
        u = np.ones((ny, nx + 1), dtype=np.float32)
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        dx = 1.0
        dt = 1.0  # Move 1 cell to the right

        advect_density_kernel_2d(density, density_new, u, v, dx=dx, dt=dt, ny=ny, nx=nx)

        # Original maximum at center
        original_max_x = np.argmax(np.sum(density, axis=0))

        # New maximum should be shifted right by approximately 1 cell
        new_max_x = np.argmax(np.sum(density_new, axis=0))

        # Allow some tolerance due to interpolation
        assert abs((new_max_x - original_max_x) - 1) <= 1, \
            f"Blob should move 1 cell right, moved {new_max_x - original_max_x}"

        # Mass should be approximately conserved
        original_mass = np.sum(density)
        new_mass = np.sum(density_new)
        assert abs(new_mass - original_mass) / original_mass < 0.1, \
            f"Mass not conserved: {original_mass} -> {new_mass}"

    def test_advect_density_conservation(self):
        """Test that advection approximately conserves mass."""
        ny, nx = 32, 32
        density = np.random.rand(ny, nx).astype(np.float32)
        density_new = np.zeros((ny, nx), dtype=np.float32)

        # Random velocity field
        u = 0.5 * np.random.rand(ny, nx + 1).astype(np.float32)
        v = 0.5 * np.random.rand(ny + 1, nx).astype(np.float32)

        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=0.01, ny=ny, nx=nx)

        # Mass should be approximately conserved
        # Note: Semi-Lagrangian advection is dissipative, so some mass loss is expected
        original_mass = np.sum(density)
        new_mass = np.sum(density_new)
        relative_error = abs(new_mass - original_mass) / original_mass

        assert relative_error < 0.4, \
            f"Mass conservation error too large: {relative_error:.4f}"

    def test_advect_u_velocity_no_velocity(self):
        """Test that u-velocity doesn't change much with zero v-velocity."""
        ny, nx = 32, 32
        u = np.random.rand(ny, nx + 1).astype(np.float32) * 2.0 - 1.0
        u_new = np.zeros((ny, nx + 1), dtype=np.float32)
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        advect_u_velocity_kernel_2d(u, u_new, v, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # With zero v-velocity (and u advecting itself), expect some change but not huge
        # The result should have similar statistics (not random)
        assert np.abs(np.mean(u_new[1:-1, 1:-1]) - np.mean(u[1:-1, 1:-1])) < 0.5
        assert np.abs(np.std(u_new[1:-1, 1:-1]) - np.std(u[1:-1, 1:-1])) < 0.5

    def test_advect_v_velocity_no_velocity(self):
        """Test that v-velocity doesn't change much with zero u-velocity."""
        ny, nx = 32, 32
        v = np.random.rand(ny + 1, nx).astype(np.float32) * 2.0 - 1.0
        v_new = np.zeros((ny + 1, nx), dtype=np.float32)
        u = np.zeros((ny, nx + 1), dtype=np.float32)

        advect_v_velocity_kernel_2d(v, v_new, u, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # With zero u-velocity (and v advecting itself), expect some change but not huge
        # The result should have similar statistics
        assert np.abs(np.mean(v_new[1:-1, 1:-1]) - np.mean(v[1:-1, 1:-1])) < 0.5
        assert np.abs(np.std(v_new[1:-1, 1:-1]) - np.std(v[1:-1, 1:-1])) < 0.5

    def test_advect_density_bounded(self):
        """Test that advected density stays within reasonable bounds."""
        ny, nx = 32, 32
        # Bounded initial density
        density = np.clip(np.random.rand(ny, nx).astype(np.float32), 0.0, 1.0)
        density_new = np.zeros((ny, nx), dtype=np.float32)

        # Moderate velocity
        u = 2.0 * np.random.rand(ny, nx + 1).astype(np.float32)
        v = 2.0 * np.random.rand(ny + 1, nx).astype(np.float32)

        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=0.01, ny=ny, nx=nx)

        # Advected density should not exceed original bounds significantly
        # (may have small interpolation artifacts)
        assert np.min(density_new) >= -0.01, f"Minimum density {np.min(density_new)} < -0.01"
        assert np.max(density_new) <= 1.01, f"Maximum density {np.max(density_new)} > 1.01"


class TestAdvection3D:
    """Tests for 3D advection kernels."""

    def test_advect_density_3d_no_velocity(self):
        """Test that 3D density doesn't change with zero velocity."""
        nz, ny, nx = 16, 16, 16
        density = np.random.rand(nz, ny, nx).astype(np.float32)
        density_new = np.zeros((nz, ny, nx), dtype=np.float32)
        u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        advect_density_kernel_3d(density, density_new, u, v, w, dx=1.0/16, dt=0.1, nz=nz, ny=ny, nx=nx)

        # With zero velocity, density should remain unchanged (except boundaries)
        interior = density_new[1:-1, 1:-1, 1:-1]
        interior_orig = density[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(interior, interior_orig, rtol=1e-5)

    def test_advect_density_3d_translation(self):
        """Test 3D density advection with uniform translation."""
        nz, ny, nx = 32, 32, 32
        density = np.zeros((nz, ny, nx), dtype=np.float32)
        density_new = np.zeros((nz, ny, nx), dtype=np.float32)

        # Create a Gaussian blob at center
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    dist2 = (z - cz)**2 + (y - cy)**2 + (x - cx)**2
                    density[z, y, x] = np.exp(-dist2 / 20.0)

        # Uniform velocity to the right (u = 1.0)
        u = np.ones((nz, ny, nx + 1), dtype=np.float32)
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        dx = 1.0
        dt = 1.0  # Move 1 cell to the right

        advect_density_kernel_3d(density, density_new, u, v, w, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx)

        # Original maximum location
        original_max_x = np.argmax(np.sum(density, axis=(0, 1)))

        # New maximum should be shifted right by approximately 1 cell
        new_max_x = np.argmax(np.sum(density_new, axis=(0, 1)))

        assert abs((new_max_x - original_max_x) - 1) <= 1, \
            f"Blob should move 1 cell right, moved {new_max_x - original_max_x}"

    def test_advect_density_3d_conservation(self):
        """Test that 3D advection approximately conserves mass."""
        nz, ny, nx = 16, 16, 16
        density = np.random.rand(nz, ny, nx).astype(np.float32)
        density_new = np.zeros((nz, ny, nx), dtype=np.float32)

        # Random velocity field
        u = 0.5 * np.random.rand(nz, ny, nx + 1).astype(np.float32)
        v = 0.5 * np.random.rand(nz, ny + 1, nx).astype(np.float32)
        w = 0.5 * np.random.rand(nz + 1, ny, nx).astype(np.float32)

        advect_density_kernel_3d(density, density_new, u, v, w, dx=1.0/16, dt=0.01, nz=nz, ny=ny, nx=nx)

        # Mass should be approximately conserved
        # Note: Semi-Lagrangian advection is dissipative, so some mass loss is expected
        original_mass = np.sum(density)
        new_mass = np.sum(density_new)
        relative_error = abs(new_mass - original_mass) / original_mass

        assert relative_error < 0.4, \
            f"Mass conservation error too large: {relative_error:.4f}"

    def test_advect_3d_velocity_components(self):
        """Test that all 3D velocity components can be advected."""
        nz, ny, nx = 16, 16, 16

        # Test u advection
        u = np.random.rand(nz, ny, nx + 1).astype(np.float32)
        u_new = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        advect_u_velocity_kernel_3d(u, u_new, v, w, dx=1.0/16, dt=0.01, nz=nz, ny=ny, nx=nx)
        assert not np.allclose(u_new[1:-1, 1:-1, 1:-1], 0.0)

        # Test v advection
        v = np.random.rand(nz, ny + 1, nx).astype(np.float32)
        v_new = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        w = np.zeros((nz + 1, ny, nx), dtype=np.float32)

        advect_v_velocity_kernel_3d(v, v_new, u, w, dx=1.0/16, dt=0.01, nz=nz, ny=ny, nx=nx)
        assert not np.allclose(v_new[1:-1, 1:-1, 1:-1], 0.0)

        # Test w advection
        w = np.random.rand(nz + 1, ny, nx).astype(np.float32)
        w_new = np.zeros((nz + 1, ny, nx), dtype=np.float32)
        u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        v = np.zeros((nz, ny + 1, nx), dtype=np.float32)

        advect_w_velocity_kernel_3d(w, w_new, u, v, dx=1.0/16, dt=0.01, nz=nz, ny=ny, nx=nx)
        assert not np.allclose(w_new[1:-1, 1:-1, 1:-1], 0.0)


class TestAdvectionStability:
    """Tests for advection stability and edge cases."""

    def test_advect_density_large_velocity(self):
        """Test advection behavior with large velocity (CFL violation)."""
        ny, nx = 32, 32
        density = np.zeros((ny, nx), dtype=np.float32)
        density[ny//2, nx//2] = 1.0
        density_new = np.zeros((ny, nx), dtype=np.float32)

        # Very large velocity (CFL >> 1)
        u = np.ones((ny, nx + 1), dtype=np.float32) * 100.0
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        # Should still run without crashes (though may be inaccurate)
        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=0.1, ny=ny, nx=nx)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(density_new))
        assert not np.any(np.isinf(density_new))

    def test_advect_negative_density(self):
        """Test that advection handles negative values correctly."""
        ny, nx = 32, 32
        # Include negative values
        density = (np.random.rand(ny, nx).astype(np.float32) - 0.5) * 2.0
        density_new = np.zeros((ny, nx), dtype=np.float32)

        u = 0.5 * np.random.rand(ny, nx + 1).astype(np.float32)
        v = 0.5 * np.random.rand(ny + 1, nx).astype(np.float32)

        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=0.01, ny=ny, nx=nx)

        # Should complete without errors
        assert not np.any(np.isnan(density_new))
        assert not np.any(np.isinf(density_new))

    def test_advect_small_timestep(self):
        """Test advection with very small timestep."""
        ny, nx = 32, 32
        density = np.random.rand(ny, nx).astype(np.float32)
        density_new = np.zeros((ny, nx), dtype=np.float32)

        u = np.ones((ny, nx + 1), dtype=np.float32)
        v = np.zeros((ny + 1, nx), dtype=np.float32)

        # Very small timestep should give results very close to original
        dt = 1e-6
        advect_density_kernel_2d(density, density_new, u, v, dx=1.0/32, dt=dt, ny=ny, nx=nx)

        # Should be almost identical to original
        # Allow for some interpolation artifacts
        np.testing.assert_allclose(density_new[1:-1, 1:-1], density[1:-1, 1:-1], rtol=0.01, atol=1e-4)
