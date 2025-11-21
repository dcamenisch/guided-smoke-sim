"""Tests for advection kernels."""

from __future__ import annotations

import math

import torch

from kernels.advection import (
    advect_density_kernel_2d,
    advect_u_velocity_kernel_2d,
    advect_v_velocity_kernel_2d,
    advect_density_kernel_3d,
    advect_u_velocity_kernel_3d,
    advect_v_velocity_kernel_3d,
    advect_w_velocity_kernel_3d,
)


DTYPE = torch.float32


def rand_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, dtype=DTYPE)


def zeros_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=DTYPE)


class TestAdvection2D:
    """Tests for 2D advection kernels."""

    def test_advect_density_no_velocity(self) -> None:
        """Test that density doesn't change with zero velocity."""
        ny, nx = 32, 32
        density = rand_tensor((ny, nx))
        density_new = zeros_tensor((ny, nx))
        u = zeros_tensor((ny, nx + 1))
        v = zeros_tensor((ny + 1, nx))

        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx
        )

        interior = density_new[1:-1, 1:-1]
        interior_orig = density[1:-1, 1:-1]
        torch.testing.assert_close(interior, interior_orig, rtol=1e-5, atol=1e-6)

    def test_advect_density_translation(self) -> None:
        """Test density advection with uniform translation."""
        ny, nx = 64, 64
        density = zeros_tensor((ny, nx))
        density_new = zeros_tensor((ny, nx))

        cy, cx = ny // 2, nx // 2
        for y in range(ny):
            for x in range(nx):
                dist2 = (y - cy) ** 2 + (x - cx) ** 2
                density[y, x] = math.exp(-dist2 / 20.0)

        u = torch.ones((ny, nx + 1), dtype=DTYPE)
        v = zeros_tensor((ny + 1, nx))

        dx = 1.0
        dt = 1.0

        advect_density_kernel_2d(density, density_new, u, v, dx=dx, dt=dt, ny=ny, nx=nx)

        original_max_x = torch.argmax(torch.sum(density, dim=0)).item()
        new_max_x = torch.argmax(torch.sum(density_new, dim=0)).item()
        assert (
            abs((new_max_x - original_max_x) - 1) <= 1
        ), f"Blob should move 1 cell right, moved {new_max_x - original_max_x}"

        original_mass = torch.sum(density).item()
        new_mass = torch.sum(density_new).item()
        assert (
            abs(new_mass - original_mass) / max(original_mass, 1e-6) < 0.1
        ), f"Mass not conserved: {original_mass} -> {new_mass}"

    def test_advect_density_conservation(self) -> None:
        """Test that advection approximately conserves mass."""
        ny, nx = 32, 32
        density = rand_tensor((ny, nx))
        density_new = zeros_tensor((ny, nx))
        u = 0.5 * rand_tensor((ny, nx + 1))
        v = 0.5 * rand_tensor((ny + 1, nx))

        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=0.01, ny=ny, nx=nx
        )

        original_mass = torch.sum(density).item()
        new_mass = torch.sum(density_new).item()
        relative_error = abs(new_mass - original_mass) / max(original_mass, 1e-6)

        assert (
            relative_error < 0.4
        ), f"Mass conservation error too large: {relative_error:.4f}"

    def test_advect_u_velocity_no_velocity(self) -> None:
        """Test that u-velocity doesn't change much with zero v-velocity."""
        ny, nx = 32, 32
        u = (rand_tensor((ny, nx + 1)) * 2.0) - 1.0
        u_new = zeros_tensor((ny, nx + 1))
        v = zeros_tensor((ny + 1, nx))

        advect_u_velocity_kernel_2d(u, u_new, v, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx)

        mean_diff = torch.abs(
            torch.mean(u_new[1:-1, 1:-1]) - torch.mean(u[1:-1, 1:-1])
        ).item()
        std_diff = torch.abs(
            torch.std(u_new[1:-1, 1:-1]) - torch.std(u[1:-1, 1:-1])
        ).item()
        assert mean_diff < 0.5
        assert std_diff < 0.5

    def test_advect_v_velocity_no_velocity(self) -> None:
        """Test that v-velocity doesn't change much with zero u-velocity."""
        ny, nx = 32, 32
        v = (rand_tensor((ny + 1, nx)) * 2.0) - 1.0
        v_new = zeros_tensor((ny + 1, nx))
        u = zeros_tensor((ny, nx + 1))

        advect_v_velocity_kernel_2d(v, v_new, u, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx)

        mean_diff = torch.abs(
            torch.mean(v_new[1:-1, 1:-1]) - torch.mean(v[1:-1, 1:-1])
        ).item()
        std_diff = torch.abs(
            torch.std(v_new[1:-1, 1:-1]) - torch.std(v[1:-1, 1:-1])
        ).item()
        assert mean_diff < 0.5
        assert std_diff < 0.5

    def test_advect_density_bounded(self) -> None:
        """Test that advected density stays within reasonable bounds."""
        ny, nx = 32, 32
        density = rand_tensor((ny, nx)).clamp_(0.0, 1.0)
        density_new = zeros_tensor((ny, nx))
        u = 2.0 * rand_tensor((ny, nx + 1))
        v = 2.0 * rand_tensor((ny + 1, nx))

        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=0.01, ny=ny, nx=nx
        )

        assert density_new.min().item() >= -0.01
        assert density_new.max().item() <= 1.01


class TestAdvection3D:
    """Tests for 3D advection kernels."""

    def test_advect_density_3d_no_velocity(self) -> None:
        """Test that 3D density doesn't change with zero velocity."""
        nz, ny, nx = 16, 16, 16
        density = rand_tensor((nz, ny, nx))
        density_new = zeros_tensor((nz, ny, nx))
        u = zeros_tensor((nz, ny, nx + 1))
        v = zeros_tensor((nz, ny + 1, nx))
        w = zeros_tensor((nz + 1, ny, nx))

        advect_density_kernel_3d(
            density, density_new, u, v, w, dx=1.0 / 16, dt=0.1, nz=nz, ny=ny, nx=nx
        )

        interior = density_new[1:-1, 1:-1, 1:-1]
        interior_orig = density[1:-1, 1:-1, 1:-1]
        torch.testing.assert_close(interior, interior_orig, rtol=1e-5, atol=1e-6)

    def test_advect_density_3d_translation(self) -> None:
        """Test 3D density advection with uniform translation."""
        nz, ny, nx = 32, 32, 32
        density = zeros_tensor((nz, ny, nx))
        density_new = zeros_tensor((nz, ny, nx))

        cz, cy, cx = nz // 2, ny // 2, nx // 2
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    dist2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
                    density[z, y, x] = math.exp(-dist2 / 20.0)

        u = torch.ones((nz, ny, nx + 1), dtype=DTYPE)
        v = zeros_tensor((nz, ny + 1, nx))
        w = zeros_tensor((nz + 1, ny, nx))

        dx = 1.0
        dt = 1.0

        advect_density_kernel_3d(
            density, density_new, u, v, w, dx=dx, dt=dt, nz=nz, ny=ny, nx=nx
        )

        original_max_x = torch.argmax(torch.sum(density, dim=(0, 1))).item()
        new_max_x = torch.argmax(torch.sum(density_new, dim=(0, 1))).item()

        assert (
            abs((new_max_x - original_max_x) - 1) <= 1
        ), f"Blob should move 1 cell right, moved {new_max_x - original_max_x}"

    def test_advect_density_3d_conservation(self) -> None:
        """Test that 3D advection approximately conserves mass."""
        nz, ny, nx = 16, 16, 16
        density = rand_tensor((nz, ny, nx))
        density_new = zeros_tensor((nz, ny, nx))
        u = 0.5 * rand_tensor((nz, ny, nx + 1))
        v = 0.5 * rand_tensor((nz, ny + 1, nx))
        w = 0.5 * rand_tensor((nz + 1, ny, nx))

        advect_density_kernel_3d(
            density, density_new, u, v, w, dx=1.0 / 16, dt=0.01, nz=nz, ny=ny, nx=nx
        )

        original_mass = torch.sum(density).item()
        new_mass = torch.sum(density_new).item()
        relative_error = abs(new_mass - original_mass) / max(original_mass, 1e-6)

        assert (
            relative_error < 0.4
        ), f"Mass conservation error too large: {relative_error:.4f}"

    def test_advect_3d_velocity_components(self) -> None:
        """Test that all 3D velocity components can be advected."""
        nz, ny, nx = 16, 16, 16

        u = rand_tensor((nz, ny, nx + 1))
        u_new = zeros_tensor((nz, ny, nx + 1))
        v = zeros_tensor((nz, ny + 1, nx))
        w = zeros_tensor((nz + 1, ny, nx))

        advect_u_velocity_kernel_3d(
            u, u_new, v, w, dx=1.0 / 16, dt=0.01, nz=nz, ny=ny, nx=nx
        )
        assert not torch.allclose(u_new[1:-1, 1:-1, 1:-1], torch.zeros(1, dtype=DTYPE))

        v = rand_tensor((nz, ny + 1, nx))
        v_new = zeros_tensor((nz, ny + 1, nx))
        u = zeros_tensor((nz, ny, nx + 1))
        w = zeros_tensor((nz + 1, ny, nx))

        advect_v_velocity_kernel_3d(
            v, v_new, u, w, dx=1.0 / 16, dt=0.01, nz=nz, ny=ny, nx=nx
        )
        assert not torch.allclose(v_new[1:-1, 1:-1, 1:-1], torch.zeros(1, dtype=DTYPE))

        w = rand_tensor((nz + 1, ny, nx))
        w_new = zeros_tensor((nz + 1, ny, nx))
        u = zeros_tensor((nz, ny, nx + 1))
        v = zeros_tensor((nz, ny + 1, nx))

        advect_w_velocity_kernel_3d(
            w, w_new, u, v, dx=1.0 / 16, dt=0.01, nz=nz, ny=ny, nx=nx
        )
        assert not torch.allclose(w_new[1:-1, 1:-1, 1:-1], torch.zeros(1, dtype=DTYPE))


class TestAdvectionStability:
    """Tests for advection stability and edge cases."""

    def test_advect_density_large_velocity(self) -> None:
        """Test advection behavior with large velocity (CFL violation)."""
        ny, nx = 32, 32
        density = zeros_tensor((ny, nx))
        density[ny // 2, nx // 2] = 1.0
        density_new = zeros_tensor((ny, nx))

        u = torch.ones((ny, nx + 1), dtype=DTYPE) * 100.0
        v = zeros_tensor((ny + 1, nx))

        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=0.1, ny=ny, nx=nx
        )

        assert not torch.isnan(density_new).any().item()
        assert not torch.isinf(density_new).any().item()

    def test_advect_negative_density(self) -> None:
        """Test that advection handles negative values correctly."""
        ny, nx = 32, 32
        density = ((rand_tensor((ny, nx)) - 0.5) * 2.0).clone()
        density_new = zeros_tensor((ny, nx))
        u = 0.5 * rand_tensor((ny, nx + 1))
        v = 0.5 * rand_tensor((ny + 1, nx))

        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=0.01, ny=ny, nx=nx
        )

        assert not torch.isnan(density_new).any().item()
        assert not torch.isinf(density_new).any().item()

    def test_advect_small_timestep(self) -> None:
        """Test advection with very small timestep."""
        ny, nx = 32, 32
        density = rand_tensor((ny, nx))
        density_new = zeros_tensor((ny, nx))
        u = torch.ones((ny, nx + 1), dtype=DTYPE)
        v = zeros_tensor((ny + 1, nx))

        dt = 1e-6
        advect_density_kernel_2d(
            density, density_new, u, v, dx=1.0 / 32, dt=dt, ny=ny, nx=nx
        )

        torch.testing.assert_close(
            density_new[1:-1, 1:-1], density[1:-1, 1:-1], rtol=0.01, atol=1e-4
        )
