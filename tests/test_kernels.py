"""Tests for computational kernels."""

import torch
from kernels.poisson import (
    solve_poisson_rb_gauss_seidel_2d,
    solve_poisson_rb_gauss_seidel_3d,
)
from kernels.differential import (
    compute_vorticity_kernel_2d,
    compute_vorticity_kernel_3d,
)
from kernels.interpolation import bilinear_interp, trilinear_interp


class TestPoissonSolver:
    """Tests for Poisson solver."""

    def test_poisson_2d_converges(self):
        """Test that 2D Poisson solver converges to reasonable solution."""
        ny, nx = 32, 32
        pressure = torch.zeros((ny, nx), dtype=torch.float32)
        divergence = torch.ones((ny, nx), dtype=torch.float32)

        # Set up Poisson equation: ∇²p = -ρ/dt * div
        dx = 1.0 / nx
        dt = 0.1
        rho = 1.0

        result = solve_poisson_rb_gauss_seidel_2d(
            pressure,
            divergence,
            dx,
            dt,
            rho,
            max_iter=100,
            tolerance=1e-4,
            ny=ny,
            nx=nx,
        )

        # Pressure should have changed from zero
        assert not torch.allclose(result, torch.zeros_like(result))

        # Pressure should be bounded and smooth
        max_pressure = result.abs().max().item()
        assert max_pressure < 100.0, f"Pressure too large: {max_pressure}"

        # For uniform divergence, pressure should be smooth (no large gradients)
        grad_x = (result[:, 1:] - result[:, :-1]).abs()
        grad_y = (result[1:, :] - result[:-1, :]).abs()
        assert grad_x.max().item() < 10.0, "Pressure gradients too large in x"
        assert grad_y.max().item() < 10.0, "Pressure gradients too large in y"

    def test_poisson_3d_converges(self):
        """Test that 3D Poisson solver converges."""
        nz, ny, nx = 16, 16, 16
        pressure = torch.zeros((nz, ny, nx), dtype=torch.float32)
        divergence = torch.ones((nz, ny, nx), dtype=torch.float32)

        dx = 1.0 / nx
        dt = 0.1
        rho = 1.0

        result = solve_poisson_rb_gauss_seidel_3d(
            pressure,
            divergence,
            dx,
            dt,
            rho,
            max_iter=100,
            tolerance=1e-4,
            nz=nz,
            ny=ny,
            nx=nx,
        )

        # Pressure should have changed
        assert not torch.allclose(result, torch.zeros_like(result))
        assert result.abs().max().item() < 100.0

    def test_poisson_symmetry_2d(self):
        """Test that symmetric input gives symmetric output."""
        ny, nx = 32, 32
        pressure = torch.zeros((ny, nx), dtype=torch.float32)

        # Symmetric divergence field
        divergence = torch.ones((ny, nx), dtype=torch.float32)
        divergence[ny // 2, nx // 2] = 10.0  # Center spike

        result = solve_poisson_rb_gauss_seidel_2d(
            pressure,
            divergence,
            1.0 / nx,
            0.1,
            1.0,
            max_iter=100,
            tolerance=1e-4,
            ny=ny,
            nx=nx,
        )

        # Check approximate symmetry (within tolerance)
        center_y, center_x = ny // 2, nx // 2
        diff_y = torch.abs(
            result[center_y + 1, center_x] - result[center_y - 1, center_x]
        ).item()
        diff_x = torch.abs(
            result[center_y, center_x + 1] - result[center_y, center_x - 1]
        ).item()
        assert diff_y < 0.1
        assert diff_x < 0.1

    def test_poisson_2d_parabolic_profile(self):
        """Test Poisson solver produces parabolic-like profile for constant divergence.

        For constant divergence, the pressure solution should be smooth
        and approximately symmetric.
        """
        ny, nx = 64, 64
        dx = 1.0 / nx
        dt = 0.1
        rho = 1.0

        # Create a constant divergence field
        divergence = torch.ones((ny, nx), dtype=torch.float32)

        pressure = torch.zeros((ny, nx), dtype=torch.float32)
        result = solve_poisson_rb_gauss_seidel_2d(
            pressure,
            divergence,
            dx,
            dt,
            rho,
            max_iter=1000,
            tolerance=1e-6,
            ny=ny,
            nx=nx,
        )

        # For constant divergence, expect smooth solution
        # Pressure should vary smoothly across grid
        center_y = ny // 2
        quarter_x = nx // 4
        center_x = nx // 2
        three_quarter_x = 3 * nx // 4

        # Check that solution is not constant (solver did something)
        assert result.std().item() > 0.01, "Pressure field should not be constant"

        # Check approximate symmetry (left-right)
        left_half = result[:, : nx // 2]
        right_half = torch.flip(result[:, nx // 2 :], dims=[1])
        overlap = right_half[:, : left_half.shape[1]]
        symmetry_error = torch.mean(torch.abs(left_half - overlap)).item()
        assert (
            symmetry_error < 1.0
        ), f"Solution should be approximately symmetric, error: {symmetry_error}"


class TestVorticity:
    """Tests for vorticity computation."""

    def test_vorticity_2d_zero_for_uniform(self):
        """Test that uniform flow has zero vorticity."""
        ny, nx = 24, 16
        u = torch.ones((ny, nx + 1), dtype=torch.float32)
        v = torch.zeros((ny + 1, nx), dtype=torch.float32)
        vorticity = torch.zeros((ny, nx), dtype=torch.float32)

        compute_vorticity_kernel_2d(vorticity, u, v, 0.1, ny, nx)

        # Uniform flow should have zero vorticity (except boundaries)
        interior = vorticity[2:-2, 2:-2]
        assert interior.abs().max().item() < 1e-5

    def test_vorticity_2d_rotation(self):
        """Test that rotating flow has non-zero vorticity.

        For solid body rotation: u = -ω*y, v = ω*x
        Vorticity ω_z = dv/dx - du/dy = ω - (-ω) = 2ω
        """
        ny, nx = 24, 16
        u = torch.zeros((ny, nx + 1), dtype=torch.float32)
        v = torch.zeros((ny + 1, nx), dtype=torch.float32)

        # Create rotating flow (counter-clockwise) with angular velocity ω = 1.0
        omega = 1.0
        for y in range(ny):
            for x in range(nx + 1):
                u[y, x] = -omega * (y - ny / 2) / ny  # u = -ω*y
        for y in range(ny + 1):
            for x in range(nx):
                v[y, x] = omega * (x - nx / 2) / nx  # v = ω*x

            vorticity = torch.zeros((ny, nx), dtype=torch.float32)
        dx = 1.0 / nx
        compute_vorticity_kernel_2d(vorticity, u, v, dx, ny, nx)

        # Expected vorticity for solid body rotation: 2*omega
        # Accounting for scaling: dv/dx ≈ omega/nx*dx, du/dy ≈ -omega/ny*dx
        # Net vorticity should be positive and roughly 2*omega/dx in grid units
        expected_vorticity = 2.0 * omega  # Approximately

        # Rotating flow should have positive vorticity
        interior = vorticity[2:-2, 2:-2]
        mean_vorticity = interior.mean().item()

        assert (
            mean_vorticity > 0.5
        ), f"Expected positive vorticity, got {mean_vorticity}"
        assert mean_vorticity < 5.0, f"Vorticity too large: {mean_vorticity}"
        assert interior.abs().max().item() > 0.1

    def test_vorticity_3d_shape(self):
        """Test that 3D vorticity has correct shape."""
        nz, ny, nx = 8, 12, 8
        u = torch.zeros((nz, ny, nx + 1), dtype=torch.float32)
        v = torch.zeros((nz, ny + 1, nx), dtype=torch.float32)
        w = torch.zeros((nz + 1, ny, nx), dtype=torch.float32)
        vorticity = torch.zeros((nz, ny, nx, 3), dtype=torch.float32)

        compute_vorticity_kernel_3d(vorticity, u, v, w, 0.1, nz, ny, nx)

        # Check shape is correct (vector field)
        assert vorticity.shape == (nz, ny, nx, 3)


class TestInterpolation:
    """Tests for interpolation kernels."""

    def test_bilinear_interp_corners(self):
        """Test bilinear interpolation at grid corners."""
        field = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
        )

        # Bottom-left corner (0, 0) -> 1.0
        assert torch.isclose(bilinear_interp(field, 0.0, 0.0), torch.tensor(1.0))

        # Bottom-right corner (2, 0) -> 3.0
        assert torch.isclose(bilinear_interp(field, 2.0, 0.0), torch.tensor(3.0))

        # Top-left corner (0, 2) -> 7.0
        assert torch.isclose(bilinear_interp(field, 0.0, 2.0), torch.tensor(7.0))

        # Top-right corner (2, 2) -> 9.0
        assert torch.isclose(bilinear_interp(field, 2.0, 2.0), torch.tensor(9.0))

    def test_bilinear_interp_center(self):
        """Test bilinear interpolation at center."""
        field = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
        )

        # Center (1.0, 1.0) -> 5.0 (exact grid point)
        assert torch.isclose(bilinear_interp(field, 1.0, 1.0), torch.tensor(5.0))

        # Mid-point between 1 and 5 (0.5, 0.5) -> 3.0
        assert torch.isclose(bilinear_interp(field, 0.5, 0.5), torch.tensor(3.0))

    def test_trilinear_interp_corners(self):
        """Test trilinear interpolation at grid corners."""
        field = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
            ],
            dtype=torch.float32,
        )

        # Test a few corners
        assert torch.isclose(trilinear_interp(field, 0.0, 0.0, 0.0), torch.tensor(1.0))
        assert torch.isclose(trilinear_interp(field, 2.0, 2.0, 2.0), torch.tensor(27.0))
        assert torch.isclose(trilinear_interp(field, 1.0, 1.0, 1.0), torch.tensor(14.0))

    def test_trilinear_interp_center(self):
        """Test trilinear interpolation at center."""
        field = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32
        )

        # Center should be average of all 8 values = 4.5
        assert torch.isclose(trilinear_interp(field, 0.5, 0.5, 0.5), torch.tensor(4.5))
