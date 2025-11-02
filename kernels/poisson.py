"""Poisson equation solvers using Jacobi iteration."""

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def solve_poisson_jacobi_2d(
    pressure, divergence, dx, dt, rho, max_iter, tolerance, ny, nx
):
    """Optimized 2D Jacobi solver with Numba JIT compilation

    Solves: ∇²p = -ρ/dt * ∇·u

    Args:
        pressure: Initial pressure field (ny, nx)
        divergence: Velocity divergence field (ny, nx)
        dx: Grid spacing
        dt: Time step
        rho: Fluid density
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        ny, nx: Grid dimensions

    Returns:
        Updated pressure field
    """
    dx2 = dx * dx
    pressure_new = pressure.copy()

    for iteration in range(max_iter):
        # Jacobi update
        for y in prange(1, ny - 1):
            for x in range(1, nx - 1):
                b = -divergence[y, x] / dt * rho
                pressure_new[y, x] = (
                    dx2 * b
                    + pressure[y - 1, x]
                    + pressure[y + 1, x]
                    + pressure[y, x - 1]
                    + pressure[y, x + 1]
                ) / 4.0

        # Swap arrays
        pressure, pressure_new = pressure_new, pressure

        # Check convergence every 10 iterations to save time
        if iteration % 10 == 0:
            residual = 0.0
            for y in prange(1, ny - 1):
                for x in range(1, nx - 1):
                    b = -divergence[y, x] / dt * rho
                    cell_residual = (
                        b
                        - (
                            4 * pressure[y, x]
                            - pressure[y - 1, x]
                            - pressure[y + 1, x]
                            - pressure[y, x - 1]
                            - pressure[y, x + 1]
                        )
                        / dx2
                    )
                    residual += cell_residual * cell_residual

            residual = np.sqrt(residual) / ((nx - 2) * (ny - 2))
            if residual < tolerance:
                break

    return pressure


@jit(nopython=True, parallel=True, cache=True)
def solve_poisson_jacobi_3d(
    pressure, divergence, dx, dt, rho, max_iter, tolerance, nz, ny, nx
):
    """Optimized 3D Jacobi solver with Numba JIT compilation

    Solves: ∇²p = -ρ/dt * ∇·u

    Args:
        pressure: Initial pressure field (nz, ny, nx)
        divergence: Velocity divergence field (nz, ny, nx)
        dx: Grid spacing
        dt: Time step
        rho: Fluid density
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        nz, ny, nx: Grid dimensions

    Returns:
        Updated pressure field
    """
    dx2 = dx * dx
    pressure_new = pressure.copy()

    for iteration in range(max_iter):
        # Jacobi update
        for z in prange(1, nz - 1):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    b = -divergence[z, y, x] / dt * rho
                    pressure_new[z, y, x] = (
                        dx2 * b
                        + pressure[z - 1, y, x]
                        + pressure[z + 1, y, x]
                        + pressure[z, y - 1, x]
                        + pressure[z, y + 1, x]
                        + pressure[z, y, x - 1]
                        + pressure[z, y, x + 1]
                    ) / 6.0

        # Swap arrays
        pressure, pressure_new = pressure_new, pressure

        # Check convergence every 10 iterations to save time
        if iteration % 10 == 0:
            residual = 0.0
            for z in prange(1, nz - 1):
                for y in range(1, ny - 1):
                    for x in range(1, nx - 1):
                        b = -divergence[z, y, x] / dt * rho
                        cell_residual = (
                            b
                            - (
                                6 * pressure[z, y, x]
                                - pressure[z - 1, y, x]
                                - pressure[z + 1, y, x]
                                - pressure[z, y - 1, x]
                                - pressure[z, y + 1, x]
                                - pressure[z, y, x - 1]
                                - pressure[z, y, x + 1]
                            )
                            / dx2
                        )
                        residual += cell_residual * cell_residual

            residual = np.sqrt(residual) / ((nx - 2) * (ny - 2) * (nz - 2))
            if residual < tolerance:
                break

    return pressure
