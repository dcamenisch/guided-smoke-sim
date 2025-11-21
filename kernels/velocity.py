"""Torch-based velocity correction kernels for pressure projection."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def correct_velocity_kernel_2d(
    u: Tensor, v: Tensor, pressure: Tensor, dx: float, dt: float, ny: int, nx: int
) -> None:
    """Subtract pressure gradient from MAC velocities in 2D."""
    grad_p_x = (pressure[1 : ny - 1, 1:nx] - pressure[1 : ny - 1, 0 : nx - 1]) / dx
    u[1 : ny - 1, 1:nx] -= dt * grad_p_x

    grad_p_y = (pressure[1:ny, 1 : nx - 1] - pressure[0 : ny - 1, 1 : nx - 1]) / dx
    v[1:ny, 1 : nx - 1] -= dt * grad_p_y


def correct_velocity_kernel_3d(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    pressure: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """Subtract pressure gradient from MAC velocities in 3D."""
    grad_p_x = (
        pressure[1 : nz - 1, 1 : ny - 1, 1:nx]
        - pressure[1 : nz - 1, 1 : ny - 1, 0 : nx - 1]
    ) / dx
    u[1 : nz - 1, 1 : ny - 1, 1:nx] -= dt * grad_p_x

    grad_p_y = (
        pressure[1 : nz - 1, 1:ny, 1 : nx - 1]
        - pressure[1 : nz - 1, 0 : ny - 1, 1 : nx - 1]
    ) / dx
    v[1 : nz - 1, 1:ny, 1 : nx - 1] -= dt * grad_p_y

    grad_p_z = (
        pressure[1:nz, 1 : ny - 1, 1 : nx - 1]
        - pressure[0 : nz - 1, 1 : ny - 1, 1 : nx - 1]
    ) / dx
    w[1:nz, 1 : ny - 1, 1 : nx - 1] -= dt * grad_p_z
