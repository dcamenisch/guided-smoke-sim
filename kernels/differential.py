"""Differential operators (curl, vorticity) for fluid simulation using Torch tensors."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def compute_vorticity_kernel_2d(
    vorticity: Tensor, u: Tensor, v: Tensor, dx: float, ny: int, nx: int
) -> None:
    """Compute 2D scalar vorticity ω = ∂v/∂x - ∂u/∂y on Torch grids."""

    if ny < 4 or nx < 4:
        vorticity.zero_()
        return

    inv_2dx = 0.5 / dx
    y_slice = slice(2, ny - 2)
    x_slice = slice(2, nx - 2)

    dvdx = (v[2 : ny - 2, 3 : nx - 1] - v[2 : ny - 2, 1 : nx - 3]) * inv_2dx
    dudy = (u[3 : ny - 1, 2 : nx - 2] - u[1 : ny - 3, 2 : nx - 2]) * inv_2dx

    vorticity[y_slice, x_slice] = dvdx - dudy


def compute_vorticity_kernel_3d(
    vorticity: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """Compute 3D vorticity vector ω = ∇ × u using Torch tensors."""

    if nz < 4 or ny < 4 or nx < 4:
        vorticity.zero_()
        return

    inv_2dx = 0.5 / dx
    z_slice = slice(2, nz - 2)
    y_slice = slice(2, ny - 2)
    x_slice = slice(2, nx - 2)

    dwdy = (
        w[2 : nz - 2, 3 : ny - 1, 2 : nx - 2] - w[2 : nz - 2, 1 : ny - 3, 2 : nx - 2]
    ) * inv_2dx
    dvdz = (
        v[3 : nz - 1, 2 : ny - 2, 2 : nx - 2] - v[1 : nz - 3, 2 : ny - 2, 2 : nx - 2]
    ) * inv_2dx
    dudz = (
        u[3 : nz - 1, 2 : ny - 2, 2 : nx - 2] - u[1 : nz - 3, 2 : ny - 2, 2 : nx - 2]
    ) * inv_2dx
    dwdx = (
        w[2 : nz - 2, 2 : ny - 2, 3 : nx - 1] - w[2 : nz - 2, 2 : ny - 2, 1 : nx - 3]
    ) * inv_2dx
    dvdx = (
        v[2 : nz - 2, 2 : ny - 2, 3 : nx - 1] - v[2 : nz - 2, 2 : ny - 2, 1 : nx - 3]
    ) * inv_2dx
    dudy = (
        u[2 : nz - 2, 3 : ny - 1, 2 : nx - 2] - u[2 : nz - 2, 1 : ny - 3, 2 : nx - 2]
    ) * inv_2dx

    vorticity[z_slice, y_slice, x_slice, 0] = dwdy - dvdz
    vorticity[z_slice, y_slice, x_slice, 1] = dudz - dwdx
    vorticity[z_slice, y_slice, x_slice, 2] = dvdx - dudy
