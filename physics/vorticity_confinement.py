"""Vorticity confinement to restore small-scale turbulent details."""

from __future__ import annotations

import torch

from core import MACGrid2D, MACGrid3D
from kernels import grid_ops


def compute_vorticity_magnitude_gradient_2d(
    vorticity: torch.Tensor,
    dx: float,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of vorticity magnitude using torch tensors."""

    omega_mag = vorticity.abs()
    grad_x = torch.zeros_like(vorticity)
    grad_y = torch.zeros_like(vorticity)

    if ny > 2 and nx > 2:
        denom = 2.0 * dx
        grad_x[1 : ny - 1, 1 : nx - 1] = (
            omega_mag[1 : ny - 1, 2:nx] - omega_mag[1 : ny - 1, 0 : nx - 2]
        ) / denom
        grad_y[1 : ny - 1, 1 : nx - 1] = (
            omega_mag[2:ny, 1 : nx - 1] - omega_mag[0 : ny - 2, 1 : nx - 1]
        ) / denom

    return grad_x, grad_y


def compute_vorticity_magnitude_gradient_3d(
    vorticity: torch.Tensor,
    dx: float,
    nz: int,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradient of vorticity magnitude in 3D using torch tensors."""

    omega_mag = torch.linalg.vector_norm(vorticity, dim=-1)
    grad_x = torch.zeros_like(omega_mag)
    grad_y = torch.zeros_like(omega_mag)
    grad_z = torch.zeros_like(omega_mag)

    denom = 2.0 * dx
    if nx > 2:
        grad_x[:, :, 1 : nx - 1] = (
            omega_mag[:, :, 2:nx] - omega_mag[:, :, 0 : nx - 2]
        ) / denom
    if ny > 2:
        grad_y[:, 1 : ny - 1, :] = (
            omega_mag[:, 2:ny, :] - omega_mag[:, 0 : ny - 2, :]
        ) / denom
    if nz > 2:
        grad_z[1 : nz - 1, :, :] = (
            omega_mag[2:nz, :, :] - omega_mag[0 : nz - 2, :, :]
        ) / denom

    return grad_x, grad_y, grad_z


def apply_vorticity_confinement_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    vorticity: torch.Tensor,
    dx: float,
    dt: float,
    epsilon: float = 0.1,
) -> None:
    """Apply vorticity confinement force to restore rotational features."""

    ny, nx = vorticity.shape
    grad_x, grad_y = compute_vorticity_magnitude_gradient_2d(vorticity, dx, ny, nx)

    mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
    eps = torch.finfo(vorticity.dtype).eps
    mask = mag > eps
    N_x = torch.zeros_like(grad_x)
    N_y = torch.zeros_like(grad_y)
    N_x[mask] = grad_x[mask] / mag[mask]
    N_y[mask] = grad_y[mask] / mag[mask]

    force.u_data.zero_()
    force.v_data.zero_()

    f_x_center = epsilon * dx * N_y * vorticity
    f_y_center = -epsilon * dx * N_x * vorticity

    grid_ops.average_center_to_u_faces_2d(f_x_center, force.u_data, ny, nx)
    grid_ops.average_center_to_v_faces_2d(f_y_center, force.v_data, ny, nx)

    grid_ops.apply_force_to_velocity_2d(velocity, force, dt)


def apply_vorticity_confinement_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    vorticity: torch.Tensor,
    dx: float,
    dt: float,
    epsilon: float = 0.1,
) -> None:
    """Apply vorticity confinement force: f = ε * h * (N × ω)."""

    nz, ny, nx, _ = vorticity.shape
    grad_x, grad_y, grad_z = compute_vorticity_magnitude_gradient_3d(
        vorticity, dx, nz, ny, nx
    )

    mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    eps = torch.finfo(vorticity.dtype).eps
    mask = mag > eps
    N_x = torch.zeros_like(grad_x)
    N_y = torch.zeros_like(grad_y)
    N_z = torch.zeros_like(grad_z)
    N_x[mask] = grad_x[mask] / mag[mask]
    N_y[mask] = grad_y[mask] / mag[mask]
    N_z[mask] = grad_z[mask] / mag[mask]

    force.u_data.zero_()
    force.v_data.zero_()
    force.w_data.zero_()

    N = torch.stack((N_x, N_y, N_z), dim=-1)
    cross = torch.cross(N, vorticity, dim=-1)
    f_center = epsilon * dx * cross

    grid_ops.average_center_to_u_faces_3d(f_center[..., 0], force.u_data, nz, ny, nx)
    grid_ops.average_center_to_v_faces_3d(f_center[..., 1], force.v_data, nz, ny, nx)
    grid_ops.average_center_to_w_faces_3d(f_center[..., 2], force.w_data, nz, ny, nx)

    grid_ops.apply_force_to_velocity_3d(velocity, force, dt)
