"""Torch-based MAC grid utility operations."""

from __future__ import annotations

from typing import Tuple

import torch

from core import MACGrid2D, MACGrid3D

Tensor = torch.Tensor


def interpolate_velocity_to_cell_center_2d(
    u: Tensor, v: Tensor, y: int, x: int
) -> Tuple[Tensor, Tensor]:
    vel_x = (u[y, x] + u[y, x + 1]) * 0.5
    vel_y = (v[y, x] + v[y + 1, x]) * 0.5
    return vel_x, vel_y


def interpolate_velocity_to_cell_center_3d(
    u: Tensor, v: Tensor, w: Tensor, z: int, y: int, x: int
) -> Tuple[Tensor, Tensor, Tensor]:
    vel_x = (u[z, y, x] + u[z, y, x + 1]) * 0.5
    vel_y = (v[z, y, x] + v[z, y + 1, x]) * 0.5
    vel_z = (w[z, y, x] + w[z + 1, y, x]) * 0.5
    return vel_x, vel_y, vel_z


def interpolate_v_to_u_face_2d(v: Tensor, y: int, x: int) -> Tensor:
    return (v[y, x] + v[y, x - 1] + v[y + 1, x - 1] + v[y + 1, x]) * 0.25


def interpolate_u_to_v_face_2d(u: Tensor, y: int, x: int) -> Tensor:
    return (u[y, x] + u[y, x + 1] + u[y - 1, x + 1] + u[y - 1, x]) * 0.25


def interpolate_v_to_u_face_3d(v: Tensor, z: int, y: int, x: int) -> Tensor:
    return (v[z, y, x] + v[z, y, x - 1] + v[z, y + 1, x - 1] + v[z, y + 1, x]) * 0.25


def interpolate_w_to_u_face_3d(w: Tensor, z: int, y: int, x: int) -> Tensor:
    return (w[z, y, x] + w[z, y, x - 1] + w[z + 1, y, x - 1] + w[z + 1, y, x]) * 0.25


def interpolate_u_to_v_face_3d(u: Tensor, z: int, y: int, x: int) -> Tensor:
    return (u[z, y, x] + u[z, y, x + 1] + u[z, y - 1, x + 1] + u[z, y - 1, x]) * 0.25


def interpolate_w_to_v_face_3d(w: Tensor, z: int, y: int, x: int) -> Tensor:
    return (w[z, y, x] + w[z, y - 1, x] + w[z + 1, y - 1, x] + w[z + 1, y, x]) * 0.25


def interpolate_u_to_w_face_3d(u: Tensor, z: int, y: int, x: int) -> Tensor:
    return (u[z, y, x] + u[z, y, x + 1] + u[z - 1, y, x + 1] + u[z - 1, y, x]) * 0.25


def interpolate_v_to_w_face_3d(v: Tensor, z: int, y: int, x: int) -> Tensor:
    return (v[z, y, x] + v[z, y + 1, x] + v[z - 1, y + 1, x] + v[z - 1, y, x]) * 0.25


def clamp_to_cell_center_2d(
    x: float, y: float, nx: int, ny: int
) -> Tuple[float, float]:
    return max(1.0, min(x, nx - 2.0)), max(1.0, min(y, ny - 2.0))


def clamp_to_cell_center_3d(
    x: float, y: float, z: float, nx: int, ny: int, nz: int
) -> Tuple[float, float, float]:
    return (
        max(1.0, min(x, nx - 2.0)),
        max(1.0, min(y, ny - 2.0)),
        max(1.0, min(z, nz - 2.0)),
    )


def clamp_to_u_face_2d(x: float, y: float, nx: int, ny: int) -> Tuple[float, float]:
    return max(1.5, min(x, nx - 1.5)), max(1.5, min(y, ny - 2.5))


def clamp_to_u_face_3d(
    x: float, y: float, z: float, nx: int, ny: int, nz: int
) -> Tuple[float, float, float]:
    return (
        max(1.5, min(x, nx - 1.5)),
        max(1.5, min(y, ny - 2.5)),
        max(1.5, min(z, nz - 2.5)),
    )


def clamp_to_v_face_2d(x: float, y: float, nx: int, ny: int) -> Tuple[float, float]:
    return max(1.5, min(x, nx - 2.5)), max(1.5, min(y, ny - 1.5))


def clamp_to_v_face_3d(
    x: float, y: float, z: float, nx: int, ny: int, nz: int
) -> Tuple[float, float, float]:
    return (
        max(1.5, min(x, nx - 2.5)),
        max(1.5, min(y, ny - 1.5)),
        max(1.5, min(z, nz - 2.5)),
    )


def clamp_to_w_face_3d(
    x: float, y: float, z: float, nx: int, ny: int, nz: int
) -> Tuple[float, float, float]:
    return (
        max(1.5, min(x, nx - 2.5)),
        max(1.5, min(y, ny - 2.5)),
        max(1.5, min(z, nz - 1.5)),
    )


def find_neighborhood_bounds_2d(
    field: Tensor, y: int, x: int, ny: int, nx: int, radius: int = 1
) -> Tuple[Tensor, Tensor]:
    val_min = field[y, x]
    val_max = field[y, x]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny_idx = y + dy
            nx_idx = x + dx
            if 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                val = field[ny_idx, nx_idx]
                val_min = torch.minimum(val_min, val)
                val_max = torch.maximum(val_max, val)
    return val_min, val_max


def find_neighborhood_bounds_3d(
    field: Tensor,
    z: int,
    y: int,
    x: int,
    nz: int,
    ny: int,
    nx: int,
    radius: int = 1,
) -> Tuple[Tensor, Tensor]:
    val_min = field[z, y, x]
    val_max = field[z, y, x]
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nz_idx = z + dz
                ny_idx = y + dy
                nx_idx = x + dx
                if 0 <= nz_idx < nz and 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                    val = field[nz_idx, ny_idx, nx_idx]
                    val_min = torch.minimum(val_min, val)
                    val_max = torch.maximum(val_max, val)
    return val_min, val_max


def find_neighborhood_bounds_1d_y_2d(
    field: Tensor, y: int, x: int, ny: int
) -> Tuple[Tensor, Tensor]:
    val_min = field[y, x]
    val_max = field[y, x]
    for dy in range(-1, 2):
        ny_idx = y + dy
        if 0 <= ny_idx < ny:
            val = field[ny_idx, x]
            val_min = torch.minimum(val_min, val)
            val_max = torch.maximum(val_max, val)
    return val_min, val_max


def reset_forces_2d(force: MACGrid2D) -> None:
    """Reset forces (differentiable version)"""
    force.u_data = torch.zeros_like(force.u_data)
    force.v_data = torch.zeros_like(force.v_data)


def reset_forces_3d(force: MACGrid3D) -> None:
    """Reset forces (differentiable version)"""
    force.u_data = torch.zeros_like(force.u_data)
    force.v_data = torch.zeros_like(force.v_data)
    force.w_data = torch.zeros_like(force.w_data)


def apply_force_to_velocity_2d(
    velocity: MACGrid2D, force: MACGrid2D, dt: float
) -> None:
    """Apply forces to velocity (differentiable version)"""
    velocity.u_data = velocity.u_data + force.u_data * dt
    velocity.v_data = velocity.v_data + force.v_data * dt


def apply_force_to_velocity_3d(
    velocity: MACGrid3D, force: MACGrid3D, dt: float
) -> None:
    """Apply forces to velocity (differentiable version)"""
    velocity.u_data = velocity.u_data + force.u_data * dt
    velocity.v_data = velocity.v_data + force.v_data * dt
    velocity.w_data = velocity.w_data + force.w_data * dt


def average_center_to_u_faces_2d(
    center_values: Tensor, u_faces: Tensor, ny: int, nx: int
) -> None:
    """Average center values to u-faces (differentiable version)"""
    u_faces_new = u_faces.clone()
    u_faces_new[1 : ny - 1, 1:nx] = 0.5 * (
        center_values[1 : ny - 1, 0 : nx - 1] + center_values[1 : ny - 1, 1:nx]
    )
    # Return by modifying the reference (for compatibility)
    u_faces.copy_(u_faces_new)


def average_center_to_v_faces_2d(
    center_values: Tensor, v_faces: Tensor, ny: int, nx: int
) -> None:
    """Average center values to v-faces (differentiable version)"""
    v_faces_new = v_faces.clone()
    v_faces_new[1:ny, 1 : nx - 1] = 0.5 * (
        center_values[0 : ny - 1, 1 : nx - 1] + center_values[1:ny, 1 : nx - 1]
    )
    v_faces.copy_(v_faces_new)


def average_center_to_u_faces_3d(
    center_values: Tensor, u_faces: Tensor, nz: int, ny: int, nx: int
) -> None:
    """Average center values to u-faces (differentiable version)"""
    u_faces_new = u_faces.clone()
    u_faces_new[1 : nz - 1, 1 : ny - 1, 1:nx] = 0.5 * (
        center_values[1 : nz - 1, 1 : ny - 1, 0 : nx - 1]
        + center_values[1 : nz - 1, 1 : ny - 1, 1:nx]
    )
    u_faces.copy_(u_faces_new)


def average_center_to_v_faces_3d(
    center_values: Tensor, v_faces: Tensor, nz: int, ny: int, nx: int
) -> None:
    """Average center values to v-faces (differentiable version)"""
    v_faces_new = v_faces.clone()
    v_faces_new[1 : nz - 1, 1:ny, 1 : nx - 1] = 0.5 * (
        center_values[1 : nz - 1, 0 : ny - 1, 1 : nx - 1]
        + center_values[1 : nz - 1, 1:ny, 1 : nx - 1]
    )
    v_faces.copy_(v_faces_new)


def average_center_to_w_faces_3d(
    center_values: Tensor, w_faces: Tensor, nz: int, ny: int, nx: int
) -> None:
    """Average center values to w-faces (differentiable version)"""
    w_faces_new = w_faces.clone()
    w_faces_new[1:nz, 1 : ny - 1, 1 : nx - 1] = 0.5 * (
        center_values[0 : nz - 1, 1 : ny - 1, 1 : nx - 1]
        + center_values[1:nz, 1 : ny - 1, 1 : nx - 1]
    )
    w_faces.copy_(w_faces_new)
