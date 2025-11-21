"""Semi-Lagrangian advection kernels with MacCormack and SSPRK3 support."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from kernels.interpolation import bilinear_interp, trilinear_interp
from kernels import grid_ops

Tensor = torch.Tensor


def _meshgrid_ij(*coords: Tensor) -> Tuple[Tensor, ...]:
    """Create IJ-indexed meshgrids with backward compatibility."""
    try:
        return torch.meshgrid(*coords, indexing="ij")
    except TypeError:  # Older PyTorch
        return torch.meshgrid(*coords)


def _normalize_coord(coord: Tensor, size: int) -> Tensor:
    """Normalize coordinates to [-1, 1] for grid_sample."""
    if size <= 1:
        raise ValueError("Grid dimension must be at least 2 for sampling")
    scale = coord.new_tensor(2.0 / float(size - 1))
    return coord * scale - 1.0


def _make_sampling_grid_2d(pos_x: Tensor, pos_y: Tensor, nx: int, ny: int) -> Tensor:
    norm_x = _normalize_coord(pos_x, nx)
    norm_y = _normalize_coord(pos_y, ny)
    return torch.stack((norm_x, norm_y), dim=-1)


def _make_sampling_grid_3d(
    pos_x: Tensor, pos_y: Tensor, pos_z: Tensor, nx: int, ny: int, nz: int
) -> Tensor:
    norm_x = _normalize_coord(pos_x, nx)
    norm_y = _normalize_coord(pos_y, ny)
    norm_z = _normalize_coord(pos_z, nz)
    return torch.stack((norm_x, norm_y, norm_z), dim=-1)


def _sample_scalar_field_2d(
    field: Tensor, pos_x: Tensor, pos_y: Tensor, nx: int, ny: int
) -> Tensor:
    grid = _make_sampling_grid_2d(pos_x, pos_y, nx, ny).unsqueeze(0)
    field_4d = field.unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        field_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, 0]


def _sample_scalar_field_3d(
    field: Tensor,
    pos_x: Tensor,
    pos_y: Tensor,
    pos_z: Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> Tensor:
    grid = _make_sampling_grid_3d(pos_x, pos_y, pos_z, nx, ny, nz).unsqueeze(0)
    field_5d = field.unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        field_5d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, 0]


def _sample_vector_field_2d(
    field: Tensor, pos_x: Tensor, pos_y: Tensor, nx: int, ny: int
) -> Tuple[Tensor, Tensor]:
    grid = _make_sampling_grid_2d(pos_x, pos_y, nx, ny).unsqueeze(0)
    field_4d = field.unsqueeze(0)
    sampled = F.grid_sample(
        field_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    return sampled[0], sampled[1]


def _sample_vector_field_3d(
    field: Tensor,
    pos_x: Tensor,
    pos_y: Tensor,
    pos_z: Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    grid = _make_sampling_grid_3d(pos_x, pos_y, pos_z, nx, ny, nz).unsqueeze(0)
    field_5d = field.unsqueeze(0)
    sampled = F.grid_sample(
        field_5d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    return sampled[0], sampled[1], sampled[2]


def _clamp_cell_center_tensor_2d(
    pos_x: Tensor, pos_y: Tensor, nx: int, ny: int
) -> Tuple[Tensor, Tensor]:
    clamp_min = pos_x.new_tensor(1.0)
    clamp_max_x = pos_x.new_tensor(max(1.0, float(nx - 2)))
    clamp_max_y = pos_y.new_tensor(max(1.0, float(ny - 2)))
    return (
        pos_x.clamp(clamp_min, clamp_max_x),
        pos_y.clamp(clamp_min, clamp_max_y),
    )


def _clamp_cell_center_tensor_3d(
    pos_x: Tensor, pos_y: Tensor, pos_z: Tensor, nx: int, ny: int, nz: int
) -> Tuple[Tensor, Tensor, Tensor]:
    clamp_min = pos_x.new_tensor(1.0)
    clamp_max_x = pos_x.new_tensor(max(1.0, float(nx - 2)))
    clamp_max_y = pos_y.new_tensor(max(1.0, float(ny - 2)))
    clamp_max_z = pos_z.new_tensor(max(1.0, float(nz - 2)))
    return (
        pos_x.clamp(clamp_min, clamp_max_x),
        pos_y.clamp(clamp_min, clamp_max_y),
        pos_z.clamp(clamp_min, clamp_max_z),
    )


def _clamp_u_face_tensor_2d(
    pos_x: Tensor, pos_y: Tensor, nx: int, ny: int
) -> Tuple[Tensor, Tensor]:
    clamp_min_x = pos_x.new_tensor(1.5)
    clamp_max_x = pos_x.new_tensor(max(1.5, float(nx - 1.5)))
    clamp_min_y = pos_y.new_tensor(1.5)
    clamp_max_y = pos_y.new_tensor(max(1.5, float(ny - 2.5)))
    return (
        pos_x.clamp(clamp_min_x, clamp_max_x),
        pos_y.clamp(clamp_min_y, clamp_max_y),
    )


def _clamp_v_face_tensor_2d(
    pos_x: Tensor, pos_y: Tensor, nx: int, ny: int
) -> Tuple[Tensor, Tensor]:
    clamp_min_x = pos_x.new_tensor(1.5)
    clamp_max_x = pos_x.new_tensor(max(1.5, float(nx - 2.5)))
    clamp_min_y = pos_y.new_tensor(1.5)
    clamp_max_y = pos_y.new_tensor(max(1.5, float(ny - 1.5)))
    return (
        pos_x.clamp(clamp_min_x, clamp_max_x),
        pos_y.clamp(clamp_min_y, clamp_max_y),
    )


def _clamp_u_face_tensor_3d(
    pos_x: Tensor, pos_y: Tensor, pos_z: Tensor, nx: int, ny: int, nz: int
) -> Tuple[Tensor, Tensor, Tensor]:
    clamp_min_x = pos_x.new_tensor(1.5)
    clamp_max_x = pos_x.new_tensor(max(1.5, float(nx - 1.5)))
    clamp_min_y = pos_y.new_tensor(1.5)
    clamp_max_y = pos_y.new_tensor(max(1.5, float(ny - 2.5)))
    clamp_min_z = pos_z.new_tensor(1.5)
    clamp_max_z = pos_z.new_tensor(max(1.5, float(nz - 2.5)))
    return (
        pos_x.clamp(clamp_min_x, clamp_max_x),
        pos_y.clamp(clamp_min_y, clamp_max_y),
        pos_z.clamp(clamp_min_z, clamp_max_z),
    )


def _clamp_v_face_tensor_3d(
    pos_x: Tensor, pos_y: Tensor, pos_z: Tensor, nx: int, ny: int, nz: int
) -> Tuple[Tensor, Tensor, Tensor]:
    clamp_min_x = pos_x.new_tensor(1.5)
    clamp_max_x = pos_x.new_tensor(max(1.5, float(nx - 2.5)))
    clamp_min_y = pos_y.new_tensor(1.5)
    clamp_max_y = pos_y.new_tensor(max(1.5, float(ny - 1.5)))
    clamp_min_z = pos_z.new_tensor(1.5)
    clamp_max_z = pos_z.new_tensor(max(1.5, float(nz - 2.5)))
    return (
        pos_x.clamp(clamp_min_x, clamp_max_x),
        pos_y.clamp(clamp_min_y, clamp_max_y),
        pos_z.clamp(clamp_min_z, clamp_max_z),
    )


def _clamp_w_face_tensor_3d(
    pos_x: Tensor, pos_y: Tensor, pos_z: Tensor, nx: int, ny: int, nz: int
) -> Tuple[Tensor, Tensor, Tensor]:
    clamp_min_x = pos_x.new_tensor(1.5)
    clamp_max_x = pos_x.new_tensor(max(1.5, float(nx - 2.5)))
    clamp_min_y = pos_y.new_tensor(1.5)
    clamp_max_y = pos_y.new_tensor(max(1.5, float(ny - 2.5)))
    clamp_min_z = pos_z.new_tensor(1.5)
    clamp_max_z = pos_z.new_tensor(max(1.5, float(nz - 1.5)))
    return (
        pos_x.clamp(clamp_min_x, clamp_max_x),
        pos_y.clamp(clamp_min_y, clamp_max_y),
        pos_z.clamp(clamp_min_z, clamp_max_z),
    )


def _as_tensor(value: Tensor) -> Tensor:
    return value


def _tensor_from_like(reference: Tensor, value: Tensor) -> Tensor:
    if value.dtype == reference.dtype and value.device == reference.device:
        return value
    return value.to(device=reference.device, dtype=reference.dtype)


def _prepare_output(reference: Tensor, value: Tensor) -> Tuple[Tensor, Tensor, bool]:
    tensor = _tensor_from_like(reference, value)
    needs_copy = tensor is not value
    return tensor, value, needs_copy


def _finalize_output(tensor: Tensor, original: Tensor, needs_copy: bool) -> None:
    if needs_copy:
        original.copy_(tensor.to(device=original.device, dtype=original.dtype))


def _velocity_is_zero(epsilon: float, *components: Tensor) -> bool:
    return all(torch.all(component.abs() <= epsilon) for component in components)


def advect_density_kernel_2d(
    density: Tensor,
    density_new: Tensor,
    u: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """2D density advection with semi-Lagrangian method.

    Args:
        density: Current density field (ny, nx)
        density_new: Output density field (ny, nx)
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    density_tensor = _as_tensor(density)
    density_new_tensor, density_new_original, density_new_needs_copy = _prepare_output(
        density_tensor, density_new
    )
    u_tensor = _tensor_from_like(density_tensor, u)
    v_tensor = _tensor_from_like(density_tensor, v)

    if ny <= 2 or nx <= 2:
        density_new_tensor.copy_(density_tensor)
        _finalize_output(
            density_new_tensor, density_new_original, density_new_needs_copy
        )
        return

    density_new_tensor.copy_(density_tensor)
    device = density_tensor.device
    dtype = density_tensor.dtype

    y_coords = torch.arange(ny, device=device, dtype=dtype)
    x_coords = torch.arange(nx, device=device, dtype=dtype)
    grid_y, grid_x = _meshgrid_ij(y_coords, x_coords)

    dt_over_dx = density_tensor.new_tensor(dt / dx)
    center_u = 0.5 * (u_tensor[:, :-1] + u_tensor[:, 1:])
    center_v = 0.5 * (v_tensor[:-1, :] + v_tensor[1:, :])

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, center_u, center_v):
        _finalize_output(
            density_new_tensor, density_new_original, density_new_needs_copy
        )
        return

    last_x = grid_x - dt_over_dx * center_u
    last_y = grid_y - dt_over_dx * center_v
    last_x, last_y = _clamp_cell_center_tensor_2d(last_x, last_y, nx, ny)

    if rk_order == 3:
        velocity_field = torch.stack((center_u, center_v), dim=0)

        pos2_x = grid_x - 0.5 * dt_over_dx * center_u
        pos2_y = grid_y - 0.5 * dt_over_dx * center_v
        pos2_x, pos2_y = _clamp_cell_center_tensor_2d(pos2_x, pos2_y, nx, ny)
        k2_x, k2_y = _sample_vector_field_2d(velocity_field, pos2_x, pos2_y, nx, ny)

        pos3_x = grid_x - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y - 0.75 * dt_over_dx * k2_y
        pos3_x, pos3_y = _clamp_cell_center_tensor_2d(pos3_x, pos3_y, nx, ny)
        k3_x, k3_y = _sample_vector_field_2d(velocity_field, pos3_x, pos3_y, nx, ny)

        last_x = grid_x - dt_over_dx * (
            2.0 / 9.0 * center_u + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y - dt_over_dx * (
            2.0 / 9.0 * center_v + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_x, last_y = _clamp_cell_center_tensor_2d(last_x, last_y, nx, ny)

    sampled_density = _sample_scalar_field_2d(density_tensor, last_x, last_y, nx, ny)
    density_new_tensor[1 : ny - 1, 1 : nx - 1] = sampled_density[1 : ny - 1, 1 : nx - 1]

    _finalize_output(density_new_tensor, density_new_original, density_new_needs_copy)


def advect_u_velocity_kernel_2d(
    u: Tensor,
    u_new: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """2D u-velocity advection on MAC grid.

    Args:
        u: Current x-velocity (ny, nx+1)
        u_new: Output x-velocity (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    u_tensor = _as_tensor(u)
    u_new_tensor, u_new_original, u_new_needs_copy = _prepare_output(u_tensor, u_new)
    v_tensor = _tensor_from_like(u_tensor, v)

    if ny <= 2 or nx <= 1:
        u_new_tensor.copy_(u_tensor)
        _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)
        return

    u_new_tensor.copy_(u_tensor)
    device = u_tensor.device
    dtype = u_tensor.dtype

    y_coords = torch.arange(ny, device=device, dtype=dtype)
    x_coords = torch.arange(nx + 1, device=device, dtype=dtype)
    grid_y, grid_x = _meshgrid_ij(y_coords, x_coords)

    vel_x_field = u_tensor
    vel_y_field = torch.zeros_like(u_tensor)
    if ny > 2 and nx > 1:
        vel_y_field[1 : ny - 1, 1:nx] = 0.25 * (
            v_tensor[1 : ny - 1, 1:nx]
            + v_tensor[1 : ny - 1, 0 : nx - 1]
            + v_tensor[2:ny, 0 : nx - 1]
            + v_tensor[2:ny, 1:nx]
        )

    inner_slice = (slice(1, ny - 1), slice(1, nx))
    grid_x_inner = grid_x[inner_slice]
    grid_y_inner = grid_y[inner_slice]
    vel_x_inner = vel_x_field[inner_slice]
    vel_y_inner = vel_y_field[inner_slice]

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, vel_x_inner, vel_y_inner):
        _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)
        return

    dt_over_dx = u_tensor.new_tensor(dt / dx)
    last_x = grid_x_inner - dt_over_dx * vel_x_inner
    last_y = grid_y_inner - dt_over_dx * vel_y_inner
    last_x, last_y = _clamp_u_face_tensor_2d(last_x, last_y, nx, ny)

    if rk_order == 3:
        velocity_field = torch.stack((vel_x_field, vel_y_field), dim=0)

        pos2_x = grid_x_inner - 0.5 * dt_over_dx * vel_x_inner
        pos2_y = grid_y_inner - 0.5 * dt_over_dx * vel_y_inner
        pos2_x, pos2_y = _clamp_u_face_tensor_2d(pos2_x, pos2_y, nx, ny)
        k2_x, k2_y = _sample_vector_field_2d(velocity_field, pos2_x, pos2_y, nx + 1, ny)

        pos3_x = grid_x_inner - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y_inner - 0.75 * dt_over_dx * k2_y
        pos3_x, pos3_y = _clamp_u_face_tensor_2d(pos3_x, pos3_y, nx, ny)
        k3_x, k3_y = _sample_vector_field_2d(velocity_field, pos3_x, pos3_y, nx + 1, ny)

        last_x = grid_x_inner - dt_over_dx * (
            2.0 / 9.0 * vel_x_inner + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y_inner - dt_over_dx * (
            2.0 / 9.0 * vel_y_inner + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_x, last_y = _clamp_u_face_tensor_2d(last_x, last_y, nx, ny)

    sampled_u = _sample_scalar_field_2d(u_tensor, last_x, last_y, nx + 1, ny)
    u_new_tensor[inner_slice] = sampled_u

    _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)


def advect_v_velocity_kernel_2d(
    v: Tensor,
    v_new: Tensor,
    u: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """2D v-velocity advection on MAC grid.

    Args:
        v: Current y-velocity (ny+1, nx)
        v_new: Output y-velocity (ny+1, nx)
        u: x-velocity component (ny, nx+1)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    v_tensor = _as_tensor(v)
    v_new_tensor, v_new_original, v_new_needs_copy = _prepare_output(v_tensor, v_new)
    u_tensor = _tensor_from_like(v_tensor, u)

    if ny <= 1 or nx <= 2:
        v_new_tensor.copy_(v_tensor)
        _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)
        return

    v_new_tensor.copy_(v_tensor)
    device = v_tensor.device
    dtype = v_tensor.dtype

    y_coords = torch.arange(ny + 1, device=device, dtype=dtype)
    x_coords = torch.arange(nx, device=device, dtype=dtype)
    grid_y, grid_x = _meshgrid_ij(y_coords, x_coords)

    vel_y_field = v_tensor
    vel_x_field = torch.zeros_like(v_tensor)
    if ny > 1 and nx > 2:
        vel_x_field[1:ny, 1 : nx - 1] = 0.25 * (
            u_tensor[1:ny, 1 : nx - 1]
            + u_tensor[1:ny, 2:nx]
            + u_tensor[0 : ny - 1, 2:nx]
            + u_tensor[0 : ny - 1, 1 : nx - 1]
        )

    inner_slice = (slice(1, ny), slice(1, nx - 1))
    grid_x_inner = grid_x[inner_slice]
    grid_y_inner = grid_y[inner_slice]
    vel_x_inner = vel_x_field[inner_slice]
    vel_y_inner = vel_y_field[inner_slice]

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, vel_x_inner, vel_y_inner):
        _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)
        return

    dt_over_dx = v_tensor.new_tensor(dt / dx)
    last_x = grid_x_inner - dt_over_dx * vel_x_inner
    last_y = grid_y_inner - dt_over_dx * vel_y_inner
    last_x, last_y = _clamp_v_face_tensor_2d(last_x, last_y, nx, ny)

    if rk_order == 3:
        velocity_field = torch.stack((vel_x_field, vel_y_field), dim=0)

        pos2_x = grid_x_inner - 0.5 * dt_over_dx * vel_x_inner
        pos2_y = grid_y_inner - 0.5 * dt_over_dx * vel_y_inner
        pos2_x, pos2_y = _clamp_v_face_tensor_2d(pos2_x, pos2_y, nx, ny)
        k2_x, k2_y = _sample_vector_field_2d(velocity_field, pos2_x, pos2_y, nx, ny + 1)

        pos3_x = grid_x_inner - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y_inner - 0.75 * dt_over_dx * k2_y
        pos3_x, pos3_y = _clamp_v_face_tensor_2d(pos3_x, pos3_y, nx, ny)
        k3_x, k3_y = _sample_vector_field_2d(velocity_field, pos3_x, pos3_y, nx, ny + 1)

        last_x = grid_x_inner - dt_over_dx * (
            2.0 / 9.0 * vel_x_inner + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y_inner - dt_over_dx * (
            2.0 / 9.0 * vel_y_inner + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_x, last_y = _clamp_v_face_tensor_2d(last_x, last_y, nx, ny)

    sampled_v = _sample_scalar_field_2d(v_tensor, last_x, last_y, nx, ny + 1)
    v_new_tensor[inner_slice] = sampled_v

    _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)


def advect_density_kernel_3d(
    density: Tensor,
    density_new: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """3D density advection with semi-Lagrangian method.

    Args:
        density: Current density field (nz, ny, nx)
        density_new: Output density field (nz, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    density_tensor = _as_tensor(density)
    density_new_tensor, density_new_original, density_new_needs_copy = _prepare_output(
        density_tensor, density_new
    )
    u_tensor = _tensor_from_like(density_tensor, u)
    v_tensor = _tensor_from_like(density_tensor, v)
    w_tensor = _tensor_from_like(density_tensor, w)

    if nz <= 2 or ny <= 2 or nx <= 2:
        density_new_tensor.copy_(density_tensor)
        _finalize_output(
            density_new_tensor, density_new_original, density_new_needs_copy
        )
        return

    density_new_tensor.copy_(density_tensor)
    device = density_tensor.device
    dtype = density_tensor.dtype

    z_coords = torch.arange(nz, device=device, dtype=dtype)
    y_coords = torch.arange(ny, device=device, dtype=dtype)
    x_coords = torch.arange(nx, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = _meshgrid_ij(z_coords, y_coords, x_coords)

    dt_over_dx = density_tensor.new_tensor(dt / dx)
    center_u = 0.5 * (u_tensor[:, :, :-1] + u_tensor[:, :, 1:])
    center_v = 0.5 * (v_tensor[:, :-1, :] + v_tensor[:, 1:, :])
    center_w = 0.5 * (w_tensor[:-1, :, :] + w_tensor[1:, :, :])

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, center_u, center_v, center_w):
        _finalize_output(
            density_new_tensor, density_new_original, density_new_needs_copy
        )
        return

    last_x = grid_x - dt_over_dx * center_u
    last_y = grid_y - dt_over_dx * center_v
    last_z = grid_z - dt_over_dx * center_w
    last_x, last_y, last_z = _clamp_cell_center_tensor_3d(
        last_x, last_y, last_z, nx, ny, nz
    )

    if rk_order == 3:
        velocity_field = torch.stack((center_u, center_v, center_w), dim=0)

        pos2_x = grid_x - 0.5 * dt_over_dx * center_u
        pos2_y = grid_y - 0.5 * dt_over_dx * center_v
        pos2_z = grid_z - 0.5 * dt_over_dx * center_w
        pos2_x, pos2_y, pos2_z = _clamp_cell_center_tensor_3d(
            pos2_x, pos2_y, pos2_z, nx, ny, nz
        )
        k2_x, k2_y, k2_z = _sample_vector_field_3d(
            velocity_field, pos2_x, pos2_y, pos2_z, nx, ny, nz
        )

        pos3_x = grid_x - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y - 0.75 * dt_over_dx * k2_y
        pos3_z = grid_z - 0.75 * dt_over_dx * k2_z
        pos3_x, pos3_y, pos3_z = _clamp_cell_center_tensor_3d(
            pos3_x, pos3_y, pos3_z, nx, ny, nz
        )
        k3_x, k3_y, k3_z = _sample_vector_field_3d(
            velocity_field, pos3_x, pos3_y, pos3_z, nx, ny, nz
        )

        last_x = grid_x - dt_over_dx * (
            2.0 / 9.0 * center_u + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y - dt_over_dx * (
            2.0 / 9.0 * center_v + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_z = grid_z - dt_over_dx * (
            2.0 / 9.0 * center_w + 3.0 / 9.0 * k2_z + 4.0 / 9.0 * k3_z
        )
        last_x, last_y, last_z = _clamp_cell_center_tensor_3d(
            last_x, last_y, last_z, nx, ny, nz
        )

    sampled_density = _sample_scalar_field_3d(
        density_tensor, last_x, last_y, last_z, nx, ny, nz
    )
    density_new_tensor[1 : nz - 1, 1 : ny - 1, 1 : nx - 1] = sampled_density[
        1 : nz - 1, 1 : ny - 1, 1 : nx - 1
    ]

    _finalize_output(density_new_tensor, density_new_original, density_new_needs_copy)


def advect_u_velocity_kernel_3d(
    u: Tensor,
    u_new: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """3D u-velocity advection on MAC grid.

    Args:
        u: Current x-velocity (nz, ny, nx+1)
        u_new: Output x-velocity (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    u_tensor = _as_tensor(u)
    u_new_tensor, u_new_original, u_new_needs_copy = _prepare_output(u_tensor, u_new)
    v_tensor = _tensor_from_like(u_tensor, v)
    w_tensor = _tensor_from_like(u_tensor, w)

    if nz <= 2 or ny <= 2 or nx <= 1:
        u_new_tensor.copy_(u_tensor)
        _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)
        return

    u_new_tensor.copy_(u_tensor)
    device = u_tensor.device
    dtype = u_tensor.dtype

    z_coords = torch.arange(nz, device=device, dtype=dtype)
    y_coords = torch.arange(ny, device=device, dtype=dtype)
    x_coords = torch.arange(nx + 1, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = _meshgrid_ij(z_coords, y_coords, x_coords)

    vel_x_field = u_tensor
    vel_y_field = torch.zeros_like(u_tensor)
    vel_z_field = torch.zeros_like(u_tensor)
    if nz > 2 and ny > 2 and nx > 1:
        vel_y_field[1 : nz - 1, 1 : ny - 1, 1:nx] = 0.25 * (
            v_tensor[1 : nz - 1, 1 : ny - 1, 1:nx]
            + v_tensor[1 : nz - 1, 1 : ny - 1, 0 : nx - 1]
            + v_tensor[1 : nz - 1, 2:ny, 0 : nx - 1]
            + v_tensor[1 : nz - 1, 2:ny, 1:nx]
        )
        vel_z_field[1 : nz - 1, 1 : ny - 1, 1:nx] = 0.25 * (
            w_tensor[1 : nz - 1, 1 : ny - 1, 1:nx]
            + w_tensor[1 : nz - 1, 1 : ny - 1, 0 : nx - 1]
            + w_tensor[2:nz, 1 : ny - 1, 0 : nx - 1]
            + w_tensor[2:nz, 1 : ny - 1, 1:nx]
        )

    inner_slice = (slice(1, nz - 1), slice(1, ny - 1), slice(1, nx))
    grid_x_inner = grid_x[inner_slice]
    grid_y_inner = grid_y[inner_slice]
    grid_z_inner = grid_z[inner_slice]
    vel_x_inner = vel_x_field[inner_slice]
    vel_y_inner = vel_y_field[inner_slice]
    vel_z_inner = vel_z_field[inner_slice]

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, vel_x_inner, vel_y_inner, vel_z_inner):
        _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)
        return

    dt_over_dx = u_tensor.new_tensor(dt / dx)
    last_x = grid_x_inner - dt_over_dx * vel_x_inner
    last_y = grid_y_inner - dt_over_dx * vel_y_inner
    last_z = grid_z_inner - dt_over_dx * vel_z_inner
    last_x, last_y, last_z = _clamp_u_face_tensor_3d(last_x, last_y, last_z, nx, ny, nz)

    if rk_order == 3:
        velocity_field = torch.stack((vel_x_field, vel_y_field, vel_z_field), dim=0)

        pos2_x = grid_x_inner - 0.5 * dt_over_dx * vel_x_inner
        pos2_y = grid_y_inner - 0.5 * dt_over_dx * vel_y_inner
        pos2_z = grid_z_inner - 0.5 * dt_over_dx * vel_z_inner
        pos2_x, pos2_y, pos2_z = _clamp_u_face_tensor_3d(
            pos2_x, pos2_y, pos2_z, nx, ny, nz
        )
        k2_x, k2_y, k2_z = _sample_vector_field_3d(
            velocity_field, pos2_x, pos2_y, pos2_z, nx + 1, ny, nz
        )

        pos3_x = grid_x_inner - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y_inner - 0.75 * dt_over_dx * k2_y
        pos3_z = grid_z_inner - 0.75 * dt_over_dx * k2_z
        pos3_x, pos3_y, pos3_z = _clamp_u_face_tensor_3d(
            pos3_x, pos3_y, pos3_z, nx, ny, nz
        )
        k3_x, k3_y, k3_z = _sample_vector_field_3d(
            velocity_field, pos3_x, pos3_y, pos3_z, nx + 1, ny, nz
        )

        last_x = grid_x_inner - dt_over_dx * (
            2.0 / 9.0 * vel_x_inner + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y_inner - dt_over_dx * (
            2.0 / 9.0 * vel_y_inner + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_z = grid_z_inner - dt_over_dx * (
            2.0 / 9.0 * vel_z_inner + 3.0 / 9.0 * k2_z + 4.0 / 9.0 * k3_z
        )
        last_x, last_y, last_z = _clamp_u_face_tensor_3d(
            last_x, last_y, last_z, nx, ny, nz
        )

    sampled_u = _sample_scalar_field_3d(
        u_tensor, last_x, last_y, last_z, nx + 1, ny, nz
    )
    u_new_tensor[inner_slice] = sampled_u

    _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)


def advect_v_velocity_kernel_3d(
    v: Tensor,
    v_new: Tensor,
    u: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """3D v-velocity advection on MAC grid.

    Args:
        v: Current y-velocity (nz, ny+1, nx)
        v_new: Output y-velocity (nz, ny+1, nx)
        u: x-velocity component (nz, ny, nx+1)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    v_tensor = _as_tensor(v)
    v_new_tensor, v_new_original, v_new_needs_copy = _prepare_output(v_tensor, v_new)
    u_tensor = _tensor_from_like(v_tensor, u)
    w_tensor = _tensor_from_like(v_tensor, w)

    if nz <= 2 or ny <= 1 or nx <= 2:
        v_new_tensor.copy_(v_tensor)
        _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)
        return

    v_new_tensor.copy_(v_tensor)
    device = v_tensor.device
    dtype = v_tensor.dtype

    z_coords = torch.arange(nz, device=device, dtype=dtype)
    y_coords = torch.arange(ny + 1, device=device, dtype=dtype)
    x_coords = torch.arange(nx, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = _meshgrid_ij(z_coords, y_coords, x_coords)

    vel_y_field = v_tensor
    vel_x_field = torch.zeros_like(v_tensor)
    vel_z_field = torch.zeros_like(v_tensor)
    if nz > 2 and ny > 1 and nx > 2:
        vel_x_field[1 : nz - 1, 1:ny, 1 : nx - 1] = 0.25 * (
            u_tensor[1 : nz - 1, 1:ny, 1 : nx - 1]
            + u_tensor[1 : nz - 1, 1:ny, 2:nx]
            + u_tensor[1 : nz - 1, 0 : ny - 1, 2:nx]
            + u_tensor[1 : nz - 1, 0 : ny - 1, 1 : nx - 1]
        )
        vel_z_field[1 : nz - 1, 1:ny, 1 : nx - 1] = 0.25 * (
            w_tensor[1 : nz - 1, 1:ny, 1 : nx - 1]
            + w_tensor[1 : nz - 1, 0 : ny - 1, 1 : nx - 1]
            + w_tensor[2:nz, 0 : ny - 1, 1 : nx - 1]
            + w_tensor[2:nz, 1:ny, 1 : nx - 1]
        )

    inner_slice = (slice(1, nz - 1), slice(1, ny), slice(1, nx - 1))
    grid_x_inner = grid_x[inner_slice]
    grid_y_inner = grid_y[inner_slice]
    grid_z_inner = grid_z[inner_slice]
    vel_x_inner = vel_x_field[inner_slice]
    vel_y_inner = vel_y_field[inner_slice]
    vel_z_inner = vel_z_field[inner_slice]

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, vel_x_inner, vel_y_inner, vel_z_inner):
        _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)
        return

    dt_over_dx = v_tensor.new_tensor(dt / dx)
    last_x = grid_x_inner - dt_over_dx * vel_x_inner
    last_y = grid_y_inner - dt_over_dx * vel_y_inner
    last_z = grid_z_inner - dt_over_dx * vel_z_inner
    last_x, last_y, last_z = _clamp_v_face_tensor_3d(last_x, last_y, last_z, nx, ny, nz)

    if rk_order == 3:
        velocity_field = torch.stack((vel_x_field, vel_y_field, vel_z_field), dim=0)

        pos2_x = grid_x_inner - 0.5 * dt_over_dx * vel_x_inner
        pos2_y = grid_y_inner - 0.5 * dt_over_dx * vel_y_inner
        pos2_z = grid_z_inner - 0.5 * dt_over_dx * vel_z_inner
        pos2_x, pos2_y, pos2_z = _clamp_v_face_tensor_3d(
            pos2_x, pos2_y, pos2_z, nx, ny, nz
        )
        k2_x, k2_y, k2_z = _sample_vector_field_3d(
            velocity_field, pos2_x, pos2_y, pos2_z, nx, ny + 1, nz
        )

        pos3_x = grid_x_inner - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y_inner - 0.75 * dt_over_dx * k2_y
        pos3_z = grid_z_inner - 0.75 * dt_over_dx * k2_z
        pos3_x, pos3_y, pos3_z = _clamp_v_face_tensor_3d(
            pos3_x, pos3_y, pos3_z, nx, ny, nz
        )
        k3_x, k3_y, k3_z = _sample_vector_field_3d(
            velocity_field, pos3_x, pos3_y, pos3_z, nx, ny + 1, nz
        )

        last_x = grid_x_inner - dt_over_dx * (
            2.0 / 9.0 * vel_x_inner + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y_inner - dt_over_dx * (
            2.0 / 9.0 * vel_y_inner + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_z = grid_z_inner - dt_over_dx * (
            2.0 / 9.0 * vel_z_inner + 3.0 / 9.0 * k2_z + 4.0 / 9.0 * k3_z
        )
        last_x, last_y, last_z = _clamp_v_face_tensor_3d(
            last_x, last_y, last_z, nx, ny, nz
        )

    sampled_v = _sample_scalar_field_3d(
        v_tensor, last_x, last_y, last_z, nx, ny + 1, nz
    )
    v_new_tensor[inner_slice] = sampled_v

    _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)


def advect_w_velocity_kernel_3d(
    w: Tensor,
    w_new: Tensor,
    u: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
    rk_order: int = 1,
) -> None:
    """3D w-velocity advection on MAC grid.

    Args:
        w: Current z-velocity (nz+1, ny, nx)
        w_new: Output z-velocity (nz+1, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Runge-Kutta order (1=Euler, 3=SSPRK3)
    """
    w_tensor = _as_tensor(w)
    w_new_tensor, w_new_original, w_new_needs_copy = _prepare_output(w_tensor, w_new)
    u_tensor = _tensor_from_like(w_tensor, u)
    v_tensor = _tensor_from_like(w_tensor, v)

    if nz <= 1 or ny <= 2 or nx <= 2:
        w_new_tensor.copy_(w_tensor)
        _finalize_output(w_new_tensor, w_new_original, w_new_needs_copy)
        return

    w_new_tensor.copy_(w_tensor)
    device = w_tensor.device
    dtype = w_tensor.dtype

    z_coords = torch.arange(nz + 1, device=device, dtype=dtype)
    y_coords = torch.arange(ny, device=device, dtype=dtype)
    x_coords = torch.arange(nx, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = _meshgrid_ij(z_coords, y_coords, x_coords)

    vel_z_field = w_tensor
    vel_x_field = torch.zeros_like(w_tensor)
    vel_y_field = torch.zeros_like(w_tensor)
    if nz > 1 and ny > 2 and nx > 2:
        vel_x_field[1:nz, 1 : ny - 1, 1 : nx - 1] = 0.25 * (
            u_tensor[1:nz, 1 : ny - 1, 1 : nx - 1]
            + u_tensor[1:nz, 1 : ny - 1, 2:nx]
            + u_tensor[0 : nz - 1, 1 : ny - 1, 2:nx]
            + u_tensor[0 : nz - 1, 1 : ny - 1, 1 : nx - 1]
        )
        vel_y_field[1:nz, 1 : ny - 1, 1 : nx - 1] = 0.25 * (
            v_tensor[1:nz, 1 : ny - 1, 1 : nx - 1]
            + v_tensor[1:nz, 2:ny, 1 : nx - 1]
            + v_tensor[0 : nz - 1, 2:ny, 1 : nx - 1]
            + v_tensor[0 : nz - 1, 1 : ny - 1, 1 : nx - 1]
        )

    inner_slice = (slice(1, nz), slice(1, ny - 1), slice(1, nx - 1))
    grid_x_inner = grid_x[inner_slice]
    grid_y_inner = grid_y[inner_slice]
    grid_z_inner = grid_z[inner_slice]
    vel_x_inner = vel_x_field[inner_slice]
    vel_y_inner = vel_y_field[inner_slice]
    vel_z_inner = vel_z_field[inner_slice]

    velocity_eps = float(torch.finfo(dtype).eps * 10.0)
    if _velocity_is_zero(velocity_eps, vel_x_inner, vel_y_inner, vel_z_inner):
        _finalize_output(w_new_tensor, w_new_original, w_new_needs_copy)
        return

    dt_over_dx = w_tensor.new_tensor(dt / dx)
    last_x = grid_x_inner - dt_over_dx * vel_x_inner
    last_y = grid_y_inner - dt_over_dx * vel_y_inner
    last_z = grid_z_inner - dt_over_dx * vel_z_inner
    last_x, last_y, last_z = _clamp_w_face_tensor_3d(last_x, last_y, last_z, nx, ny, nz)

    if rk_order == 3:
        velocity_field = torch.stack((vel_x_field, vel_y_field, vel_z_field), dim=0)

        pos2_x = grid_x_inner - 0.5 * dt_over_dx * vel_x_inner
        pos2_y = grid_y_inner - 0.5 * dt_over_dx * vel_y_inner
        pos2_z = grid_z_inner - 0.5 * dt_over_dx * vel_z_inner
        pos2_x, pos2_y, pos2_z = _clamp_w_face_tensor_3d(
            pos2_x, pos2_y, pos2_z, nx, ny, nz
        )
        k2_x, k2_y, k2_z = _sample_vector_field_3d(
            velocity_field, pos2_x, pos2_y, pos2_z, nx, ny, nz + 1
        )

        pos3_x = grid_x_inner - 0.75 * dt_over_dx * k2_x
        pos3_y = grid_y_inner - 0.75 * dt_over_dx * k2_y
        pos3_z = grid_z_inner - 0.75 * dt_over_dx * k2_z
        pos3_x, pos3_y, pos3_z = _clamp_w_face_tensor_3d(
            pos3_x, pos3_y, pos3_z, nx, ny, nz
        )
        k3_x, k3_y, k3_z = _sample_vector_field_3d(
            velocity_field, pos3_x, pos3_y, pos3_z, nx, ny, nz + 1
        )

        last_x = grid_x_inner - dt_over_dx * (
            2.0 / 9.0 * vel_x_inner + 3.0 / 9.0 * k2_x + 4.0 / 9.0 * k3_x
        )
        last_y = grid_y_inner - dt_over_dx * (
            2.0 / 9.0 * vel_y_inner + 3.0 / 9.0 * k2_y + 4.0 / 9.0 * k3_y
        )
        last_z = grid_z_inner - dt_over_dx * (
            2.0 / 9.0 * vel_z_inner + 3.0 / 9.0 * k2_z + 4.0 / 9.0 * k3_z
        )
        last_x, last_y, last_z = _clamp_w_face_tensor_3d(
            last_x, last_y, last_z, nx, ny, nz
        )

    sampled_w = _sample_scalar_field_3d(
        w_tensor, last_x, last_y, last_z, nx, ny, nz + 1
    )
    w_new_tensor[inner_slice] = sampled_w

    _finalize_output(w_new_tensor, w_new_original, w_new_needs_copy)


# ============= MacCormack Advection (2D) =============


def advect_density_maccormack_2d(
    density: Tensor,
    density_new: Tensor,
    u: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 2D density field with reduced dissipation."""

    density_tensor = _as_tensor(density)
    density_new_tensor, density_new_original, density_new_needs_copy = _prepare_output(
        density_tensor, density_new
    )
    u_tensor = _tensor_from_like(density_tensor, u)
    v_tensor = _tensor_from_like(density_tensor, v)

    phi_hat = torch.zeros_like(density_tensor)
    phi_hat_hat = torch.zeros_like(density_tensor)

    # Forward step (semi-Lagrangian)
    advect_density_kernel_2d(
        density_tensor,
        phi_hat,
        u_tensor,
        v_tensor,
        dx=dx,
        dt=dt,
        ny=ny,
        nx=nx,
    )

    # Backward step (reverse time)
    advect_density_kernel_2d(
        phi_hat,
        phi_hat_hat,
        u_tensor,
        v_tensor,
        dx=dx,
        dt=-dt,
        ny=ny,
        nx=nx,
    )

    # Correction and clamping
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            correction = phi_hat[y, x] + 0.5 * (
                density_tensor[y, x] - phi_hat_hat[y, x]
            )

            neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_2d(
                density_tensor, y, x, ny, nx
            )

            density_new_tensor[y, x] = torch.clamp(
                correction, min=neighbors_min, max=neighbors_max
            )

    _finalize_output(density_new_tensor, density_new_original, density_new_needs_copy)


def advect_u_velocity_maccormack_2d(
    u: Tensor,
    u_new: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 2D u-velocity on a MAC grid."""

    u_tensor = _as_tensor(u)
    u_new_tensor, u_new_original, u_new_needs_copy = _prepare_output(u_tensor, u_new)
    v_tensor = _tensor_from_like(u_tensor, v)

    phi_hat = torch.zeros_like(u_tensor)
    phi_hat_hat = torch.zeros_like(u_tensor)

    advect_u_velocity_kernel_2d(
        u_tensor,
        phi_hat,
        v_tensor,
        dx=dx,
        dt=dt,
        ny=ny,
        nx=nx,
    )

    advect_u_velocity_kernel_2d(
        phi_hat,
        phi_hat_hat,
        v_tensor,
        dx=dx,
        dt=-dt,
        ny=ny,
        nx=nx,
    )

    for y in range(1, ny - 1):
        for x in range(1, nx):
            correction = phi_hat[y, x] + 0.5 * (u_tensor[y, x] - phi_hat_hat[y, x])

            neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_1d_y_2d(
                u_tensor, y, x, ny
            )

            u_new_tensor[y, x] = torch.clamp(
                correction, min=neighbors_min, max=neighbors_max
            )

    _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)


def advect_v_velocity_maccormack_2d(
    v: Tensor,
    v_new: Tensor,
    u: Tensor,
    dx: float,
    dt: float,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 2D v-velocity on a MAC grid."""

    v_tensor = _as_tensor(v)
    v_new_tensor, v_new_original, v_new_needs_copy = _prepare_output(v_tensor, v_new)
    u_tensor = _tensor_from_like(v_tensor, u)

    phi_hat = torch.zeros_like(v_tensor)
    phi_hat_hat = torch.zeros_like(v_tensor)

    advect_v_velocity_kernel_2d(
        v_tensor,
        phi_hat,
        u_tensor,
        dx=dx,
        dt=dt,
        ny=ny,
        nx=nx,
    )

    advect_v_velocity_kernel_2d(
        phi_hat,
        phi_hat_hat,
        u_tensor,
        dx=dx,
        dt=-dt,
        ny=ny,
        nx=nx,
    )

    for y in range(1, ny):
        for x in range(1, nx - 1):
            correction = phi_hat[y, x] + 0.5 * (v_tensor[y, x] - phi_hat_hat[y, x])

            neighbors_min = v_tensor[y, x]
            neighbors_max = v_tensor[y, x]

            for dx_offset in range(-1, 2):
                nx_idx = x + dx_offset
                if 0 <= nx_idx < nx:
                    val = v_tensor[y, nx_idx]
                    neighbors_min = torch.minimum(neighbors_min, val)
                    neighbors_max = torch.maximum(neighbors_max, val)

            v_new_tensor[y, x] = torch.clamp(
                correction, min=neighbors_min, max=neighbors_max
            )

    _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)


def advect_density_maccormack_3d(
    density: Tensor,
    density_new: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 3D density field with reduced dissipation."""

    density_tensor = _as_tensor(density)
    density_new_tensor, density_new_original, density_new_needs_copy = _prepare_output(
        density_tensor, density_new
    )
    u_tensor = _tensor_from_like(density_tensor, u)
    v_tensor = _tensor_from_like(density_tensor, v)
    w_tensor = _tensor_from_like(density_tensor, w)

    phi_hat = torch.zeros_like(density_tensor)
    phi_hat_hat = torch.zeros_like(density_tensor)

    advect_density_kernel_3d(
        density_tensor,
        phi_hat,
        u_tensor,
        v_tensor,
        w_tensor,
        dx=dx,
        dt=dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    advect_density_kernel_3d(
        phi_hat,
        phi_hat_hat,
        u_tensor,
        v_tensor,
        w_tensor,
        dx=dx,
        dt=-dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    density_tensor[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_3d(
                    density_tensor, z, y, x, nz, ny, nx
                )

                density_new_tensor[z, y, x] = torch.clamp(
                    correction, min=neighbors_min, max=neighbors_max
                )

    _finalize_output(density_new_tensor, density_new_original, density_new_needs_copy)


def advect_u_velocity_maccormack_3d(
    u: Tensor,
    u_new: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 3D u-velocity on a MAC grid."""

    u_tensor = _as_tensor(u)
    u_new_tensor, u_new_original, u_new_needs_copy = _prepare_output(u_tensor, u_new)
    v_tensor = _tensor_from_like(u_tensor, v)
    w_tensor = _tensor_from_like(u_tensor, w)

    phi_hat = torch.zeros_like(u_tensor)
    phi_hat_hat = torch.zeros_like(u_tensor)

    advect_u_velocity_kernel_3d(
        u_tensor,
        phi_hat,
        v_tensor,
        w_tensor,
        dx=dx,
        dt=dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    advect_u_velocity_kernel_3d(
        phi_hat,
        phi_hat_hat,
        v_tensor,
        w_tensor,
        dx=dx,
        dt=-dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                correction = phi_hat[z, y, x] + 0.5 * (
                    u_tensor[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = u_tensor[z, y, x]
                neighbors_max = u_tensor[z, y, x]

                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        nz_idx = z + dz
                        ny_idx = y + dy
                        if 0 <= nz_idx < nz and 0 <= ny_idx < ny:
                            val = u_tensor[nz_idx, ny_idx, x]
                            neighbors_min = torch.minimum(neighbors_min, val)
                            neighbors_max = torch.maximum(neighbors_max, val)

                u_new_tensor[z, y, x] = torch.clamp(
                    correction, min=neighbors_min, max=neighbors_max
                )

    _finalize_output(u_new_tensor, u_new_original, u_new_needs_copy)


def advect_v_velocity_maccormack_3d(
    v: Tensor,
    v_new: Tensor,
    u: Tensor,
    w: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 3D v-velocity on a MAC grid."""

    v_tensor = _as_tensor(v)
    v_new_tensor, v_new_original, v_new_needs_copy = _prepare_output(v_tensor, v_new)
    u_tensor = _tensor_from_like(v_tensor, u)
    w_tensor = _tensor_from_like(v_tensor, w)

    phi_hat = torch.zeros_like(v_tensor)
    phi_hat_hat = torch.zeros_like(v_tensor)

    advect_v_velocity_kernel_3d(
        v_tensor,
        phi_hat,
        u_tensor,
        w_tensor,
        dx=dx,
        dt=dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    advect_v_velocity_kernel_3d(
        phi_hat,
        phi_hat_hat,
        u_tensor,
        w_tensor,
        dx=dx,
        dt=-dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    for z in range(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    v_tensor[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = v_tensor[z, y, x]
                neighbors_max = v_tensor[z, y, x]

                for dz in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        nz_idx = z + dz
                        nx_idx = x + dx_offset
                        if 0 <= nz_idx < nz and 0 <= nx_idx < nx:
                            val = v_tensor[nz_idx, y, nx_idx]
                            neighbors_min = torch.minimum(neighbors_min, val)
                            neighbors_max = torch.maximum(neighbors_max, val)

                v_new_tensor[z, y, x] = torch.clamp(
                    correction, min=neighbors_min, max=neighbors_max
                )

    _finalize_output(v_new_tensor, v_new_original, v_new_needs_copy)


def advect_w_velocity_maccormack_3d(
    w: Tensor,
    w_new: Tensor,
    u: Tensor,
    v: Tensor,
    dx: float,
    dt: float,
    nz: int,
    ny: int,
    nx: int,
) -> None:
    """MacCormack advection for 3D w-velocity on a MAC grid."""

    w_tensor = _as_tensor(w)
    w_new_tensor, w_new_original, w_new_needs_copy = _prepare_output(w_tensor, w_new)
    u_tensor = _tensor_from_like(w_tensor, u)
    v_tensor = _tensor_from_like(w_tensor, v)

    phi_hat = torch.zeros_like(w_tensor)
    phi_hat_hat = torch.zeros_like(w_tensor)

    advect_w_velocity_kernel_3d(
        w_tensor,
        phi_hat,
        u_tensor,
        v_tensor,
        dx=dx,
        dt=dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    advect_w_velocity_kernel_3d(
        phi_hat,
        phi_hat_hat,
        u_tensor,
        v_tensor,
        dx=dx,
        dt=-dt,
        nz=nz,
        ny=ny,
        nx=nx,
    )

    for z in range(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    w_tensor[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = w_tensor[z, y, x]
                neighbors_max = w_tensor[z, y, x]

                for dy in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        ny_idx = y + dy
                        nx_idx = x + dx_offset
                        if 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                            val = w_tensor[z, ny_idx, nx_idx]
                            neighbors_min = torch.minimum(neighbors_min, val)
                            neighbors_max = torch.maximum(neighbors_max, val)

                w_new_tensor[z, y, x] = torch.clamp(
                    correction, min=neighbors_min, max=neighbors_max
                )

    _finalize_output(w_new_tensor, w_new_original, w_new_needs_copy)
