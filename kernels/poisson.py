"""Poisson equation solvers implemented with Torch tensors."""

from __future__ import annotations

from typing import Any, Tuple

import torch

Tensor = torch.Tensor


def _ensure_tensors(pressure: Any, divergence: Any) -> Tuple[Tensor, Tensor]:
    """Ensure inputs are torch tensors sharing dtype/device."""

    if not torch.is_tensor(pressure):
        pressure = torch.as_tensor(pressure, dtype=torch.float32)
    if not torch.is_tensor(divergence):
        divergence = torch.as_tensor(divergence, dtype=pressure.dtype)
    else:
        divergence = divergence.to(dtype=pressure.dtype, device=pressure.device)

    if divergence.device != pressure.device:
        divergence = divergence.to(pressure.device)

    return pressure, divergence


def _checkerboard_mask(
    shape: Tuple[int, ...], parity: int, device: torch.device
) -> Tensor:
    if len(shape) == 2:
        i = torch.arange(shape[0], device=device).view(-1, 1)
        j = torch.arange(shape[1], device=device).view(1, -1)
        mask = ((i + j) % 2) == parity
    else:
        i = torch.arange(shape[0], device=device).view(-1, 1, 1)
        j = torch.arange(shape[1], device=device).view(1, -1, 1)
        k = torch.arange(shape[2], device=device).view(1, 1, -1)
        mask = ((i + j + k) % 2) == parity
    return mask


def _residual_l2_2d(pressure: Tensor, rhs: Tensor, dx2: float) -> float:
    if rhs.numel() == 0:
        return 0.0
    interior = pressure[1:-1, 1:-1]
    laplacian = (
        -4.0 * interior
        + pressure[0:-2, 1:-1]
        + pressure[2:, 1:-1]
        + pressure[1:-1, 0:-2]
        + pressure[1:-1, 2:]
    ) / dx2
    residual = rhs - laplacian
    value = torch.sqrt(torch.sum(residual * residual)) / rhs.numel()
    return float(value)


def _residual_l2_3d(pressure: Tensor, rhs: Tensor, dx2: float) -> float:
    if rhs.numel() == 0:
        return 0.0
    interior = pressure[1:-1, 1:-1, 1:-1]
    laplacian = (
        -6.0 * interior
        + pressure[0:-2, 1:-1, 1:-1]
        + pressure[2:, 1:-1, 1:-1]
        + pressure[1:-1, 0:-2, 1:-1]
        + pressure[1:-1, 2:, 1:-1]
        + pressure[1:-1, 1:-1, 0:-2]
        + pressure[1:-1, 1:-1, 2:]
    ) / dx2
    residual = rhs - laplacian
    value = torch.sqrt(torch.sum(residual * residual)) / rhs.numel()
    return float(value)


def _gauss_seidel_update_2d(
    pressure: Tensor, rhs: Tensor, dx2: float, mask: Tensor
) -> None:
    interior = pressure[1:-1, 1:-1]
    neighbor_sum = (
        pressure[0:-2, 1:-1]
        + pressure[2:, 1:-1]
        + pressure[1:-1, 0:-2]
        + pressure[1:-1, 2:]
    )
    updated = (dx2 * rhs + neighbor_sum) * 0.25
    interior = torch.where(mask, updated, interior)
    pressure[1:-1, 1:-1] = interior


def _gauss_seidel_update_3d(
    pressure: Tensor, rhs: Tensor, dx2: float, mask: Tensor
) -> None:
    interior = pressure[1:-1, 1:-1, 1:-1]
    neighbor_sum = (
        pressure[0:-2, 1:-1, 1:-1]
        + pressure[2:, 1:-1, 1:-1]
        + pressure[1:-1, 0:-2, 1:-1]
        + pressure[1:-1, 2:, 1:-1]
        + pressure[1:-1, 1:-1, 0:-2]
        + pressure[1:-1, 1:-1, 2:]
    )
    updated = (dx2 * rhs + neighbor_sum) / 6.0
    interior = torch.where(mask, updated, interior)
    pressure[1:-1, 1:-1, 1:-1] = interior


@torch.no_grad()
def solve_poisson_rb_gauss_seidel_2d(
    pressure: Tensor,
    divergence: Tensor,
    dx: float,
    dt: float,
    rho: float,
    max_iter: int,
    tolerance: float,
    ny: int,
    nx: int,
) -> Tensor:
    """Torch-based 2D Red-Black Gauss-Seidel Poisson solver."""

    pressure, divergence = _ensure_tensors(pressure, divergence)
    if pressure.ndim != 2:
        raise ValueError("Expected 2D pressure grid")

    if ny != pressure.shape[0] or nx != pressure.shape[1]:
        raise ValueError("Provided grid dimensions do not match pressure shape")

    if ny < 3 or nx < 3:
        return pressure

    dx2 = dx * dx
    rhs = -divergence[1:-1, 1:-1] / dt * rho
    mask_red = _checkerboard_mask(rhs.shape, 0, pressure.device)
    mask_black = ~mask_red

    for iteration in range(max_iter):
        _gauss_seidel_update_2d(pressure, rhs, dx2, mask_red)
        _gauss_seidel_update_2d(pressure, rhs, dx2, mask_black)

        if iteration % 10 == 0:
            residual = _residual_l2_2d(pressure, rhs, dx2)
            if residual < tolerance:
                break

    return pressure


@torch.no_grad()
def solve_poisson_rb_gauss_seidel_3d(
    pressure: Tensor,
    divergence: Tensor,
    dx: float,
    dt: float,
    rho: float,
    max_iter: int,
    tolerance: float,
    nz: int,
    ny: int,
    nx: int,
) -> Tensor:
    """Torch-based 3D Red-Black Gauss-Seidel Poisson solver."""

    pressure, divergence = _ensure_tensors(pressure, divergence)
    if pressure.ndim != 3:
        raise ValueError("Expected 3D pressure grid")

    if (nz, ny, nx) != pressure.shape:
        raise ValueError("Provided grid dimensions do not match pressure shape")

    if nz < 3 or ny < 3 or nx < 3:
        return pressure

    dx2 = dx * dx
    rhs = -divergence[1:-1, 1:-1, 1:-1] / dt * rho
    mask_red = _checkerboard_mask(rhs.shape, 0, pressure.device)
    mask_black = ~mask_red

    for iteration in range(max_iter):
        _gauss_seidel_update_3d(pressure, rhs, dx2, mask_red)
        _gauss_seidel_update_3d(pressure, rhs, dx2, mask_black)

        if iteration % 10 == 0:
            residual = _residual_l2_3d(pressure, rhs, dx2)
            if residual < tolerance:
                break

    return pressure


@torch.no_grad()
def solve_poisson_jacobi_2d(
    pressure: Tensor,
    divergence: Tensor,
    dx: float,
    dt: float,
    rho: float,
    max_iter: int,
    tolerance: float,
    ny: int,
    nx: int,
) -> Tensor:
    """Torch-based 2D Jacobi solver for the Poisson equation."""

    pressure, divergence = _ensure_tensors(pressure, divergence)
    if pressure.ndim != 2:
        raise ValueError("Expected 2D pressure grid")
    if ny != pressure.shape[0] or nx != pressure.shape[1]:
        raise ValueError("Provided grid dimensions do not match pressure shape")

    if ny < 3 or nx < 3:
        return pressure

    dx2 = dx * dx
    pressure_new = pressure.clone()
    rhs = -divergence / dt * rho

    for iteration in range(max_iter):
        neighbor_sum = (
            pressure[0:-2, 1:-1]
            + pressure[2:, 1:-1]
            + pressure[1:-1, 0:-2]
            + pressure[1:-1, 2:]
        )
        pressure_new[1:-1, 1:-1] = (dx2 * rhs[1:-1, 1:-1] + neighbor_sum) * 0.25

        pressure, pressure_new = pressure_new, pressure

        if iteration % 10 == 0:
            residual = _residual_l2_2d(pressure, rhs[1:-1, 1:-1], dx2)
            if residual < tolerance:
                break

    return pressure


@torch.no_grad()
def solve_poisson_jacobi_3d(
    pressure: Tensor,
    divergence: Tensor,
    dx: float,
    dt: float,
    rho: float,
    max_iter: int,
    tolerance: float,
    nz: int,
    ny: int,
    nx: int,
) -> Tensor:
    """Torch-based 3D Jacobi solver for the Poisson equation."""

    pressure, divergence = _ensure_tensors(pressure, divergence)
    if pressure.ndim != 3:
        raise ValueError("Expected 3D pressure grid")
    if (nz, ny, nx) != pressure.shape:
        raise ValueError("Provided grid dimensions do not match pressure shape")

    if nz < 3 or ny < 3 or nx < 3:
        return pressure

    dx2 = dx * dx
    pressure_new = pressure.clone()
    rhs = -divergence / dt * rho

    for iteration in range(max_iter):
        neighbor_sum = (
            pressure[0:-2, 1:-1, 1:-1]
            + pressure[2:, 1:-1, 1:-1]
            + pressure[1:-1, 0:-2, 1:-1]
            + pressure[1:-1, 2:, 1:-1]
            + pressure[1:-1, 1:-1, 0:-2]
            + pressure[1:-1, 1:-1, 2:]
        )
        pressure_new[1:-1, 1:-1, 1:-1] = (
            dx2 * rhs[1:-1, 1:-1, 1:-1] + neighbor_sum
        ) / 6.0

        pressure, pressure_new = pressure_new, pressure

        if iteration % 10 == 0:
            residual = _residual_l2_3d(pressure, rhs[1:-1, 1:-1, 1:-1], dx2)
            if residual < tolerance:
                break

    return pressure
