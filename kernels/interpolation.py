"""Interpolation helpers implemented with Torch tensors."""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch

Tensor = torch.Tensor


def _interp_indices(
    coord: Union[Tensor, float, int], size: int
) -> Tuple[float, int, int]:
    """Clamp coordinates to valid interpolation range."""
    if size < 2:
        raise ValueError("Interpolation requires at least two samples per axis")

    coord_value = float(coord)
    if coord_value <= 0.0:
        return 0.0, 0, 1
    if coord_value >= size - 1:
        low = size - 2
        return float(size - 1), low, low + 1

    low = int(math.floor(coord_value))
    high = low + 1
    return coord_value, low, high


def bilinear_interp(field: Tensor, x: float, y: float) -> Tensor:
    """Fast bilinear interpolation for 2D fields

    Args:
        field: 2D array to interpolate from
        x: x-coordinate (can be fractional)
        y: y-coordinate (can be fractional)

    Returns:
        Interpolated value at (x, y)
    """
    y_value, y_low, y_high = _interp_indices(y, field.shape[0])
    x_value, x_low, x_high = _interp_indices(x, field.shape[1])

    x_weight = x_value - x_low
    y_weight = y_value - y_low

    return (
        (1 - x_weight) * (1 - y_weight) * field[y_low, x_low]
        + x_weight * (1 - y_weight) * field[y_low, x_high]
        + (1 - x_weight) * y_weight * field[y_high, x_low]
        + x_weight * y_weight * field[y_high, x_high]
    )


def trilinear_interp(field: Tensor, x: float, y: float, z: float) -> Tensor:
    """Fast trilinear interpolation for 3D fields

    Args:
        field: 3D array to interpolate from
        x: x-coordinate (can be fractional)
        y: y-coordinate (can be fractional)
        z: z-coordinate (can be fractional)

    Returns:
        Interpolated value at (x, y, z)
    """
    z_value, z_low, z_high = _interp_indices(z, field.shape[0])
    y_value, y_low, y_high = _interp_indices(y, field.shape[1])
    x_value, x_low, x_high = _interp_indices(x, field.shape[2])

    x_weight = x_value - x_low
    y_weight = y_value - y_low
    z_weight = z_value - z_low

    return (
        (1 - x_weight) * (1 - y_weight) * (1 - z_weight) * field[z_low, y_low, x_low]
        + x_weight * (1 - y_weight) * (1 - z_weight) * field[z_low, y_low, x_high]
        + (1 - x_weight) * y_weight * (1 - z_weight) * field[z_low, y_high, x_low]
        + x_weight * y_weight * (1 - z_weight) * field[z_low, y_high, x_high]
        + (1 - x_weight) * (1 - y_weight) * z_weight * field[z_high, y_low, x_low]
        + x_weight * (1 - y_weight) * z_weight * field[z_high, y_low, x_high]
        + (1 - x_weight) * y_weight * z_weight * field[z_high, y_high, x_low]
        + x_weight * y_weight * z_weight * field[z_high, y_high, x_high]
    )
