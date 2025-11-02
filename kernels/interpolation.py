"""Interpolation kernels for fluid simulation."""

from numba import jit


@jit(nopython=True, cache=True)
def bilinear_interp(field, x, y):
    """Fast bilinear interpolation for 2D fields

    Args:
        field: 2D array to interpolate from
        x: x-coordinate (can be fractional)
        y: y-coordinate (can be fractional)

    Returns:
        Interpolated value at (x, y)
    """
    x_low = int(x)
    y_low = int(y)
    x_high = x_low + 1
    y_high = y_low + 1

    x_weight = x - x_low
    y_weight = y - y_low

    return (
        (1 - x_weight) * (1 - y_weight) * field[y_low, x_low]
        + x_weight * (1 - y_weight) * field[y_low, x_high]
        + (1 - x_weight) * y_weight * field[y_high, x_low]
        + x_weight * y_weight * field[y_high, x_high]
    )


@jit(nopython=True, cache=True)
def trilinear_interp(field, x, y, z):
    """Fast trilinear interpolation for 3D fields

    Args:
        field: 3D array to interpolate from
        x: x-coordinate (can be fractional)
        y: y-coordinate (can be fractional)
        z: z-coordinate (can be fractional)

    Returns:
        Interpolated value at (x, y, z)
    """
    x_low = int(x)
    y_low = int(y)
    z_low = int(z)
    x_high = x_low + 1
    y_high = y_low + 1
    z_high = z_low + 1

    x_weight = x - x_low
    y_weight = y - y_low
    z_weight = z - z_low

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
