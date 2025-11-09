"""Numba-JIT optimized MAC grid operations for fluid simulation.

Provides velocity interpolation, boundary clamping, averaging, neighborhood
operations, and force utilities. All functions maintain separate 2D/3D versions.
"""

import numpy as np
from numba import jit, prange
from core import MACGrid2D, MACGrid3D


@jit(nopython=True, cache=True)
def interpolate_velocity_to_cell_center_2d(u, v, y, x):
    """Interpolate MAC grid velocities to cell center.

    Args:
        u: x-velocity on x-faces (ny, nx+1)
        v: y-velocity on y-faces (ny+1, nx)
        y, x: Cell center coordinates

    Returns:
        (vel_x, vel_y) at cell center
    """
    vel_x = (u[y, x] + u[y, x + 1]) * 0.5
    vel_y = (v[y, x] + v[y + 1, x]) * 0.5
    return vel_x, vel_y


@jit(nopython=True, cache=True)
def interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x):
    """Interpolate MAC grid velocities to cell center.

    Args:
        u: x-velocity on x-faces (nz, ny, nx+1)
        v: y-velocity on y-faces (nz, ny+1, nx)
        w: z-velocity on z-faces (nz+1, ny, nx)
        z, y, x: Cell center coordinates

    Returns:
        (vel_x, vel_y, vel_z) at cell center
    """
    vel_x = (u[z, y, x] + u[z, y, x + 1]) * 0.5
    vel_y = (v[z, y, x] + v[z, y + 1, x]) * 0.5
    vel_z = (w[z, y, x] + w[z + 1, y, x]) * 0.5
    return vel_x, vel_y, vel_z


@jit(nopython=True, cache=True)
def interpolate_v_to_u_face_2d(v, y, x):
    """Average 4 v-velocity values to u-face location.

    Args:
        v: y-velocity on y-faces (ny+1, nx)
        y, x: u-face coordinates

    Returns:
        v-velocity at u-face
    """
    return (v[y, x] + v[y, x - 1] + v[y + 1, x - 1] + v[y + 1, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_u_to_v_face_2d(u, y, x):
    """Average 4 u-velocity values to v-face location.

    Args:
        u: x-velocity on x-faces (ny, nx+1)
        y, x: v-face coordinates

    Returns:
        u-velocity at v-face
    """
    return (u[y, x] + u[y, x + 1] + u[y - 1, x + 1] + u[y - 1, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_v_to_u_face_3d(v, z, y, x):
    """Average 4 v-velocity values to u-face location."""
    return (v[z, y, x] + v[z, y, x - 1] + v[z, y + 1, x - 1] + v[z, y + 1, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_w_to_u_face_3d(w, z, y, x):
    """Average 4 w-velocity values to u-face location."""
    return (w[z, y, x] + w[z, y, x - 1] + w[z + 1, y, x - 1] + w[z + 1, y, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_u_to_v_face_3d(u, z, y, x):
    """Average 4 u-velocity values to v-face location."""
    return (u[z, y, x] + u[z, y, x + 1] + u[z, y - 1, x + 1] + u[z, y - 1, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_w_to_v_face_3d(w, z, y, x):
    """Average 4 w-velocity values to v-face location."""
    return (w[z, y, x] + w[z, y - 1, x] + w[z + 1, y - 1, x] + w[z + 1, y, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_u_to_w_face_3d(u, z, y, x):
    """Average 4 u-velocity values to w-face location."""
    return (u[z, y, x] + u[z, y, x + 1] + u[z - 1, y, x + 1] + u[z - 1, y, x]) * 0.25


@jit(nopython=True, cache=True)
def interpolate_v_to_w_face_3d(v, z, y, x):
    """Average 4 v-velocity values to w-face location."""
    return (v[z, y, x] + v[z, y + 1, x] + v[z - 1, y + 1, x] + v[z - 1, y, x]) * 0.25


@jit(nopython=True, cache=True)
def clamp_to_cell_center_2d(x, y, nx, ny):
    """Clamp position to valid cell center region."""
    x_clamped = max(1.0, min(x, nx - 2.0))
    y_clamped = max(1.0, min(y, ny - 2.0))
    return x_clamped, y_clamped


@jit(nopython=True, cache=True)
def clamp_to_cell_center_3d(x, y, z, nx, ny, nz):
    """Clamp position to valid cell center region."""
    x_clamped = max(1.0, min(x, nx - 2.0))
    y_clamped = max(1.0, min(y, ny - 2.0))
    z_clamped = max(1.0, min(z, nz - 2.0))
    return x_clamped, y_clamped, z_clamped


@jit(nopython=True, cache=True)
def clamp_to_u_face_2d(x, y, nx, ny):
    """Clamp position to valid u-face region."""
    x_clamped = max(1.5, min(x, nx - 1.5))
    y_clamped = max(1.5, min(y, ny - 2.5))
    return x_clamped, y_clamped


@jit(nopython=True, cache=True)
def clamp_to_u_face_3d(x, y, z, nx, ny, nz):
    """Clamp position to valid u-face region."""
    x_clamped = max(1.5, min(x, nx - 1.5))
    y_clamped = max(1.5, min(y, ny - 2.5))
    z_clamped = max(1.5, min(z, nz - 2.5))
    return x_clamped, y_clamped, z_clamped


@jit(nopython=True, cache=True)
def clamp_to_v_face_2d(x, y, nx, ny):
    """Clamp position to valid v-face region."""
    x_clamped = max(1.5, min(x, nx - 2.5))
    y_clamped = max(1.5, min(y, ny - 1.5))
    return x_clamped, y_clamped


@jit(nopython=True, cache=True)
def clamp_to_v_face_3d(x, y, z, nx, ny, nz):
    """Clamp position to valid v-face region."""
    x_clamped = max(1.5, min(x, nx - 2.5))
    y_clamped = max(1.5, min(y, ny - 1.5))
    z_clamped = max(1.5, min(z, nz - 2.5))
    return x_clamped, y_clamped, z_clamped


@jit(nopython=True, cache=True)
def clamp_to_w_face_3d(x, y, z, nx, ny, nz):
    """Clamp position to valid w-face region."""
    x_clamped = max(1.5, min(x, nx - 2.5))
    y_clamped = max(1.5, min(y, ny - 2.5))
    z_clamped = max(1.5, min(z, nz - 1.5))
    return x_clamped, y_clamped, z_clamped


@jit(nopython=True, cache=True)
def find_neighborhood_bounds_2d(field, y, x, ny, nx, radius=1):
    """Find min/max in neighborhood for MacCormack clamping.

    Args:
        field: Scalar field (ny, nx)
        y, x: Center coordinates
        ny, nx: Grid dimensions
        radius: Neighborhood radius (default: 1)

    Returns:
        (val_min, val_max) in neighborhood
    """
    val_min = field[y, x]
    val_max = field[y, x]

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny_idx = y + dy
            nx_idx = x + dx
            if 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                val = field[ny_idx, nx_idx]
                val_min = min(val_min, val)
                val_max = max(val_max, val)

    return val_min, val_max


@jit(nopython=True, cache=True)
def find_neighborhood_bounds_3d(field, z, y, x, nz, ny, nx, radius=1):
    """Find min/max in neighborhood for MacCormack clamping.

    Args:
        field: Scalar field (nz, ny, nx)
        z, y, x: Center coordinates
        nz, ny, nx: Grid dimensions
        radius: Neighborhood radius (default: 1)

    Returns:
        (val_min, val_max) in neighborhood
    """
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
                    val_min = min(val_min, val)
                    val_max = max(val_max, val)

    return val_min, val_max


@jit(nopython=True, cache=True)
def find_neighborhood_bounds_1d_y_2d(field, y, x, ny):
    """Find min/max along y-axis for velocity MacCormack clamping.

    Args:
        field: Velocity component field
        y, x: Face coordinates
        ny: Grid dimension in y

    Returns:
        (val_min, val_max) in neighborhood
    """
    val_min = field[y, x]
    val_max = field[y, x]

    for dy in range(-1, 2):
        ny_idx = y + dy
        if 0 <= ny_idx < ny:
            val = field[ny_idx, x]
            val_min = min(val_min, val)
            val_max = max(val_max, val)

    return val_min, val_max


def reset_forces_2d(force: MACGrid2D):
    """Reset all force components to zero."""
    force.u_data.fill(0)
    force.v_data.fill(0)


def reset_forces_3d(force: MACGrid3D):
    """Reset all force components to zero."""
    force.u_data.fill(0)
    force.v_data.fill(0)
    force.w_data.fill(0)


def apply_force_to_velocity_2d(velocity: MACGrid2D, force: MACGrid2D, dt: float):
    """Apply force to velocity: v += dt * f."""
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data


def apply_force_to_velocity_3d(velocity: MACGrid3D, force: MACGrid3D, dt: float):
    """Apply force to velocity: v += dt * f."""
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data
    velocity.w_data += dt * force.w_data


@jit(nopython=True, parallel=True, cache=True)
def average_center_to_u_faces_2d(center_values, u_faces, ny, nx):
    """Average cell-centered values to u-faces."""
    for y in prange(1, ny - 1):
        for x in range(1, nx):
            u_faces[y, x] = 0.5 * (center_values[y, x - 1] + center_values[y, x])


@jit(nopython=True, parallel=True, cache=True)
def average_center_to_v_faces_2d(center_values, v_faces, ny, nx):
    """Average cell-centered values to v-faces."""
    for y in prange(1, ny):
        for x in range(1, nx - 1):
            v_faces[y, x] = 0.5 * (center_values[y - 1, x] + center_values[y, x])


@jit(nopython=True, parallel=True, cache=True)
def average_center_to_u_faces_3d(center_values, u_faces, nz, ny, nx):
    """Average cell-centered values to u-faces."""
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                u_faces[z, y, x] = 0.5 * (center_values[z, y, x - 1] + center_values[z, y, x])


@jit(nopython=True, parallel=True, cache=True)
def average_center_to_v_faces_3d(center_values, v_faces, nz, ny, nx):
    """Average cell-centered values to v-faces."""
    for z in prange(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                v_faces[z, y, x] = 0.5 * (center_values[z, y - 1, x] + center_values[z, y, x])


@jit(nopython=True, parallel=True, cache=True)
def average_center_to_w_faces_3d(center_values, w_faces, nz, ny, nx):
    """Average cell-centered values to w-faces."""
    for z in prange(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                w_faces[z, y, x] = 0.5 * (center_values[z - 1, y, x] + center_values[z, y, x])
