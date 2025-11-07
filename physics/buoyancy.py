"""Buoyancy force implementations."""

import numpy as np
from core import MACGrid2D, MACGrid3D


def apply_buoyancy_force_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    density: np.ndarray,
    dt: float,
    nx: int,
    alpha: float = 0.1,
):
    """Apply buoyancy force to 2D velocity field

    Buoyancy force is proportional to density and acts in the upward (y) direction.

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        density: Density field (ny, nx)
        dt: Time step
        nx: Grid resolution in x (used for scaling)
        alpha: Buoyancy coefficient (default: 0.1)
    """
    # Reset forces
    force.u_data.fill(0)
    force.v_data.fill(0)

    # Scaling factor to match C++ implementation
    scaling_factor = 64.0 / nx

    # Buoyancy force proportional to density
    buoyancy = alpha * density

    # Average adjacent cells to interior y-faces
    buoyancy_at_faces = 0.5 * (buoyancy[:-1, :] + buoyancy[1:, :])

    # Apply to interior v-faces (y-velocity)
    force.v_data[1:-1, :] += buoyancy_at_faces * scaling_factor

    # Update velocities with forces
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data


def apply_buoyancy_force_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    density: np.ndarray,
    dt: float,
    nx: int,
    alpha: float = 0.1,
):
    """Apply buoyancy force to 3D velocity field

    Buoyancy force is proportional to density and acts in the upward (y) direction.

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        density: Density field (nz, ny, nx)
        dt: Time step
        nx: Grid resolution in x (used for scaling)
        alpha: Buoyancy coefficient (default: 0.1)
    """
    # Reset forces
    force.u_data.fill(0)
    force.v_data.fill(0)
    force.w_data.fill(0)

    # Scaling factor to match C++ implementation
    scaling_factor = 64.0 / nx

    # Buoyancy force proportional to density
    buoyancy = alpha * density

    # Average adjacent cells to interior y-faces
    buoyancy_at_faces = 0.5 * (buoyancy[:, :-1, :] + buoyancy[:, 1:, :])

    # Apply to interior v-faces (y-velocity)
    force.v_data[:, 1:-1, :] += buoyancy_at_faces * scaling_factor

    # Update all velocity components
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data
    velocity.w_data += dt * force.w_data
