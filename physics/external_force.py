"""External force field implementations."""

import numpy as np
from core import MACGrid2D, MACGrid3D


def apply_external_force_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    force_field_u: np.ndarray,
    force_field_v: np.ndarray,
    dt: float,
):
    """Apply external force field to 2D velocity

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        force_field_u: External force in x-direction (ny, nx+1)
        force_field_v: External force in y-direction (ny+1, nx)
        dt: Time step
    """
    force.u_data[:] = force_field_u
    force.v_data[:] = force_field_v

    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data


def apply_external_force_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    force_field_u: np.ndarray,
    force_field_v: np.ndarray,
    force_field_w: np.ndarray,
    dt: float,
):
    """Apply external force field to 3D velocity

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        force_field_u: External force in x-direction (nz, ny, nx+1)
        force_field_v: External force in y-direction (nz, ny+1, nx)
        force_field_w: External force in z-direction (nz+1, ny, nx)
        dt: Time step
    """
    force.u_data[:] = force_field_u
    force.v_data[:] = force_field_v
    force.w_data[:] = force_field_w

    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data
    velocity.w_data += dt * force.w_data
