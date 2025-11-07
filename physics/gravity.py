"""Gravity force implementations."""

import numpy as np
from core import MACGrid2D, MACGrid3D
from kernels import grid_ops


def apply_gravity_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    dt: float,
    g: float = -9.81,
):
    """Apply uniform gravity force to 2D velocity field

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        dt: Time step
        g: Gravitational acceleration (default: -9.81, negative is downward)
    """
    # Reset forces
    grid_ops.reset_forces_2d(force)

    # Apply gravity to v-velocity (y-direction)
    force.v_data[:] = g

    velocity.v_data += dt * force.v_data


def apply_gravity_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    dt: float,
    g: float = -9.81,
):
    """Apply uniform gravity force to 3D velocity field

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        dt: Time step
        g: Gravitational acceleration (default: -9.81, negative is downward)
    """
    # Reset forces
    grid_ops.reset_forces_3d(force)

    # Apply gravity to v-velocity (y-direction)
    force.v_data[:] = g

    velocity.v_data += dt * force.v_data
