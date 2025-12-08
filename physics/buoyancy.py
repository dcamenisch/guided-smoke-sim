"""Buoyancy force implementations."""

from __future__ import annotations

import torch

from core import MACGrid2D, MACGrid3D
from kernels import grid_ops


def apply_buoyancy_force_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    density: torch.Tensor,
    dt: float,
    nx: int,
    alpha: float = 0.1,
) -> None:
    """Apply buoyancy force proportional to density in upward direction.

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        density: Density field (ny, nx)
        dt: Time step
        nx: Grid resolution for scaling
        alpha: Buoyancy coefficient (default: 0.1)
    """
    grid_ops.reset_forces_2d(force)

    scaling_factor = density.new_tensor(64.0 / nx)

    # Buoyancy force proportional to density
    buoyancy = density * alpha

    # Average adjacent cells to interior y-faces
    buoyancy_at_faces = 0.5 * (buoyancy[:-1, :] + buoyancy[1:, :])

    # Apply to interior v-faces (y-velocity) - differentiable version
    v_force = force.v_data.clone()
    v_force[1:-1, :] = v_force[1:-1, :] + buoyancy_at_faces * scaling_factor
    force.v_data = v_force

    # Update velocities with forces
    grid_ops.apply_force_to_velocity_2d(velocity, force, dt)


def apply_buoyancy_force_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    density: torch.Tensor,
    dt: float,
    nx: int,
    alpha: float = 0.1,
) -> None:
    """Apply buoyancy force proportional to density in upward direction.

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        density: Density field (nz, ny, nx)
        dt: Time step
        nx: Grid resolution for scaling
        alpha: Buoyancy coefficient (default: 0.1)
    """
    grid_ops.reset_forces_3d(force)

    scaling_factor = density.new_tensor(64.0 / nx)

    # Buoyancy force proportional to density
    buoyancy = density * alpha

    # Average adjacent cells to interior y-faces
    buoyancy_at_faces = 0.5 * (buoyancy[:, :-1, :] + buoyancy[:, 1:, :])

    # Apply to interior v-faces (y-velocity) - differentiable version
    v_force = force.v_data.clone()
    v_force[:, 1:-1, :] = v_force[:, 1:-1, :] + buoyancy_at_faces * scaling_factor
    force.v_data = v_force

    # Update all velocity components
    grid_ops.apply_force_to_velocity_3d(velocity, force, dt)
