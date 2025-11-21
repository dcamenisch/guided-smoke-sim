"""External force field implementations."""

from __future__ import annotations

import torch

from core import MACGrid2D, MACGrid3D


def apply_external_force_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    force_field_u: torch.Tensor,
    force_field_v: torch.Tensor,
    dt: float,
) -> None:
    """Apply external force field to 2D velocity

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        force_field_u: External force in x-direction (ny, nx+1)
        force_field_v: External force in y-direction (ny+1, nx)
        dt: Time step
    """
    force.u_data.copy_(force_field_u.to(force.u_data.device, dtype=force.u_data.dtype))
    force.v_data.copy_(force_field_v.to(force.v_data.device, dtype=force.v_data.dtype))

    velocity.u_data.add_(force.u_data, alpha=dt)
    velocity.v_data.add_(force.v_data, alpha=dt)


def apply_external_force_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    force_field_u: torch.Tensor,
    force_field_v: torch.Tensor,
    force_field_w: torch.Tensor,
    dt: float,
) -> None:
    """Apply external force field to 3D velocity

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        force_field_u: External force in x-direction (nz, ny, nx+1)
        force_field_v: External force in y-direction (nz, ny+1, nx)
        force_field_w: External force in z-direction (nz+1, ny, nx)
        dt: Time step
    """
    force.u_data.copy_(force_field_u.to(force.u_data.device, dtype=force.u_data.dtype))
    force.v_data.copy_(force_field_v.to(force.v_data.device, dtype=force.v_data.dtype))
    force.w_data.copy_(force_field_w.to(force.w_data.device, dtype=force.w_data.dtype))

    velocity.u_data.add_(force.u_data, alpha=dt)
    velocity.v_data.add_(force.v_data, alpha=dt)
    velocity.w_data.add_(force.w_data, alpha=dt)
