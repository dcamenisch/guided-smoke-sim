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
    # Preserve gradients - only convert if necessary
    if force_field_u.device != force.u_data.device or force_field_u.dtype != force.u_data.dtype:
        force.u_data = force_field_u.to(force.u_data.device, dtype=force.u_data.dtype)
    else:
        force.u_data = force_field_u
        
    if force_field_v.device != force.v_data.device or force_field_v.dtype != force.v_data.dtype:
        force.v_data = force_field_v.to(force.v_data.device, dtype=force.v_data.dtype)
    else:
        force.v_data = force_field_v

    velocity.u_data = velocity.u_data + force.u_data * dt
    velocity.v_data = velocity.v_data + force.v_data * dt


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
    # Preserve gradients - only convert if necessary
    if force_field_u.device != force.u_data.device or force_field_u.dtype != force.u_data.dtype:
        force.u_data = force_field_u.to(force.u_data.device, dtype=force.u_data.dtype)
    else:
        force.u_data = force_field_u
        
    if force_field_v.device != force.v_data.device or force_field_v.dtype != force.v_data.dtype:
        force.v_data = force_field_v.to(force.v_data.device, dtype=force.v_data.dtype)
    else:
        force.v_data = force_field_v
        
    if force_field_w.device != force.w_data.device or force_field_w.dtype != force.w_data.dtype:
        force.w_data = force_field_w.to(force.w_data.device, dtype=force.w_data.dtype)
    else:
        force.w_data = force_field_w

    velocity.u_data = velocity.u_data + force.u_data * dt
    velocity.v_data = velocity.v_data + force.v_data * dt
    velocity.w_data = velocity.w_data + force.w_data * dt
