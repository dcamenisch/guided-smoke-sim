"""Numba-optimized kernels for fluid simulation."""

from kernels.interpolation import bilinear_interp, trilinear_interp
from kernels.poisson import solve_poisson_jacobi_2d, solve_poisson_jacobi_3d
from kernels.advection import (
    advect_density_kernel_2d,
    advect_u_velocity_kernel_2d,
    advect_v_velocity_kernel_2d,
    advect_density_kernel_3d,
    advect_u_velocity_kernel_3d,
    advect_v_velocity_kernel_3d,
    advect_w_velocity_kernel_3d,
    advect_density_maccormack_2d,
    advect_u_velocity_maccormack_2d,
    advect_v_velocity_maccormack_2d,
    advect_density_maccormack_3d,
    advect_u_velocity_maccormack_3d,
    advect_v_velocity_maccormack_3d,
    advect_w_velocity_maccormack_3d,
)
from kernels.velocity import correct_velocity_kernel_2d, correct_velocity_kernel_3d
from kernels.operators import compute_vorticity_kernel_2d, compute_vorticity_kernel_3d

__all__ = [
    "bilinear_interp",
    "trilinear_interp",
    "solve_poisson_jacobi_2d",
    "solve_poisson_jacobi_3d",
    "advect_density_kernel_2d",
    "advect_u_velocity_kernel_2d",
    "advect_v_velocity_kernel_2d",
    "advect_density_kernel_3d",
    "advect_u_velocity_kernel_3d",
    "advect_v_velocity_kernel_3d",
    "advect_w_velocity_kernel_3d",
    "advect_density_maccormack_2d",
    "advect_u_velocity_maccormack_2d",
    "advect_v_velocity_maccormack_2d",
    "advect_density_maccormack_3d",
    "advect_u_velocity_maccormack_3d",
    "advect_v_velocity_maccormack_3d",
    "advect_w_velocity_maccormack_3d",
    "correct_velocity_kernel_2d",
    "correct_velocity_kernel_3d",
    "compute_vorticity_kernel_2d",
    "compute_vorticity_kernel_3d",
]
