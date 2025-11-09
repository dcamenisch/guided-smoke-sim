"""Advection kernels using semi-Lagrangian method."""

import numpy as np
from numba import jit, prange
from kernels.interpolation import bilinear_interp, trilinear_interp
from kernels import grid_ops


# ============= 2D Advection Kernels =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel_2d(density, density_new, u, v, dx, dt, ny, nx, rk_order=1):
    """Optimized 2D density advection with Numba

    Uses semi-Lagrangian advection: trace particle backwards and interpolate
    Supports 1st order (Euler) and 3rd order (SSPRK3) backtracing

    Args:
        density: Current density field (ny, nx)
        density_new: Output density field (ny, nx)
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            if rk_order == 1:
                # First-order Euler backtracing
                last_x_velocity, last_y_velocity = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, y, x)

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity

                # Clamp to valid density region
                last_x, last_y = grid_ops.clamp_to_cell_center_2d(last_x, last_y, nx, ny)

            elif rk_order == 3:
                # Third-order SSPRK3 backtracing
                # Stage 1: Euler step
                vel_x, vel_y = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, y, x)
                k1_x = vel_x
                k1_y = vel_y

                # Stage 2: Evaluate at position displaced by 0.5 * dt * k1
                pos2_x = x - 0.5 * dt / dx * k1_x
                pos2_y = y - 0.5 * dt / dx * k1_y
                pos2_x, pos2_y = grid_ops.clamp_to_cell_center_2d(pos2_x, pos2_y, nx, ny)

                # Interpolate velocity at stage 2 position
                k2_x = bilinear_interp(u[:, :-1] if u.shape[1] > nx else u, pos2_x, pos2_y)
                k2_y = bilinear_interp(v[:-1, :] if v.shape[0] > ny else v, pos2_x, pos2_y)

                # Stage 3: Evaluate at position displaced by 0.75 * dt * k2
                pos3_x = x - 0.75 * dt / dx * k2_x
                pos3_y = y - 0.75 * dt / dx * k2_y
                pos3_x, pos3_y = grid_ops.clamp_to_cell_center_2d(pos3_x, pos3_y, nx, ny)

                # Interpolate velocity at stage 3 position
                k3_x = bilinear_interp(u[:, :-1] if u.shape[1] > nx else u, pos3_x, pos3_y)
                k3_y = bilinear_interp(v[:-1, :] if v.shape[0] > ny else v, pos3_x, pos3_y)

                # Final position: weighted combination (2/9, 3/9, 4/9)
                last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)

                # Clamp to valid density region
                last_x, last_y = grid_ops.clamp_to_cell_center_2d(last_x, last_y, nx, ny)

            else:
                # Default to Euler if unknown order
                last_x_velocity, last_y_velocity = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, y, x)
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_x, last_y = grid_ops.clamp_to_cell_center_2d(last_x, last_y, nx, ny)

            # Bilinear interpolation
            density_new[y, x] = bilinear_interp(density, last_x, last_y)


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_kernel_2d(u, u_new, v, dx, dt, ny, nx, rk_order=1):
    """Optimized 2D u-velocity advection on MAC grid

    Args:
        u: Current x-velocity (ny, nx+1)
        u_new: Output x-velocity (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for y in prange(1, ny - 1):
        for x in range(1, nx):  # u goes to nx
            if rk_order == 1:
                # First-order Euler backtracing
                last_x_velocity = u[y, x]
                last_y_velocity = grid_ops.interpolate_v_to_u_face_2d(v, y, x)

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity

                # Clamp to MAC grid boundaries for u
                last_x, last_y = grid_ops.clamp_to_u_face_2d(last_x, last_y, nx, ny)

            elif rk_order == 3:
                # Third-order SSPRK3 backtracing
                # Stage 1
                k1_x = u[y, x]
                k1_y = grid_ops.interpolate_v_to_u_face_2d(v, y, x)

                # Stage 2
                pos2_x = x - 0.5 * dt / dx * k1_x
                pos2_y = y - 0.5 * dt / dx * k1_y
                pos2_x, pos2_y = grid_ops.clamp_to_u_face_2d(pos2_x, pos2_y, nx, ny)

                k2_x = bilinear_interp(u, pos2_x, pos2_y)
                k2_y = bilinear_interp(v[:-1, :] if v.shape[0] > ny else v, pos2_x, pos2_y)

                # Stage 3
                pos3_x = x - 0.75 * dt / dx * k2_x
                pos3_y = y - 0.75 * dt / dx * k2_y
                pos3_x, pos3_y = grid_ops.clamp_to_u_face_2d(pos3_x, pos3_y, nx, ny)

                k3_x = bilinear_interp(u, pos3_x, pos3_y)
                k3_y = bilinear_interp(v[:-1, :] if v.shape[0] > ny else v, pos3_x, pos3_y)

                # Final position
                last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)

                last_x, last_y = grid_ops.clamp_to_u_face_2d(last_x, last_y, nx, ny)

            else:
                # Default to Euler
                last_x_velocity = u[y, x]
                last_y_velocity = grid_ops.interpolate_v_to_u_face_2d(v, y, x)
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_x, last_y = grid_ops.clamp_to_u_face_2d(last_x, last_y, nx, ny)

            # Bilinear interpolation
            u_new[y, x] = bilinear_interp(u, last_x, last_y)


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_kernel_2d(v, v_new, u, dx, dt, ny, nx, rk_order=1):
    """Optimized 2D v-velocity advection on MAC grid

    Args:
        v: Current y-velocity (ny+1, nx)
        v_new: Output y-velocity (ny+1, nx)
        u: x-velocity component (ny, nx+1)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for y in prange(1, ny):  # v goes to ny
        for x in range(1, nx - 1):
            if rk_order == 1:
                # First-order Euler backtracing
                last_y_velocity = v[y, x]
                last_x_velocity = grid_ops.interpolate_u_to_v_face_2d(u, y, x)

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity

                # Clamp to MAC grid boundaries for v
                last_x, last_y = grid_ops.clamp_to_v_face_2d(last_x, last_y, nx, ny)

            elif rk_order == 3:
                # Third-order SSPRK3 backtracing
                # Stage 1
                k1_y = v[y, x]
                k1_x = grid_ops.interpolate_u_to_v_face_2d(u, y, x)

                # Stage 2
                pos2_x = x - 0.5 * dt / dx * k1_x
                pos2_y = y - 0.5 * dt / dx * k1_y
                pos2_x, pos2_y = grid_ops.clamp_to_v_face_2d(pos2_x, pos2_y, nx, ny)

                k2_x = bilinear_interp(u[:, :-1] if u.shape[1] > nx else u, pos2_x, pos2_y)
                k2_y = bilinear_interp(v, pos2_x, pos2_y)

                # Stage 3
                pos3_x = x - 0.75 * dt / dx * k2_x
                pos3_y = y - 0.75 * dt / dx * k2_y
                pos3_x, pos3_y = grid_ops.clamp_to_v_face_2d(pos3_x, pos3_y, nx, ny)

                k3_x = bilinear_interp(u[:, :-1] if u.shape[1] > nx else u, pos3_x, pos3_y)
                k3_y = bilinear_interp(v, pos3_x, pos3_y)

                # Final position
                last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)

                last_x, last_y = grid_ops.clamp_to_v_face_2d(last_x, last_y, nx, ny)

            else:
                # Default to Euler
                last_y_velocity = v[y, x]
                last_x_velocity = grid_ops.interpolate_u_to_v_face_2d(u, y, x)
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_x, last_y = grid_ops.clamp_to_v_face_2d(last_x, last_y, nx, ny)

            # Bilinear interpolation
            v_new[y, x] = bilinear_interp(v, last_x, last_y)


# ============= 3D Advection Kernels =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel_3d(density, density_new, u, v, w, dx, dt, nz, ny, nx, rk_order=1):
    """Optimized 3D density advection with Numba

    Uses semi-Lagrangian advection: trace particle backwards and interpolate
    Supports 1st order (Euler) and 3rd order (SSPRK3) backtracing

    Args:
        density: Current density field (nz, ny, nx)
        density_new: Output density field (nz, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if rk_order == 1:
                    # First-order Euler backtracing
                    last_x_velocity, last_y_velocity, last_z_velocity = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x)

                    # Trace backwards
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity

                    # Clamp
                    last_x, last_y, last_z = grid_ops.clamp_to_cell_center_3d(last_x, last_y, last_z, nx, ny, nz)

                elif rk_order == 3:
                    # Third-order SSPRK3 backtracing
                    # Stage 1
                    vel_x, vel_y, vel_z = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x)
                    k1_x, k1_y, k1_z = vel_x, vel_y, vel_z

                    # Stage 2
                    pos2_x = x - 0.5 * dt / dx * k1_x
                    pos2_y = y - 0.5 * dt / dx * k1_y
                    pos2_z = z - 0.5 * dt / dx * k1_z
                    pos2_x, pos2_y, pos2_z = grid_ops.clamp_to_cell_center_3d(pos2_x, pos2_y, pos2_z, nx, ny, nz)

                    k2_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos2_x, pos2_y, pos2_z)
                    k2_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos2_x, pos2_y, pos2_z)
                    k2_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos2_x, pos2_y, pos2_z)

                    # Stage 3
                    pos3_x = x - 0.75 * dt / dx * k2_x
                    pos3_y = y - 0.75 * dt / dx * k2_y
                    pos3_z = z - 0.75 * dt / dx * k2_z
                    pos3_x, pos3_y, pos3_z = grid_ops.clamp_to_cell_center_3d(pos3_x, pos3_y, pos3_z, nx, ny, nz)

                    k3_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos3_x, pos3_y, pos3_z)
                    k3_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos3_x, pos3_y, pos3_z)
                    k3_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos3_x, pos3_y, pos3_z)

                    # Final position
                    last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                    last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)
                    last_z = z - dt / dx * (2.0/9.0 * k1_z + 3.0/9.0 * k2_z + 4.0/9.0 * k3_z)

                    last_x, last_y, last_z = grid_ops.clamp_to_cell_center_3d(last_x, last_y, last_z, nx, ny, nz)

                else:
                    # Default to Euler
                    last_x_velocity, last_y_velocity, last_z_velocity = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x)
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity
                    last_x, last_y, last_z = grid_ops.clamp_to_cell_center_3d(last_x, last_y, last_z, nx, ny, nz)

                # Trilinear interpolation
                density_new[z, y, x] = trilinear_interp(density, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_kernel_3d(u, u_new, v, w, dx, dt, nz, ny, nx, rk_order=1):
    """Optimized 3D u-velocity advection on MAC grid

    Args:
        u: Current x-velocity (nz, ny, nx+1)
        u_new: Output x-velocity (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):  # u goes to nx
                if rk_order == 1:
                    # First-order Euler backtracing
                    last_x_velocity = u[z, y, x]
                    last_y_velocity = grid_ops.interpolate_v_to_u_face_3d(v, z, y, x)
                    last_z_velocity = grid_ops.interpolate_w_to_u_face_3d(w, z, y, x)

                    # Trace backwards
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity

                    # Clamp to MAC grid boundaries for u
                    last_x, last_y, last_z = grid_ops.clamp_to_u_face_3d(last_x, last_y, last_z, nx, ny, nz)

                elif rk_order == 3:
                    # Third-order SSPRK3 backtracing
                    # Stage 1
                    k1_x = u[z, y, x]
                    k1_y = grid_ops.interpolate_v_to_u_face_3d(v, z, y, x)
                    k1_z = grid_ops.interpolate_w_to_u_face_3d(w, z, y, x)

                    # Stage 2
                    pos2_x = x - 0.5 * dt / dx * k1_x
                    pos2_y = y - 0.5 * dt / dx * k1_y
                    pos2_z = z - 0.5 * dt / dx * k1_z
                    pos2_x, pos2_y, pos2_z = grid_ops.clamp_to_u_face_3d(pos2_x, pos2_y, pos2_z, nx, ny, nz)

                    k2_x = trilinear_interp(u, pos2_x, pos2_y, pos2_z)
                    k2_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos2_x, pos2_y, pos2_z)
                    k2_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos2_x, pos2_y, pos2_z)

                    # Stage 3
                    pos3_x = x - 0.75 * dt / dx * k2_x
                    pos3_y = y - 0.75 * dt / dx * k2_y
                    pos3_z = z - 0.75 * dt / dx * k2_z
                    pos3_x, pos3_y, pos3_z = grid_ops.clamp_to_u_face_3d(pos3_x, pos3_y, pos3_z, nx, ny, nz)

                    k3_x = trilinear_interp(u, pos3_x, pos3_y, pos3_z)
                    k3_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos3_x, pos3_y, pos3_z)
                    k3_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos3_x, pos3_y, pos3_z)

                    # Final position
                    last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                    last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)
                    last_z = z - dt / dx * (2.0/9.0 * k1_z + 3.0/9.0 * k2_z + 4.0/9.0 * k3_z)

                    last_x, last_y, last_z = grid_ops.clamp_to_u_face_3d(last_x, last_y, last_z, nx, ny, nz)

                else:
                    # Default to Euler
                    last_x_velocity = u[z, y, x]
                    last_y_velocity = grid_ops.interpolate_v_to_u_face_3d(v, z, y, x)
                    last_z_velocity = grid_ops.interpolate_w_to_u_face_3d(w, z, y, x)
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity
                    last_x, last_y, last_z = grid_ops.clamp_to_u_face_3d(last_x, last_y, last_z, nx, ny, nz)

                # Trilinear interpolation
                u_new[z, y, x] = trilinear_interp(u, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_kernel_3d(v, v_new, u, w, dx, dt, nz, ny, nx, rk_order=1):
    """Optimized 3D v-velocity advection on MAC grid

    Args:
        v: Current y-velocity (nz, ny+1, nx)
        v_new: Output y-velocity (nz, ny+1, nx)
        u: x-velocity component (nz, ny, nx+1)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for z in prange(1, nz - 1):
        for y in range(1, ny):  # v goes to ny
            for x in range(1, nx - 1):
                if rk_order == 1:
                    # First-order Euler backtracing
                    last_y_velocity = v[z, y, x]
                    last_x_velocity = grid_ops.interpolate_u_to_v_face_3d(u, z, y, x)
                    last_z_velocity = grid_ops.interpolate_w_to_v_face_3d(w, z, y, x)

                    # Trace backwards
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity

                    # Clamp to MAC grid boundaries for v
                    last_x, last_y, last_z = grid_ops.clamp_to_v_face_3d(last_x, last_y, last_z, nx, ny, nz)

                elif rk_order == 3:
                    # Third-order SSPRK3 backtracing
                    # Stage 1
                    k1_y = v[z, y, x]
                    k1_x = grid_ops.interpolate_u_to_v_face_3d(u, z, y, x)
                    k1_z = grid_ops.interpolate_w_to_v_face_3d(w, z, y, x)

                    # Stage 2
                    pos2_x = x - 0.5 * dt / dx * k1_x
                    pos2_y = y - 0.5 * dt / dx * k1_y
                    pos2_z = z - 0.5 * dt / dx * k1_z
                    pos2_x, pos2_y, pos2_z = grid_ops.clamp_to_v_face_3d(pos2_x, pos2_y, pos2_z, nx, ny, nz)

                    k2_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos2_x, pos2_y, pos2_z)
                    k2_y = trilinear_interp(v, pos2_x, pos2_y, pos2_z)
                    k2_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos2_x, pos2_y, pos2_z)

                    # Stage 3
                    pos3_x = x - 0.75 * dt / dx * k2_x
                    pos3_y = y - 0.75 * dt / dx * k2_y
                    pos3_z = z - 0.75 * dt / dx * k2_z
                    pos3_x, pos3_y, pos3_z = grid_ops.clamp_to_v_face_3d(pos3_x, pos3_y, pos3_z, nx, ny, nz)

                    k3_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos3_x, pos3_y, pos3_z)
                    k3_y = trilinear_interp(v, pos3_x, pos3_y, pos3_z)
                    k3_z = trilinear_interp(w[:-1, :, :] if w.shape[0] > nz else w, pos3_x, pos3_y, pos3_z)

                    # Final position
                    last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                    last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)
                    last_z = z - dt / dx * (2.0/9.0 * k1_z + 3.0/9.0 * k2_z + 4.0/9.0 * k3_z)

                    last_x, last_y, last_z = grid_ops.clamp_to_v_face_3d(last_x, last_y, last_z, nx, ny, nz)

                else:
                    # Default to Euler
                    last_y_velocity = v[z, y, x]
                    last_x_velocity = grid_ops.interpolate_u_to_v_face_3d(u, z, y, x)
                    last_z_velocity = grid_ops.interpolate_w_to_v_face_3d(w, z, y, x)
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity
                    last_x, last_y, last_z = grid_ops.clamp_to_v_face_3d(last_x, last_y, last_z, nx, ny, nz)

                # Trilinear interpolation
                v_new[z, y, x] = trilinear_interp(v, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_w_velocity_kernel_3d(w, w_new, u, v, dx, dt, nz, ny, nx, rk_order=1):
    """Optimized 3D w-velocity advection on MAC grid

    Args:
        w: Current z-velocity (nz+1, ny, nx)
        w_new: Output z-velocity (nz+1, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
        rk_order: Order of Runge-Kutta for backtracing (1 or 3)
    """
    for z in prange(1, nz):  # w goes to nz
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if rk_order == 1:
                    # First-order Euler backtracing
                    last_z_velocity = w[z, y, x]
                    last_x_velocity = grid_ops.interpolate_u_to_w_face_3d(u, z, y, x)
                    last_y_velocity = grid_ops.interpolate_v_to_w_face_3d(v, z, y, x)

                    # Trace backwards
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity

                    # Clamp to MAC grid boundaries for w
                    last_x, last_y, last_z = grid_ops.clamp_to_w_face_3d(last_x, last_y, last_z, nx, ny, nz)

                elif rk_order == 3:
                    # Third-order SSPRK3 backtracing
                    # Stage 1
                    k1_z = w[z, y, x]
                    k1_x = grid_ops.interpolate_u_to_w_face_3d(u, z, y, x)
                    k1_y = grid_ops.interpolate_v_to_w_face_3d(v, z, y, x)

                    # Stage 2
                    pos2_x = x - 0.5 * dt / dx * k1_x
                    pos2_y = y - 0.5 * dt / dx * k1_y
                    pos2_z = z - 0.5 * dt / dx * k1_z
                    pos2_x, pos2_y, pos2_z = grid_ops.clamp_to_w_face_3d(pos2_x, pos2_y, pos2_z, nx, ny, nz)

                    k2_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos2_x, pos2_y, pos2_z)
                    k2_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos2_x, pos2_y, pos2_z)
                    k2_z = trilinear_interp(w, pos2_x, pos2_y, pos2_z)

                    # Stage 3
                    pos3_x = x - 0.75 * dt / dx * k2_x
                    pos3_y = y - 0.75 * dt / dx * k2_y
                    pos3_z = z - 0.75 * dt / dx * k2_z
                    pos3_x, pos3_y, pos3_z = grid_ops.clamp_to_w_face_3d(pos3_x, pos3_y, pos3_z, nx, ny, nz)

                    k3_x = trilinear_interp(u[:, :, :-1] if u.shape[2] > nx else u, pos3_x, pos3_y, pos3_z)
                    k3_y = trilinear_interp(v[:, :-1, :] if v.shape[1] > ny else v, pos3_x, pos3_y, pos3_z)
                    k3_z = trilinear_interp(w, pos3_x, pos3_y, pos3_z)

                    # Final position
                    last_x = x - dt / dx * (2.0/9.0 * k1_x + 3.0/9.0 * k2_x + 4.0/9.0 * k3_x)
                    last_y = y - dt / dx * (2.0/9.0 * k1_y + 3.0/9.0 * k2_y + 4.0/9.0 * k3_y)
                    last_z = z - dt / dx * (2.0/9.0 * k1_z + 3.0/9.0 * k2_z + 4.0/9.0 * k3_z)

                    last_x, last_y, last_z = grid_ops.clamp_to_w_face_3d(last_x, last_y, last_z, nx, ny, nz)

                else:
                    # Default to Euler
                    last_z_velocity = w[z, y, x]
                    last_x_velocity = grid_ops.interpolate_u_to_w_face_3d(u, z, y, x)
                    last_y_velocity = grid_ops.interpolate_v_to_w_face_3d(v, z, y, x)
                    last_x = x - dt / dx * last_x_velocity
                    last_y = y - dt / dx * last_y_velocity
                    last_z = z - dt / dx * last_z_velocity
                    last_x, last_y, last_z = grid_ops.clamp_to_w_face_3d(last_x, last_y, last_z, nx, ny, nz)

                # Trilinear interpolation
                w_new[z, y, x] = trilinear_interp(w, last_x, last_y, last_z)


# ============= MacCormack Advection (2D) =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_maccormack_2d(density, density_new, u, v, dx, dt, ny, nx):
    """MacCormack advection for 2D density field

    Two-step method that reduces numerical dissipation:
    1. Forward step (standard semi-Lagrangian)
    2. Backward correction step
    3. Clamping to prevent overshoots

    Args:
        density: Current density field (ny, nx)
        density_new: Output density field (ny, nx)
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    # Temporary arrays for intermediate steps
    phi_hat = np.zeros_like(density)
    phi_hat_hat = np.zeros_like(density)

    # Step 1: Forward advection (standard semi-Lagrangian)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Interpolate velocity to cell center
            vel_x, vel_y = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, y, x)

            # Trace backwards
            last_x = x - dt / dx * vel_x
            last_y = y - dt / dx * vel_y

            # Clamp
            last_x, last_y = grid_ops.clamp_to_cell_center_2d(last_x, last_y, nx, ny)

            phi_hat[y, x] = bilinear_interp(density, last_x, last_y)

    # Step 2: Backward advection from phi_hat
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Interpolate velocity to cell center
            vel_x, vel_y = grid_ops.interpolate_velocity_to_cell_center_2d(u, v, y, x)

            # Trace FORWARD
            next_x = x + dt / dx * vel_x
            next_y = y + dt / dx * vel_y

            # Clamp
            next_x, next_y = grid_ops.clamp_to_cell_center_2d(next_x, next_y, nx, ny)

            phi_hat_hat[y, x] = bilinear_interp(phi_hat, next_x, next_y)

    # Step 3: Error correction and clamping
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # MacCormack correction
            correction = phi_hat[y, x] + 0.5 * (density[y, x] - phi_hat_hat[y, x])

            # Find min/max in neighborhood for clamping
            # This prevents overshoots and maintains stability
            neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_2d(density, y, x, ny, nx)

            # Clamp correction to neighborhood bounds
            density_new[y, x] = max(neighbors_min, min(correction, neighbors_max))


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_maccormack_2d(u, u_new, v, dx, dt, ny, nx):
    """MacCormack advection for 2D u-velocity on MAC grid

    Args:
        u: Current x-velocity (ny, nx+1)
        u_new: Output x-velocity (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(u)
    phi_hat_hat = np.zeros_like(u)

    # Forward step
    for y in prange(1, ny - 1):
        for x in range(1, nx):
            vel_x = u[y, x]
            vel_y = grid_ops.interpolate_v_to_u_face_2d(v, y, x)

            last_x = x - dt / dx * vel_x
            last_y = y - dt / dx * vel_y

            last_x, last_y = grid_ops.clamp_to_u_face_2d(last_x, last_y, nx, ny)

            phi_hat[y, x] = bilinear_interp(u, last_x, last_y)

    # Backward step
    for y in prange(1, ny - 1):
        for x in range(1, nx):
            vel_x = phi_hat[y, x]
            vel_y = grid_ops.interpolate_v_to_u_face_2d(v, y, x)

            next_x = x + dt / dx * vel_x
            next_y = y + dt / dx * vel_y

            next_x, next_y = grid_ops.clamp_to_u_face_2d(next_x, next_y, nx, ny)

            phi_hat_hat[y, x] = bilinear_interp(phi_hat, next_x, next_y)

    # Correction with clamping
    for y in prange(1, ny - 1):
        for x in range(1, nx):
            correction = phi_hat[y, x] + 0.5 * (u[y, x] - phi_hat_hat[y, x])

            # Find neighborhood bounds
            neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_1d_y_2d(u, y, x, ny)

            u_new[y, x] = max(neighbors_min, min(correction, neighbors_max))


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_maccormack_2d(v, v_new, u, dx, dt, ny, nx):
    """MacCormack advection for 2D v-velocity on MAC grid

    Args:
        v: Current y-velocity (ny+1, nx)
        v_new: Output y-velocity (ny+1, nx)
        u: x-velocity component (ny, nx+1)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(v)
    phi_hat_hat = np.zeros_like(v)

    # Forward step
    for y in prange(1, ny):
        for x in range(1, nx - 1):
            vel_y = v[y, x]
            vel_x = grid_ops.interpolate_u_to_v_face_2d(u, y, x)

            last_x = x - dt / dx * vel_x
            last_y = y - dt / dx * vel_y

            last_x, last_y = grid_ops.clamp_to_v_face_2d(last_x, last_y, nx, ny)

            phi_hat[y, x] = bilinear_interp(v, last_x, last_y)

    # Backward step
    for y in prange(1, ny):
        for x in range(1, nx - 1):
            vel_y = phi_hat[y, x]
            vel_x = grid_ops.interpolate_u_to_v_face_2d(u, y, x)

            next_x = x + dt / dx * vel_x
            next_y = y + dt / dx * vel_y

            next_x, next_y = grid_ops.clamp_to_v_face_2d(next_x, next_y, nx, ny)

            phi_hat_hat[y, x] = bilinear_interp(phi_hat, next_x, next_y)

    # Correction with clamping
    for y in prange(1, ny):
        for x in range(1, nx - 1):
            correction = phi_hat[y, x] + 0.5 * (v[y, x] - phi_hat_hat[y, x])

            # Find neighborhood bounds (1D along x-axis)
            neighbors_min = v[y, x]
            neighbors_max = v[y, x]

            for dx_offset in range(-1, 2):
                nx_idx = x + dx_offset
                if 0 <= nx_idx < nx:
                    val = v[y, nx_idx]
                    neighbors_min = min(neighbors_min, val)
                    neighbors_max = max(neighbors_max, val)

            v_new[y, x] = max(neighbors_min, min(correction, neighbors_max))


# ============= MacCormack Advection (3D) =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_maccormack_3d(density, density_new, u, v, w, dx, dt, nz, ny, nx):
    """MacCormack advection for 3D density field

    Args:
        density: Current density field (nz, ny, nx)
        density_new: Output density field (nz, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(density)
    phi_hat_hat = np.zeros_like(density)

    # Forward step
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                vel_x, vel_y, vel_z = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x)

                last_x = x - dt / dx * vel_x
                last_y = y - dt / dx * vel_y
                last_z = z - dt / dx * vel_z

                last_x, last_y, last_z = grid_ops.clamp_to_cell_center_3d(last_x, last_y, last_z, nx, ny, nz)

                phi_hat[z, y, x] = trilinear_interp(density, last_x, last_y, last_z)

    # Backward step
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                vel_x, vel_y, vel_z = grid_ops.interpolate_velocity_to_cell_center_3d(u, v, w, z, y, x)

                next_x = x + dt / dx * vel_x
                next_y = y + dt / dx * vel_y
                next_z = z + dt / dx * vel_z

                next_x, next_y, next_z = grid_ops.clamp_to_cell_center_3d(next_x, next_y, next_z, nx, ny, nz)

                phi_hat_hat[z, y, x] = trilinear_interp(phi_hat, next_x, next_y, next_z)

    # Correction with clamping
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    density[z, y, x] - phi_hat_hat[z, y, x]
                )

                # Find neighborhood bounds
                neighbors_min, neighbors_max = grid_ops.find_neighborhood_bounds_3d(density, z, y, x, nz, ny, nx)

                density_new[z, y, x] = max(
                    neighbors_min, min(correction, neighbors_max)
                )


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_maccormack_3d(u, u_new, v, w, dx, dt, nz, ny, nx):
    """MacCormack advection for 3D u-velocity on MAC grid

    Args:
        u: Current x-velocity (nz, ny, nx+1)
        u_new: Output x-velocity (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(u)
    phi_hat_hat = np.zeros_like(u)

    # Forward step
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                vel_x = u[z, y, x]
                vel_y = grid_ops.interpolate_v_to_u_face_3d(v, z, y, x)
                vel_z = grid_ops.interpolate_w_to_u_face_3d(w, z, y, x)

                last_x = x - dt / dx * vel_x
                last_y = y - dt / dx * vel_y
                last_z = z - dt / dx * vel_z

                last_x, last_y, last_z = grid_ops.clamp_to_u_face_3d(last_x, last_y, last_z, nx, ny, nz)

                phi_hat[z, y, x] = trilinear_interp(u, last_x, last_y, last_z)

    # Backward step
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                vel_x = phi_hat[z, y, x]
                vel_y = grid_ops.interpolate_v_to_u_face_3d(v, z, y, x)
                vel_z = grid_ops.interpolate_w_to_u_face_3d(w, z, y, x)

                next_x = x + dt / dx * vel_x
                next_y = y + dt / dx * vel_y
                next_z = z + dt / dx * vel_z

                next_x, next_y, next_z = grid_ops.clamp_to_u_face_3d(next_x, next_y, next_z, nx, ny, nz)

                phi_hat_hat[z, y, x] = trilinear_interp(phi_hat, next_x, next_y, next_z)

    # Correction with clamping
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                correction = phi_hat[z, y, x] + 0.5 * (
                    u[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = u[z, y, x]
                neighbors_max = u[z, y, x]

                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        nz_idx = z + dz
                        ny_idx = y + dy
                        if 0 <= nz_idx < nz and 0 <= ny_idx < ny:
                            val = u[nz_idx, ny_idx, x]
                            neighbors_min = min(neighbors_min, val)
                            neighbors_max = max(neighbors_max, val)

                u_new[z, y, x] = max(neighbors_min, min(correction, neighbors_max))


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_maccormack_3d(v, v_new, u, w, dx, dt, nz, ny, nx):
    """MacCormack advection for 3D v-velocity on MAC grid

    Args:
        v: Current y-velocity (nz, ny+1, nx)
        v_new: Output y-velocity (nz, ny+1, nx)
        u: x-velocity component (nz, ny, nx+1)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(v)
    phi_hat_hat = np.zeros_like(v)

    # Forward step
    for z in prange(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                vel_y = v[z, y, x]
                vel_x = grid_ops.interpolate_u_to_v_face_3d(u, z, y, x)
                vel_z = grid_ops.interpolate_w_to_v_face_3d(w, z, y, x)

                last_x = x - dt / dx * vel_x
                last_y = y - dt / dx * vel_y
                last_z = z - dt / dx * vel_z

                last_x, last_y, last_z = grid_ops.clamp_to_v_face_3d(last_x, last_y, last_z, nx, ny, nz)

                phi_hat[z, y, x] = trilinear_interp(v, last_x, last_y, last_z)

    # Backward step
    for z in prange(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                vel_y = phi_hat[z, y, x]
                vel_x = grid_ops.interpolate_u_to_v_face_3d(u, z, y, x)
                vel_z = grid_ops.interpolate_w_to_v_face_3d(w, z, y, x)

                next_x = x + dt / dx * vel_x
                next_y = y + dt / dx * vel_y
                next_z = z + dt / dx * vel_z

                next_x, next_y, next_z = grid_ops.clamp_to_v_face_3d(next_x, next_y, next_z, nx, ny, nz)

                phi_hat_hat[z, y, x] = trilinear_interp(phi_hat, next_x, next_y, next_z)

    # Correction with clamping
    for z in prange(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    v[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = v[z, y, x]
                neighbors_max = v[z, y, x]

                for dz in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        nz_idx = z + dz
                        nx_idx = x + dx_offset
                        if 0 <= nz_idx < nz and 0 <= nx_idx < nx:
                            val = v[nz_idx, y, nx_idx]
                            neighbors_min = min(neighbors_min, val)
                            neighbors_max = max(neighbors_max, val)

                v_new[z, y, x] = max(neighbors_min, min(correction, neighbors_max))


@jit(nopython=True, parallel=True, cache=True)
def advect_w_velocity_maccormack_3d(w, w_new, u, v, dx, dt, nz, ny, nx):
    """MacCormack advection for 3D w-velocity on MAC grid

    Args:
        w: Current z-velocity (nz+1, ny, nx)
        w_new: Output z-velocity (nz+1, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    phi_hat = np.zeros_like(w)
    phi_hat_hat = np.zeros_like(w)

    # Forward step
    for z in prange(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                vel_z = w[z, y, x]
                vel_x = grid_ops.interpolate_u_to_w_face_3d(u, z, y, x)
                vel_y = grid_ops.interpolate_v_to_w_face_3d(v, z, y, x)

                last_x = x - dt / dx * vel_x
                last_y = y - dt / dx * vel_y
                last_z = z - dt / dx * vel_z

                last_x, last_y, last_z = grid_ops.clamp_to_w_face_3d(last_x, last_y, last_z, nx, ny, nz)

                phi_hat[z, y, x] = trilinear_interp(w, last_x, last_y, last_z)

    # Backward step
    for z in prange(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                vel_z = phi_hat[z, y, x]
                # Use grid_ops helpers for consistency
                vel_x = grid_ops.interpolate_u_to_w_face_3d(u, z, y, x)
                vel_y = grid_ops.interpolate_v_to_w_face_3d(v, z, y, x)

                next_x = x + dt / dx * vel_x
                next_y = y + dt / dx * vel_y
                next_z = z + dt / dx * vel_z

                next_x = max(1.5, min(next_x, nx - 2.5))
                next_y = max(1.5, min(next_y, ny - 2.5))
                next_z = max(1.5, min(next_z, nz - 1.5))

                phi_hat_hat[z, y, x] = trilinear_interp(phi_hat, next_x, next_y, next_z)

    # Correction with clamping
    for z in prange(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                correction = phi_hat[z, y, x] + 0.5 * (
                    w[z, y, x] - phi_hat_hat[z, y, x]
                )

                neighbors_min = w[z, y, x]
                neighbors_max = w[z, y, x]

                for dy in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        ny_idx = y + dy
                        nx_idx = x + dx_offset
                        if 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                            val = w[z, ny_idx, nx_idx]
                            neighbors_min = min(neighbors_min, val)
                            neighbors_max = max(neighbors_max, val)

                w_new[z, y, x] = max(neighbors_min, min(correction, neighbors_max))
