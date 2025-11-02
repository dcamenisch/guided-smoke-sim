"""Advection kernels using semi-Lagrangian method."""

from numba import jit, prange
from kernels.interpolation import bilinear_interp, trilinear_interp


# ============= 2D Advection Kernels =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel_2d(density, density_new, u, v, dx, dt, ny, nx):
    """Optimized 2D density advection with Numba

    Uses semi-Lagrangian advection: trace particle backwards and interpolate

    Args:
        density: Current density field (ny, nx)
        density_new: Output density field (ny, nx)
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Interpolate velocity to cell center
            last_x_velocity = (u[y, x] + u[y, x + 1]) * 0.5
            last_y_velocity = (v[y, x] + v[y + 1, x]) * 0.5

            # Trace backwards
            last_x = x - dt / dx * last_x_velocity
            last_y = y - dt / dx * last_y_velocity

            # Clamp to valid density region
            last_x = max(1.0, min(last_x, nx - 2.0))
            last_y = max(1.0, min(last_y, ny - 2.0))

            # Bilinear interpolation
            density_new[y, x] = bilinear_interp(density, last_x, last_y)


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_kernel_2d(u, u_new, v, dx, dt, ny, nx):
    """Optimized 2D u-velocity advection on MAC grid

    Args:
        u: Current x-velocity (ny, nx+1)
        u_new: Output x-velocity (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    for y in prange(1, ny - 1):
        for x in range(1, nx):  # u goes to nx
            # u is at x-face, interpolate v to this location
            last_x_velocity = u[y, x]

            # Average 4 v values around u-face
            last_y_velocity = (
                v[y, x] + v[y, x - 1] + v[y + 1, x - 1] + v[y + 1, x]
            ) * 0.25

            # Trace backwards
            last_x = x - dt / dx * last_x_velocity
            last_y = y - dt / dx * last_y_velocity

            # Clamp to MAC grid boundaries for u
            last_x = max(1.5, min(last_x, nx - 1.5))
            last_y = max(1.5, min(last_y, ny - 2.5))

            # Bilinear interpolation
            u_new[y, x] = bilinear_interp(u, last_x, last_y)


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_kernel_2d(v, v_new, u, dx, dt, ny, nx):
    """Optimized 2D v-velocity advection on MAC grid

    Args:
        v: Current y-velocity (ny+1, nx)
        v_new: Output y-velocity (ny+1, nx)
        u: x-velocity component (ny, nx+1)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    for y in prange(1, ny):  # v goes to ny
        for x in range(1, nx - 1):
            # v is at y-face, interpolate u to this location
            last_y_velocity = v[y, x]

            # Average 4 u values around v-face
            last_x_velocity = (
                u[y, x] + u[y, x + 1] + u[y - 1, x + 1] + u[y - 1, x]
            ) * 0.25

            # Trace backwards
            last_x = x - dt / dx * last_x_velocity
            last_y = y - dt / dx * last_y_velocity

            # Clamp to MAC grid boundaries for v
            last_x = max(1.5, min(last_x, nx - 2.5))
            last_y = max(1.5, min(last_y, ny - 1.5))

            # Bilinear interpolation
            v_new[y, x] = bilinear_interp(v, last_x, last_y)


# ============= 3D Advection Kernels =============


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel_3d(density, density_new, u, v, w, dx, dt, nz, ny, nx):
    """Optimized 3D density advection with Numba

    Uses semi-Lagrangian advection: trace particle backwards and interpolate

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
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                # Interpolate velocity to cell center
                last_x_velocity = (u[z, y, x] + u[z, y, x + 1]) * 0.5
                last_y_velocity = (v[z, y, x] + v[z, y + 1, x]) * 0.5
                last_z_velocity = (w[z, y, x] + w[z + 1, y, x]) * 0.5

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_z = z - dt / dx * last_z_velocity

                # Clamp
                last_x = max(1.0, min(last_x, nx - 2.0))
                last_y = max(1.0, min(last_y, ny - 2.0))
                last_z = max(1.0, min(last_z, nz - 2.0))

                # Trilinear interpolation
                density_new[z, y, x] = trilinear_interp(density, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_u_velocity_kernel_3d(u, u_new, v, w, dx, dt, nz, ny, nx):
    """Optimized 3D u-velocity advection on MAC grid

    Args:
        u: Current x-velocity (nz, ny, nx+1)
        u_new: Output x-velocity (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):  # u goes to nx
                # u is at x-face, interpolate v and w to this location
                last_x_velocity = u[z, y, x]

                # Average 4 v values around u-face
                last_y_velocity = (
                    v[z, y, x - 1] + v[z, y + 1, x - 1] + v[z, y, x] + v[z, y + 1, x]
                ) * 0.25

                # Average 4 w values around u-face
                last_z_velocity = (
                    w[z, y, x - 1] + w[z + 1, y, x - 1] + w[z, y, x] + w[z + 1, y, x]
                ) * 0.25

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_z = z - dt / dx * last_z_velocity

                # Clamp to MAC grid boundaries for u
                last_x = max(1.5, min(last_x, nx - 1.5))
                last_y = max(1.5, min(last_y, ny - 2.5))
                last_z = max(1.5, min(last_z, nz - 2.5))

                # Trilinear interpolation
                u_new[z, y, x] = trilinear_interp(u, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_v_velocity_kernel_3d(v, v_new, u, w, dx, dt, nz, ny, nx):
    """Optimized 3D v-velocity advection on MAC grid

    Args:
        v: Current y-velocity (nz, ny+1, nx)
        v_new: Output y-velocity (nz, ny+1, nx)
        u: x-velocity component (nz, ny, nx+1)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    for z in prange(1, nz - 1):
        for y in range(1, ny):  # v goes to ny
            for x in range(1, nx - 1):
                # v is at y-face, interpolate u and w to this location
                last_y_velocity = v[z, y, x]

                # Average 4 u values around v-face
                last_x_velocity = (
                    u[z, y - 1, x] + u[z, y - 1, x + 1] + u[z, y, x] + u[z, y, x + 1]
                ) * 0.25

                # Average 4 w values around v-face
                last_z_velocity = (
                    w[z, y - 1, x] + w[z + 1, y - 1, x] + w[z, y, x] + w[z + 1, y, x]
                ) * 0.25

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_z = z - dt / dx * last_z_velocity

                # Clamp to MAC grid boundaries for v
                last_x = max(1.5, min(last_x, nx - 2.5))
                last_y = max(1.5, min(last_y, ny - 1.5))
                last_z = max(1.5, min(last_z, nz - 2.5))

                # Trilinear interpolation
                v_new[z, y, x] = trilinear_interp(v, last_x, last_y, last_z)


@jit(nopython=True, parallel=True, cache=True)
def advect_w_velocity_kernel_3d(w, w_new, u, v, dx, dt, nz, ny, nx):
    """Optimized 3D w-velocity advection on MAC grid

    Args:
        w: Current z-velocity (nz+1, ny, nx)
        w_new: Output z-velocity (nz+1, ny, nx)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    for z in prange(1, nz):  # w goes to nz
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                # w is at z-face, interpolate u and v to this location
                last_z_velocity = w[z, y, x]

                # Average 4 u values around w-face
                last_x_velocity = (
                    u[z - 1, y, x] + u[z - 1, y, x + 1] + u[z, y, x] + u[z, y, x + 1]
                ) * 0.25

                # Average 4 v values around w-face
                last_y_velocity = (
                    v[z - 1, y, x] + v[z - 1, y + 1, x] + v[z, y, x] + v[z, y + 1, x]
                ) * 0.25

                # Trace backwards
                last_x = x - dt / dx * last_x_velocity
                last_y = y - dt / dx * last_y_velocity
                last_z = z - dt / dx * last_z_velocity

                # Clamp to MAC grid boundaries for w
                last_x = max(1.5, min(last_x, nx - 2.5))
                last_y = max(1.5, min(last_y, ny - 2.5))
                last_z = max(1.5, min(last_z, nz - 1.5))

                # Trilinear interpolation
                w_new[z, y, x] = trilinear_interp(w, last_x, last_y, last_z)
