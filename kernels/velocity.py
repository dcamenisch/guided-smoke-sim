"""Velocity correction kernels for pressure projection."""

from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def correct_velocity_kernel_2d(u, v, pressure, dx, dt, ny, nx):
    """Optimized 2D velocity correction with Numba

    Corrects velocity by subtracting pressure gradient:
    u_new = u - dt * ∇p

    Args:
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        pressure: Pressure field (ny, nx)
        dx: Grid spacing
        dt: Time step
        ny, nx: Grid dimensions
    """
    # Correct u (x-velocity)
    for y in prange(1, ny - 1):
        for x in range(1, nx):  # Goes to nx (not nx-1) for MAC grid
            grad_p_x = (pressure[y, x] - pressure[y, x - 1]) / dx
            u[y, x] -= dt * grad_p_x

    # Correct v (y-velocity)
    for y in prange(1, ny):  # Goes to ny (not ny-1) for MAC grid
        for x in range(1, nx - 1):
            grad_p_y = (pressure[y, x] - pressure[y - 1, x]) / dx
            v[y, x] -= dt * grad_p_y


@jit(nopython=True, parallel=True, cache=True)
def correct_velocity_kernel_3d(u, v, w, pressure, dx, dt, nz, ny, nx):
    """Optimized 3D velocity correction with Numba

    Corrects velocity by subtracting pressure gradient:
    u_new = u - dt * ∇p

    Args:
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        pressure: Pressure field (nz, ny, nx)
        dx: Grid spacing
        dt: Time step
        nz, ny, nx: Grid dimensions
    """
    # Correct u
    for z in prange(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                grad_p_x = (pressure[z, y, x] - pressure[z, y, x - 1]) / dx
                u[z, y, x] -= dt * grad_p_x

    # Correct v
    for z in prange(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                grad_p_y = (pressure[z, y, x] - pressure[z, y - 1, x]) / dx
                v[z, y, x] -= dt * grad_p_y

    # Correct w
    for z in prange(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                grad_p_z = (pressure[z, y, x] - pressure[z - 1, y, x]) / dx
                w[z, y, x] -= dt * grad_p_z
