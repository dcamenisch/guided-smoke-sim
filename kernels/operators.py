"""Differential operators (curl, vorticity) for fluid simulation."""

from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def compute_vorticity_kernel_2d(vorticity, u, v, dx, ny, nx):
    """Optimized 2D vorticity computation with Numba

    Computes scalar vorticity: ω = ∂v/∂x - ∂u/∂y

    Args:
        vorticity: Output vorticity field (ny, nx)
        u: x-velocity component (ny, nx+1)
        v: y-velocity component (ny+1, nx)
        dx: Grid spacing
        ny, nx: Grid dimensions
    """
    inv_2dx = 0.5 / dx

    for y in prange(2, ny - 2):
        for x in range(2, nx - 2):
            # ω = ∂v/∂x - ∂u/∂y (scalar in 2D)
            dvdx = (v[y, x + 1] - v[y, x - 1]) * inv_2dx
            dudy = (u[y + 1, x] - u[y - 1, x]) * inv_2dx
            vorticity[y, x] = dvdx - dudy


@jit(nopython=True, parallel=True, cache=True)
def compute_vorticity_kernel_3d(vorticity, u, v, w, dx, nz, ny, nx):
    """Optimized 3D vorticity computation with Numba

    Computes vorticity vector: ω = ∇ × u

    Args:
        vorticity: Output vorticity field (nz, ny, nx, 3)
        u: x-velocity component (nz, ny, nx+1)
        v: y-velocity component (nz, ny+1, nx)
        w: z-velocity component (nz+1, ny, nx)
        dx: Grid spacing
        nz, ny, nx: Grid dimensions
    """
    inv_2dx = 0.5 / dx

    for z in prange(2, nz - 2):
        for y in range(2, ny - 2):
            for x in range(2, nx - 2):
                # ω_x = ∂w/∂y - ∂v/∂z
                dwdy = (w[z, y + 1, x] - w[z, y - 1, x]) * inv_2dx
                dvdz = (v[z + 1, y, x] - v[z - 1, y, x]) * inv_2dx
                vorticity[z, y, x, 0] = dwdy - dvdz

                # ω_y = ∂u/∂z - ∂w/∂x
                dudz = (u[z + 1, y, x] - u[z - 1, y, x]) * inv_2dx
                dwdx = (w[z, y, x + 1] - w[z, y, x - 1]) * inv_2dx
                vorticity[z, y, x, 1] = dudz - dwdx

                # ω_z = ∂v/∂x - ∂u/∂y
                dvdx = (v[z, y, x + 1] - v[z, y, x - 1]) * inv_2dx
                dudy = (u[z, y + 1, x] - u[z, y - 1, x]) * inv_2dx
                vorticity[z, y, x, 2] = dvdx - dudy
