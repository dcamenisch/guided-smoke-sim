"""Vorticity confinement force implementations.

Vorticity confinement restores small-scale turbulent details lost to numerical
dissipation by amplifying vorticity in the flow.
"""

import numpy as np
from numba import jit
from core import MACGrid2D, MACGrid3D


@jit(nopython=True)
def compute_vorticity_magnitude_gradient_2d(
    vorticity: np.ndarray,
    dx: float,
    ny: int,
    nx: int,
) -> tuple:
    """Compute gradient of vorticity magnitude for 2D

    Args:
        vorticity: Vorticity field (ny, nx) - scalar in 2D
        dx: Grid spacing
        ny, nx: Grid dimensions

    Returns:
        Tuple of (grad_x, grad_y) arrays
    """
    grad_x = np.zeros((ny, nx), dtype=np.float32)
    grad_y = np.zeros((ny, nx), dtype=np.float32)

    # Compute magnitude of vorticity (already scalar in 2D)
    omega_mag = np.abs(vorticity)

    # Central differences for gradient
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            grad_x[y, x] = (omega_mag[y, x + 1] - omega_mag[y, x - 1]) / (2.0 * dx)
            grad_y[y, x] = (omega_mag[y + 1, x] - omega_mag[y - 1, x]) / (2.0 * dx)

    return grad_x, grad_y


@jit(nopython=True)
def compute_vorticity_magnitude_gradient_3d(
    vorticity: np.ndarray,
    dx: float,
    nz: int,
    ny: int,
    nx: int,
) -> tuple:
    """Compute gradient of vorticity magnitude for 3D

    Args:
        vorticity: Vorticity field (nz, ny, nx, 3) - vector in 3D
        dx: Grid spacing
        nz, ny, nx: Grid dimensions

    Returns:
        Tuple of (grad_x, grad_y, grad_z) arrays
    """
    grad_x = np.zeros((nz, ny, nx), dtype=np.float32)
    grad_y = np.zeros((nz, ny, nx), dtype=np.float32)
    grad_z = np.zeros((nz, ny, nx), dtype=np.float32)

    # Compute magnitude of vorticity vector
    omega_mag = np.zeros((nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                omega_mag[z, y, x] = np.sqrt(
                    vorticity[z, y, x, 0] ** 2
                    + vorticity[z, y, x, 1] ** 2
                    + vorticity[z, y, x, 2] ** 2
                )

    # Central differences for gradient
    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                grad_x[z, y, x] = (omega_mag[z, y, x + 1] - omega_mag[z, y, x - 1]) / (
                    2.0 * dx
                )
                grad_y[z, y, x] = (omega_mag[z, y + 1, x] - omega_mag[z, y - 1, x]) / (
                    2.0 * dx
                )
                grad_z[z, y, x] = (omega_mag[z + 1, y, x] - omega_mag[z - 1, y, x]) / (
                    2.0 * dx
                )

    return grad_x, grad_y, grad_z


def apply_vorticity_confinement_2d(
    force: MACGrid2D,
    velocity: MACGrid2D,
    vorticity: np.ndarray,
    dx: float,
    dt: float,
    epsilon: float = 0.1,
):
    """Apply vorticity confinement force to 2D velocity field

    Vorticity confinement adds a force perpendicular to the gradient of
    vorticity magnitude, scaled by the vorticity itself. This restores
    rotational features that are lost due to numerical dissipation.

    Args:
        force: MACGrid2D to store forces
        velocity: MACGrid2D containing current velocities
        vorticity: Vorticity field (ny, nx) - scalar in 2D
        dx: Grid spacing
        dt: Time step
        epsilon: Confinement strength (default: 0.1)
    """
    ny, nx = vorticity.shape

    # Compute gradient of vorticity magnitude
    grad_x, grad_y = compute_vorticity_magnitude_gradient_2d(vorticity, dx, ny, nx)

    # Normalize gradient to get direction N
    # N points in the direction of increasing vorticity magnitude
    N_x = np.zeros_like(grad_x)
    N_y = np.zeros_like(grad_y)

    for y in range(ny):
        for x in range(nx):
            mag = np.sqrt(grad_x[y, x] ** 2 + grad_y[y, x] ** 2)
            if mag > 1e-10:
                N_x[y, x] = grad_x[y, x] / mag
                N_y[y, x] = grad_y[y, x] / mag

    # Vorticity confinement force: f = ε * h * (N × ω)
    # In 2D: ω is scalar (perpendicular to plane), so:
    # f_x = ε * dx * N_y * ω (perpendicular to N, in plane)
    # f_y = -ε * dx * N_x * ω

    # Reset forces
    force.u_data.fill(0)
    force.v_data.fill(0)

    # Apply force at cell centers, then average to MAC grid faces
    f_x_center = epsilon * dx * N_y * vorticity
    f_y_center = -epsilon * dx * N_x * vorticity

    # Average to u-faces (x-velocity locations)
    for y in range(1, ny - 1):
        for x in range(1, nx):
            force.u_data[y, x] = 0.5 * (f_x_center[y, x - 1] + f_x_center[y, x])

    # Average to v-faces (y-velocity locations)
    for y in range(1, ny):
        for x in range(1, nx - 1):
            force.v_data[y, x] = 0.5 * (f_y_center[y - 1, x] + f_y_center[y, x])

    # Update velocities
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data


def apply_vorticity_confinement_3d(
    force: MACGrid3D,
    velocity: MACGrid3D,
    vorticity: np.ndarray,
    dx: float,
    dt: float,
    epsilon: float = 0.1,
):
    """Apply vorticity confinement force to 3D velocity field

    Vorticity confinement adds a force perpendicular to the gradient of
    vorticity magnitude, scaled by the vorticity vector. This restores
    rotational features that are lost due to numerical dissipation.

    Formula: f = ε * h * (N × ω)
    where:
    - ε is the confinement parameter
    - h is the grid spacing (dx)
    - N is the normalized gradient of |ω|
    - ω is the vorticity vector

    Args:
        force: MACGrid3D to store forces
        velocity: MACGrid3D containing current velocities
        vorticity: Vorticity field (nz, ny, nx, 3) - vector in 3D
        dx: Grid spacing
        dt: Time step
        epsilon: Confinement strength (default: 0.1)
    """
    nz, ny, nx, _ = vorticity.shape

    # Compute gradient of vorticity magnitude
    grad_x, grad_y, grad_z = compute_vorticity_magnitude_gradient_3d(
        vorticity, dx, nz, ny, nx
    )

    # Normalize gradient to get direction N
    N_x = np.zeros((nz, ny, nx), dtype=np.float32)
    N_y = np.zeros((nz, ny, nx), dtype=np.float32)
    N_z = np.zeros((nz, ny, nx), dtype=np.float32)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                mag = np.sqrt(
                    grad_x[z, y, x] ** 2 + grad_y[z, y, x] ** 2 + grad_z[z, y, x] ** 2
                )
                if mag > 1e-10:
                    N_x[z, y, x] = grad_x[z, y, x] / mag
                    N_y[z, y, x] = grad_y[z, y, x] / mag
                    N_z[z, y, x] = grad_z[z, y, x] / mag

    # Vorticity confinement force: f = ε * h * (N × ω)
    # Cross product: N × ω
    f_x_center = np.zeros((nz, ny, nx), dtype=np.float32)
    f_y_center = np.zeros((nz, ny, nx), dtype=np.float32)
    f_z_center = np.zeros((nz, ny, nx), dtype=np.float32)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                # Cross product components
                f_x_center[z, y, x] = (
                    epsilon
                    * dx
                    * (
                        N_y[z, y, x] * vorticity[z, y, x, 2]
                        - N_z[z, y, x] * vorticity[z, y, x, 1]
                    )
                )
                f_y_center[z, y, x] = (
                    epsilon
                    * dx
                    * (
                        N_z[z, y, x] * vorticity[z, y, x, 0]
                        - N_x[z, y, x] * vorticity[z, y, x, 2]
                    )
                )
                f_z_center[z, y, x] = (
                    epsilon
                    * dx
                    * (
                        N_x[z, y, x] * vorticity[z, y, x, 1]
                        - N_y[z, y, x] * vorticity[z, y, x, 0]
                    )
                )

    # Reset forces
    force.u_data.fill(0)
    force.v_data.fill(0)
    force.w_data.fill(0)

    # Average to u-faces (x-velocity locations)
    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx):
                force.u_data[z, y, x] = 0.5 * (
                    f_x_center[z, y, x - 1] + f_x_center[z, y, x]
                )

    # Average to v-faces (y-velocity locations)
    for z in range(1, nz - 1):
        for y in range(1, ny):
            for x in range(1, nx - 1):
                force.v_data[z, y, x] = 0.5 * (
                    f_y_center[z, y - 1, x] + f_y_center[z, y, x]
                )

    # Average to w-faces (z-velocity locations)
    for z in range(1, nz):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                force.w_data[z, y, x] = 0.5 * (
                    f_z_center[z - 1, y, x] + f_z_center[z, y, x]
                )

    # Update velocities
    velocity.u_data += dt * force.u_data
    velocity.v_data += dt * force.v_data
    velocity.w_data += dt * force.w_data
