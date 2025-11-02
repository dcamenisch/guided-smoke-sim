import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, prange

from macgrid import MACGrid3


# ============= Numba-optimized kernels =============


@jit(nopython=True, parallel=True, cache=True)
def solve_poisson_jacobi(
    pressure, divergence, dx, dt, rho, max_iter, tolerance, nz, ny, nx
):
    """Optimized Jacobi solver with Numba JIT compilation"""
    dx2 = dx * dx
    pressure_new = pressure.copy()

    for iteration in range(max_iter):
        # Jacobi update
        for z in prange(1, nz - 1):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    b = -divergence[z, y, x] / dt * rho
                    pressure_new[z, y, x] = (
                        dx2 * b
                        + pressure[z - 1, y, x]
                        + pressure[z + 1, y, x]
                        + pressure[z, y - 1, x]
                        + pressure[z, y + 1, x]
                        + pressure[z, y, x - 1]
                        + pressure[z, y, x + 1]
                    ) / 6.0

        # Swap arrays
        pressure, pressure_new = pressure_new, pressure

        # Check convergence every 10 iterations to save time
        if iteration % 10 == 0:
            residual = 0.0
            for z in prange(1, nz - 1):
                for y in range(1, ny - 1):
                    for x in range(1, nx - 1):
                        b = -divergence[z, y, x] / dt * rho
                        cell_residual = (
                            b
                            - (
                                6 * pressure[z, y, x]
                                - pressure[z - 1, y, x]
                                - pressure[z + 1, y, x]
                                - pressure[z, y - 1, x]
                                - pressure[z, y + 1, x]
                                - pressure[z, y, x - 1]
                                - pressure[z, y, x + 1]
                            )
                            / dx2
                        )
                        residual += cell_residual * cell_residual

            residual = np.sqrt(residual) / ((nx - 2) * (ny - 2) * (nz - 2))
            if residual < tolerance:
                break

    return pressure


@jit(nopython=True, parallel=True, cache=True)
def correct_velocity_kernel(u, v, w, pressure, dx, dt, nz, ny, nx):
    """Optimized velocity correction with Numba"""
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


@jit(nopython=True, cache=True)
def trilinear_interp(field, x, y, z):
    """Fast trilinear interpolation"""
    x_low = int(x)
    y_low = int(y)
    z_low = int(z)
    x_high = x_low + 1
    y_high = y_low + 1
    z_high = z_low + 1

    x_weight = x - x_low
    y_weight = y - y_low
    z_weight = z - z_low

    return (
        (1 - x_weight) * (1 - y_weight) * (1 - z_weight) * field[z_low, y_low, x_low]
        + x_weight * (1 - y_weight) * (1 - z_weight) * field[z_low, y_low, x_high]
        + (1 - x_weight) * y_weight * (1 - z_weight) * field[z_low, y_high, x_low]
        + x_weight * y_weight * (1 - z_weight) * field[z_low, y_high, x_high]
        + (1 - x_weight) * (1 - y_weight) * z_weight * field[z_high, y_low, x_low]
        + x_weight * (1 - y_weight) * z_weight * field[z_high, y_low, x_high]
        + (1 - x_weight) * y_weight * z_weight * field[z_high, y_high, x_low]
        + x_weight * y_weight * z_weight * field[z_high, y_high, x_high]
    )


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel(density, density_new, u, v, w, dx, dt, nz, ny, nx):
    """Optimized density advection with Numba"""
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
def advect_u_velocity_kernel(u, u_new, v, w, dx, dt, nz, ny, nx):
    """Optimized u-velocity advection on MAC grid"""
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
def advect_v_velocity_kernel(v, v_new, u, w, dx, dt, nz, ny, nx):
    """Optimized v-velocity advection on MAC grid"""
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
def advect_w_velocity_kernel(w, w_new, u, v, dx, dt, nz, ny, nx):
    """Optimized w-velocity advection on MAC grid"""
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


@jit(nopython=True, parallel=True, cache=True)
def compute_vorticity_kernel(vorticity, u, v, w, dx, nz, ny, nx):
    """Optimized vorticity computation with Numba"""
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


class SmokeSimulator:
    def __init__(self, nx=64, ny=96, nz=3):
        self.nx, self.ny, self.nz = nx, ny, nz

        self.dx = 1.0 / nx

        self.velocity = MACGrid3(nx, ny, nz, self.dx)
        self.force = MACGrid3(nx, ny, nz, self.dx)

        self.density = np.zeros((nz, ny, nx), dtype=np.float32)
        self.pressure = np.zeros((nz, ny, nx), dtype=np.float32)
        self.divergence = np.zeros((nz, ny, nx), dtype=np.float32)
        self.vorticity = np.zeros((nz, ny, nx, 3), dtype=np.float32)

        self.dt = 0.07
        self.tolerance = 1e-5
        self.max_iterations = 1000

    def step(self):
        # Add smoke source
        self.add_source()

        # Apply forces
        self.apply_forces()

        # Remove divergence
        self.solve_pressure()

        # Advect everything
        self.advect()

        # Reset forces
        self.force.reset()

    def add_source(self):
        # Vectorized source addition
        z_start, z_end = int(0.4 * self.nz), int(0.6 * self.nz) + 1
        y_start, y_end = int(0.1 * self.ny), int(0.15 * self.ny) + 1
        x_start, x_end = int(0.4 * self.nx), int(0.6 * self.nx) + 1
        self.density[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    def apply_forces(self):
        scaling_factor = 64.0 / self.nx

        alpha = 0.1
        buoyancy = alpha * self.density

        # Average adjacent cells to interior faces
        buoyancy_at_faces = 0.5 * (buoyancy[:, :-1, :] + buoyancy[:, 1:, :])

        # Apply to interior v-faces
        self.force.y_data[:, 1:-1, :] += buoyancy_at_faces * scaling_factor

        # Update all velocity components
        self.velocity.x_data += self.dt * self.force.x_data
        self.velocity.y_data += self.dt * self.force.y_data
        self.velocity.z_data += self.dt * self.force.z_data

    def set_boundry_conditions(self):
        # Set x components of velocity at boundaries
        self.velocity.x_data[:, :, 0] = self.velocity.x_data[:, :, 2]
        self.velocity.x_data[:, :, -1] = self.velocity.x_data[:, :, -3]

        self.velocity.x_data[:, 0, :] = 0
        self.velocity.x_data[:, -1, :] = 0

        self.velocity.x_data[0, :, :] = 0
        self.velocity.x_data[-1, :, :] = 0

        # Set y components of velocity at boundaries
        self.velocity.y_data[:, :, 0] = 0
        self.velocity.y_data[:, :, -1] = 0

        self.velocity.y_data[:, 0, :] = self.velocity.y_data[:, 2, :]
        self.velocity.y_data[:, -1, :] = self.velocity.y_data[:, -3, :]

        self.velocity.y_data[0, :, :] = 0
        self.velocity.y_data[-1, :, :] = 0

        # Set z components of velocity at boundaries
        self.velocity.z_data[:, :, 0] = 0
        self.velocity.z_data[:, :, -1] = 0

        self.velocity.z_data[:, 0, :] = 0
        self.velocity.z_data[:, -1, :] = 0

        self.velocity.z_data[0, :, :] = self.velocity.z_data[1, :, :]
        self.velocity.z_data[-1, :, :] = self.velocity.z_data[-2, :, :]

        # Additional boundary conditions for pressure
        self.pressure[:, :, 0] = self.pressure[:, :, 1]
        self.pressure[:, :, -1] = self.pressure[:, :, -2]
        self.pressure[:, 0, :] = self.pressure[:, 1, :]
        self.pressure[:, -1, :] = self.pressure[:, -2, :]
        self.pressure[0, :, :] = self.pressure[1, :, :]
        self.pressure[-1, :, :] = self.pressure[-2, :, :]

    def compute_divergence(self):
        # Get velocity components from MAC grid
        u = self.velocity.x_data
        v = self.velocity.y_data
        w = self.velocity.z_data

        # Compute partial derivatives for interior cells
        dudx = (u[:, :, 1:] - u[:, :, :-1]) / self.dx
        dvdy = (v[:, 1:, :] - v[:, :-1, :]) / self.dx
        dwdz = (w[1:, :, :] - w[:-1, :, :]) / self.dx

        # Sum all partial derivatives - all have shape (nz, ny, nx)
        self.divergence[:, :, :] = dudx + dvdy + dwdz

    def solve_poisson(self):
        rho = 1.0
        self.pressure = solve_poisson_jacobi(
            self.pressure,
            self.divergence,
            self.dx,
            self.dt,
            rho,
            self.max_iterations,
            self.tolerance,
            self.nz,
            self.ny,
            self.nx,
        )

    def correct_velocity(self):
        correct_velocity_kernel(
            self.velocity.x_data,
            self.velocity.y_data,
            self.velocity.z_data,
            self.pressure,
            self.dx,
            self.dt,
            self.nz,
            self.ny,
            self.nx,
        )

    def solve_pressure(self):
        self.set_boundry_conditions()
        self.compute_divergence()
        self.solve_poisson()
        self.correct_velocity()
        self.compute_vorticity()
        self.compute_divergence()

    def advect(self):
        self.advect_density()
        self.advect_velocity()

    def advect_density(self):
        density_tmp = np.zeros_like(self.density)
        advect_density_kernel(
            self.density,
            density_tmp,
            self.velocity.x_data,
            self.velocity.y_data,
            self.velocity.z_data,
            self.dx,
            self.dt,
            self.nz,
            self.ny,
            self.nx,
        )
        self.density = density_tmp

    def advect_velocity(self):
        """Advect velocity components using semi-Lagrangian method (optimized)"""
        u = self.velocity.x_data
        v = self.velocity.y_data
        w = self.velocity.z_data

        u_tmp = np.zeros_like(u)
        v_tmp = np.zeros_like(v)
        w_tmp = np.zeros_like(w)

        # Use optimized Numba kernels
        advect_u_velocity_kernel(
            u, u_tmp, v, w, self.dx, self.dt, self.nz, self.ny, self.nx
        )
        advect_v_velocity_kernel(
            v, v_tmp, u, w, self.dx, self.dt, self.nz, self.ny, self.nx
        )
        advect_w_velocity_kernel(
            w, w_tmp, u, v, self.dx, self.dt, self.nz, self.ny, self.nx
        )

        # Copy back
        self.velocity.x_data = u_tmp
        self.velocity.y_data = v_tmp
        self.velocity.z_data = w_tmp

    def compute_vorticity(self):
        compute_vorticity_kernel(
            self.vorticity,
            self.velocity.x_data,
            self.velocity.y_data,
            self.velocity.z_data,
            self.dx,
            self.nz,
            self.ny,
            self.nx,
        )

    def render_slice(self, ax, slice_type="mid_z"):
        """Render a 2D slice of the 3D volume"""
        ax.clear()

        if slice_type == "mid_z":
            # Show xy slice at middle z
            z_slice = self.nz // 2
            data = self.density[z_slice, :, :]
            title = f"XY Slice (z={z_slice})"
        elif slice_type == "mid_y":
            # Show xz slice at middle y
            y_slice = self.ny // 2
            data = self.density[:, y_slice, :]
            title = f"XZ Slice (y={y_slice})"
        else:
            # Show yz slice at middle x
            x_slice = self.nx // 2
            data = self.density[:, :, x_slice]
            title = f"YZ Slice (x={x_slice})"

        ax.imshow(
            data, cmap="hot", origin="lower", vmin=0, vmax=1, interpolation="none"
        )
        ax.set_title(f"{title}")
        ax.axis("off")

    def render_volume_projection(self, ax):
        """Render maximum intensity projection"""
        ax.clear()

        # Maximum intensity projection along z-axis
        projection = np.max(self.density, axis=0)

        ax.imshow(
            projection,
            cmap="hot",
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="none",
        )
        ax.set_title(f"Max Projection (along Z)")
        ax.axis("off")


# Run 3D simulation
if __name__ == "__main__":
    sim = SmokeSimulator(nx=48, ny=72, nz=3)

    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Calculate substeps to maintain similar physical time per frame
    # Goal: Simulate more time per frame for faster visual progression
    target_time_per_frame = 0.3  # Increased from 0.1 for faster visual changes

    def animate(frame):
        # Run multiple substeps per frame for stability and correct timing
        print(f"Frame {frame}: Advancing simulation...")
        sim.step()

        # Render different views
        sim.render_slice(axes[0, 0], "mid_z")
        sim.render_slice(axes[0, 1], "mid_y")
        sim.render_slice(axes[1, 0], "mid_x")
        sim.render_volume_projection(axes[1, 1])

        return []

    print("Starting animation...")
    anim = FuncAnimation(fig, animate, frames=200, interval=30)  # Faster frame rate
    plt.tight_layout()
    plt.show()
