"""
2D Smoke Simulation - Based on 3D version but with 2D MAC grid
This should match the C++ reference implementation exactly
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, prange


# ============= Numba-optimized kernels for 2D =============


@jit(nopython=True, parallel=True, cache=True)
def solve_poisson_jacobi_2d(
    pressure, divergence, dx, dt, rho, max_iter, tolerance, ny, nx
):
    """Optimized 2D Jacobi solver with Numba JIT compilation"""
    dx2 = dx * dx
    pressure_new = pressure.copy()

    for iteration in range(max_iter):
        # Jacobi update
        for y in prange(1, ny - 1):
            for x in range(1, nx - 1):
                b = -divergence[y, x] / dt * rho
                pressure_new[y, x] = (
                    dx2 * b
                    + pressure[y - 1, x]
                    + pressure[y + 1, x]
                    + pressure[y, x - 1]
                    + pressure[y, x + 1]
                ) / 4.0

        # Swap arrays
        pressure, pressure_new = pressure_new, pressure

        # Check convergence every 10 iterations to save time
        if iteration % 10 == 0:
            residual = 0.0
            for y in prange(1, ny - 1):
                for x in range(1, nx - 1):
                    b = -divergence[y, x] / dt * rho
                    cell_residual = (
                        b
                        - (
                            4 * pressure[y, x]
                            - pressure[y - 1, x]
                            - pressure[y + 1, x]
                            - pressure[y, x - 1]
                            - pressure[y, x + 1]
                        )
                        / dx2
                    )
                    residual += cell_residual * cell_residual

            residual = np.sqrt(residual) / ((nx - 2) * (ny - 2))
            if residual < tolerance:
                break

    return pressure


@jit(nopython=True, parallel=True, cache=True)
def correct_velocity_kernel_2d(u, v, pressure, dx, dt, ny, nx):
    """Optimized 2D velocity correction with Numba"""
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


@jit(nopython=True, cache=True)
def bilinear_interp(field, x, y):
    """Fast bilinear interpolation"""
    x_low = int(x)
    y_low = int(y)
    x_high = x_low + 1
    y_high = y_low + 1

    x_weight = x - x_low
    y_weight = y - y_low

    return (
        (1 - x_weight) * (1 - y_weight) * field[y_low, x_low]
        + x_weight * (1 - y_weight) * field[y_low, x_high]
        + (1 - x_weight) * y_weight * field[y_high, x_low]
        + x_weight * y_weight * field[y_high, x_high]
    )


@jit(nopython=True, parallel=True, cache=True)
def advect_density_kernel_2d(density, density_new, u, v, dx, dt, ny, nx):
    """Optimized 2D density advection with Numba"""
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
    """Optimized 2D u-velocity advection on MAC grid"""
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
    """Optimized 2D v-velocity advection on MAC grid"""
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


@jit(nopython=True, parallel=True, cache=True)
def compute_vorticity_kernel_2d(vorticity, u, v, dx, ny, nx):
    """Optimized 2D vorticity computation with Numba"""
    inv_2dx = 0.5 / dx

    for y in prange(2, ny - 2):
        for x in range(2, nx - 2):
            # ω = ∂v/∂x - ∂u/∂y (scalar in 2D)
            dvdx = (v[y, x + 1] - v[y, x - 1]) * inv_2dx
            dudy = (u[y + 1, x] - u[y - 1, x]) * inv_2dx
            vorticity[y, x] = dvdx - dudy


class MACGrid2D:
    """2D MAC grid for staggered velocity storage"""

    def __init__(self, nx, ny, dx):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        # u stored at x-faces (ny, nx+1)
        # v stored at y-faces (ny+1, nx)
        self.u_data = np.zeros((ny, nx + 1), dtype=np.float32)
        self.v_data = np.zeros((ny + 1, nx), dtype=np.float32)

    def reset(self):
        """Reset all velocity components to zero"""
        self.u_data.fill(0.0)
        self.v_data.fill(0.0)


class SmokeSimulator2D:
    def __init__(self, nx=128, ny=192):
        self.nx, self.ny = nx, ny
        self.dx = 1.0 / nx

        self.velocity = MACGrid2D(nx, ny, self.dx)
        self.force = MACGrid2D(nx, ny, self.dx)

        self.density = np.zeros((ny, nx), dtype=np.float32)
        self.pressure = np.zeros((ny, nx), dtype=np.float32)
        self.divergence = np.zeros((ny, nx), dtype=np.float32)
        self.vorticity = np.zeros((ny, nx), dtype=np.float32)

        self.dt = 0.07
        self.tolerance = 1e-5
        self.max_iterations = 1000

    def step(self):
        """Main simulation step - matches C++ FluidApp::step()"""
        # Add smoke source
        self.add_source()

        # Apply forces
        self.apply_forces()

        # Remove divergence (pressure projection)
        self.solve_pressure()

        # Advect everything
        self.advect()

        # Reset forces
        self.force.reset()

    def add_source(self):
        """Add smoke source - matches C++ applySource(0.45, 0.55, 0.1, 0.15)"""
        y_start, y_end = int(0.1 * self.ny), int(0.15 * self.ny) + 1
        x_start, x_end = int(0.45 * self.nx), int(0.55 * self.nx) + 1
        self.density[y_start:y_end, x_start:x_end] = 1.0

    def apply_forces(self):
        """Apply buoyancy force and update velocity - matches C++ addBuoyancyForce() + applyForce()"""
        scaling_factor = 64.0 / self.nx

        # Buoyancy force proportional to density
        alpha = 0.1
        buoyancy = alpha * self.density

        # Average adjacent cells to interior y-faces
        buoyancy_at_faces = 0.5 * (buoyancy[:-1, :] + buoyancy[1:, :])

        # Apply to interior v-faces (y-velocity)
        self.force.v_data[1:-1, :] += buoyancy_at_faces * scaling_factor

        # Update velocities with forces (applyForce)
        self.velocity.u_data += self.dt * self.force.u_data
        self.velocity.v_data += self.dt * self.force.v_data

    def set_boundary_conditions(self):
        """Set boundary conditions - matches C++ setBoundaryConditions()"""
        # x-velocity (u) boundary conditions
        u = self.velocity.u_data
        ny, nx_u = u.shape

        # Left/right boundaries: du/dx = 0 (Neumann BC)
        u[:, 0] = u[:, 2]
        u[:, -1] = u[:, -3]

        # Top/bottom boundaries: u = 0 (Dirichlet BC)
        u[0, :] = 0
        u[-1, :] = 0

        # y-velocity (v) boundary conditions
        v = self.velocity.v_data
        ny_v, nx = v.shape

        # Top/bottom boundaries: dv/dy = 0 (Neumann BC)
        v[0, :] = v[2, :]
        v[-1, :] = v[-3, :]

        # Left/right boundaries: v = 0 (Dirichlet BC)
        v[:, 0] = 0
        v[:, -1] = 0

        # Pressure boundary conditions
        p = self.pressure

        # All boundaries: dp/dn = 0 (Neumann BC)
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

    def compute_divergence(self):
        """Compute velocity divergence - matches C++ computeDivergence()"""
        u = self.velocity.u_data
        v = self.velocity.v_data

        # Compute partial derivatives for interior cells
        dudx = (u[:, 1:] - u[:, :-1]) / self.dx
        dvdy = (v[1:, :] - v[:-1, :]) / self.dx

        # Sum all partial derivatives - both have shape (ny, nx)
        self.divergence[:, :] = dudx + dvdy

    def solve_poisson(self):
        """Solve Poisson equation for pressure - matches C++ solvePoisson()"""
        rho = 1.0
        self.pressure = solve_poisson_jacobi_2d(
            self.pressure,
            self.divergence,
            self.dx,
            self.dt,
            rho,
            self.max_iterations,
            self.tolerance,
            self.ny,
            self.nx,
        )

    def correct_velocity(self):
        """Correct velocity with pressure gradient - matches C++ correctVelocity()"""
        correct_velocity_kernel_2d(
            self.velocity.u_data,
            self.velocity.v_data,
            self.pressure,
            self.dx,
            self.dt,
            self.ny,
            self.nx,
        )

    def compute_vorticity(self):
        """Compute vorticity field - matches C++ computeVorticity()"""
        compute_vorticity_kernel_2d(
            self.vorticity,
            self.velocity.u_data,
            self.velocity.v_data,
            self.dx,
            self.ny,
            self.nx,
        )

    def solve_pressure(self):
        """Full pressure solve step - matches C++ solvePressure()"""
        self.set_boundary_conditions()
        self.compute_divergence()
        self.solve_poisson()
        self.correct_velocity()
        self.compute_vorticity()
        self.compute_divergence()  # For debugging

    def advect(self):
        """Advect all quantities - matches C++ advectValues()"""
        self.advect_density()
        self.advect_velocity()

    def advect_density(self):
        """Advect density using semi-Lagrangian method - matches C++ advectDensitySL()"""
        density_tmp = np.zeros_like(self.density)
        advect_density_kernel_2d(
            self.density,
            density_tmp,
            self.velocity.u_data,
            self.velocity.v_data,
            self.dx,
            self.dt,
            self.ny,
            self.nx,
        )
        self.density = density_tmp

    def advect_velocity(self):
        """Advect velocity using semi-Lagrangian method - matches C++ advectVelocitySL()"""
        u = self.velocity.u_data
        v = self.velocity.v_data

        u_tmp = np.zeros_like(u)
        v_tmp = np.zeros_like(v)

        # Use optimized Numba kernels
        advect_u_velocity_kernel_2d(u, u_tmp, v, self.dx, self.dt, self.ny, self.nx)
        advect_v_velocity_kernel_2d(v, v_tmp, u, self.dx, self.dt, self.ny, self.nx)

        # Copy back
        self.velocity.u_data = u_tmp
        self.velocity.v_data = v_tmp

    def get_velocity_magnitude(self):
        """Get velocity magnitude at cell centers for visualization"""
        # Average u to cell centers
        u_center = 0.5 * (self.velocity.u_data[:, :-1] + self.velocity.u_data[:, 1:])
        # Average v to cell centers
        v_center = 0.5 * (self.velocity.v_data[:-1, :] + self.velocity.v_data[1:, :])

        return np.sqrt(u_center**2 + v_center**2)


# Visualization and testing
if __name__ == "__main__":
    print("Starting 2D Smoke Simulation...")
    print("This matches the C++ reference implementation")

    # Use same resolution as C++ default
    sim = SmokeSimulator2D(nx=128, ny=192)

    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    def animate(frame):
        """Animation function"""
        print(f"Frame {frame}: Running simulation step...")
        sim.step()

        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Density
        ax = axes[0, 0]
        im = ax.imshow(
            sim.density, origin="lower", cmap="hot", vmin=0, vmax=1, aspect="auto"
        )
        ax.set_title(f"Density (Frame {frame})")
        ax.axis("off")

        # Velocity magnitude
        ax = axes[0, 1]
        vel_mag = sim.get_velocity_magnitude()
        im = ax.imshow(vel_mag, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title("Velocity Magnitude")
        ax.axis("off")

        # Divergence
        ax = axes[1, 0]
        div_max = np.abs(sim.divergence).max()
        im = ax.imshow(
            sim.divergence,
            origin="lower",
            cmap="RdBu_r",
            vmin=-0.01,
            vmax=0.01,
            aspect="auto",
        )
        ax.set_title(f"Divergence (max={div_max:.2e})")
        ax.axis("off")

        # Vorticity
        ax = axes[1, 1]
        im = ax.imshow(sim.vorticity, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title("Vorticity")
        ax.axis("off")

        return []

    print("Starting animation...")
    anim = FuncAnimation(fig, animate, frames=200, interval=30)
    plt.tight_layout()
    plt.show()
