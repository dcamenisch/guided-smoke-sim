"""2D smoke simulator implementation."""

import numpy as np
from simulation.base_simulator import BaseSimulator
from physics.buoyancy import apply_buoyancy_force_2d
from physics.vorticity_confinement import apply_vorticity_confinement_2d
from core import MACGrid2D
from kernels import (
    solve_poisson_jacobi_2d,
    correct_velocity_kernel_2d,
    advect_density_kernel_2d,
    advect_u_velocity_kernel_2d,
    advect_v_velocity_kernel_2d,
    advect_density_maccormack_2d,
    advect_u_velocity_maccormack_2d,
    advect_v_velocity_maccormack_2d,
    compute_vorticity_kernel_2d,
)


class SmokeSimulator2D(BaseSimulator):
    """2D smoke simulator using MAC grid and semi-Lagrangian advection

    Matches the C++ reference implementation exactly.
    """

    def __init__(
        self,
        nx=128,
        ny=192,
        dt=0.07,
        tolerance=1e-5,
        max_iterations=1000,
        use_maccormack=True,
        vorticity_epsilon=0.0,
    ):
        """Initialize 2D smoke simulator

        Args:
            nx: Grid resolution in x-direction
            ny: Grid resolution in y-direction
            dt: Time step
            tolerance: Convergence tolerance for pressure solver
            max_iterations: Maximum iterations for pressure solver
            use_maccormack: Use MacCormack advection (True) or semi-Lagrangian (False)
            vorticity_epsilon: Vorticity confinement strength (0.0 = disabled, 0.1-0.5 typical)
        """
        super().__init__(dt=dt, tolerance=tolerance, max_iterations=max_iterations)

        self.nx, self.ny = nx, ny
        self.dx = 1.0 / nx
        self.use_maccormack = use_maccormack
        self.vorticity_epsilon = vorticity_epsilon

        # MAC grids for velocity and force
        self.velocity = MACGrid2D(nx, ny, self.dx)
        self.force = MACGrid2D(nx, ny, self.dx)

        # Scalar fields
        self.density = np.zeros((ny, nx), dtype=np.float32)
        self.pressure = np.zeros((ny, nx), dtype=np.float32)
        self.divergence = np.zeros((ny, nx), dtype=np.float32)
        self.vorticity = np.zeros((ny, nx), dtype=np.float32)

    def add_source(self):
        """Add smoke source - matches C++ applySource(0.45, 0.55, 0.1, 0.15)"""
        y_start, y_end = int(0.1 * self.ny), int(0.15 * self.ny) + 1
        x_start, x_end = int(0.45 * self.nx), int(0.55 * self.nx) + 1
        self.density[y_start:y_end, x_start:x_end] = 1.0

    def apply_forces(self):
        """Apply buoyancy force and update velocity

        Matches C++ addBuoyancyForce() + applyForce()
        """
        apply_buoyancy_force_2d(
            self.force,
            self.velocity,
            self.density,
            self.dt,
            self.nx,
            alpha=0.1,
        )

    def set_boundary_conditions(self):
        """Set boundary conditions - matches C++ setBoundaryConditions()"""
        # x-velocity (u) boundary conditions
        u = self.velocity.u_data

        # Left/right boundaries: du/dx = 0 (Neumann BC)
        u[:, 0] = u[:, 2]
        u[:, -1] = u[:, -3]

        # Top/bottom boundaries: u = 0 (Dirichlet BC)
        u[0, :] = 0
        u[-1, :] = 0

        # y-velocity (v) boundary conditions
        v = self.velocity.v_data

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

        # Apply vorticity confinement if enabled
        if self.vorticity_epsilon > 0.0:
            apply_vorticity_confinement_2d(
                self.force,
                self.velocity,
                self.vorticity,
                self.dx,
                self.dt,
                epsilon=self.vorticity_epsilon,
            )

    def advect_density(self):
        """Advect density using MacCormack or semi-Lagrangian method"""
        density_tmp = np.zeros_like(self.density)

        if self.use_maccormack:
            advect_density_maccormack_2d(
                self.density,
                density_tmp,
                self.velocity.u_data,
                self.velocity.v_data,
                self.dx,
                self.dt,
                self.ny,
                self.nx,
            )
        else:
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
        """Advect velocity using MacCormack or semi-Lagrangian method"""
        u = self.velocity.u_data
        v = self.velocity.v_data

        u_tmp = np.zeros_like(u)
        v_tmp = np.zeros_like(v)

        if self.use_maccormack:
            advect_u_velocity_maccormack_2d(
                u, u_tmp, v, self.dx, self.dt, self.ny, self.nx
            )
            advect_v_velocity_maccormack_2d(
                v, v_tmp, u, self.dx, self.dt, self.ny, self.nx
            )
        else:
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

    def export_to_npz(self, filepath, timestep=None):
        """Export 2D simulation state to NPZ format

        Args:
            filepath: Path to save the NPZ file
            timestep: Optional timestep number to include in metadata
        """
        data = {
            # Simulation parameters
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dt": self.dt,
            # Velocity field (on MAC grid faces)
            "u_velocity": self.velocity.u_data,
            "v_velocity": self.velocity.v_data,
            # Scalar fields (on cell centers)
            "density": self.density,
            "pressure": self.pressure,
            "divergence": self.divergence,
            "vorticity": self.vorticity,
        }

        if timestep is not None:
            data["timestep"] = timestep

        np.savez_compressed(filepath, **data)
        print(f"Exported 2D simulation state to {filepath}")
