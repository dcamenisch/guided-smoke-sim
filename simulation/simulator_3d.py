"""3D smoke simulator implementation."""

import numpy as np
from simulation.base_simulator import BaseSimulator
from core import MACGrid3D
from kernels import (
    solve_poisson_jacobi_3d,
    correct_velocity_kernel_3d,
    advect_density_kernel_3d,
    advect_u_velocity_kernel_3d,
    advect_v_velocity_kernel_3d,
    advect_w_velocity_kernel_3d,
    compute_vorticity_kernel_3d,
)


class SmokeSimulator3D(BaseSimulator):
    """3D smoke simulator using MAC grid and semi-Lagrangian advection

    Matches the C++ reference implementation exactly.
    """

    def __init__(
        self, nx=64, ny=96, nz=3, dt=0.07, tolerance=1e-5, max_iterations=1000
    ):
        """Initialize 3D smoke simulator

        Args:
            nx: Grid resolution in x-direction
            ny: Grid resolution in y-direction
            nz: Grid resolution in z-direction
            dt: Time step
            tolerance: Convergence tolerance for pressure solver
            max_iterations: Maximum iterations for pressure solver
        """
        super().__init__(dt=dt, tolerance=tolerance, max_iterations=max_iterations)

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = 1.0 / nx

        # MAC grids for velocity and force
        self.velocity = MACGrid3D(nx, ny, nz, self.dx)
        self.force = MACGrid3D(nx, ny, nz, self.dx)

        # Scalar fields
        self.density = np.zeros((nz, ny, nx), dtype=np.float32)
        self.pressure = np.zeros((nz, ny, nx), dtype=np.float32)
        self.divergence = np.zeros((nz, ny, nx), dtype=np.float32)
        self.vorticity = np.zeros((nz, ny, nx, 3), dtype=np.float32)

    def add_source(self):
        """Add smoke source - matches C++ applySource()"""
        z_start, z_end = int(0.475 * self.nz), int(0.525 * self.nz) + 1
        y_start, y_end = int(0.0 * self.ny), int(0.025 * self.ny) + 1
        x_start, x_end = int(0.475 * self.nx), int(0.525 * self.nx) + 1
        self.density[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    def apply_forces(self):
        """Apply buoyancy force and update velocity

        Matches C++ addBuoyancyForce() + applyForce()
        """
        scaling_factor = 64.0 / self.nx

        # Buoyancy force proportional to density
        alpha = 0.1
        buoyancy = alpha * self.density

        # Average adjacent cells to interior y-faces
        buoyancy_at_faces = 0.5 * (buoyancy[:, :-1, :] + buoyancy[:, 1:, :])

        # Apply to interior v-faces (y-velocity)
        self.force.v_data[:, 1:-1, :] += buoyancy_at_faces * scaling_factor

        # Update all velocity components
        self.velocity.u_data += self.dt * self.force.u_data
        self.velocity.v_data += self.dt * self.force.v_data
        self.velocity.w_data += self.dt * self.force.w_data

    def set_boundary_conditions(self):
        """Set boundary conditions - matches C++ setBoundaryConditions()"""
        # x-velocity (u) boundary conditions
        u = self.velocity.u_data

        # Left/right boundaries: du/dx = 0 (Neumann BC)
        u[:, :, 0] = u[:, :, 2]
        u[:, :, -1] = u[:, :, -3]

        # Top/bottom boundaries: u = 0 (Dirichlet BC)
        u[:, 0, :] = 0
        u[:, -1, :] = 0

        # Front/back boundaries: u = 0 (Dirichlet BC)
        u[0, :, :] = 0
        u[-1, :, :] = 0

        # y-velocity (v) boundary conditions
        v = self.velocity.v_data

        # Left/right boundaries: v = 0 (Dirichlet BC)
        v[:, :, 0] = 0
        v[:, :, -1] = 0

        # Top/bottom boundaries: dv/dy = 0 (Neumann BC)
        v[:, 0, :] = v[:, 2, :]
        v[:, -1, :] = v[:, -3, :]

        # Front/back boundaries: v = 0 (Dirichlet BC)
        v[0, :, :] = 0
        v[-1, :, :] = 0

        # z-velocity (w) boundary conditions
        w = self.velocity.w_data

        # Left/right boundaries: w = 0 (Dirichlet BC)
        w[:, :, 0] = 0
        w[:, :, -1] = 0

        # Top/bottom boundaries: w = 0 (Dirichlet BC)
        w[:, 0, :] = 0
        w[:, -1, :] = 0

        # Front/back boundaries: dw/dz = 0 (Neumann BC)
        w[0, :, :] = w[1, :, :]
        w[-1, :, :] = w[-2, :, :]

        # Pressure boundary conditions
        p = self.pressure

        # All boundaries: dp/dn = 0 (Neumann BC)
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]

    def compute_divergence(self):
        """Compute velocity divergence - matches C++ computeDivergence()"""
        u = self.velocity.u_data
        v = self.velocity.v_data
        w = self.velocity.w_data

        # Compute partial derivatives for interior cells
        dudx = (u[:, :, 1:] - u[:, :, :-1]) / self.dx
        dvdy = (v[:, 1:, :] - v[:, :-1, :]) / self.dx
        dwdz = (w[1:, :, :] - w[:-1, :, :]) / self.dx

        # Sum all partial derivatives - all have shape (nz, ny, nx)
        self.divergence[:, :, :] = dudx + dvdy + dwdz

    def solve_poisson(self):
        """Solve Poisson equation for pressure - matches C++ solvePoisson()"""
        rho = 1.0
        self.pressure = solve_poisson_jacobi_3d(
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
        """Correct velocity with pressure gradient - matches C++ correctVelocity()"""
        correct_velocity_kernel_3d(
            self.velocity.u_data,
            self.velocity.v_data,
            self.velocity.w_data,
            self.pressure,
            self.dx,
            self.dt,
            self.nz,
            self.ny,
            self.nx,
        )

    def compute_vorticity(self):
        """Compute vorticity field - matches C++ computeVorticity()"""
        compute_vorticity_kernel_3d(
            self.vorticity,
            self.velocity.u_data,
            self.velocity.v_data,
            self.velocity.w_data,
            self.dx,
            self.nz,
            self.ny,
            self.nx,
        )

    def advect_density(self):
        """Advect density using semi-Lagrangian method - matches C++ advectDensitySL()"""
        density_tmp = np.zeros_like(self.density)
        advect_density_kernel_3d(
            self.density,
            density_tmp,
            self.velocity.u_data,
            self.velocity.v_data,
            self.velocity.w_data,
            self.dx,
            self.dt,
            self.nz,
            self.ny,
            self.nx,
        )
        self.density = density_tmp

    def advect_velocity(self):
        """Advect velocity using semi-Lagrangian method - matches C++ advectVelocitySL()"""
        u = self.velocity.u_data
        v = self.velocity.v_data
        w = self.velocity.w_data

        u_tmp = np.zeros_like(u)
        v_tmp = np.zeros_like(v)
        w_tmp = np.zeros_like(w)

        # Use optimized Numba kernels
        advect_u_velocity_kernel_3d(
            u, u_tmp, v, w, self.dx, self.dt, self.nz, self.ny, self.nx
        )
        advect_v_velocity_kernel_3d(
            v, v_tmp, u, w, self.dx, self.dt, self.nz, self.ny, self.nx
        )
        advect_w_velocity_kernel_3d(
            w, w_tmp, u, v, self.dx, self.dt, self.nz, self.ny, self.nx
        )

        # Copy back
        self.velocity.u_data = u_tmp
        self.velocity.v_data = v_tmp
        self.velocity.w_data = w_tmp

    def export_to_npz(self, filepath, timestep=None):
        """Export 3D simulation state to NPZ format

        Args:
            filepath: Path to save the NPZ file
            timestep: Optional timestep number to include in metadata
        """
        data = {
            # Simulation parameters
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dx": self.dx,
            "dt": self.dt,
            # Velocity field (on MAC grid faces)
            "u_velocity": self.velocity.u_data,
            "v_velocity": self.velocity.v_data,
            "w_velocity": self.velocity.w_data,
            # Scalar fields (on cell centers)
            "density": self.density,
            "pressure": self.pressure,
            "divergence": self.divergence,
            "vorticity": self.vorticity,
        }

        if timestep is not None:
            data["timestep"] = timestep

        np.savez_compressed(filepath, **data)
        print(f"Exported 3D simulation state to {filepath}")
