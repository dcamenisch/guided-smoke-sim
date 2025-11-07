"""Unified smoke simulator supporting both 2D and 3D simulations."""

import numpy as np
from simulation.base_simulator import BaseSimulator
from physics.buoyancy import apply_buoyancy_force_2d, apply_buoyancy_force_3d
from physics.vorticity_confinement import (
    apply_vorticity_confinement_2d,
    apply_vorticity_confinement_3d,
)
from core import MACGrid2D, MACGrid3D
from kernels.poisson import (
    solve_poisson_rb_gauss_seidel_2d,
    solve_poisson_rb_gauss_seidel_3d,
)
from kernels.velocity import correct_velocity_kernel_2d, correct_velocity_kernel_3d
from kernels.advection import (
    advect_density_kernel_2d,
    advect_u_velocity_kernel_2d,
    advect_v_velocity_kernel_2d,
    advect_density_maccormack_2d,
    advect_u_velocity_maccormack_2d,
    advect_v_velocity_maccormack_2d,
    advect_density_kernel_3d,
    advect_u_velocity_kernel_3d,
    advect_v_velocity_kernel_3d,
    advect_w_velocity_kernel_3d,
    advect_density_maccormack_3d,
    advect_u_velocity_maccormack_3d,
    advect_v_velocity_maccormack_3d,
    advect_w_velocity_maccormack_3d,
)
from kernels.differential import (
    compute_vorticity_kernel_2d,
    compute_vorticity_kernel_3d,
)


class SmokeSimulator(BaseSimulator):
    """Unified smoke simulator supporting both 2D and 3D

    Matches the C++ reference implementation exactly.
    Automatically detects dimensionality based on nz parameter:
    - nz=None → 2D simulation
    - nz=int → 3D simulation
    """

    def __init__(
        self,
        nx=128,
        ny=192,
        nz=None,
        dt=0.07,
        tolerance=1e-5,
        max_iterations=1000,
        use_maccormack=True,
        vorticity_epsilon=0.0,
        cfl_target=1.0,
        dt_min=0.001,
        dt_max=0.1,
    ):
        """Initialize smoke simulator

        Args:
            nx: Grid resolution in x-direction
            ny: Grid resolution in y-direction
            nz: Grid resolution in z-direction (None for 2D, int for 3D)
            dt: Initial time step (used as dt_max if not specified)
            tolerance: Convergence tolerance for pressure solver
            max_iterations: Maximum iterations for pressure solver
            use_maccormack: Use MacCormack advection (True) or semi-Lagrangian (False)
            vorticity_epsilon: Vorticity confinement strength (0.0 = disabled, 0.1-0.5 typical)
            cfl_target: Target CFL number (typically 1.0-5.0)
            dt_min: Minimum allowed time step
            dt_max: Maximum allowed time step (defaults to dt parameter)
        """
        super().__init__(
            dt=dt,
            tolerance=tolerance,
            max_iterations=max_iterations,
            cfl_target=cfl_target,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        # Determine dimensionality
        self.ndim = 2 if nz is None else 3
        self.nx, self.ny = nx, ny
        self.nz = nz if nz is not None else 1  # For internal consistency
        self.dx = 1.0 / nx
        self.use_maccormack = use_maccormack
        self.vorticity_epsilon = vorticity_epsilon

        # Create appropriate MAC grids based on dimensionality
        if self.ndim == 2:
            self.velocity = MACGrid2D(nx, ny, self.dx)
            self.force = MACGrid2D(nx, ny, self.dx)
            # Scalar fields for 2D
            self.density = np.zeros((ny, nx), dtype=np.float32)
            self.pressure = np.zeros((ny, nx), dtype=np.float32)
            self.divergence = np.zeros((ny, nx), dtype=np.float32)
            self.vorticity = np.zeros((ny, nx), dtype=np.float32)
        else:
            self.velocity = MACGrid3D(nx, ny, nz, self.dx)
            self.force = MACGrid3D(nx, ny, nz, self.dx)
            # Scalar fields for 3D
            self.density = np.zeros((nz, ny, nx), dtype=np.float32)
            self.pressure = np.zeros((nz, ny, nx), dtype=np.float32)
            self.divergence = np.zeros((nz, ny, nx), dtype=np.float32)
            self.vorticity = np.zeros((nz, ny, nx, 3), dtype=np.float32)

    def add_source(self):
        """Add smoke source - dimension-specific locations"""
        if self.ndim == 2:
            # 2D: matches C++ applySource(0.45, 0.55, 0.1, 0.15)
            y_start, y_end = int(0.1 * self.ny), int(0.15 * self.ny) + 1
            x_start, x_end = int(0.45 * self.nx), int(0.55 * self.nx) + 1
            self.density[y_start:y_end, x_start:x_end] = 1.0
        else:
            # 3D: matches C++ applySource()
            z_start, z_end = int(0.475 * self.nz), int(0.525 * self.nz) + 1
            y_start, y_end = int(0.0 * self.ny), int(0.025 * self.ny) + 1
            x_start, x_end = int(0.475 * self.nx), int(0.525 * self.nx) + 1
            self.density[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

    def apply_forces(self):
        """Apply buoyancy force and update velocity

        Matches C++ addBuoyancyForce() + applyForce()
        """
        if self.ndim == 2:
            apply_buoyancy_force_2d(
                self.force,
                self.velocity,
                self.density,
                self.dt,
                self.nx,
                alpha=0.1,
            )
        else:
            apply_buoyancy_force_3d(
                self.force,
                self.velocity,
                self.density,
                self.dt,
                self.nx,
                alpha=0.1,
            )

    def set_boundary_conditions(self):
        """Set boundary conditions - matches C++ setBoundaryConditions()"""
        if self.ndim == 2:
            self._set_boundary_conditions_2d()
        else:
            self._set_boundary_conditions_3d()

    def _set_boundary_conditions_2d(self):
        """Set 2D boundary conditions"""
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

    def _set_boundary_conditions_3d(self):
        """Set 3D boundary conditions"""
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

        if self.ndim == 2:
            # Compute partial derivatives for interior cells
            dudx = (u[:, 1:] - u[:, :-1]) / self.dx
            dvdy = (v[1:, :] - v[:-1, :]) / self.dx

            # Sum all partial derivatives - both have shape (ny, nx)
            self.divergence[:, :] = dudx + dvdy
        else:
            w = self.velocity.w_data

            # Compute partial derivatives for interior cells
            dudx = (u[:, :, 1:] - u[:, :, :-1]) / self.dx
            dvdy = (v[:, 1:, :] - v[:, :-1, :]) / self.dx
            dwdz = (w[1:, :, :] - w[:-1, :, :]) / self.dx

            # Sum all partial derivatives - all have shape (nz, ny, nx)
            self.divergence[:, :, :] = dudx + dvdy + dwdz

    def solve_poisson(self):
        """Solve Poisson equation for pressure using Red-Black Gauss-Seidel"""
        rho = 1.0
        if self.ndim == 2:
            self.pressure = solve_poisson_rb_gauss_seidel_2d(
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
        else:
            self.pressure = solve_poisson_rb_gauss_seidel_3d(
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
        if self.ndim == 2:
            correct_velocity_kernel_2d(
                self.velocity.u_data,
                self.velocity.v_data,
                self.pressure,
                self.dx,
                self.dt,
                self.ny,
                self.nx,
            )
        else:
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
        if self.ndim == 2:
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
        else:
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

            # Apply vorticity confinement if enabled
            if self.vorticity_epsilon > 0.0:
                apply_vorticity_confinement_3d(
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

        if self.ndim == 2:
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
        else:
            if self.use_maccormack:
                advect_density_maccormack_3d(
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
            else:
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
        """Advect velocity using MacCormack or semi-Lagrangian method"""
        u = self.velocity.u_data
        v = self.velocity.v_data

        u_tmp = np.zeros_like(u)
        v_tmp = np.zeros_like(v)

        if self.ndim == 2:
            if self.use_maccormack:
                advect_u_velocity_maccormack_2d(
                    u, u_tmp, v, self.dx, self.dt, self.ny, self.nx
                )
                advect_v_velocity_maccormack_2d(
                    v, v_tmp, u, self.dx, self.dt, self.ny, self.nx
                )
            else:
                advect_u_velocity_kernel_2d(
                    u, u_tmp, v, self.dx, self.dt, self.ny, self.nx
                )
                advect_v_velocity_kernel_2d(
                    v, v_tmp, u, self.dx, self.dt, self.ny, self.nx
                )

            # Copy back
            self.velocity.u_data = u_tmp
            self.velocity.v_data = v_tmp
        else:
            w = self.velocity.w_data
            w_tmp = np.zeros_like(w)

            if self.use_maccormack:
                advect_u_velocity_maccormack_3d(
                    u, u_tmp, v, w, self.dx, self.dt, self.nz, self.ny, self.nx
                )
                advect_v_velocity_maccormack_3d(
                    v, v_tmp, u, w, self.dx, self.dt, self.nz, self.ny, self.nx
                )
                advect_w_velocity_maccormack_3d(
                    w, w_tmp, u, v, self.dx, self.dt, self.nz, self.ny, self.nx
                )
            else:
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

    def get_velocity_magnitude(self):
        """Get velocity magnitude at cell centers for visualization"""
        if self.ndim == 2:
            # Average u to cell centers
            u_center = 0.5 * (
                self.velocity.u_data[:, :-1] + self.velocity.u_data[:, 1:]
            )
            # Average v to cell centers
            v_center = 0.5 * (
                self.velocity.v_data[:-1, :] + self.velocity.v_data[1:, :]
            )

            return np.sqrt(u_center**2 + v_center**2)
        else:
            # For 3D, return magnitude for visualization (averaged to cell centers)
            u_center = 0.5 * (
                self.velocity.u_data[:, :, :-1] + self.velocity.u_data[:, :, 1:]
            )
            v_center = 0.5 * (
                self.velocity.v_data[:, :-1, :] + self.velocity.v_data[:, 1:, :]
            )
            w_center = 0.5 * (
                self.velocity.w_data[:-1, :, :] + self.velocity.w_data[1:, :, :]
            )

            return np.sqrt(u_center**2 + v_center**2 + w_center**2)

    def compute_adaptive_timestep(self):
        """Compute adaptive time step based on CFL condition

        The CFL (Courant-Friedrichs-Lewy) condition ensures numerical stability
        by requiring that fluid particles don't travel more than one grid cell
        per time step.

        Returns:
            float: Adaptive time step value clamped between dt_min and dt_max
        """
        if self.ndim == 2:
            # Find maximum velocity components in 2D
            max_u = np.max(np.abs(self.velocity.u_data))
            max_v = np.max(np.abs(self.velocity.v_data))
            max_velocity = max(max_u, max_v)
        else:
            # Find maximum velocity components in 3D
            max_u = np.max(np.abs(self.velocity.u_data))
            max_v = np.max(np.abs(self.velocity.v_data))
            max_w = np.max(np.abs(self.velocity.w_data))
            max_velocity = max(max_u, max_v, max_w)

        # Compute CFL-based time step
        if max_velocity > 1e-10:
            # CFL condition: dt = CFL * dx / max_velocity
            dt_cfl = self.cfl_target * self.dx / max_velocity

            # Clamp to reasonable bounds
            dt = np.clip(dt_cfl, self.dt_min, self.dt_max)

            # Update current CFL number for diagnostics
            self.current_cfl = (max_velocity * dt) / self.dx
        else:
            # No motion, use maximum allowed time step
            dt = self.dt_max
            self.current_cfl = 0.0

        return dt

    def export_to_npz(self, filepath, timestep=None):
        """Export simulation state to NPZ format

        Args:
            filepath: Path to save the NPZ file
            timestep: Optional timestep number to include in metadata
        """
        if self.ndim == 2:
            data = {
                # Simulation parameters
                "ndim": self.ndim,
                "nx": self.nx,
                "ny": self.ny,
                "dx": self.dx,
                "dt": self.dt,
                "simulation_time": self.simulation_time,
                # Velocity field (on MAC grid faces)
                "u_velocity": self.velocity.u_data,
                "v_velocity": self.velocity.v_data,
                # Scalar fields (on cell centers)
                "density": self.density,
                "pressure": self.pressure,
                "divergence": self.divergence,
                "vorticity": self.vorticity,
            }
        else:
            data = {
                # Simulation parameters
                "ndim": self.ndim,
                "nx": self.nx,
                "ny": self.ny,
                "nz": self.nz,
                "dx": self.dx,
                "dt": self.dt,
                "simulation_time": self.simulation_time,
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
        print(f"Exported {self.ndim}D simulation state to {filepath}")
