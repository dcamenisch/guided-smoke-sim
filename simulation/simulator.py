"""Unified smoke simulator supporting both 2D and 3D simulations using Torch."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from physics.buoyancy import apply_buoyancy_force_2d, apply_buoyancy_force_3d
from physics.vorticity_confinement import (
    apply_vorticity_confinement_2d,
    apply_vorticity_confinement_3d,
)
from physics.external_force import apply_external_force_2d, apply_external_force_3d
from core import MACGrid2D, MACGrid3D
from kernels.linear_solver import ConjugateGradientSolver
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

Tensor = torch.Tensor


def _to_numpy(array: Any) -> Any:
    """Detach Torch tensors to NumPy for serialization."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


class SmokeSimulator:
    """Unified smoke simulator for 2D and 3D.

    Dimensionality is determined by nz parameter:
    - nz=None → 2D simulation
    - nz=int → 3D simulation
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 192,
        nz: Optional[int] = None,
        dt: float = 0.0416667,
        tolerance: float = 1e-5,
        max_iterations: int = 1000,
        use_maccormack: bool = False,
        advection_rk_order: int = 3,
        vorticity_epsilon: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        enable_buoyancy: bool = True,
        buoyancy_alpha: float = 0.1,
    ) -> None:
        """Initialize smoke simulator.

        Args:
            nx: Grid resolution in x
            ny: Grid resolution in y
            nz: Grid resolution in z (None for 2D)
            dt: Initial time step
            tolerance: Pressure solver convergence tolerance
            max_iterations: Pressure solver max iterations
            use_maccormack: Use MacCormack (True) or semi-Lagrangian (False)
            advection_rk_order: RK order for semi-Lagrangian (1 or 3)
            vorticity_epsilon: Vorticity confinement strength (0.0=off)
            device: Device to run on
            dtype: Data type
            enable_buoyancy: Whether to apply buoyancy
            buoyancy_alpha: Buoyancy coefficient
        """
        self.dt = dt
        self.dt_initial = dt
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.simulation_time = 0.0

        # Determine dimensionality
        self.ndim = 2 if nz is None else 3
        self.nx, self.ny = nx, ny
        self.nz = nz if nz is not None else 1  # For internal consistency
        self.dx = 1.0 / nx
        self.use_maccormack = use_maccormack
        self.advection_rk_order = advection_rk_order
        self.vorticity_epsilon = vorticity_epsilon
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_buoyancy = enable_buoyancy
        self.buoyancy_alpha = buoyancy_alpha

        # Create appropriate MAC grids based on dimensionality
        if self.ndim == 2:
            self.velocity = MACGrid2D(
                nx, ny, self.dx, device=self.device, dtype=self.dtype
            )
            self.force = MACGrid2D(
                nx, ny, self.dx, device=self.device, dtype=self.dtype
            )
            # Scalar fields for 2D
            self.density = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
            self.pressure = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
            self.divergence = torch.zeros(
                (ny, nx), dtype=self.dtype, device=self.device
            )
            self.vorticity = torch.zeros((ny, nx), dtype=self.dtype, device=self.device)
        else:
            self.velocity = MACGrid3D(
                nx, ny, nz, self.dx, device=self.device, dtype=self.dtype
            )
            self.force = MACGrid3D(
                nx, ny, nz, self.dx, device=self.device, dtype=self.dtype
            )
            # Scalar fields for 3D
            self.density = torch.zeros(
                (nz, ny, nx), dtype=self.dtype, device=self.device
            )
            self.pressure = torch.zeros(
                (nz, ny, nx), dtype=self.dtype, device=self.device
            )
            self.divergence = torch.zeros(
                (nz, ny, nx), dtype=self.dtype, device=self.device
            )
            self.vorticity = torch.zeros(
                (nz, ny, nx, 3), dtype=self.dtype, device=self.device
            )

        # Initialize sparse solver
        self.cg_solver = ConjugateGradientSolver(nx, ny, self.nz, self.dx)

        # Initialize control force fields as None
        self.control_force_u: Optional[torch.Tensor] = None
        self.control_force_v: Optional[torch.Tensor] = None
        self.control_force_w: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset simulation state to initial conditions."""
        self.simulation_time = 0.0
        self.velocity.reset()
        self.force.reset()

        # Reset scalar fields
        self.density.zero_()
        self.pressure.zero_()
        self.divergence.zero_()
        self.vorticity.zero_()

        # Reset control forces
        self.control_force_u = None
        self.control_force_v = None
        self.control_force_w = None

    def step(self) -> None:
        """Execute one simulation step.

        1. Advect density and velocity
        2. Apply forces (buoyancy + external)
        3. Apply boundary conditions (after forces)
        4. Pressure projection
        """
        self.advect()
        self.apply_forces()
        self.set_boundary_conditions()
        self.solve_pressure()
        self.force.reset()
        self.simulation_time += self.dt

    def solve_pressure(self) -> None:
        """Solve pressure projection to enforce incompressibility."""
        self.compute_divergence()
        self.solve_poisson()
        self.correct_velocity()
        self.compute_vorticity()
        self.compute_divergence()

    def advect(self) -> None:
        """Advect all quantities through the velocity field."""
        self.advect_density()
        self.advect_velocity()

    def add_source(self) -> None:
        """Add smoke source at specified location."""
        # if self.ndim == 2:
        #     y_start, y_end = int(0.1 * self.ny), int(0.15 * self.ny) + 1
        #     x_start, x_end = int(0.45 * self.nx), int(0.55 * self.nx) + 1
        #     self.density[y_start:y_end, x_start:x_end] = 1.0
        # else:
        #     z_start, z_end = int(0.475 * self.nz), int(0.525 * self.nz) + 1
        #     y_start, y_end = int(0.0 * self.ny), int(0.025 * self.ny) + 1
        #     x_start, x_end = int(0.475 * self.nx), int(0.525 * self.nx) + 1
        #     self.density[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0

        radius = 0.1
        density_value = 1.0
        center_x = 0.5
        center_y = 0.15
        center_z = 0.5

        if self.ndim == 2:
            y_coords = (
                torch.arange(self.ny, device=self.device, dtype=self.dtype) + 0.5
            ) / self.ny
            x_coords = (
                torch.arange(self.nx, device=self.device, dtype=self.dtype) + 0.5
            ) / self.nx
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            y_scale = self.ny / self.nx if self.nx else 1.0
            dist_sq = (xx - center_x) ** 2 + ((yy - center_y) * y_scale) ** 2
            mask = dist_sq <= radius**2
            self.density = self.density.where(
                ~mask, torch.full_like(self.density, density_value)
            )
        else:
            z_coords = (
                torch.arange(self.nz, device=self.device, dtype=self.dtype) + 0.5
            ) / self.nz
            y_coords = (
                torch.arange(self.ny, device=self.device, dtype=self.dtype) + 0.5
            ) / self.ny
            x_coords = (
                torch.arange(self.nx, device=self.device, dtype=self.dtype) + 0.5
            ) / self.nx
            zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
            y_scale = self.ny / self.nx if self.nx else 1.0
            z_scale = self.nz / self.nx if self.nx else 1.0
            dist_sq = (
                (xx - center_x) ** 2
                + ((yy - center_y) * y_scale) ** 2
                + ((zz - center_z) * z_scale) ** 2
            )
            mask = dist_sq <= radius**2
            self.density = self.density.where(
                ~mask, torch.full_like(self.density, density_value)
            )

    def apply_forces(self) -> None:
        """Apply buoyancy and control forces to velocity field."""
        # Apply Buoyancy
        if self.enable_buoyancy:
            if self.ndim == 2:
                apply_buoyancy_force_2d(
                    self.force,
                    self.velocity,
                    self.density,
                    self.dt,
                    self.nx,
                    alpha=self.buoyancy_alpha,
                )
            else:
                apply_buoyancy_force_3d(
                    self.force,
                    self.velocity,
                    self.density,
                    self.dt,
                    self.nx,
                    alpha=self.buoyancy_alpha,
                )

        # Apply Control Forces
        if self.control_force_u is not None:
            self._apply_control_force()

    def set_control_force(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> None:
        """Set external control force fields.

        Args:
            u: x-component of force
            v: y-component of force
            w: z-component of force (3D only)
        """
        self.control_force_u = u
        self.control_force_v = v
        if self.ndim == 3:
            self.control_force_w = w

    def _apply_control_force(self) -> None:
        """Apply the set control force to the velocity field."""
        if self.ndim == 2:
            apply_external_force_2d(
                self.force,
                self.velocity,
                self.control_force_u,
                self.control_force_v,
                self.dt,
            )
        else:
            apply_external_force_3d(
                self.force,
                self.velocity,
                self.control_force_u,
                self.control_force_v,
                self.control_force_w,
                self.dt,
            )

    def set_boundary_conditions(self) -> None:
        """Set boundary conditions on velocity field."""
        if self.ndim == 2:
            self._set_boundary_conditions_2d()
        else:
            self._set_boundary_conditions_3d()

    def _set_boundary_conditions_2d(self) -> None:
        """Set 2D boundary conditions

        Open boundaries at top, left, and right (outflow)
        Closed boundary at bottom (no-slip wall)
        """
        # x-velocity (u) boundary conditions - create new tensor to preserve gradients
        u = self.velocity.u_data.clone()

        # Left/right boundaries: open (outflow - extrapolate)
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

        # Bottom boundary: no-slip wall (u = 0)
        u[0, :] = 0

        # Top boundary: open (outflow - extrapolate)
        u[-1, :] = u[-2, :]

        self.velocity.u_data = u

        # y-velocity (v) boundary conditions
        v = self.velocity.v_data.clone()

        # Bottom boundary: no penetration (v = 0)
        v[0, :] = 0

        # Top boundary: open (outflow - extrapolate)
        v[-1, :] = v[-2, :]

        # Left/right boundaries: open (outflow - extrapolate)
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]

        self.velocity.v_data = v

        # Pressure boundary conditions
        p = self.pressure.clone()

        # All boundaries: dp/dn = 0 (Neumann BC for all)
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        self.pressure = p

    def _set_boundary_conditions_3d(self) -> None:
        """Set 3D boundary conditions

        Open boundaries at top and all sides (outflow)
        Closed boundary at bottom (no-slip wall)
        """
        # x-velocity (u) boundary conditions - create new tensor
        u = self.velocity.u_data.clone()

        # Left/right boundaries: open (outflow - extrapolate)
        u[:, :, 0] = u[:, :, 1]
        u[:, :, -1] = u[:, :, -2]

        # Bottom boundary: no-slip wall (u = 0)
        u[:, 0, :] = 0

        # Top boundary: open (outflow - extrapolate)
        u[:, -1, :] = u[:, -2, :]

        # Front/back boundaries: open (outflow - extrapolate)
        u[0, :, :] = u[1, :, :]
        u[-1, :, :] = u[-2, :, :]

        self.velocity.u_data = u

        # y-velocity (v) boundary conditions
        v = self.velocity.v_data.clone()

        # Left/right boundaries: open (outflow - extrapolate)
        v[:, :, 0] = v[:, :, 1]
        v[:, :, -1] = v[:, :, -2]

        # Bottom boundary: no penetration (v = 0)
        v[:, 0, :] = 0

        # Top boundary: open (outflow - extrapolate)
        v[:, -1, :] = v[:, -2, :]

        # Front/back boundaries: open (outflow - extrapolate)
        v[0, :, :] = v[1, :, :]
        v[-1, :, :] = v[-2, :, :]

        self.velocity.v_data = v

        # z-velocity (w) boundary conditions
        w = self.velocity.w_data.clone()

        # Left/right boundaries: open (outflow - extrapolate)
        w[:, :, 0] = w[:, :, 1]
        w[:, :, -1] = w[:, :, -2]

        # Bottom boundary: no-slip wall (w = 0)
        w[:, 0, :] = 0

        # Top boundary: open (outflow - extrapolate)
        w[:, -1, :] = w[:, -2, :]

        # Front/back boundaries: open (outflow - extrapolate)
        w[0, :, :] = w[1, :, :]
        w[-1, :, :] = w[-2, :, :]

        self.velocity.w_data = w

        # Pressure boundary conditions
        p = self.pressure.clone()

        # All boundaries: dp/dn = 0 (Neumann BC for all)
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]

        self.pressure = p

    def compute_divergence(self) -> None:
        """Compute velocity divergence."""
        u = self.velocity.u_data
        v = self.velocity.v_data

        if self.ndim == 2:
            # Compute partial derivatives for interior cells
            dudx = (u[:, 1:] - u[:, :-1]) / self.dx
            dvdy = (v[1:, :] - v[:-1, :]) / self.dx

            # Sum all partial derivatives - both have shape (ny, nx)
            self.divergence = dudx + dvdy
        else:
            w = self.velocity.w_data

            # Compute partial derivatives for interior cells
            dudx = (u[:, :, 1:] - u[:, :, :-1]) / self.dx
            dvdy = (v[:, 1:, :] - v[:, :-1, :]) / self.dx
            dwdz = (w[1:, :, :] - w[:-1, :, :]) / self.dx

            # Sum all partial derivatives - all have shape (nz, ny, nx)
            self.divergence = dudx + dvdy + dwdz

    def solve_poisson(self) -> None:
        """Solve Poisson equation for pressure using Conjugate Gradient."""
        # All cells are active (fluid domain)
        active_mask = torch.ones_like(self.divergence, dtype=torch.bool)

        # RHS for Poisson: -div(u) / dt (assuming rho=1)
        rhs = -self.divergence / self.dt

        self.pressure = self.cg_solver.solve(
            rhs,
            active_mask,
            p_init=self.pressure,
            tol=self.tolerance,
            max_iter=self.max_iterations,
        )

    def correct_velocity(self) -> None:
        """Correct velocity with pressure gradient."""
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

        # Apply boundary conditions after velocity correction
        self.set_boundary_conditions()

    def compute_vorticity(self) -> None:
        """Compute vorticity field."""
        vorticity_tmp = torch.zeros_like(self.vorticity)

        if self.ndim == 2:
            compute_vorticity_kernel_2d(
                vorticity_tmp,
                self.velocity.u_data,
                self.velocity.v_data,
                self.dx,
                self.ny,
                self.nx,
            )
            self.vorticity = vorticity_tmp

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
                vorticity_tmp,
                self.velocity.u_data,
                self.velocity.v_data,
                self.velocity.w_data,
                self.dx,
                self.nz,
                self.ny,
                self.nx,
            )
            self.vorticity = vorticity_tmp

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

    def advect_density(self) -> None:
        """Advect density using MacCormack or semi-Lagrangian method"""
        density_tmp = torch.zeros_like(self.density)

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
                    self.advection_rk_order,
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
                    self.advection_rk_order,
                )
        self.density = density_tmp

    def advect_velocity(self) -> None:
        """Advect velocity using MacCormack or semi-Lagrangian method"""
        u = self.velocity.u_data
        v = self.velocity.v_data

        u_tmp = torch.zeros_like(u)
        v_tmp = torch.zeros_like(v)

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
                    u,
                    u_tmp,
                    v,
                    self.dx,
                    self.dt,
                    self.ny,
                    self.nx,
                    self.advection_rk_order,
                )
                advect_v_velocity_kernel_2d(
                    v,
                    v_tmp,
                    u,
                    self.dx,
                    self.dt,
                    self.ny,
                    self.nx,
                    self.advection_rk_order,
                )

            # Copy back
            self.velocity.u_data = u_tmp
            self.velocity.v_data = v_tmp
        else:
            w = self.velocity.w_data
            w_tmp = torch.zeros_like(w)

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
                    u,
                    u_tmp,
                    v,
                    w,
                    self.dx,
                    self.dt,
                    self.nz,
                    self.ny,
                    self.nx,
                    self.advection_rk_order,
                )
                advect_v_velocity_kernel_3d(
                    v,
                    v_tmp,
                    u,
                    w,
                    self.dx,
                    self.dt,
                    self.nz,
                    self.ny,
                    self.nx,
                    self.advection_rk_order,
                )
                advect_w_velocity_kernel_3d(
                    w,
                    w_tmp,
                    u,
                    v,
                    self.dx,
                    self.dt,
                    self.nz,
                    self.ny,
                    self.nx,
                    self.advection_rk_order,
                )

            # Copy back
            self.velocity.u_data = u_tmp
            self.velocity.v_data = v_tmp
            self.velocity.w_data = w_tmp

    def get_velocity_magnitude(self) -> Tensor:
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

            return torch.sqrt(u_center**2 + v_center**2)
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

            return torch.sqrt(u_center**2 + v_center**2 + w_center**2)

    def export_to_npz(self, filepath: str, timestep: Optional[int] = None) -> None:
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

        numpy_data = {key: _to_numpy(value) for key, value in data.items()}
        np.savez_compressed(filepath, **numpy_data)
        print(f"Exported {self.ndim}D simulation state to {filepath}")
