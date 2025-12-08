"""Base simulator class with shared logic for smoke simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseSimulator(ABC):
    """Abstract base class for smoke simulators.

    Provides common simulation workflow and shared methods.
    Subclasses implement dimension-specific details.
    """

    def __init__(
        self,
        dt: float = 0.07,
        tolerance: float = 1e-5,
        max_iterations: int = 1000,
        cfl_target: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        """Initialize simulator parameters.

        Args:
            dt: Initial time step
            tolerance: Pressure solver convergence tolerance
            max_iterations: Maximum pressure solver iterations
            cfl_target: Target CFL number (typically 1.0-5.0)
            dt_min: Minimum time step
            dt_max: Maximum time step
        """
        self.dt = dt
        self.dt_initial = dt  # Store initial dt for reference
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.cfl_target = cfl_target
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.current_cfl = 0.0  # Track actual CFL number
        self.simulation_time = 0.0  # Track total accumulated simulation time

    def step(self) -> None:
        """Execute one simulation step with adaptive time stepping.

        1. Add source
        2. Advect density and velocity
        3. Apply forces (buoyancy + external)
        4. Apply boundary conditions (after forces)
        5. Pressure projection
        """
        self.dt = self.compute_adaptive_timestep()
        self.add_source()
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

    # Abstract methods that subclasses must implement

    @abstractmethod
    def add_source(self) -> None:
        """Add smoke source to density field"""
        pass

    @abstractmethod
    def apply_forces(self) -> None:
        """Apply buoyancy force and update velocity"""
        pass

    @abstractmethod
    def set_boundary_conditions(self) -> None:
        """Set boundary conditions for velocity and pressure"""
        pass

    @abstractmethod
    def compute_divergence(self) -> None:
        """Compute velocity divergence"""
        pass

    @abstractmethod
    def solve_poisson(self) -> None:
        """Solve Poisson equation for pressure"""
        pass

    @abstractmethod
    def correct_velocity(self) -> None:
        """Correct velocity with pressure gradient"""
        pass

    @abstractmethod
    def compute_vorticity(self) -> None:
        """Compute vorticity field"""
        pass

    @abstractmethod
    def advect_density(self) -> None:
        """Advect density using semi-Lagrangian method"""
        pass

    @abstractmethod
    def advect_velocity(self) -> None:
        """Advect velocity using semi-Lagrangian method"""
        pass

    @abstractmethod
    def compute_adaptive_timestep(self) -> float:
        """Compute adaptive time step based on CFL condition

        Returns:
            float: Adaptive time step value
        """
        pass

    @abstractmethod
    def export_to_npz(self, filepath: str, timestep: Optional[int] = None) -> None:
        """Export simulation state to NPZ format

        Args:
            filepath: Path to save the NPZ file
            timestep: Optional timestep number to include in metadata
        """
        pass
