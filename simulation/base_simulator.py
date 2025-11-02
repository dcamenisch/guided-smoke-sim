"""Base simulator class with shared logic for smoke simulation."""

import numpy as np
from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    """Abstract base class for smoke simulators

    Provides common simulation workflow and shared methods.
    Subclasses must implement dimension-specific details.
    """

    def __init__(self, dt=0.07, tolerance=1e-5, max_iterations=1000):
        """Initialize base simulator parameters

        Args:
            dt: Time step
            tolerance: Convergence tolerance for pressure solver
            max_iterations: Maximum iterations for pressure solver
        """
        self.dt = dt
        self.tolerance = tolerance
        self.max_iterations = max_iterations

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

    # Abstract methods that subclasses must implement

    @abstractmethod
    def add_source(self):
        """Add smoke source to density field"""
        pass

    @abstractmethod
    def apply_forces(self):
        """Apply buoyancy force and update velocity"""
        pass

    @abstractmethod
    def set_boundary_conditions(self):
        """Set boundary conditions for velocity and pressure"""
        pass

    @abstractmethod
    def compute_divergence(self):
        """Compute velocity divergence"""
        pass

    @abstractmethod
    def solve_poisson(self):
        """Solve Poisson equation for pressure"""
        pass

    @abstractmethod
    def correct_velocity(self):
        """Correct velocity with pressure gradient"""
        pass

    @abstractmethod
    def compute_vorticity(self):
        """Compute vorticity field"""
        pass

    @abstractmethod
    def advect_density(self):
        """Advect density using semi-Lagrangian method"""
        pass

    @abstractmethod
    def advect_velocity(self):
        """Advect velocity using semi-Lagrangian method"""
        pass

    @abstractmethod
    def export_to_npz(self, filepath, timestep=None):
        """Export simulation state to NPZ format

        Args:
            filepath: Path to save the NPZ file
            timestep: Optional timestep number to include in metadata
        """
        pass
