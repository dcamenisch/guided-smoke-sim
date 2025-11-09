"""Dataclasses for configuring smoke simulation instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.simulator import SmokeSimulator


@dataclass
class SimulationConfig:
    """Configuration for creating a :class:`SmokeSimulator` instance."""

    nx: int = 128
    ny: int = 192
    nz: int | None = None
    dt: float = 0.07
    tolerance: float = 1e-5
    max_iterations: int = 1000
    use_maccormack: bool = True
    advection_rk_order: int = 1
    vorticity_epsilon: float = 0.0
    cfl_target: float = 1.0
    dt_min: float = 0.001
    dt_max: float = 0.1

    def create_simulator(self) -> "SmokeSimulator":
        """Instantiate a simulator with the current configuration."""

        from simulation import SmokeSimulator

        return SmokeSimulator(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dt=self.dt,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            use_maccormack=self.use_maccormack,
            advection_rk_order=self.advection_rk_order,
            vorticity_epsilon=self.vorticity_epsilon,
            cfl_target=self.cfl_target,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
        )
