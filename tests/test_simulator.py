"""Tests for smoke simulator."""

import pytest
import torch

from simulation import SmokeSimulator


class TestSimulatorInitialization:
    """Tests for simulator initialization."""

    def test_2d_initialization(self):
        """Test 2D simulator initialization."""
        sim = SmokeSimulator(nx=16, ny=24, nz=None)

        assert sim.ndim == 2
        assert sim.nx == 16
        assert sim.ny == 24
        assert sim.nz == 1
        assert sim.density.shape == (24, 16)
        assert sim.pressure.shape == (24, 16)
        assert sim.vorticity.shape == (24, 16)

    def test_3d_initialization(self):
        """Test 3D simulator initialization."""
        sim = SmokeSimulator(nx=8, ny=12, nz=8)

        assert sim.ndim == 3
        assert sim.nx == 8
        assert sim.ny == 12
        assert sim.nz == 8
        assert sim.density.shape == (8, 12, 8)
        assert sim.pressure.shape == (8, 12, 8)
        assert sim.vorticity.shape == (8, 12, 8, 3)

    def test_adaptive_time_stepping(self):
        """Test adaptive time stepping is enabled."""
        sim = SmokeSimulator(nx=16, ny=24, nz=None, cfl_target=1.5)

        assert sim.cfl_target == 1.5
        assert sim.dt_min == 0.001
        assert sim.dt_max == 0.1
        assert sim.simulation_time == 0.0


class TestSimulatorMethods:
    """Tests for simulator methods."""

    def test_add_source_2d(self, sim_2d_small):
        """Test smoke source in 2D."""
        # Initially no density
        assert torch.allclose(
            sim_2d_small.density, torch.zeros_like(sim_2d_small.density)
        )

        # Add source
        sim_2d_small.add_source()

        # Check density added in correct region
        assert torch.max(sim_2d_small.density).item() > 0.0
        assert torch.count_nonzero(sim_2d_small.density > 0).item() > 0

    def test_add_source_3d(self, sim_3d_small):
        """Test smoke source in 3D."""
        # Initially no density
        assert torch.allclose(
            sim_3d_small.density, torch.zeros_like(sim_3d_small.density)
        )

        # Add source
        sim_3d_small.add_source()

        # Check density added
        assert torch.max(sim_3d_small.density).item() > 0.0
        assert torch.count_nonzero(sim_3d_small.density > 0).item() > 0

    def test_step_increases_time(self, sim_2d_small):
        """Test that step increases simulation time."""
        initial_time = sim_2d_small.simulation_time

        sim_2d_small.step()

        assert sim_2d_small.simulation_time > initial_time

    def test_multiple_steps(self, sim_2d_small):
        """Test multiple simulation steps."""
        for _ in range(5):
            sim_2d_small.step()

        # Should have advanced in time
        assert sim_2d_small.simulation_time > 0.0

    def test_density_stays_bounded_2d(self, sim_2d_medium):
        """Test density remains in reasonable bounds."""
        # Run simulation for a while
        for _ in range(4):
            sim_2d_medium.step()

        # Density should be non-negative
        assert torch.all(sim_2d_medium.density >= 0.0).item()

        # Density shouldn't explode
        assert torch.max(sim_2d_medium.density).item() < 10.0

    def test_density_stays_bounded_3d(self, sim_3d_medium):
        """Test density remains in reasonable bounds."""
        # Run simulation for a while
        for _ in range(4):
            sim_3d_medium.step()

        # Density should be non-negative
        assert torch.all(sim_3d_medium.density >= 0.0).item()

        # Density shouldn't explode
        assert torch.max(sim_3d_medium.density).item() < 10.0


class TestDivergenceFree:
    """Tests for divergence-free constraint."""

    def test_divergence_after_projection_2d(self, sim_2d_medium):
        """Test that velocity is divergence-free after pressure projection."""
        # Add some initial velocity
        sim_2d_medium.velocity.u_data[:] = 0.1
        sim_2d_medium.velocity.v_data[:] = 0.1

        # Apply pressure projection
        sim_2d_medium.solve_pressure()

        # Check divergence is small
        interior_div = sim_2d_medium.divergence[1:-1, 1:-1]
        assert torch.max(torch.abs(interior_div)).item() < 1e-3

    def test_divergence_after_projection_3d(self, sim_3d_medium):
        """Test that velocity is divergence-free after pressure projection."""
        # Add some initial velocity
        sim_3d_medium.velocity.u_data[:] = 0.1
        sim_3d_medium.velocity.v_data[:] = 0.1
        sim_3d_medium.velocity.w_data[:] = 0.1

        # Apply pressure projection
        sim_3d_medium.solve_pressure()

        # Check divergence is small
        interior_div = sim_3d_medium.divergence[1:-1, 1:-1, 1:-1]
        assert torch.max(torch.abs(interior_div)).item() < 1e-3


class TestAdaptiveTimeStep:
    """Tests for adaptive time stepping."""

    def test_dt_adapts_to_velocity(self, sim_2d_small):
        """Test that dt adapts based on velocity magnitude."""
        # Low velocity -> large dt
        sim_2d_small.velocity.u_data[:] = 0.01
        sim_2d_small.velocity.v_data[:] = 0.01
        dt_low = sim_2d_small.compute_adaptive_timestep()

        # High velocity -> small dt
        sim_2d_small.velocity.u_data[:] = 1.0
        sim_2d_small.velocity.v_data[:] = 1.0
        dt_high = sim_2d_small.compute_adaptive_timestep()

        assert dt_high < dt_low

    def test_dt_respects_bounds(self, sim_2d_small):
        """Test that dt stays within min/max bounds."""
        # Very high velocity
        sim_2d_small.velocity.u_data[:] = 100.0
        dt = sim_2d_small.compute_adaptive_timestep()
        assert dt >= sim_2d_small.dt_min

        # Very low velocity
        sim_2d_small.velocity.u_data[:] = 0.0001
        dt = sim_2d_small.compute_adaptive_timestep()
        assert dt <= sim_2d_small.dt_max

    def test_cfl_tracking(self, sim_2d_small):
        """Test that CFL number is tracked."""
        sim_2d_small.velocity.u_data[:] = 0.5
        sim_2d_small.compute_adaptive_timestep()

        assert sim_2d_small.current_cfl >= 0.0
        assert sim_2d_small.current_cfl < 10.0  # Reasonable range
