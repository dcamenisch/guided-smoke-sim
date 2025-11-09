"""Integration tests for full simulation workflow."""

import pytest
import numpy as np


class TestFullSimulation2D:
    """Integration tests for complete 2D simulation."""

    def test_smoke_rises(self, sim_2d_medium):
        """Test that smoke rises due to buoyancy."""
        # Run simulation with vorticity confinement
        sim = sim_2d_medium
        sim.vorticity_epsilon = 0.2

        # Get initial density center of mass
        initial_density = sim.density.copy()
        initial_com_y = np.sum(np.arange(sim.ny)[:, None] * initial_density) / (
            np.sum(initial_density) + 1e-10
        )

        # Run simulation
        for _ in range(50):
            sim.step()

        # Get final center of mass
        final_com_y = np.sum(np.arange(sim.ny)[:, None] * sim.density) / (
            np.sum(sim.density) + 1e-10
        )

        # Smoke should have risen (higher y position)
        assert final_com_y > initial_com_y

    def test_mass_conservation_approximate(self, sim_2d_medium):
        """Test approximate mass conservation (allowing for boundary losses and sources)."""
        sim = sim_2d_medium

        # Run for a few steps to let source stabilize
        for _ in range(5):
            sim.step()

        # Measure mass after stabilization
        mass_before = np.sum(sim.density)

        # Run a few more steps (sources are continuously added)
        for _ in range(5):
            sim.step()

        mass_after = np.sum(sim.density)

        # Mass should increase or stay similar (sources add mass)
        # But shouldn't explode exponentially
        assert mass_after >= mass_before * 0.5  # Allow some dissipation
        assert mass_after <= mass_before * 3.0  # Shouldn't grow too fast

    def test_stability_long_run(self, sim_2d_small):
        """Test that simulation remains stable over many steps."""
        sim = sim_2d_small

        # Run for many steps
        for i in range(100):
            sim.step()

            # Check for NaN or inf
            assert not np.any(np.isnan(sim.density))
            assert not np.any(np.isinf(sim.density))
            assert not np.any(np.isnan(sim.velocity.u_data))
            assert not np.any(np.isnan(sim.velocity.v_data))

            # Check bounds
            assert np.max(np.abs(sim.velocity.u_data)) < 100.0
            assert np.max(np.abs(sim.velocity.v_data)) < 100.0


class TestFullSimulation3D:
    """Integration tests for complete 3D simulation."""

    def test_smoke_rises_3d(self, sim_3d_medium):
        """Test that smoke rises in 3D simulation."""
        sim = sim_3d_medium
        sim.vorticity_epsilon = 0.2

        # Get initial center of mass (y-axis)
        initial_com_y = np.sum(np.arange(sim.ny)[None, :, None] * sim.density) / (
            np.sum(sim.density) + 1e-10
        )

        # Run simulation
        for _ in range(30):
            sim.step()

        # Get final center of mass
        final_com_y = np.sum(np.arange(sim.ny)[None, :, None] * sim.density) / (
            np.sum(sim.density) + 1e-10
        )

        # Smoke should have risen
        assert final_com_y > initial_com_y

    def test_stability_long_run_3d(self, sim_3d_small):
        """Test 3D simulation stability."""
        sim = sim_3d_small

        for i in range(50):
            sim.step()

            # Check for NaN or inf
            assert not np.any(np.isnan(sim.density))
            assert not np.any(np.isnan(sim.velocity.u_data))
            assert not np.any(np.isnan(sim.velocity.v_data))
            assert not np.any(np.isnan(sim.velocity.w_data))

            # Check bounds
            assert np.max(np.abs(sim.velocity.w_data)) < 100.0


class TestAdvectionMethods:
    """Tests comparing advection methods."""

    def test_maccormack_vs_semilagrangian(self):
        """Compare MacCormack and semi-Lagrangian advection."""
        from simulation import SmokeSimulator

        # Create two identical simulations
        sim_mac = SmokeSimulator(nx=32, ny=48, nz=None, use_maccormack=True, dt=0.01)
        sim_sl = SmokeSimulator(nx=32, ny=48, nz=None, use_maccormack=False, dt=0.01)

        # Run both for same number of steps
        for _ in range(10):
            sim_mac.step()
            sim_sl.step()

        # Both should produce valid results
        assert not np.any(np.isnan(sim_mac.density))
        assert not np.any(np.isnan(sim_sl.density))

        # They should be somewhat different (MacCormack less diffusive)
        # but both should have density present
        assert np.sum(sim_mac.density) > 0.0
        assert np.sum(sim_sl.density) > 0.0
