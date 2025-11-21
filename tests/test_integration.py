"""Integration tests for full simulation workflow."""

import pytest
import torch


class TestFullSimulation2D:
    """Integration tests for complete 2D simulation."""

    def test_smoke_rises(self, sim_2d_medium):
        """Test that smoke rises due to buoyancy."""
        # Run simulation with vorticity confinement
        sim = sim_2d_medium
        sim.vorticity_epsilon = 0.2

        # Get initial density center of mass
        initial_density = sim.density.clone()
        y_indices = torch.arange(
            sim.ny, device=sim.density.device, dtype=sim.density.dtype
        ).view(-1, 1)
        initial_com_y = (
            torch.sum(y_indices * initial_density)
            / (
                torch.sum(initial_density)
                + torch.tensor(
                    1e-10, device=sim.density.device, dtype=sim.density.dtype
                )
            )
        ).item()

        # Run simulation
        for _ in range(6):
            sim.step()

        # Get final center of mass
        final_com_y = (
            torch.sum(y_indices * sim.density)
            / (
                torch.sum(sim.density)
                + torch.tensor(
                    1e-10, device=sim.density.device, dtype=sim.density.dtype
                )
            )
        ).item()

        # Smoke should have risen (higher y position)
        assert final_com_y > initial_com_y

    def test_mass_conservation_approximate(self, sim_2d_medium):
        """Test approximate mass conservation (allowing for boundary losses and sources)."""
        sim = sim_2d_medium

        # Run for a few steps to let source stabilize
        for _ in range(2):
            sim.step()

        # Measure mass after stabilization
        mass_before = torch.sum(sim.density).item()

        # Run a few more steps (sources are continuously added)
        for _ in range(2):
            sim.step()

        mass_after = torch.sum(sim.density).item()

        # Mass should increase or stay similar (sources add mass)
        # But shouldn't explode exponentially
        assert mass_after >= mass_before * 0.5  # Allow some dissipation
        assert mass_after <= mass_before * 3.0  # Shouldn't grow too fast

    def test_stability_long_run(self, sim_2d_small):
        """Test that simulation remains stable over many steps."""
        sim = sim_2d_small

        # Run for many steps
        for i in range(8):
            sim.step()

            # Check for NaN or inf
            assert not torch.isnan(sim.density).any().item()
            assert not torch.isinf(sim.density).any().item()
            assert not torch.isnan(sim.velocity.u_data).any().item()
            assert not torch.isnan(sim.velocity.v_data).any().item()

            # Check bounds
            assert torch.max(torch.abs(sim.velocity.u_data)).item() < 100.0
            assert torch.max(torch.abs(sim.velocity.v_data)).item() < 100.0


class TestFullSimulation3D:
    """Integration tests for complete 3D simulation."""

    def test_smoke_rises_3d(self, sim_3d_medium):
        """Test that smoke rises in 3D simulation."""
        sim = sim_3d_medium
        sim.vorticity_epsilon = 0.2

        # Get initial center of mass (y-axis)
        y_indices = torch.arange(
            sim.ny, device=sim.density.device, dtype=sim.density.dtype
        ).view(1, -1, 1)
        initial_com_y = (
            torch.sum(y_indices * sim.density)
            / (
                torch.sum(sim.density)
                + torch.tensor(
                    1e-10, device=sim.density.device, dtype=sim.density.dtype
                )
            )
        ).item()

        # Run simulation
        for _ in range(5):
            sim.step()

        # Get final center of mass
        final_com_y = (
            torch.sum(y_indices * sim.density)
            / (
                torch.sum(sim.density)
                + torch.tensor(
                    1e-10, device=sim.density.device, dtype=sim.density.dtype
                )
            )
        ).item()

        # Smoke should have risen
        assert final_com_y > initial_com_y

    def test_stability_long_run_3d(self, sim_3d_small):
        """Test 3D simulation stability."""
        sim = sim_3d_small

        for i in range(8):
            sim.step()

            # Check for NaN or inf
            assert not torch.isnan(sim.density).any().item()
            assert not torch.isnan(sim.velocity.u_data).any().item()
            assert not torch.isnan(sim.velocity.v_data).any().item()
            assert not torch.isnan(sim.velocity.w_data).any().item()

            # Check bounds
            assert torch.max(torch.abs(sim.velocity.w_data)).item() < 100.0


class TestAdvectionMethods:
    """Tests comparing advection methods."""

    def test_maccormack_vs_semilagrangian(self):
        """Compare MacCormack and semi-Lagrangian advection."""
        from simulation import SmokeSimulator

        # Create two identical simulations
        sim_mac = SmokeSimulator(nx=32, ny=48, nz=None, use_maccormack=True, dt=0.01)
        sim_sl = SmokeSimulator(nx=32, ny=48, nz=None, use_maccormack=False, dt=0.01)

        # Run both for same number of steps
        for _ in range(3):
            sim_mac.step()
            sim_sl.step()

        # Both should produce valid results
        assert not torch.isnan(sim_mac.density).any().item()
        assert not torch.isnan(sim_sl.density).any().item()

        # They should be somewhat different (MacCormack less diffusive)
        # but both should have density present
        assert torch.sum(sim_mac.density).item() > 0.0
        assert torch.sum(sim_sl.density).item() > 0.0
