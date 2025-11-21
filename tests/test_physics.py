"""Tests for physics forces."""

import pytest
import torch
from physics.buoyancy import apply_buoyancy_force_2d, apply_buoyancy_force_3d
from physics.vorticity_confinement import (
    apply_vorticity_confinement_2d,
    apply_vorticity_confinement_3d,
)
from physics.gravity import apply_gravity_2d, apply_gravity_3d
from physics.external_force import apply_external_force_2d, apply_external_force_3d
from core import MACGrid2D, MACGrid3D


class TestBuoyancy:
    """Tests for buoyancy forces."""

    def test_buoyancy_2d_upward(self, macgrid_2d):
        """Test that buoyancy force is upward for positive density."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        density = torch.ones((24, 16), dtype=torch.float32)

        # Apply buoyancy
        # Expected: buoyancy = alpha * density * scaling_factor * dt
        # scaling_factor = 64.0 / nx = 64.0 / 16 = 4.0
        # buoyancy = 0.1 * 1.0 * 4.0 * 0.1 = 0.04
        apply_buoyancy_force_2d(force, velocity, density, dt=0.1, nx=16, alpha=0.1)

        # v-velocity should increase (upward) by approximately the expected amount
        expected_velocity = 0.04
        max_v = velocity.v_data.max().item()
        assert max_v > 0.01  # At least 25% of expected
        assert max_v < 0.1  # Not more than 2.5x expected
        assert abs(max_v - expected_velocity) < 0.02

    def test_buoyancy_3d_upward(self, macgrid_3d):
        """Test that buoyancy force is upward in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        density = torch.ones((8, 12, 8), dtype=torch.float32)

        # Apply buoyancy
        # Expected: buoyancy = alpha * density * scaling_factor * dt
        # scaling_factor = 64.0 / nx = 64.0 / 8 = 8.0
        # buoyancy = 0.1 * 1.0 * 8.0 * 0.1 = 0.08
        apply_buoyancy_force_3d(force, velocity, density, dt=0.1, nx=8, alpha=0.1)

        expected_velocity = 0.08
        max_v = velocity.v_data.max().item()
        assert max_v > 0.02
        assert max_v < 0.2
        assert abs(max_v - expected_velocity) < 0.04

    def test_buoyancy_proportional_to_density(self, macgrid_2d):
        """Test that buoyancy force is proportional to density."""
        velocity1 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force1 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        density1 = torch.ones((24, 16), dtype=torch.float32)

        velocity2 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force2 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        density2 = torch.ones((24, 16), dtype=torch.float32) * 2.0

        apply_buoyancy_force_2d(force1, velocity1, density1, dt=0.1, nx=16, alpha=0.1)
        apply_buoyancy_force_2d(force2, velocity2, density2, dt=0.1, nx=16, alpha=0.1)

        # Double density should give roughly double velocity change
        ratio = (velocity2.v_data.max() / velocity1.v_data.max()).item()
        assert 1.8 < ratio < 2.2


class TestVorticityConfinement:
    """Tests for vorticity confinement."""

    def test_vorticity_confinement_2d_no_effect_on_zero(self, macgrid_2d):
        """Test that vorticity confinement has no effect on zero vorticity."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        vorticity = torch.zeros((24, 16), dtype=torch.float32)

        u_initial = velocity.u_data.clone()
        v_initial = velocity.v_data.clone()

        # Apply vorticity confinement
        apply_vorticity_confinement_2d(
            force, velocity, vorticity, dx=1.0 / 16, dt=0.1, epsilon=0.1
        )

        torch.testing.assert_close(velocity.u_data, u_initial, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(velocity.v_data, v_initial, atol=1e-5, rtol=1e-5)

    def test_vorticity_confinement_3d_no_effect_on_zero(self, macgrid_3d):
        """Test that vorticity confinement has no effect on zero vorticity."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        vorticity = torch.zeros((8, 12, 8, 3), dtype=torch.float32)

        u_initial = velocity.u_data.clone()
        v_initial = velocity.v_data.clone()
        w_initial = velocity.w_data.clone()

        apply_vorticity_confinement_3d(
            force, velocity, vorticity, dx=1.0 / 8, dt=0.1, epsilon=0.1
        )

        torch.testing.assert_close(velocity.u_data, u_initial, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(velocity.v_data, v_initial, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(velocity.w_data, w_initial, atol=1e-5, rtol=1e-5)

    def test_vorticity_confinement_2d_amplifies_rotation(self, macgrid_2d):
        """Test that vorticity confinement amplifies rotational flow."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        cy, cx = 12.0, 8.0
        y_coords = torch.arange(24, dtype=torch.float32).unsqueeze(1)
        x_coords = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        dist2 = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        vorticity = torch.exp(-dist2 / 10.0)

        assert velocity.u_data.abs().max().item() == 0.0
        assert velocity.v_data.abs().max().item() == 0.0

        # Apply confinement with significant epsilon
        apply_vorticity_confinement_2d(
            force, velocity, vorticity, dx=1.0 / 16, dt=0.1, epsilon=1.0
        )

        # Velocity should be affected by the vorticity gradient
        u_final = velocity.u_data.abs().max().item()
        v_final = velocity.v_data.abs().max().item()

        total_velocity_change = u_final + v_final
        assert (
            total_velocity_change > 0.005
        ), f"Expected velocity change > 0.005, got {total_velocity_change}"
        assert (
            total_velocity_change < 1.0
        ), f"Velocity change too large: {total_velocity_change}"


class TestGravity:
    """Tests for gravity forces."""

    def test_gravity_2d_downward(self):
        """Test that gravity accelerates fluid downward in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        torch.testing.assert_close(velocity.v_data, torch.zeros_like(velocity.v_data))

        # Apply gravity with g = -9.81 (downward)
        dt = 0.1
        g = -9.81
        apply_gravity_2d(force, velocity, dt=dt, g=g)

        # v-velocity should become negative (downward)
        # Expected: v_new = v_old + g * dt = 0 + (-9.81) * 0.1 = -0.981
        expected_v = g * dt
        expected_tensor = torch.full_like(velocity.v_data, expected_v)
        torch.testing.assert_close(
            velocity.v_data, expected_tensor, rtol=1e-5, atol=1e-6
        )

        torch.testing.assert_close(velocity.u_data, torch.zeros_like(velocity.u_data))

    def test_gravity_2d_upward(self):
        """Test gravity with positive g (upward acceleration)."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        dt = 0.1
        g = 5.0  # Upward
        apply_gravity_2d(force, velocity, dt=dt, g=g)

        # v-velocity should become positive (upward)
        expected_v = g * dt
        expected_tensor = torch.full_like(velocity.v_data, expected_v)
        torch.testing.assert_close(
            velocity.v_data, expected_tensor, rtol=1e-5, atol=1e-6
        )

    def test_gravity_2d_accumulation(self):
        """Test that gravity accumulates over multiple steps."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        dt = 0.1
        g = -9.81

        # Apply gravity twice
        apply_gravity_2d(force, velocity, dt=dt, g=g)
        apply_gravity_2d(force, velocity, dt=dt, g=g)

        # Velocity should be 2 * g * dt
        expected_v = 2.0 * g * dt
        expected_tensor = torch.full_like(velocity.v_data, expected_v)
        torch.testing.assert_close(
            velocity.v_data, expected_tensor, rtol=1e-5, atol=1e-6
        )

    def test_gravity_3d_downward(self):
        """Test that gravity accelerates fluid downward in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)

        dt = 0.1
        g = -9.81
        apply_gravity_3d(force, velocity, dt=dt, g=g)

        expected_v = g * dt
        expected_tensor = torch.full_like(velocity.v_data, expected_v)
        torch.testing.assert_close(
            velocity.v_data, expected_tensor, rtol=1e-5, atol=1e-6
        )

        torch.testing.assert_close(velocity.u_data, torch.zeros_like(velocity.u_data))
        torch.testing.assert_close(velocity.w_data, torch.zeros_like(velocity.w_data))

    def test_gravity_zero(self):
        """Test that zero gravity produces no acceleration."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        apply_gravity_2d(force, velocity, dt=0.1, g=0.0)

        torch.testing.assert_close(velocity.u_data, torch.zeros_like(velocity.u_data))
        torch.testing.assert_close(velocity.v_data, torch.zeros_like(velocity.v_data))


class TestExternalForce:
    """Tests for external force fields."""

    def test_external_force_2d_uniform(self):
        """Test uniform external force in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        force_u = torch.ones((24, 17), dtype=torch.float32) * 2.0
        force_v = torch.ones((25, 16), dtype=torch.float32) * 3.0

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should be force * dt
        expected_u = torch.full_like(velocity.u_data, 2.0 * dt)
        expected_v = torch.full_like(velocity.v_data, 3.0 * dt)

        torch.testing.assert_close(velocity.u_data, expected_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.v_data, expected_v, rtol=1e-5, atol=1e-6)

    def test_external_force_2d_varying(self):
        """Test spatially varying external force in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        torch.manual_seed(0)
        force_u = torch.rand((24, 17), dtype=torch.float32)
        force_v = torch.rand((25, 16), dtype=torch.float32)

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Each cell should have force * dt
        expected_u = force_u * dt
        expected_v = force_v * dt

        torch.testing.assert_close(velocity.u_data, expected_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.v_data, expected_v, rtol=1e-5, atol=1e-6)

    def test_external_force_2d_accumulation(self):
        """Test that external forces accumulate over multiple applications."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        force_u = torch.ones((24, 17), dtype=torch.float32) * 5.0
        force_v = torch.ones((25, 16), dtype=torch.float32) * 10.0

        dt = 0.1

        # Apply twice
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should accumulate: 2 * force * dt
        expected_u = torch.full_like(velocity.u_data, 2.0 * 5.0 * dt)
        expected_v = torch.full_like(velocity.v_data, 2.0 * 10.0 * dt)

        torch.testing.assert_close(velocity.u_data, expected_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.v_data, expected_v, rtol=1e-5, atol=1e-6)

    def test_external_force_3d_uniform(self):
        """Test uniform external force in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)

        force_u = torch.ones((8, 12, 9), dtype=torch.float32) * 2.0
        force_v = torch.ones((8, 13, 8), dtype=torch.float32) * 3.0
        force_w = torch.ones((9, 12, 8), dtype=torch.float32) * 4.0

        dt = 0.1
        apply_external_force_3d(force, velocity, force_u, force_v, force_w, dt=dt)

        expected_u = torch.full_like(velocity.u_data, 2.0 * dt)
        expected_v = torch.full_like(velocity.v_data, 3.0 * dt)
        expected_w = torch.full_like(velocity.w_data, 4.0 * dt)

        torch.testing.assert_close(velocity.u_data, expected_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.v_data, expected_v, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.w_data, expected_w, rtol=1e-5, atol=1e-6)

    def test_external_force_zero(self):
        """Test that zero external force produces no change."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        velocity.u_data.fill_(5.0)
        velocity.v_data.fill_(3.0)

        # Zero force
        force_u = torch.zeros((24, 17), dtype=torch.float32)
        force_v = torch.zeros((25, 16), dtype=torch.float32)

        apply_external_force_2d(force, velocity, force_u, force_v, dt=0.1)

        # Velocity should remain unchanged
        expected_u = torch.full_like(velocity.u_data, 5.0)
        expected_v = torch.full_like(velocity.v_data, 3.0)
        torch.testing.assert_close(velocity.u_data, expected_u)
        torch.testing.assert_close(velocity.v_data, expected_v)

    def test_external_force_negative(self):
        """Test external force with negative values."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Negative forces
        force_u = torch.ones((24, 17), dtype=torch.float32) * -10.0
        force_v = torch.ones((25, 16), dtype=torch.float32) * -5.0

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should be negative
        expected_u = torch.full_like(velocity.u_data, -10.0 * dt)
        expected_v = torch.full_like(velocity.v_data, -5.0 * dt)

        torch.testing.assert_close(velocity.u_data, expected_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(velocity.v_data, expected_v, rtol=1e-5, atol=1e-6)
