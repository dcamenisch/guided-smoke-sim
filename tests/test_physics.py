"""Tests for physics forces."""

import pytest
import numpy as np
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
        density = np.ones((24, 16), dtype=np.float32)

        # Apply buoyancy
        # Expected: buoyancy = alpha * density * scaling_factor * dt
        # scaling_factor = 64.0 / nx = 64.0 / 16 = 4.0
        # buoyancy = 0.1 * 1.0 * 4.0 * 0.1 = 0.04
        apply_buoyancy_force_2d(force, velocity, density, dt=0.1, nx=16, alpha=0.1)

        # v-velocity should increase (upward) by approximately the expected amount
        expected_velocity = 0.04
        assert np.max(velocity.v_data) > 0.01  # At least 25% of expected
        assert np.max(velocity.v_data) < 0.1   # Not more than 2.5x expected
        # More precise check: should be close to expected value
        assert np.abs(np.max(velocity.v_data) - expected_velocity) < 0.02

    def test_buoyancy_3d_upward(self, macgrid_3d):
        """Test that buoyancy force is upward in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        density = np.ones((8, 12, 8), dtype=np.float32)

        # Apply buoyancy
        # Expected: buoyancy = alpha * density * scaling_factor * dt
        # scaling_factor = 64.0 / nx = 64.0 / 8 = 8.0
        # buoyancy = 0.1 * 1.0 * 8.0 * 0.1 = 0.08
        apply_buoyancy_force_3d(force, velocity, density, dt=0.1, nx=8, alpha=0.1)

        # v-velocity should increase (upward) by approximately the expected amount
        expected_velocity = 0.08
        assert np.max(velocity.v_data) > 0.02  # At least 25% of expected
        assert np.max(velocity.v_data) < 0.2   # Not more than 2.5x expected
        # More precise check: should be close to expected value
        assert np.abs(np.max(velocity.v_data) - expected_velocity) < 0.04

    def test_buoyancy_proportional_to_density(self, macgrid_2d):
        """Test that buoyancy force is proportional to density."""
        velocity1 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force1 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        density1 = np.ones((24, 16), dtype=np.float32)

        velocity2 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force2 = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        density2 = np.ones((24, 16), dtype=np.float32) * 2.0

        apply_buoyancy_force_2d(force1, velocity1, density1, dt=0.1, nx=16, alpha=0.1)
        apply_buoyancy_force_2d(force2, velocity2, density2, dt=0.1, nx=16, alpha=0.1)

        # Double density should give roughly double velocity change
        ratio = np.max(velocity2.v_data) / np.max(velocity1.v_data)
        assert 1.8 < ratio < 2.2  # Allow some tolerance


class TestVorticityConfinement:
    """Tests for vorticity confinement."""

    def test_vorticity_confinement_2d_no_effect_on_zero(self, macgrid_2d):
        """Test that vorticity confinement has no effect on zero vorticity."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        vorticity = np.zeros((24, 16), dtype=np.float32)

        # Store initial velocity
        u_initial = velocity.u_data.copy()
        v_initial = velocity.v_data.copy()

        # Apply vorticity confinement
        apply_vorticity_confinement_2d(
            force, velocity, vorticity, dx=1.0 / 16, dt=0.1, epsilon=0.1
        )

        # Velocity should not change much for zero vorticity
        assert np.allclose(velocity.u_data, u_initial, atol=1e-5)
        assert np.allclose(velocity.v_data, v_initial, atol=1e-5)

    def test_vorticity_confinement_3d_no_effect_on_zero(self, macgrid_3d):
        """Test that vorticity confinement has no effect on zero vorticity."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        vorticity = np.zeros((8, 12, 8, 3), dtype=np.float32)

        u_initial = velocity.u_data.copy()
        v_initial = velocity.v_data.copy()
        w_initial = velocity.w_data.copy()

        apply_vorticity_confinement_3d(
            force, velocity, vorticity, dx=1.0 / 8, dt=0.1, epsilon=0.1
        )

        assert np.allclose(velocity.u_data, u_initial, atol=1e-5)
        assert np.allclose(velocity.v_data, v_initial, atol=1e-5)
        assert np.allclose(velocity.w_data, w_initial, atol=1e-5)

    def test_vorticity_confinement_2d_amplifies_rotation(self, macgrid_2d):
        """Test that vorticity confinement amplifies rotational flow."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Create vorticity field with spatial variation (gradient)
        vorticity = np.zeros((24, 16), dtype=np.float32)
        # Create a vortex with gradient (stronger at center)
        cy, cx = 12, 8
        for y in range(24):
            for x in range(16):
                dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                vorticity[y, x] = np.exp(-(dist**2) / 10.0)  # Gaussian vortex

        # Store initial velocity (should be zero)
        u_initial = np.max(np.abs(velocity.u_data))
        v_initial = np.max(np.abs(velocity.v_data))
        assert u_initial == 0.0 and v_initial == 0.0

        # Apply confinement with significant epsilon
        apply_vorticity_confinement_2d(
            force, velocity, vorticity, dx=1.0 / 16, dt=0.1, epsilon=1.0
        )

        # Velocity should be affected by the vorticity gradient
        u_final = np.max(np.abs(velocity.u_data))
        v_final = np.max(np.abs(velocity.v_data))

        # At least one velocity component should have changed significantly
        # With epsilon=1.0, dt=0.1, and max vorticity ~1.0, expect forces on the order of 0.01-0.1
        total_velocity_change = u_final + v_final
        assert total_velocity_change > 0.005, f"Expected velocity change > 0.005, got {total_velocity_change}"
        assert total_velocity_change < 1.0, f"Velocity change too large: {total_velocity_change}"


class TestGravity:
    """Tests for gravity forces."""

    def test_gravity_2d_downward(self):
        """Test that gravity accelerates fluid downward in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Initial velocity is zero
        assert np.allclose(velocity.v_data, 0.0)

        # Apply gravity with g = -9.81 (downward)
        dt = 0.1
        g = -9.81
        apply_gravity_2d(force, velocity, dt=dt, g=g)

        # v-velocity should become negative (downward)
        # Expected: v_new = v_old + g * dt = 0 + (-9.81) * 0.1 = -0.981
        expected_v = g * dt
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

        # u-velocity should remain zero
        np.testing.assert_allclose(velocity.u_data, 0.0)

    def test_gravity_2d_upward(self):
        """Test gravity with positive g (upward acceleration)."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        dt = 0.1
        g = 5.0  # Upward
        apply_gravity_2d(force, velocity, dt=dt, g=g)

        # v-velocity should become positive (upward)
        expected_v = g * dt
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

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
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

    def test_gravity_3d_downward(self):
        """Test that gravity accelerates fluid downward in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)

        dt = 0.1
        g = -9.81
        apply_gravity_3d(force, velocity, dt=dt, g=g)

        expected_v = g * dt
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

        # u and w should remain zero
        np.testing.assert_allclose(velocity.u_data, 0.0)
        np.testing.assert_allclose(velocity.w_data, 0.0)

    def test_gravity_zero(self):
        """Test that zero gravity produces no acceleration."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        apply_gravity_2d(force, velocity, dt=0.1, g=0.0)

        # All velocities should remain zero
        np.testing.assert_allclose(velocity.u_data, 0.0)
        np.testing.assert_allclose(velocity.v_data, 0.0)


class TestExternalForce:
    """Tests for external force fields."""

    def test_external_force_2d_uniform(self):
        """Test uniform external force in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Create uniform force field
        force_u = np.ones((24, 17), dtype=np.float32) * 2.0
        force_v = np.ones((25, 16), dtype=np.float32) * 3.0

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should be force * dt
        expected_u = 2.0 * dt
        expected_v = 3.0 * dt

        np.testing.assert_allclose(velocity.u_data, expected_u, rtol=1e-5)
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

    def test_external_force_2d_varying(self):
        """Test spatially varying external force in 2D."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Create spatially varying force
        force_u = np.random.rand(24, 17).astype(np.float32)
        force_v = np.random.rand(25, 16).astype(np.float32)

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Each cell should have force * dt
        expected_u = force_u * dt
        expected_v = force_v * dt

        np.testing.assert_allclose(velocity.u_data, expected_u, rtol=1e-5)
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

    def test_external_force_2d_accumulation(self):
        """Test that external forces accumulate over multiple applications."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        force_u = np.ones((24, 17), dtype=np.float32) * 5.0
        force_v = np.ones((25, 16), dtype=np.float32) * 10.0

        dt = 0.1

        # Apply twice
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should accumulate: 2 * force * dt
        expected_u = 2.0 * 5.0 * dt
        expected_v = 2.0 * 10.0 * dt

        np.testing.assert_allclose(velocity.u_data, expected_u, rtol=1e-5)
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)

    def test_external_force_3d_uniform(self):
        """Test uniform external force in 3D."""
        velocity = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)
        force = MACGrid3D(nx=8, ny=12, nz=8, dx=1.0 / 8)

        force_u = np.ones((8, 12, 9), dtype=np.float32) * 2.0
        force_v = np.ones((8, 13, 8), dtype=np.float32) * 3.0
        force_w = np.ones((9, 12, 8), dtype=np.float32) * 4.0

        dt = 0.1
        apply_external_force_3d(force, velocity, force_u, force_v, force_w, dt=dt)

        expected_u = 2.0 * dt
        expected_v = 3.0 * dt
        expected_w = 4.0 * dt

        np.testing.assert_allclose(velocity.u_data, expected_u, rtol=1e-5)
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)
        np.testing.assert_allclose(velocity.w_data, expected_w, rtol=1e-5)

    def test_external_force_zero(self):
        """Test that zero external force produces no change."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Set initial velocity
        velocity.u_data.fill(5.0)
        velocity.v_data.fill(3.0)

        # Zero force
        force_u = np.zeros((24, 17), dtype=np.float32)
        force_v = np.zeros((25, 16), dtype=np.float32)

        apply_external_force_2d(force, velocity, force_u, force_v, dt=0.1)

        # Velocity should remain unchanged
        np.testing.assert_allclose(velocity.u_data, 5.0)
        np.testing.assert_allclose(velocity.v_data, 3.0)

    def test_external_force_negative(self):
        """Test external force with negative values."""
        velocity = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)
        force = MACGrid2D(nx=16, ny=24, dx=1.0 / 16)

        # Negative forces
        force_u = np.ones((24, 17), dtype=np.float32) * -10.0
        force_v = np.ones((25, 16), dtype=np.float32) * -5.0

        dt = 0.1
        apply_external_force_2d(force, velocity, force_u, force_v, dt=dt)

        # Velocity should be negative
        expected_u = -10.0 * dt
        expected_v = -5.0 * dt

        np.testing.assert_allclose(velocity.u_data, expected_u, rtol=1e-5)
        np.testing.assert_allclose(velocity.v_data, expected_v, rtol=1e-5)
