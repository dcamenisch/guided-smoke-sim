"""Physics module containing force implementations."""

from physics.buoyancy import apply_buoyancy_force_2d, apply_buoyancy_force_3d
from physics.gravity import apply_gravity_2d, apply_gravity_3d
from physics.external_force import apply_external_force_2d, apply_external_force_3d
from physics.vorticity_confinement import (
    apply_vorticity_confinement_2d,
    apply_vorticity_confinement_3d,
)

__all__ = [
    "apply_buoyancy_force_2d",
    "apply_buoyancy_force_3d",
    "apply_gravity_2d",
    "apply_gravity_3d",
    "apply_external_force_2d",
    "apply_external_force_3d",
    "apply_vorticity_confinement_2d",
    "apply_vorticity_confinement_3d",
]
