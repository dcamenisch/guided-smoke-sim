import numpy as np
import torch

from solver import FluidSolver
from grid import GridType
from shapes import ShapeType
from . import utils


def main():
    dim = 2
    res = 128
    total_steps = 100

    if dim == 2:
        gs = (res, res)
        source_center = torch.Tensor([0.0, -1.0])
        source_size = torch.Tensor([0.2, 0.2])
    else:
        gs = (res, res, res)
        source_center = torch.Tensor([0.0, -1.0, 0.0])
        source_size = torch.Tensor([0.2, 0.2, 0.2])

    s = FluidSolver(grid_size=gs)

    flags = s.create_grid(GridType.FLAG)
    vel = s.create_grid(GridType.MAC)
    density = s.create_grid(GridType.REAL)

    ext = s.create_extforces()
    adv = s.create_advection()
    pre = s.create_pressure_projection()

    real_op = s.create_grid_operator(GridType.REAL)
    level_op = s.create_grid_operator(GridType.LEVEL)

    flags, phi_obs = ext.init_flags(phi=True, boundary_width=0, wall="xXyY")
    obs_size = torch.Tensor([0.2])
    obs_center = torch.Tensor([0.0, -0.5])
    sphere = s.create_shape(ShapeType.SPHERE)
    phi_sphere = sphere.compute_level_set(center=obs_center, size=obs_size)
    phi_obs = level_op.union(phi_sphere, phi_obs)

    source = s.create_shape(ShapeType.BOX)
    source_value = 1.0
    gravity = [0, -4e-3, 0]

    utils.mkdir("./demo_data/")
    for step in range(total_steps):
        print("demo: simulate at frame {}.".format(step))
        density = source.apply_to_grid(
            density, source_value, center=source_center, size=source_size
        )
        density = adv.advect_semi_lagrange(flags, phi_obs, vel, density, order=1)
        vel = adv.advect_semi_lagrange(flags, phi_obs, vel, vel, order=1)
        vel = ext.add_buoyancy(flags, phi_obs, vel, density, gravity)
        vel, pressure = pre.solve_pressure(flags, phi_obs, vel)
        s.step()
        real_op.save_img(density, filename="./demo_data/{:03}.png".format(step + 1))


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=1000, suppress=False)
    torch.set_printoptions(precision=3, linewidth=1000, profile="full")
    main()
