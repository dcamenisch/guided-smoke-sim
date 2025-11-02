import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
from lib.dfluids.grid import GridType
from lib.dfluids.shapes import ShapeType
from lib.dfluids.solver import FluidSolver
import lib.dfluids.utils as utils


class Simulation(object):
    """Base class for all diff-fluids simulations"""

    def __init__(self, opt):
        self.opt = opt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize solver
        if len(opt.res) == 1:
            if opt.dim == 2:
                gs = (opt.res[0], opt.res[0])
            else:
                gs = (opt.res[0], opt.res[0], opt.res[0])
        else:
            assert len(opt.res) == opt.dim
            gs = opt.res
        self.solver = FluidSolver(grid_size=gs, dt=opt.dt)
        # set up operator wrappers
        self.adv = self.solver.create_advection()
        self.ext = self.solver.create_extforces()
        self.pre = self.solver.create_pressure_projection(
            apply_precon=opt.apply_precon, cg_accu=opt.cg_accu
        )
        # set up grid operators
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.mac_op = self.solver.create_grid_operator(GridType.MAC)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)
        self.node_op = self.solver.create_grid_operator(GridType.NODE)
        # initiate flag domain
        assert len(opt.wall_bnd) == opt.dim * 2
        self.flags, self.phi_bnd = self.ext.init_flags(phi=True, wall=opt.wall_bnd)
        if opt.use_obstacle:
            self.obs_center = torch.Tensor(opt.obs_center)
            self.obs_size = torch.Tensor(opt.obs_size)
            obstacle = self.solver.create_shape(ShapeType.SPHERE)
            phi_obs = obstacle.compute_level_set(
                center=self.obs_center, size=self.obs_size
            )
            self.phi_obs = self.level_op.union(phi_obs, self.phi_bnd)
        else:
            self.phi_obs = self.phi_bnd
        # set source
        self.source_center = torch.Tensor(opt.source_center)
        self.source_size = torch.Tensor(opt.source_size)
        if opt.source_shape == "BOX":
            self.source = self.solver.create_shape(ShapeType.BOX)
        elif opt.source_shape == "SPHERE":
            self.source = self.solver.create_shape(ShapeType.SPHERE)
        else:
            raise ValueError(
                f"Simulation: opt.source_shape {opt.source_shape} not recognized."
            )
        self.source_value = self.opt.source_value
        # set buoyancy force
        if len(opt.buoyancy) == 1:
            self.gravity = [0, opt.buoyancy[0], 0]
        elif len(opt.buoyancy) == 3:
            self.gravity = opt.buoyancy
        else:
            raise ValueError(
                f"Simulation: opt.buoyancy {opt.bouyancy} must have length 1 or 3."
            )
        # set init vel and density as zeros
        self.vel = self.solver.create_grid(GridType.MAC)
        self.density = self.solver.create_grid(GridType.REAL)
        # load fields from npz file
        if opt.load_npz:
            filename = os.path.join(opt.save_path, "npz", f"{opt.start_step:03d}.npz")
            keys = np.load(filename).files
            assert os.path.exists(filename)
            self.force = self.mac_op.load_npz(filename=filename, key="force")
            if "vel" in keys:
                self.vel = self.real_op.load_npz(filename=filename, key="vel")
            if "density" in keys:
                self.density = self.real_op.load_npz(filename=filename, key="density")
        else:
            # set force as uniform wind force
            if opt.dim == 2:
                wind = torch.Tensor([opt.wind_x, opt.wind_y])
            else:
                wind = torch.Tensor([opt.wind_x, opt.wind_y, opt.wind_z])
            self.force = self.mac_op.get_const_grid(wind)

    def simulate_save(self):
        for step in range(
            self.opt.start_step, self.opt.start_step + self.opt.total_steps
        ):
            print("Dfluids simulation: simulate_save at frame {}".format(step))
            if step < self.opt.source_steps:
                self.density = self.source.apply_to_grid(
                    self.density,
                    self.source_value,
                    center=self.source_center,
                    size=self.source_size,
                )
            self.density = self.adv.advect_semi_lagrange(
                self.flags,
                self.phi_obs,
                self.vel,
                self.density,
                order=self.opt.advection_order,
                rk_order=self.opt.advection_rk_order,
            )
            self.vel = self.adv.advect_semi_lagrange(
                self.flags,
                self.phi_obs,
                self.vel,
                self.vel,
                order=self.opt.advection_order,
                rk_order=self.opt.advection_rk_order,
            )

            self.vel = self.ext.add_buoyancy(
                self.flags, self.phi_obs, self.vel, self.density, self.gravity
            )
            self.vel = self.ext.add_extforces(
                self.flags, self.phi_obs, self.vel, self.force
            )

            self.vel, self.pressure = self.pre.solve_pressure(
                self.flags, self.phi_obs, self.vel
            )

            self._save_quantity(step=step + 1)

            self.solver.step()

    def _save_all_img(self, step):
        # save density
        utils.mkdir(os.path.join(self.opt.save_path, "density"))
        filename_density = os.path.join(
            self.opt.save_path, "density", f"{step:03d}.png"
        )
        self.real_op.save_img(
            self.density,
            filename=filename_density,
            colormap="OrRd" if self.solver.is_2d() else "RGB",
        )
        # save velocity
        utils.mkdir(os.path.join(self.opt.save_path, "vel"))
        filename_vel = os.path.join(self.opt.save_path, "vel", f"{step:03d}.png")
        self.mac_op.save_img(self.vel, filename=filename_vel)
        # save pressure
        if self.opt.save_pressure:
            utils.mkdir(os.path.join(self.opt.save_path, "pressure"))
            filename_pressure = os.path.join(
                self.opt.save_path, "pressure", f"{step:03d}.png"
            )
            self.real_op.save_img(
                self.pressure, filename=filename_pressure, colormap="RdBu"
            )
        # save divergence of velocity
        if self.opt.save_div:
            utils.mkdir(os.path.join(self.opt.save_path, "div"))
            filename_div = os.path.join(self.opt.save_path, "div", f"{step:03d}.png")
            self.real_op.save_img(
                self.mac_op.get_divergence(self.vel),
                filename=filename_div,
                colormap="RDBU_BAR",
            )
        # save vorticity
        if self.opt.save_vort:
            utils.mkdir(os.path.join(self.opt.save_path, "vort"))
            filename_vort = os.path.join(self.opt.save_path, "vort", f"{step:03d}.png")
            self.node_op.save_img(
                self.mac_op.get_curl(self.vel),
                filename=filename_vort,
                colormap="RdBu",
            )

    def _save_all_npz(self, step):
        """Save all quantities as single npz for each step.
           Whether the npz is compressed or not depends on self.save_compressed.

        Args:
            step (int): The name of the file, which is the step (int).

        """
        utils.mkdir(os.path.join(self.opt.save_path, "npz"))
        filename = os.path.join(self.opt.save_path, "npz", f"{step:03d}.npz")
        save_fcn = np.savez_compressed if self.opt.save_compressed else np.savez
        save_fcn(
            filename,
            density=self.real_op.grid2array(self.density),
            vel=self.mac_op.grid2array(self.vel),
            force=self.mac_op.grid2array(self.force),
            pressure=self.real_op.grid2array(self.pressure)
            if self.opt.save_pressure
            else None,
            divergence=self.mac_op.grid2array(self.mac_op.get_divergence(self.vel))
            if self.opt.save_div
            else None,
            vorticity=self.mac_op.grid2array(self.mac_op.get_curl(self.vel))
            if self.opt.save_vort
            else None,
        )

    def _save_quantity(self, step):
        """Save all quantities as images and npz.
           Whether to save as npz depends on self.opt.save_npz.

        Args:
            step (int): The name of the file, which is the step (int).

        """
        self._save_all_img(step)
        if self.opt.save_npz:
            self._save_all_npz(step)
