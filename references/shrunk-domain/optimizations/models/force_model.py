import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.getcwd())
from lib.dfluids.shapes import ShapeType
from lib.dfluids.grid import GridType


class ForceModel(torch.nn.Module):
    def __init__(self, solver, keyframe_opt, use_stream_fcn=False, use_sim_wind=False):
        """Initialize the shaping wind model. The parameter(s) of the model
        is the control force(s).

        Args:
            solver: FluidSolver object
            opt: ApplicationOption object
            n_forces(int): number of force as parameter for this model
            step2label(fcn): function that maps time step
                to parameter label
        """
        super().__init__()
        self.keyframe_opt = keyframe_opt
        self.solver = solver
        self.device = self.solver.get_device()
        self.use_stream_fcn = use_stream_fcn
        self.node_op = self.solver.create_grid_operator(GridType.NODE)
        self._initialize_simulation()
        self._initialize_parameters_()
        self.use_sim_wind = use_sim_wind
        if self.use_sim_wind:
            if self.solver.is_2d():
                wind = torch.Tensor([keyframe_opt.wind_x, keyframe_opt.wind_y])
            else:
                wind = torch.Tensor(
                    [keyframe_opt.wind_x, keyframe_opt.wind_y, keyframe_opt.wind_z]
                )
            self.wind_force = self.mac_op.get_const_grid(wind)

    def _initialize_parameters_(self):
        grid_type = GridType.NODE if self.use_stream_fcn else GridType.MAC
        self.param = torch.nn.Parameter(self.solver.create_grid(grid_type))

    def _initialize_simulation(self):
        self.ext = self.solver.create_extforces()
        self.adv = self.solver.create_advection()
        try:
            self.pre = self.solver.create_pressure_projection(
                apply_precon=self.keyframe_opt.apply_precon,
                cg_accu=self.keyframe_opt.cg_accu,
            )
        except AttributeError:
            self.pre = self.solver.create_pressure_projection()
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.mac_op = self.solver.create_grid_operator(GridType.MAC)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)
        if self.keyframe_opt.source_shape.lower() == "box":
            self.source = self.solver.create_shape(ShapeType.BOX)
        elif self.keyframe_opt.source_shape.lower() == "sphere":
            self.source = self.solver.create_shape(ShapeType.SPHERE)
        else:
            raise ValueError("source shape from keyframe_opt not supported.")
        self.obs = self.solver.create_shape(ShapeType.SPHERE)
        try:
            wall = self.keyframe_opt.wall_bnd
        except AttributeError:
            wall = "xXyY" if self.solver.is_2d() else "xXyYzZ"
        self.flags, self.phi_bnd = self.ext.init_flags(
            phi=True, boundary_width=0, wall=wall
        )
        self.source_value = self.keyframe_opt.source_value
        self.source_center = torch.Tensor(self.keyframe_opt.source_center)
        self.source_size = torch.Tensor(self.keyframe_opt.source_size)
        self.source_steps = self.keyframe_opt.source_steps
        if isinstance(self.keyframe_opt.buoyancy, float):
            self.gravity = torch.Tensor([0, self.keyframe_opt.buoyancy, 0])
        elif len(self.keyframe_opt.buoyancy) == 1:
            self.gravity = torch.Tensor([0, self.keyframe_opt.buoyancy[0], 0])
        elif len(self.keyframe_opt.buoyancy) == 3:
            self.gravity = torch.Tensor(self.keyframe_opt.buoyancy)
        else:
            raise ValueError(
                "ShapingWindMode:_initialize_simulation(): "
                "self.keyframe_opt.buoyancy has a wrong format."
            )
        self.advection_order = self.keyframe_opt.advection_order
        try:
            if self.opt.advection_rk_order is None:
                raise AttributeError
            else:
                self.advection_rk_order = self.opt.advection_rk_order
        except AttributeError:
            self.advection_rk_order = self.keyframe_opt.advection_rk_order
        if self.keyframe_opt.use_obstacle:
            self.obs_size = torch.Tensor(self.keyframe_opt.obs_size)
            self.obs_center = torch.Tensor(self.keyframe_opt.obs_center)
            self.phi_obs = self.obs.compute_level_set(
                center=self.obs_center, size=self.obs_size
            )
            self.phi_obs = self.level_op.union(self.phi_bnd, self.phi_obs)
        else:
            self.phi_obs = self.phi_bnd

    def reorganize_parameters_(self, **kwargs):
        pass

    def compute_force(self, *args, **kwargs):
        if self.use_stream_fcn:
            self.force = self.node_op.get_curl(self.param)
        else:
            # + 0.0 to avoid registering self.force as nn.Parameter
            self.force = self.param + 0.0

    def get_force(self):
        assert hasattr(self, "force")
        return self.force

    def forward(self, density, vel, t, force_rescale_factor=1.0):
        assert self.force is not None
        force = self.force * force_rescale_factor
        if t < self.source_steps:
            density = self.source.apply_to_grid(
                density,
                self.source_value,
                center=self.source_center,
                size=self.source_size,
            )
        density = self.adv.advect_semi_lagrange(
            self.flags,
            self.phi_obs,
            vel,
            density,
            order=self.advection_order,
            rk_order=self.advection_rk_order,
        )
        vel = self.adv.advect_semi_lagrange(
            self.flags,
            self.phi_obs,
            vel,
            vel,
            order=self.advection_order,
            rk_order=self.advection_rk_order,
        )
        vel = self.ext.add_buoyancy(
            self.flags, self.phi_obs, vel, density, self.gravity
        )
        if self.use_sim_wind:
            vel = self.ext.add_extforces(self.flags, self.phi_obs, vel, self.wind_force)
        vel = self.ext.add_extforces(self.flags, self.phi_obs, vel, force)
        vel, pressure = self.pre.solve_pressure(self.flags, self.phi_obs, vel)
        return density, vel
