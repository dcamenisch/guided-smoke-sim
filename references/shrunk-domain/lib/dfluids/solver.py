import torch

from .grid import GridFactory, GridOperatorFactory
from .shapes import ShapeFactory
from .advection import Advection
from .extforces import ExtForces
from .pressure import PressureProjection
from .ls_solver import LsSolverFactory


class FluidSolver(object):
    def __init__(self, grid_size, dt=1.0, dtype=torch.float32):
        """Initialize a fluid solver.
        Args:
            grid_size(list): in the order of x, y, (z), i.e. [W, H, (D)]
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = list(grid_size)
        self.dim = len(grid_size)
        self.dtype = dtype
        if self.dim == 2:
            W, H = grid_size
            self.real_size = torch.Size([1, H, W])
            self.mac_size = torch.Size([2, H + 1, W + 1])
            self.node_size = torch.Size([1, H + 1, W + 1])
            self.level_size = torch.Size([1, H + 1, W + 1])
        elif self.dim == 3:
            W, H, D = grid_size
            self.real_size = torch.Size([1, D, H, W])
            self.mac_size = torch.Size([3, D + 1, H + 1, W + 1])
            self.node_size = torch.Size([3, D + 1, H + 1, W + 1])
            self.level_size = torch.Size([1, D + 1, H + 1, W + 1])
        else:
            raise ValueError("Dimension of the grid must be 2 or 3.")
        self.dt = dt
        self.frame = 0
        self.time_per_frame = 0.0
        self.time_total = 0.0
        self.frame_length = 1.0
        self._create_factories()

    def _create_factories(self):
        self.grid_factory = GridFactory(solver=self)
        self.grid_operator_factory = GridOperatorFactory(solver=self)
        self.shape_factory = ShapeFactory(solver=self)
        self.ls_solver_factory = LsSolverFactory(solver=self)

    def step(self):
        """Advances the solver one time step"""
        self.time_per_frame += self.dt
        self.time_total += self.dt

        if self.time_per_frame >= self.frame_length:
            self.frame += 1

            # re-calculate total time to prevent drift
            self.time_total = self.frame * self.frame_length
            self.time_per_frame = 0

    def create_grid(self, grid_type):
        """create grid according to grid_type"""
        return self.grid_factory.create(grid_type=grid_type, dtype=self.dtype)

    def create_grid_operator(self, grid_type):
        """create grid operator to transform grid"""
        return self.grid_operator_factory.create(grid_type=grid_type)

    def create_shape(self, shape_type, **kwargs):
        """create grid according to grid_type"""
        return self.shape_factory.create(shape_type=shape_type, **kwargs)

    def create_advection(self):
        """create advection operators wrapper"""
        return Advection(solver=self)

    def create_extforces(self):
        """create extforces operators wrapper"""
        return ExtForces(solver=self)

    def create_pressure_projection(self, apply_precon=False, cg_accu=1e-6):
        """create pressure projection operators wrapper"""
        return PressureProjection(
            solver=self, apply_precon=apply_precon, cg_accu=cg_accu
        )

    def create_ls_solver(self, apply_precon=False):
        """create linear system solver"""
        return self.ls_solver_factory.create(apply_precon=apply_precon)

    def get_dtype(self):
        return self.dtype

    def get_dt(self):
        return self.dt

    def get_dx(self):
        return 1.0 / max(self.grid_size)

    def get_time(self):
        return self.time_total

    def get_frame(self):
        return self.frame

    def is_2d(self):
        return self.dim == 2

    def is_3d(self):
        return self.dim == 3

    def get_grid_size(self):
        return self.grid_size

    def get_torch_size_flag(self):
        return self.real_size

    def get_torch_size_real(self):
        return self.real_size

    def get_torch_size_mac(self):
        return self.mac_size

    def get_torch_size_level(self):
        return self.level_size

    def get_torch_size_node(self):
        return self.node_size

    def get_num_cells(self):
        n_cells = 1
        for size in self.grid_size:
            n_cells *= size
        return n_cells

    def get_device(self):
        return self.device

    def set_device(self, device="cuda"):
        if device not in ["cuda", "cpu"]:
            raise ValueError(
                "FluidSolver.set_device: device {}"
                "must be either cuda or cpu.".format(device)
            )
        self.device = torch.device(device)
        # reset the device of the factories
        self._create_factories()
