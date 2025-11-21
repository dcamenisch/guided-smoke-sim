import torch

from .grid import CellType, GridType


class PressureProjection(object):
    def __init__(self, solver, apply_precon=False, cg_accu=1e-6):
        """
        Args:
            solver(FluidSolver): FluidSolver object in current simulation,
                to provide basic information such as dx and dt.
        """
        self.solver = solver
        self.device = self.solver.get_device()
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)
        self.ls_solver = self.solver.create_ls_solver(apply_precon=apply_precon)
        self.cg_accu = cg_accu
        self.apply_precon = apply_precon
        self.A = None
        self.ext = self.solver.create_extforces()

    def solve_pressure(self, flags, phi_obs, vel):
        rhs = self._compute_pressure_rhs(flags, phi_obs, vel)
        pressure = self._solve_poisson_system(flags, phi_obs, rhs)
        vel = self._correct_velocity(vel, pressure)
        vel = self.ext.set_wall_bcs(phi_obs, vel)
        return vel, pressure

    def _compute_pressure_rhs(self, flags, phi_obs, vel):
        fractions = self.level_op.get_fractions(phi_obs)
        if self.solver.is_2d():
            rhs = torch.where(
                flags == int(CellType.FLUID),
                -(
                    fractions[0:1, :-1, 1:] * vel[0:1, :-1, 1:]
                    - fractions[0:1, :-1, :-1] * vel[0:1, :-1, :-1]
                    + fractions[1:2, 1:, :-1] * vel[1:2, 1:, :-1]
                    - fractions[1:2, :-1, :-1] * vel[1:2, :-1, :-1]
                ),
                self.solver.create_grid(GridType.REAL),
            )
        else:
            rhs = torch.where(
                flags == int(CellType.FLUID),
                -(
                    fractions[0:1, :-1, :-1, 1:] * vel[0:1, :-1, :-1, 1:]
                    - fractions[0:1, :-1, :-1, :-1] * vel[0:1, :-1, :-1, :-1]
                    + fractions[1:2, :-1, 1:, :-1] * vel[1:2, :-1, 1:, :-1]
                    - fractions[1:2, :-1, :-1, :-1] * vel[1:2, :-1, :-1, :-1]
                    + fractions[2:3, 1:, :-1, :-1] * vel[2:3, 1:, :-1, :-1]
                    - fractions[2:3, :-1, :-1, :-1] * vel[2:3, :-1, :-1, :-1]
                ),
                self.solver.create_grid(GridType.REAL),
            )
        return rhs

    def _solve_poisson_system(self, flags, phi_obs, rhs):
        if self.A is None:
            self.A = self.ls_solver.build_pressure_matrix(flags, phi_obs)
        if self.apply_precon and not self.ls_solver.precon_built():
            self.ls_solver.build_preconditioner(self.A)
        pressure = self.ls_solver.solve_linear_system(
            rhs, self.A, direct=False, tol=self.cg_accu
        )
        return pressure

    def _correct_velocity(self, vel, pressure):
        pressure_grad = self.real_op.get_gradient(pressure)
        vel_corrected = vel - pressure_grad
        return vel_corrected
