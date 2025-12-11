import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())
from lib.PyTorch_LBFGS import FullBatchLBFGS
from lib.dfluids.solver import FluidSolver
from lib.dfluids.grid import GridType
import lib.dfluids.utils as utils


class BaseOptimization(object):
    def __init__(self, opt):
        self.opt = opt
        if len(self.opt.res) == 1:
            self.grid_size = list([self.opt.res[0] for _ in range(self.opt.dim)])
        else:
            assert len(self.opt.res) == self.opt.dim
            self.grid_size = self.opt.res
        self.solver = FluidSolver(grid_size=self.grid_size, dt=opt.dt)
        self.device = self.solver.get_device()
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.mac_op = self.solver.create_grid_operator(GridType.MAC)
        self.node_op = self.solver.create_grid_operator(GridType.NODE)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)

    def set_optimization(self, phase):
        """Processes and sets up the details of optimization for the model."""
        raise NotImplementedError

    def simulate(self):
        """Runs a simulation using current parameters of the model
        and saves all the states for visulization.
        """
        raise NotImplementedError

    def compute_objective(self, phase):
        """Defines the way to compute objective"""
        raise NotImplementedError

    def optimize_parameters(self, phase, is_first_step=False):
        """Defines the procedure to optimize parameters of the model."""

        def closure():
            self.optimizer.zero_grad()
            self.compute_objective(phase)
            self.objective.backward()
            return self.objective

        def closure_pytorch_lbfgs():
            self.optimizer.zero_grad()
            self.compute_objective(phase)
            return self.objective

        message = None
        fail = False
        if self.opt.optimizer == "pytorch_lbfgs":
            if is_first_step:
                closure_pytorch_lbfgs()
                self.objective.backward()
            options = {
                "closure": closure_pytorch_lbfgs,
                "current_objective": self.objective,
                "max_ls": self.opt.lbfgs_max_ls,
                "inplace": self.opt.lbfgs_inplace,
            }
            if self.opt.lbfgs_line_search_fcn == "None":
                self.optimizer.step(options)
            elif self.opt.lbfgs_line_search_fcn == "Wolfe":
                _, _, t, _, F_eval, G_eval, _, fail = self.optimizer.step(options)
                message = (
                    "# fcn eval: {}, # grad eval: {}, "
                    "final step size: {}, failure of line search: {}".format(
                        F_eval, G_eval, t, fail
                    )
                )
            elif self.opt.lbfgs_line_search_fcn == "Armijo":
                options.update({"damping": True})
                _, t, _, F_eval, _, fail = self.optimizer.step(options)
                message = (
                    "# fcn eval: {}, final step size: {}, "
                    "failure of line search: {}".format(F_eval, t, fail)
                )
                self.objective.backward()
        else:
            self.optimizer.step(closure)
        return message, fail

    def create_model(self):
        raise NotImplementedError

    def get_current_objective(self):
        """Returns all current objectives in a python dict."""
        raise NotImplementedError

    def load_state_dict(self, label, format="pth"):
        """Load state dict from pth file.

        Args:
            label (str): The label of iteration to be loaded.
            format (str): The format of file.

        Raises:
            ValueError: If the format is not "pth"

        Returns:
            state_dict

        """
        if format == "pth":
            load_path = os.path.join(
                self.opt.save_path, "param", "itr_{}.pth".format(label)
            )
            state_dict = torch.load(load_path, map_location=self.device)
        else:
            raise ValueError("load state dict currently only support pth format.")
        return state_dict

    def save(self, label):
        """Save state dict as pth.

        Args:
            label (str): The label of iteration.

        """
        utils.mkdir(os.path.join(self.opt.save_path, "param"))
        save_path = os.path.join(
            self.opt.save_path, "param", "itr_{}.pth".format(label)
        )
        torch.save(self.model.state_dict(), save_path)

    def create_optimizer(self, parameters, lr):
        """Creates a torch.optim.Optimizer for gradient-based optimization.

        Args:
            parameters: The parameters of the model.
            lr (float): The learning rate.

        Returns:
            optimizer

        """

        return FullBatchLBFGS(
            parameters,
            lr=lr,
            line_search=self.opt.lbfgs_line_search_fcn,
            history_size=self.opt.lbfgs_history_size,
        )
