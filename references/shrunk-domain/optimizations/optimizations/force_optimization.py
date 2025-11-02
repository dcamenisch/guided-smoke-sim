import os
import sys
import torch
import torchvision
import numpy as np

sys.path.append(os.getcwd())
from optimizations.optimizations.base_optimization import BaseOptimization
from optimizations.objectives import L2Regularizer, TotalVariation
from optimizations.models.force_model import ForceModel
from lib.dfluids.grid import GridType
import lib.dfluids.utils as utils
from lib.dfluids.solver import FluidSolver


class ForceOptimization(BaseOptimization):
    """Baseline learner for keyframe wind. Assumes there is only one phase
    of learning process.
    """

    def __init__(self, opt):
        super().__init__(opt)
        if len(self.opt.phase_itrs) != len(opt.lr):
            raise ValueError(
                (
                    f"Number of optimization phases {len(self.opt.phase_itrs)} "
                    "is different with number of learning rates {len(self.opt.lr)}"
                )
            )
        # asseartion between optimization sim and keyframe sim
        if self.opt.dim != self.opt.keyframe_opt.dim:
            raise ValueError("Optimization has different dim with simulation.")
        if len(self.opt.keyframe_opt.res) == 1:
            keyframe_grid_size = list(
                [self.opt.keyframe_opt.res[0] for _ in range(self.opt.dim)]
            )
        else:
            assert len(self.opt.keyframe_opt.res) == self.opt.dim
            keyframe_grid_size = self.opt.keyframe_opt.res
        if self.grid_size != keyframe_grid_size:
            raise ValueError("Optimization  has different grid size with simulation.")

        # set up simulation range
        self.n_steps = self.opt.keyframes[-1] - self.opt.init_frame + 1
        self.start_step = 0
        self.end_step = self.n_steps - 1

        # initialize density field
        self._init_density()
        # initialize velocity field
        self._init_vel()

        # create model
        self.create_model()

        # set up major objective functions
        if self.opt.distance_metric == "l1":
            self.objective_pointwise = torch.nn.L1Loss(reduction="mean")
        elif self.opt.distance_metric == "l2":
            self.objective_pointwise = torch.nn.MSELoss(reduction="mean")
        self.objective_list = [
            self.objective_pointwise for i in range(len(self.opt.phase_itrs))
        ]
        # set up logging objectives
        if self.opt.log_mse or self.opt.log_normed_mse:
            self.objective_mse = torch.nn.MSELoss()

        # set up regularizers
        if self.opt.use_tikinov_reg:
            self.tikinov = L2Regularizer()
        if self.opt.use_tv_reg:
            self.tv = TotalVariation()
        self.lambd_tikinov_reg = self.opt.lambd_tikinov_reg
        self.lambd_tv_reg = self.opt.lambd_tv_reg

        # set targets
        self.targets = {}
        target_vis_list = []
        for keyframe in self.opt.keyframes:
            try:
                target = self.real_op.load_npz(
                    filename=os.path.join(
                        self.opt.keyframes_load_path,
                        "npz",
                        f"{keyframe:03d}.npz",
                    ),
                    key="density",
                )
                target_vis = (
                    target if self.opt.dim == 2 else self.real_op.get_rendered(target)
                )
            # For loading rendered image as target in 3D
            except RuntimeError:
                if len(self.opt.res) == 1:
                    gs_2d = [self.opt.res[0], self.opt.res[0]]
                else:
                    gs_2d = [self.opt.res[0], self.opt.res[1]]
                solver_2d = FluidSolver(grid_size=gs_2d)
                real_op_2d = solver_2d.create_grid_operator(GridType.REAL)
                target = real_op_2d.load_npy(
                    filename=os.path.join(
                        self.opt.keyframes_load_path,
                        "npz",
                        f"{keyframe:03d}.npz",
                    )
                )
                target_vis = target
            self.targets.update({keyframe: target})
            target_vis_list.append(torch.flip(target_vis, dims=(1,)))

        # visualize grid of targets
        torchvision.utils.save_image(
            target_vis_list,
            os.path.join(self.opt.save_path, "targets.png"),
            nrow=len(self.opt.keyframes),
            padding=0,
        )

    def _init_density(self):
        """Initialize density."""
        # load initial density from simulation dir
        if self.opt.load_init_density_sim:
            self.density_init = self.real_op.load_npy(
                filename=os.path.join(
                    self.opt.keyframes_load_path,
                    "d",
                    "{:03d}.npy".format(self.opt.init_frame),
                )
            )
        # load initial density from optimization dir
        elif self.opt.load_init_density:
            self.density_init = self.real_op.load_npy(
                filename=os.path.join(
                    self.opt.save_path, "d", "{:03d}.npy".format(self.opt.init_frame)
                )
            )
        else:
            self.density_init = self.solver.create_grid(GridType.REAL)

    def _init_vel(self):
        """Initialize velocity."""
        # load initial velocity from simulation dir
        if self.opt.load_init_vel_sim:
            self.vel_init = self.mac_op.load_npy(
                filename=os.path.join(
                    self.opt.keyframes_load_path,
                    "v",
                    "{:03d}.npy".format(self.opt.init_frame),
                ),
            )
        # load initial velocity from optimization dir
        elif self.opt.load_init_vel:
            self.vel_init = self.mac_op.load_npy(
                filename=os.path.join(
                    self.opt.save_path, "v", "{:03d}.npy".format(self.opt.init_frame)
                ),
                trim_boundary=False,
            )
        else:
            self.vel_init = self.solver.create_grid(GridType.MAC)

    def create_model(self):
        """Create 'ForceModel' as the model."""
        self.model = ForceModel(
            self.solver,
            self.opt.keyframe_opt,
            use_stream_fcn=self.opt.use_stream_fcn,
            use_sim_wind=self.opt.use_sim_wind,
        )

    def set_optimization(self, phase):
        """Set optimizer for every phase.

        Args:
            phase (int): The index of phase.

        """
        self.optimizer = self.create_optimizer(
            parameters=self.model.parameters(), lr=self.opt.lr[phase]
        )

    def load(self, label, format="pth"):
        state_dict = BaseOptimization.load_state_dict(self, label, format)
        self.model.load_state_dict(state_dict)

    def compute_objective(self, phase):
        """Defines the way to compute objective.

        Args:
            phase (int): The index of phase.

        Returns:
            self.objective (torch.Tensor)

        """
        self.objective = 0.0
        self.tikinov_objective = 0.0
        self.mse_objective = 0.0
        self.mse_normed_objective = 0.0
        self.tv_objective = 0.0

        density = self.density_init.detach()
        vel = self.vel_init.detach()

        self.model.compute_force()

        # simulation loop
        for t in range(self.start_step, self.end_step):
            density, vel = self.model(density, vel, t)
            frame = t + 1 + self.opt.init_frame
            # computes major objectives
            if frame in list(self.targets.keys()):
                target = self.targets[frame]
                input = density
                # 2D target in 3D
                if self.opt.dim == 3 and len(target.size()) == 3:
                    input = self.real_op.get_rendered(input)
                self.objective += self.objective_list[phase](input, target)
                # logs objectives
                if self.opt.log_mse:
                    self.mse_objective += self.objective_mse(input, target)
                if self.opt.log_normed_mse:
                    ratio = np.prod(target.size()) / torch.sum(target > 0).item()
                    self.mse_normed_objective += (
                        self.objective_mse(input, target) * ratio
                    )

        # computes regularization losses
        for force in self.model.get_force():
            if self.opt.use_tikinov_reg:
                self.tikinov_objective += self.regularizer(force)
            if self.opt.use_tv_reg:
                self.tv_objective += self.tv(force)

        # accumulates all objectives
        self.objective = (
            self.objective
            + self.lambd_tikinov_reg * self.tikinov_objective
            + self.lambd_tv_reg * self.tv_objective
        )

        return self.objective

    def get_current_objective(self):
        """Return the objective dict.

        Returns:
            objective_dict (dict)
        """
        objective_dict = {"total_objective": self.objective.item()}
        if self.opt.log_mse:
            objective_dict.update({"mse_objective": self.mse_objective.item()})
        if self.opt.log_normed_mse:
            objective_dict.update(
                {"mse_normed_objective": self.mse_normed_objective.item()}
            )
        if self.opt.use_tikinov_reg:
            objective_dict.update({"tikinov_objective": self.tikinov_objective.item()})
        if self.opt.use_tv_reg:
            objective_dict.update({"tv_objective": self.tv_objective.item()})
        return objective_dict

    def simulate(self):
        self.model.compute_force()
        if self.opt.load_param_label is not None:
            self.param_label = self.opt.load_param_label
        else:
            self.param_label = "init"
        # visualize model parameters
        if self.opt.save_force:
            force = self.model.get_force()
            force_dirname = os.path.join(
                self.opt.save_path, f"sim_{self.param_label}/force"
            )
            utils.mkdir(force_dirname)
            # save lic plot
            self.mac_op.save_img(
                self.mac_op.get_centered(force),
                filename=os.path.join(force_dirname, "force_lic.png"),
                colormap="LIC",
            )
            # save arrow plot
            self.mac_op.save_img(
                self.mac_op.get_centered(force),
                filename=os.path.join(force_dirname, "force_arrow.png"),
                colormap="ARROW",
            )
            if self.opt.save_npz:
                np.savez(
                    os.path.join(force_dirname, "force.npz"),
                    force=self.mac_op.grid2array(force),
                )
            if self.opt.save_div:
                self.real_op.save_img(
                    self.mac_op.get_divergence(force),
                    filename=os.path.join(force_dirname, "force_div.png"),
                    colormap="RDBU_BAR",
                )

        density = self.density_init.detach()
        vel = self.vel_init.detach()
        force_rescale_factor = 1.0
        keyframe_density_list = []

        # visualize init state
        self._save_quantity(self.opt.init_frame, density, vel)

        # simulation loop
        for t in range(self.opt.total_steps):
            frame = t + 1 + self.opt.init_frame
            print("KeyframeWindLearner.simulate: frame {}".format(frame))
            if frame > self.opt.keyframes[-1]:
                force_rescale_factor *= 1 - self.opt.force_decay
            density, vel = self.model(density, vel, t, force_rescale_factor)
            self._save_quantity(frame, density, vel)
            if frame in self.opt.keyframes:
                if self.opt.dim == 2:
                    density_vis = density
                else:
                    density_vis = self.real_op.get_rendered(density)
                keyframe_density_list.append(torch.flip(density_vis, dims=(1,)))

        # visualize grid of simulation at target frames
        torchvision.utils.save_image(
            keyframe_density_list,
            os.path.join(self.opt.save_path, "sim_{}.png".format(self.param_label)),
            nrow=len(self.opt.keyframes),
            padding=0,
        )

    def _save_all_img(self, step, density, vel):
        # save density
        save_path = os.path.join(self.opt.save_path, f"sim_{self.param_label}")
        utils.mkdir(os.path.join(save_path, "density"))
        self.real_op.save_img(
            density,
            filename=os.path.join(save_path, "density", f"{step:03d}.png"),
            colormap="OrRd" if self.solver.is_2d() else "RGB",
        )
        # save velocity
        if self.opt.save_vel:
            utils.mkdir(os.path.join(save_path, "vel"))
            self.mac_op.save_img(
                vel, filename=os.path.join(save_path, "vel", f"{step:03d}.png")
            )
        # save divergence of velocity
        if self.opt.save_div:
            utils.mkdir(os.path.join(save_path, "div"))
            self.real_op.save_img(
                self.mac_op.get_divergence(vel),
                filename=os.path.join(save_path, "div", f"{step:03d}.png"),
                colormap="RDBU_BAR",
            )

    def _save_all_npz(self, step, density, vel):
        """Save all quantities as single npz for each step.
           Whether the npz is compressed or not depends on self.save_compressed.

        Args:
            step (int): The name of the file, which is the step (int).

        """
        save_path = os.path.join(self.opt.save_path, f"sim_{self.param_label}")
        utils.mkdir(os.path.join(save_path, "npz"))
        filename = os.path.join(save_path, "npz", f"{step:03d}.npz")
        save_fcn = np.savez_compressed if self.opt.save_compressed else np.savez
        save_fcn(
            filename,
            density=self.real_op.grid2array(density),
            vel=self.mac_op.grid2array(vel),
            divergence=self.mac_op.grid2array(self.mac_op.get_divergence(vel))
            if self.opt.save_div
            else None,
        )

    def _save_quantity(self, step, density, vel):
        """Save all quantities as images and npz.
           Whether to save as npz depends on self.opt.save_npz.

        Args:
            step (int): The name of the file, which is the step (int).

        """
        self._save_all_img(step, density, vel)
        if self.opt.save_npz:
            self._save_all_npz(step, density, vel)
