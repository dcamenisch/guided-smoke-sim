import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.getcwd())
from optimizations.models.force_model import ForceModel


class ProgressiveForceModel(ForceModel):
    def __init__(
        self, solver, keyframe_opt, init_size, use_stream_fcn=False, use_sim_wind=False
    ):
        """Initialize the shaping wind model. The parameter(s) of the model
        is the control force(s).

        Args:
            solver: FluidSolver object
            keyframe_opt: Configuration of simulation
            init_size (int): The initial radii for the parameter
            use_stream_fcn (bool): True of using stream function
            use_sim_wind (bool): True of using sim wind
        """
        self.init_size = init_size
        super().__init__(solver, keyframe_opt, use_stream_fcn, use_sim_wind)

    def _get_full_size(self, size):
        """Get the parameter size.

        Args:
            size (int): The current radii.

        Returns:
            target_size (torch.Size)

        """
        size = (size - 1) * 0.5
        if self.solver.is_2d():
            W, H = self.solver.get_grid_size()
            shortest = min(H, W)
            size_x = round(W / shortest * size) * 2 + 1
            size_y = round(H / shortest * size) * 2 + 1
        else:
            W, H, D = self.solver.get_grid_size()
            shortest = min(D, min(H, W))
            size_x = round(W / shortest * size) * 2 + 1
            size_y = round(H / shortest * size) * 2 + 1
            size_z = round(D / shortest * size) * 2 + 1
        if self.use_stream_fcn:
            if self.solver.is_2d():
                target_size = torch.Size((1, size_y, size_x))
            else:
                target_size = torch.Size((3, size_z, size_y, size_x))
        else:
            if self.solver.is_2d():
                target_size = torch.Size((2, size_y, size_x))
            else:
                target_size = torch.Size((3, size_z, size_y, size_x))
        return target_size

    def _initialize_parameters_(self):
        """Initial parameter using init size."""
        target_size = self._get_full_size(self.init_size)
        self.param = torch.nn.Parameter(
            torch.zeros(target_size, device=self.device, dtype=self.solver.get_dtype())
        )

    def reorganize_parameters_(self, size=None, **kwargs):
        """Downsample/Upsample the parameters according to size

        Args:
            size (int): The current radii.

        """
        target_size = self._get_full_size(size)[1:]
        if self.use_stream_fcn:
            cur_size = min(self.param.size()[1:])
            rescale_factor = float(size - 1) / (
                cur_size - 1
            )  # accounts for staggered grid size
        else:
            rescale_factor = 1.0

        param = (
            F.interpolate(
                self.param.unsqueeze(0),
                size=target_size,
                mode="bilinear" if self.solver.is_2d() else "trilinear",
                align_corners=True,
            ).squeeze(0)
            * rescale_factor
        )
        self.param = torch.nn.Parameter(param)

    def compute_force(self, *args, **kwargs):
        target_size = self.solver.get_torch_size_mac()[1:]
        if self.use_stream_fcn:
            size = min(self.solver.get_torch_size_mac()[1:])
            cur_size = min(self.param.size()[2:])
            rescale_factor = float(size - 1) / (
                cur_size - 1
            )  # accounts for staggered grid size
        else:
            rescale_factor = 1.0
        if self.param.size()[1:] != target_size:
            param = (
                F.interpolate(
                    self.param.unsqueeze(0),
                    size=target_size,
                    mode="bilinear" if self.solver.is_2d() else "trilinear",
                    align_corners=True,
                ).squeeze(0)
                * rescale_factor
            )
        else:
            param = self.param
        if self.use_stream_fcn:
            self.force = self.node_op.get_curl(param)
        else:
            self.force = param
