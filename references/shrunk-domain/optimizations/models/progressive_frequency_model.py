import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from optimizations.models.force_model import ForceModel
from optimizations.frequency import FourierDecomposition


class ProgressiveFrequencyModel(ForceModel):
    def __init__(
        self, solver, keyframe_opt, init_size, use_stream_fcn=False, use_sim_wind=False
    ):
        """Initialize the shaping wind fourier model. The parameter of the model
        is the fourier space of control force(s) with 0 freq in the center (fftshifted).

        Args:
            solver: FluidSolver object
            keyframe_opt: Configuration of simulation
            init_size (int): Initial radius parameter
            use_stream_fcn (bool): True of using stream function
            use_sim_wind (bool): True of using sim wind
        """
        self.init_size = init_size
        super(ProgressiveFrequencyModel, self).__init__(
            solver, keyframe_opt, use_stream_fcn, use_sim_wind
        )
        self.signal_ndim = 3 if self.solver.is_3d() else 2
        self.freq_decom = FourierDecomposition(self.solver)

    def _initialize_parameters_(self):
        """Initial parameter using init size."""
        target_size = self._get_full_size(self.init_size)
        self.param = torch.nn.Parameter(
            torch.zeros(target_size, device=self.device, dtype=self.solver.get_dtype())
        )

    def reorganize_parameters_(self, size, **kwargs):
        """Enlarges frequency bands through padding.

        Args:
            size (int): The current radius.

        """
        assert isinstance(size, int)
        target_size = self._get_full_size(size)
        self.param = torch.nn.Parameter(self._pad(self.param, target_size))

    def _get_full_size(self, size):
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
                target_size = torch.Size((1, size_y, size_x, 2))
            else:
                target_size = torch.Size((3, size_z, size_y, size_x, 2))
        else:
            if self.solver.is_2d():
                target_size = torch.Size((2, size_y, size_x, 2))
            else:
                target_size = torch.Size((3, size_z, size_y, size_x, 2))
        return target_size

    def _pad(self, tensor, target_size):
        """pad tensor to target size

        Args:
            tensor(torch.Tensor): of size [C, (D), H, W, 2],
                D, H, W should be odd numbers
                i.e. resolution being even number
            target_size(torch.Size): spatial target size
                in format [C, (D), H, W, 2]
        """
        if tensor.size() == target_size:
            return tensor
        pad = tuple(
            np.repeat(
                (
                    np.array(target_size[1:-1][::-1])
                    - np.array(tensor.size()[1:-1][::-1])
                )
                * 0.5,
                repeats=2,
            ).astype(int)
        )  # (padx, padx, pady, pady, padz, padz)
        freq_r, freq_i = torch.unbind(tensor, dim=-1)
        freq_r = F.pad(freq_r, pad)
        freq_i = F.pad(freq_i, pad)
        tensor = torch.stack((freq_r, freq_i), dim=-1)
        return tensor

    def compute_force(self):
        """Computes force from param"""
        # pad the parameters
        if self.use_stream_fcn:
            target_size = self.solver.get_torch_size_node() + (2,)
        else:
            target_size = self.solver.get_torch_size_mac() + (2,)
        freq = self._pad(self.param, target_size)
        self.phi = self.freq_decom.ifft(freq, self.signal_ndim, normalized=True)
        if self.use_stream_fcn:
            self.force = self.node_op.get_curl(self.phi)
        else:
            self.force = self.phi
