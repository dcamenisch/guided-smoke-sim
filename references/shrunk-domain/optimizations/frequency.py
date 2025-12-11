import torch
import sys
import os

sys.path.append(os.getcwd())
import lib.dfluids.utils as utils


class FourierDecomposition(object):
    def __init__(self, solver):
        self.solver = solver
        self.device = self.solver.get_device()
        self.dim = 2 if self.solver.is_2d() else 3

    def ifft(self, fshift, signal_ndim, normalized=True):
        """Wrapper for torch.ifft to perform 2D/3D inverse fft on shifted frequency

        Args:
            fshift(torch.Tensor): of size [C, (T), (D), H, W, 2]
                or [C, (T), (D+1), H+1, W+1, 2],
                with the last dimension representing real and complex components.

        Returns:
            grid(torch.Tensor): of size [C, (T), (D), H, W],
                or [C, (T), (D+1), H+1, W+1], real valued
        """
        assert signal_ndim > 0 and signal_ndim < 5
        f = self._batch_ifftshift(fshift)
        if signal_ndim < 4:
            grid = torch.ifft(f, signal_ndim=signal_ndim, normalized=normalized)
        else:
            grid = self._ifft4_noshift(f, normalized=normalized)
        return grid[..., 0]

    def _ifft4_noshift(self, f, normalized):
        """Wrapper of torch.fft to perform 4D fft on input grid.
        The operation is achieved by combining a 3D fft and 1D fft.

        Args:
            f(torch.Tensor): of size [C, T, D, H, W, 2]
                or [C, T, D+1, H+1, W+1, 2], with the last dimension
                representing real and complex components.

        Returns:
            grid(torch.Tensor) of size [C, T, D, H, W, 2]
                or [C, T, D+1, H+1, W+1, 2], with the last dimension
                representing real and complex components. ifftshift is
                not performed.
        """
        x_p = torch.ifft(f, signal_ndim=3, normalized=normalized)
        grid = torch.ifft(
            x_p.permute(0, 2, 3, 4, 1, 5), signal_ndim=1, normalized=normalized
        ).permute(0, 4, 1, 2, 3, 5)
        return grid

    def _roll_n(self, X, axis, n):
        """Helper function for fftshift"""
        f_idx = tuple(
            slice(None, None, None) if i != axis else slice(0, n, None)
            for i in range(X.dim())
        )
        b_idx = tuple(
            slice(None, None, None) if i != axis else slice(n, None, None)
            for i in range(X.dim())
        )
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def _batch_ifftshift(self, x):
        """Performs batched inverse fftshift in 2d.

        Args:
            x(torch.Tensor): of size [B, (D), H, W, 2]

        Returns:
            shift(torch.Tensor): of size [B, (D), H, W, 2]
        """
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self._roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self._roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
