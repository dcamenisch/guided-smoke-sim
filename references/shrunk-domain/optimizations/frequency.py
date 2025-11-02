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

    def fft(self, grid, signal_ndim, normalized=True):
        """Wrapper for torch.fft to perform 2D/3D fft on input grid.
        Also shifts the zero freq to the center of the frequency domain.

        Args:
            grid(torch.Tensor): of size [C, (T), (D), H, W]
                or [C, (T), (D+1), H+1, W+1], real valued

        Returns:
            fshift(torch.Tensor): of size [C, (T), (D), H, W, 2]
                or [C, (T), (D+1), H+1, W+1, 2],
                with the last dimension representing real and complex components.
        """
        assert signal_ndim > 0 and signal_ndim < 5
        zero_grid = torch.zeros(
            grid.size(), device=self.device, dtype=self.solver.get_dtype()
        )
        grid = torch.stack((grid, zero_grid), dim=-1)
        if signal_ndim < 4:
            f = torch.fft(grid, signal_ndim=signal_ndim, normalized=normalized)
        else:
            f = self._fft4_noshift(grid, normalized=normalized)
        fshift = self._batch_fftshift(f)
        return fshift

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
        # return grid.norm(p=2, dim=-1)

    def get_spectral(self, fshift):
        """Computes the manitude of fourier frequency of given grid,
        used together with self.fft

        Args:
            fshift(torch.Tensor): of size [C, (D), H, W, 2]
                or [C, (D+1), H+1, W+1, 2],
                fourier decomposition with zero shifted to the center

        Returns:
            sp(torch.Tensor): of size [C, (D), H, W],
                or [C, (D+1), H+1, W+1],
                real valued magnitude of the fourier decomposition.
        """
        sp = fshift.norm(p=2, dim=-1)
        return sp

    def get_spatial_bandpass(self, fshift, inner, outer, sigma=0.0):
        """Applies the band pass to spatial dimensions of a fourier decomposition
        used together with self.fft. Implementation based on
        https://aip.scitation.org/doi/10.1063/1.4871106

        Args:
            fshift(torch.Tensor): of size [C, (T), (D), H, W, 2]
                or [C, (T), (D+1), H+1, W+1, 2],
                fourier decomposition with zero shifted to the center
            inner(int): The lower limit of the pass-through
                frequency band (in pixels, with respect to the shortest side).
            outer(int): The upper limit of the pass-through
                frequency band (in pixels, with respect to the shortest side).
            sigma(float): The standard deviation of gauss curv
                on the boundary of hard cut-off. (0.0 for hard cut-off)

        Returns:
            fshift_bandpass(torch.Tensor): of size [C, (T), (D), H, W],
                or [C, (T), (D+1), H+1, W+1]
        """
        assert inner <= outer
        if self.dim == 2:
            W, H = self.solver.get_grid_size()
            shortest = min(W, H)
        else:
            W, H, D = self.solver.get_grid_size()
            shortest = min(D, min(H, W))
        mgrid = utils.create_mesh_grid(
            fshift.size()[(-self.dim - 2) : -1],
            device=self.device,
            dtype=self.solver.get_dtype(),
        )
        # normalize frequency to [-1, 1]
        inner = inner / shortest * 2.0
        outer = outer / shortest * 2.0
        # rescale mgrid s.t. all dimensions get normalized to [-1, 1]
        mgrid[0, ...] = mgrid[0, ...] / W * 2.0 - 1.0
        mgrid[1, ...] = mgrid[1, ...] / H * 2.0 - 1.0
        dist2 = (mgrid[0, ...]) ** 2 + (mgrid[1, ...]) ** 2
        if self.dim == 3:
            mgrid[2, ...] = mgrid[2, ...] / D * 2.0 - 1.0
            dist2 = dist2 + (mgrid[2, ...]) ** 2
        dist = torch.sqrt(dist2)
        gain = torch.zeros(
            fshift.size()[:-1], device=self.device, dtype=self.solver.get_dtype()
        )
        gain[..., (dist >= inner) * (dist <= outer)] = 1.0
        if sigma > 0:
            gauss_inner = torch.exp(-((dist - inner) ** 2) / (2 * sigma ** 2))
            gauss_outer = torch.exp(-((dist - outer) ** 2) / (2 * sigma ** 2))
            gain = torch.max(torch.max(gain, gauss_inner), gauss_outer)
        gain = torch.stack((gain, gain), dim=-1)
        fshift = fshift * gain
        return fshift

    def get_temporal_bandpass(self, fshift, inner, outer, sigma=0.0):
        """Applies the band pass to the temporal dimension of fourier decomposition.
        Currently only supports hard cut-off (sigma does not play a role)

        Args:
            fshift(torch.Tensor): of size [C, T, (D), H, W, 2]
                or [C, T, (D+1), H+1, W+1, 2],
                fourier decomposition with zero shifted to the center
            inner(int): The lower limit of the pass-through
                frequency band (in pixels).
            outer(int): The upper limit of the pass-through
                frequency band (in pixels).

        Returns:
            fshift_bandpass(torch.Tensor): of size [C, T, (D), H, W],
                or [C, T, (D+1), H+1, W+1]
        """
        assert inner <= outer
        T = fshift.size()[1]
        mgrid = torch.linspace(0, T - 1, steps=T, device=self.device)
        # # normalize frequency to [-1, 1]
        inner = inner / (T - 1) * 2.0
        outer = outer / (T - 1) * 2.0
        dist = (mgrid / (T - 1) * 2.0 - 1.0).abs()
        gain = torch.zeros(
            fshift.size()[:-1], device=self.device, dtype=self.solver.get_dtype()
        )
        gain[:, (dist >= inner) * (dist <= outer), :] = 1.0
        gain = torch.stack((gain, gain), dim=-1)
        fshift = fshift * gain
        return fshift

    def _fft4_noshift(self, grid, normalized):
        """Wrapper of torch.fft to perform 4D fft on input grid.
        The operation is achieved by combining a 3D fft and 1D fft.

        Args:
            grid(torch.Tensor): of size [C, T, D, H, W, 2]
                or [C, T, D+1, H+1, W+1, 2], with the last dimension
                representing real and complex components.

        Returns:
            f(torch.Tensor) of size [C, T, D, H, W, 2]
                or [C, T, D+1, H+1, W+1, 2], with the last dimension
                representing real and complex components. fftshift is
                not performed.
        """
        f_p = torch.fft(grid, signal_ndim=3, normalized=normalized)
        f = torch.fft(
            f_p.permute(0, 2, 3, 4, 1, 5), signal_ndim=1, normalized=normalized
        ).permute(0, 4, 1, 2, 3, 5)
        # ############################## ALTERNATIVE ###########################
        # f_p = torch.fft(grid, signal_ndim=2, normalized=normalized)
        # f = torch.fft(
        #     f_p.permute(0, 3, 4, 1, 2, 5),
        #     signal_ndim=2, normalized=normalized
        # ).permute(0, 3, 4, 1, 2, 5)
        # ######################################################################
        return f

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

    # https://github.com/tomrunia/PyTorchSteerablePyramid/blob/master/steerable/math_utils.py
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

    # https://github.com/tomrunia/PyTorchSteerablePyramid/blob/master/steerable/math_utils.py
    def _batch_fftshift(self, x):
        """Performs batched fftshift in 2d.

        Args:
            x(torch.Tensor): of size [B, (D), H, W, 2]

        Returns:
            shift(torch.Tensor): of size [B, (D), H, W, 2]
        """
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim) // 2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self._roll_n(real, axis=dim, n=n_shift)
            imag = self._roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    # https://github.com/tomrunia/PyTorchSteerablePyramid/blob/master/steerable/math_utils.py
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
