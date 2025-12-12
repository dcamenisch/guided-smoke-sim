from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.fft
from simulation.simulator import SmokeSimulator


class FrequencyModel(torch.nn.Module):
    """Model that optimizes a vector potential in the frequency domain."""

    def __init__(
        self,
        simulator: SmokeSimulator,
        initial_resolution: int = 4,
        use_stream_fcn: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize model.

        Args:
            simulator: The smoke simulator instance.
            initial_resolution: Initial size of the frequency band (radius parameter).
            use_stream_fcn: If True, use stream function (divergence-free).
                           If False, optimize force directly.
            device: Torch device.
        """
        super().__init__()
        self.sim = simulator
        self.device = device or simulator.device
        self.ndim = simulator.ndim
        self.use_stream_fcn = use_stream_fcn

        # Resolution represents the radius parameter for progressive frequency bands
        self.resolution = initial_resolution

        # Force field cache
        self.force = None

        # Initialize parameters with aspect-ratio aware shape
        self._initialize_parameters()

    def _get_freq_size(self, resolution: int) -> Tuple[int, ...]:
        """Compute frequency parameter size accounting for aspect ratio.

        Matches reference: progressive_frequency_model.py:113-129
        """
        size = (resolution - 1) * 0.5

        if self.ndim == 2:
            W, H = self.sim.nx, self.sim.ny
            shortest = min(H, W)
            size_x = int(round(W / shortest * size) * 2 + 1)
            size_y = int(round(H / shortest * size) * 2 + 1)

            if self.use_stream_fcn:
                # Stream function is scalar in 2D
                return (1, size_y, size_x, 2)
            else:
                # Direct force is vector in 2D
                return (2, size_y, size_x, 2)
        else:
            W, H, D = self.sim.nx, self.sim.ny, self.sim.nz
            shortest = min(D, min(H, W))
            size_x = int(round(W / shortest * size) * 2 + 1)
            size_y = int(round(H / shortest * size) * 2 + 1)
            size_z = int(round(D / shortest * size) * 2 + 1)

            # Stream function (vector potential) is always vector in 3D
            return (3, size_z, size_y, size_x, 2)

    def _initialize_parameters(self) -> None:
        """Initialize frequency domain parameters.

        Matches reference: progressive_frequency_model.py:103-108
        """
        param_shape = self._get_freq_size(self.resolution)

        # Parameters represent Fourier coefficients (real, imag)
        # Initialized to zeros (reference convention)
        self.param = torch.nn.Parameter(
            torch.zeros(param_shape, dtype=self.sim.dtype, device=self.device)
        )
        self.param.requires_grad = True

    def reorganize_parameters(self, resolution: int) -> None:
        """Enlarge frequency bands through padding (progressive optimization).

        Matches reference: progressive_frequency_model.py:110-115 (reorganize_parameters_)
        Pads frequency domain with zeros to increase resolution without
        changing already-learned low-frequency content.

        Args:
            resolution: New resolution (size = 2*resolution + 1).
        """
        if resolution <= self.resolution:
            return

        old_param = self.param.data
        new_shape = self._get_freq_size(resolution)

        new_param = self._pad_to_target_size(old_param, new_shape)

        self.resolution = resolution
        self.param = torch.nn.Parameter(new_param)
        self.param.requires_grad = True

    def set_resolution(self, new_resolution: int) -> None:
        """Alias for reorganize_parameters for backward compatibility."""
        self.reorganize_parameters(new_resolution)

    def _pad_to_target_size(
        self, tensor: torch.Tensor, target_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """Pad tensor to target size, centering the original content.

        Matches reference: progressive_frequency_model.py:131-154 (_pad)

        Args:
            tensor: Input of size [C, (D), H, W, 2]
            target_size: Target size in format [C, (D), H, W, 2]
        """
        if tensor.size() == target_size:
            return tensor

        # Compute padding for each spatial dimension (reversed for F.pad)
        # F.pad expects (left, right, top, bottom, front, back)
        spatial_diff = (
            np.array(target_size[1:-1][::-1]) - np.array(tensor.size()[1:-1][::-1])
        ) * 0.5
        pad = tuple(np.repeat(spatial_diff, repeats=2).astype(int))

        # Pad real and imaginary parts separately
        freq_r, freq_i = torch.unbind(tensor, dim=-1)
        freq_r = F.pad(freq_r, pad)
        freq_i = F.pad(freq_i, pad)

        return torch.stack((freq_r, freq_i), dim=-1)

    def compute_force(self) -> None:
        """Compute the force field from frequency parameters.

        Matches reference: progressive_frequency_model.py:156-166
        Stores result in self.force for later retrieval via get_force().
        """
        if self.ndim == 2:
            self.force = self._compute_force_2d()
        else:
            self.force = self._compute_force_3d()

    def get_force(self) -> Tuple[torch.Tensor, ...]:
        """Get the computed force field.

        Returns:
            Tuple of force components (u, v) for 2D or (u, v, w) for 3D.
        """
        assert self.force is not None, "Must call compute_force() first"
        return self.force

    def _pad_to_full_size(self, params_c: torch.Tensor) -> torch.Tensor:
        """Pad frequency parameters to full grid size for IFFT.

        Pads current frequency parameters to match the grid size needed
        for inverse FFT to produce forces on the MAC grid.
        """
        if self.ndim == 2:
            full_h, full_w = self.sim.ny + 1, self.sim.nx + 1
            freq_shape = params_c.shape[-2:]

            if freq_shape[0] > full_h or freq_shape[1] > full_w:
                # Crop if params exceed full size
                ch = (freq_shape[0] - full_h) // 2
                cw = (freq_shape[1] - full_w) // 2
                return params_c[..., ch : ch + full_h, cw : cw + full_w]
            else:
                # Pad with zeros
                full_params = torch.zeros(
                    (*params_c.shape[:-2], full_h, full_w),
                    dtype=params_c.dtype,
                    device=self.device,
                )
                start_h = (full_h - freq_shape[0]) // 2
                start_w = (full_w - freq_shape[1]) // 2
                full_params[
                    ...,
                    start_h : start_h + freq_shape[0],
                    start_w : start_w + freq_shape[1],
                ] = params_c
                return full_params
        else:
            full_d, full_h, full_w = self.sim.nz + 1, self.sim.ny + 1, self.sim.nx + 1
            freq_shape = params_c.shape[-3:]

            full_params = torch.zeros(
                (*params_c.shape[:-3], full_d, full_h, full_w),
                dtype=params_c.dtype,
                device=self.device,
            )
            start_d = (full_d - freq_shape[0]) // 2
            start_h = (full_h - freq_shape[1]) // 2
            start_w = (full_w - freq_shape[2]) // 2
            full_params[
                ...,
                start_d : start_d + freq_shape[0],
                start_h : start_h + freq_shape[1],
                start_w : start_w + freq_shape[2],
            ] = params_c
            return full_params

    def _compute_force_2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D force field.

        Returns:
            (force_u, force_v) for MAC grid.
        """
        # Convert to complex and pad to full size
        params_c = torch.view_as_complex(self.param)
        full_params = self._pad_to_full_size(params_c)

        # IFFT shift and inverse transform
        full_params_shifted = torch.fft.ifftshift(full_params, dim=(-2, -1))
        phi_complex = torch.fft.ifft2(full_params_shifted, dim=(-2, -1), norm="ortho")
        phi = phi_complex.real  # Potential in spatial domain

        if self.use_stream_fcn:
            # Stream function: force = curl(psi) = (dpsi/dy, -dpsi/dx)
            psi = phi[0]  # Scalar stream function (H+1, W+1)
            dx = self.sim.dx

            # Compute gradients
            grad_x = (psi[:, 1:] - psi[:, :-1]) / dx
            grad_x = F.pad(grad_x, pad=(0, 1, 0, 0))

            grad_y = (psi[1:, :] - psi[:-1, :]) / dx
            grad_y = F.pad(grad_y, pad=(0, 0, 0, 1))

            # Curl for MAC grid
            force_u = grad_y[:-1, :]  # (ny, nx+1)
            force_v = -grad_x[:, :-1]  # (ny+1, nx)
        else:
            # Direct force optimization (phi is already the force)
            force_u = phi[0, :-1, :]  # (ny, nx+1)
            force_v = phi[1, :, :-1]  # (ny+1, nx)

        return force_u, force_v

    def _compute_force_3d(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 3D force field from vector potential.

        Returns:
            (force_u, force_v, force_w) for MAC grid.
        """
        # Convert to complex and pad to full size
        params_c = torch.view_as_complex(self.param)
        full_params = self._pad_to_full_size(params_c)

        # IFFT shift and inverse transform
        full_params_shifted = torch.fft.ifftshift(full_params, dim=(-3, -2, -1))
        phi_complex = torch.fft.ifftn(
            full_params_shifted, dim=(-3, -2, -1), norm="ortho"
        )
        phi = phi_complex.real  # (3, D+1, H+1, W+1)

        if self.use_stream_fcn:
            # Vector potential: force = curl(A)
            psi_x, psi_y, psi_z = phi[0], phi[1], phi[2]
            dx = self.sim.dx

            # Compute gradients
            dpsi_x_dz = (psi_x[1:, :, :] - psi_x[:-1, :, :]) / dx
            dpsi_x_dy = (psi_x[:, 1:, :] - psi_x[:, :-1, :]) / dx
            dpsi_y_dx = (psi_y[:, :, 1:] - psi_y[:, :, :-1]) / dx
            dpsi_y_dz = (psi_y[1:, :, :] - psi_y[:-1, :, :]) / dx
            dpsi_z_dx = (psi_z[:, :, 1:] - psi_z[:, :, :-1]) / dx
            dpsi_z_dy = (psi_z[:, 1:, :] - psi_z[:, :-1, :]) / dx

            # Curl components
            curl_x = dpsi_z_dy[:-1, :, :] - dpsi_y_dz[:, :-1, :]
            curl_y = dpsi_x_dz[:, :, :-1] - dpsi_z_dx[:-1, :, :]
            curl_z = dpsi_y_dx[:, :-1, :] - dpsi_x_dy[:, :, :-1]

            # Pad to MAC grid sizes
            curl_x = F.pad(curl_x, pad=(0, 0, 0, 1, 0, 1))
            curl_y = F.pad(curl_y, pad=(0, 1, 0, 0, 0, 1))
            curl_z = F.pad(curl_z, pad=(0, 1, 0, 1, 0, 0))

            # Extract for MAC grid
            force_u = curl_x[: self.sim.nz, : self.sim.ny, :]
            force_v = curl_y[: self.sim.nz, :, : self.sim.nx]
            force_w = curl_z[:, : self.sim.ny, : self.sim.nx]
        else:
            # Direct force optimization
            force_u = phi[0, : self.sim.nz, : self.sim.ny, :]
            force_v = phi[1, : self.sim.nz, :, : self.sim.nx]
            force_w = phi[2, :, : self.sim.ny, : self.sim.nx]

        return force_u, force_v, force_w
