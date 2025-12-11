"""Controller for frequency-aware force field optimization.

Implements the "Honey, I Shrunk the Domain" approach from EUROGRAPHICS 2021.
Uses vector potential (stream function) in Fourier domain for divergence-free forces.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
import torch.utils.checkpoint
import numpy as np
from simulation.simulator import SmokeSimulator


class FrequencyModel(torch.nn.Module):
    """Model that optimizes a vector potential in the frequency domain.

    Corresponds to ProgressiveFrequencyModel in the reference implementation.
    Handles parameters, force computation, and frequency padding.
    """

    def __init__(
        self,
        simulator: SmokeSimulator,
        initial_resolution: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize model.

        Args:
            simulator: The smoke simulator instance.
            initial_resolution: Initial size of the frequency band (radius parameter).
            device: Torch device.
        """
        super().__init__()
        self.sim = simulator
        self.device = device or simulator.device
        self.ndim = simulator.ndim

        # Resolution represents the radius parameter for progressive frequency bands
        self.resolution = initial_resolution

        # Initialize parameters with aspect-ratio aware shape
        self._initialize_parameters()

    def _get_freq_size(self, resolution: int) -> Tuple[int, ...]:
        """Compute frequency parameter size accounting for aspect ratio.

        Matches reference: progressive_frequency_model.py:51-57
        """
        size = (resolution - 1) * 0.5

        if self.ndim == 2:
            W, H = self.sim.nx, self.sim.ny
            shortest = min(H, W)
            size_x = int(round(W / shortest * size) * 2 + 1)
            size_y = int(round(H / shortest * size) * 2 + 1)
            # Stream function is scalar in 2D, shape: (1, H_freq, W_freq, 2)
            return (1, size_y, size_x, 2)
        else:
            W, H, D = self.sim.nx, self.sim.ny, self.sim.nz
            shortest = min(D, min(H, W))
            size_x = int(round(W / shortest * size) * 2 + 1)
            size_y = int(round(H / shortest * size) * 2 + 1)
            size_z = int(round(D / shortest * size) * 2 + 1)
            # Stream function is vector in 3D, shape: (3, D_freq, H_freq, W_freq, 2)
            return (3, size_z, size_y, size_x, 2)

    def _initialize_parameters(self) -> None:
        """Initialize frequency domain parameters."""
        param_shape = self._get_freq_size(self.resolution)

        # Parameters represent Fourier coefficients (real, imag)
        # Initialized to zeros - optimization will find the solution
        self.params = torch.nn.Parameter(
            torch.zeros(param_shape, dtype=self.sim.dtype, device=self.device)
        )
        self.params.requires_grad = True

    def set_resolution(self, new_resolution: int) -> None:
        """Increase the frequency resolution (progressive optimization).

        Matches reference: progressive_frequency_model.py:40-49 (reorganize_parameters_)
        Pads frequency domain with zeros to increase resolution without
        changing already-learned low-frequency content.
        """
        if new_resolution <= self.resolution:
            return

        old_params = self.params.data
        new_shape = self._get_freq_size(new_resolution)

        new_params = torch.zeros(new_shape, dtype=self.sim.dtype, device=self.device)

        # Copy old params to center of new params (DC at center due to fftshift)
        if self.ndim == 2:
            old_h, old_w = old_params.shape[1], old_params.shape[2]
            new_h, new_w = new_shape[1], new_shape[2]
            diff_h = (new_h - old_h) // 2
            diff_w = (new_w - old_w) // 2
            new_params[:, diff_h : diff_h + old_h, diff_w : diff_w + old_w, :] = (
                old_params
            )
        else:
            old_d, old_h, old_w = old_params.shape[1:4]
            new_d, new_h, new_w = new_shape[1:4]
            diff_d = (new_d - old_d) // 2
            diff_h = (new_h - old_h) // 2
            diff_w = (new_w - old_w) // 2
            new_params[
                :,
                diff_d : diff_d + old_d,
                diff_h : diff_h + old_h,
                diff_w : diff_w + old_w,
                :,
            ] = old_params

        self.resolution = new_resolution
        self.params = torch.nn.Parameter(new_params)
        self.params.requires_grad = True

    def compute_force(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the divergence-free force field from current potential.

        Returns:
            (force_u, force_v, force_w) where force_w is None for 2D.
        """
        if self.ndim == 2:
            return self._compute_force_2d()
        else:
            return self._compute_force_3d()

    def _pad_to_full_size(self, params_c: torch.Tensor) -> torch.Tensor:
        """Pad frequency parameters to full grid size.

        Matches reference: progressive_frequency_model.py:76-102
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

    def _compute_force_2d(self) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Compute 2D force field from stream function."""
        # 1. Convert params to complex
        params_c = torch.view_as_complex(self.params)  # (1, H_freq, W_freq)

        # 2. Pad to full grid size
        full_params = self._pad_to_full_size(params_c)  # (1, H+1, W+1)

        # 3. IFFT shift (move DC from center to corners for IFFT)
        full_params_shifted = torch.fft.ifftshift(full_params, dim=(-2, -1))

        # 4. IFFT with orthonormal normalization (matches reference's normalized=True)
        psi_complex = torch.fft.ifft2(full_params_shifted, dim=(-2, -1), norm="ortho")
        psi = psi_complex.real[0]  # (H+1, W+1)

        # 5. Compute Curl: f = curl(psi) = (dpsi/dy, -dpsi/dx)
        # Matches reference: grid.py:1009-1017
        dx = self.sim.dx

        # grad_x = dpsi/dx (shape: H+1, W)
        grad_x = (psi[:, 1:] - psi[:, :-1]) / dx
        grad_x = F.pad(grad_x, pad=(0, 1, 0, 0))  # Pad to (H+1, W+1)

        # grad_y = dpsi/dy (shape: H, W+1)
        grad_y = (psi[1:, :] - psi[:-1, :]) / dx
        grad_y = F.pad(grad_y, pad=(0, 0, 0, 1))  # Pad to (H+1, W+1)

        # curl = (grad_y, -grad_x) stacked as (2, H+1, W+1)
        # For MAC grid: u is (ny, nx+1), v is (ny+1, nx)
        # curl[0] = grad_y for u-component
        # curl[1] = -grad_x for v-component

        # Extract appropriate slices for MAC grid
        force_u = grad_y[:-1, :]  # (ny, nx+1)
        force_v = -grad_x[:, :-1]  # (ny+1, nx)

        return force_u, force_v, None

    def _compute_force_3d(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 3D force field from vector potential.

        Matches reference: grid.py:1018-1038
        """
        # 1. Convert params to complex
        params_c = torch.view_as_complex(self.params)  # (3, D_freq, H_freq, W_freq)

        # 2. Pad to full grid size
        full_params = self._pad_to_full_size(params_c)  # (3, D+1, H+1, W+1)

        # 3. IFFT shift and IFFT
        full_params_shifted = torch.fft.ifftshift(full_params, dim=(-3, -2, -1))
        psi_complex = torch.fft.ifftn(
            full_params_shifted, dim=(-3, -2, -1), norm="ortho"
        )
        psi = psi_complex.real  # (3, D+1, H+1, W+1)

        psi_x = psi[0]  # x-component of vector potential
        psi_y = psi[1]  # y-component
        psi_z = psi[2]  # z-component

        dx = self.sim.dx

        # Compute finite difference gradients (reference grid.py:1024-1029)
        # d(psi_x)/dz
        dpsi_x_dz = (psi_x[1:, :, :] - psi_x[:-1, :, :]) / dx
        # d(psi_x)/dy
        dpsi_x_dy = (psi_x[:, 1:, :] - psi_x[:, :-1, :]) / dx
        # d(psi_y)/dx
        dpsi_y_dx = (psi_y[:, :, 1:] - psi_y[:, :, :-1]) / dx
        # d(psi_y)/dz
        dpsi_y_dz = (psi_y[1:, :, :] - psi_y[:-1, :, :]) / dx
        # d(psi_z)/dx
        dpsi_z_dx = (psi_z[:, :, 1:] - psi_z[:, :, :-1]) / dx
        # d(psi_z)/dy
        dpsi_z_dy = (psi_z[:, 1:, :] - psi_z[:, :-1, :]) / dx

        # Curl components (reference grid.py:1031-1033)
        # curl_x = d(psi_z)/dy - d(psi_y)/dz
        curl_x = dpsi_z_dy[:-1, :, :] - dpsi_y_dz[:, :-1, :]
        # curl_y = d(psi_x)/dz - d(psi_z)/dx
        curl_y = dpsi_x_dz[:, :, :-1] - dpsi_z_dx[:-1, :, :]
        # curl_z = d(psi_y)/dx - d(psi_x)/dy
        curl_z = dpsi_y_dx[:, :-1, :] - dpsi_x_dy[:, :, :-1]

        # Pad back to MAC grid sizes (reference grid.py:1035-1037)
        # u-face: (nz, ny, nx+1)
        curl_x = F.pad(curl_x, pad=(0, 0, 0, 1, 0, 1))
        # v-face: (nz, ny+1, nx)
        curl_y = F.pad(curl_y, pad=(0, 1, 0, 0, 0, 1))
        # w-face: (nz+1, ny, nx)
        curl_z = F.pad(curl_z, pad=(0, 1, 0, 1, 0, 0))

        # Extract for MAC grid (matching simulator's expected shapes)
        # force_u: (nz, ny, nx+1) -> matches curl_x after padding
        # force_v: (nz, ny+1, nx) -> matches curl_y after padding
        # force_w: (nz+1, ny, nx) -> matches curl_z after padding
        force_u = curl_x[: self.sim.nz, : self.sim.ny, :]
        force_v = curl_y[: self.sim.nz, :, : self.sim.nx]
        force_w = curl_z[:, : self.sim.ny, : self.sim.nx]

        return force_u, force_v, force_w


class FrequencyOptimizer:
    """Optimizer that manages the progressive frequency optimization strategy.

    Corresponds to ProgressiveFrequencyOptimization in the reference implementation.
    Handles the optimization loop, LBFGS, and automatic mask growing.
    """

    def __init__(
        self,
        simulator: SmokeSimulator,
        initial_resolution: int,
        max_resolution: int,
        keyframes: Dict[int, torch.Tensor],
        keyframe_weights: Dict[int, float],
        base_wind: torch.Tensor,
        num_frames: int,
        source_mask: torch.Tensor,
        source_val: torch.Tensor,
        convergence_threshold: float = 0.01,
    ) -> None:
        self.sim = simulator
        self.model = FrequencyModel(simulator, initial_resolution)
        self.max_resolution = max_resolution
        self.keyframes = keyframes
        self.weights = keyframe_weights
        self.base_wind = base_wind
        self.num_frames = num_frames
        self.source_mask = source_mask
        self.source_val = source_val
        self.convergence_threshold = convergence_threshold

        self.prev_loss: Optional[float] = None
        self.epochs_at_resolution = 0
        self.min_epochs_per_resolution = 20
        self.loss_history: List[float] = []
        self.best_loss = float("inf")
        self.best_params = None

        self._create_optimizer()

    def _create_optimizer(self) -> None:
        """Create LBFGS optimizer for current parameters."""
        self.optimizer = optim.LBFGS(
            [self.model.params],
            lr=1.0,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

    def step(self) -> float:
        """Perform one optimization step (one LBFGS step)."""

        def closure():
            self.optimizer.zero_grad()

            # Initialize state
            current_density = torch.zeros(
                (self.sim.ny, self.sim.nx), device=self.sim.device
            )
            current_u = torch.zeros(
                (self.sim.ny, self.sim.nx + 1), device=self.sim.device
            )
            current_v = torch.zeros(
                (self.sim.ny + 1, self.sim.nx), device=self.sim.device
            )
            current_pressure = torch.zeros(
                (self.sim.ny, self.sim.nx), device=self.sim.device
            )

            total_loss = torch.tensor(0.0, device=self.sim.device)

            # Get force once (single force field for all frames)
            force_u, force_v, _ = self.model.compute_force()

            def run_simulation_step(density, u, v, pressure, f_u, f_v):
                # Set state
                self.sim.density = density
                self.sim.velocity.u_data = u
                self.sim.velocity.v_data = v
                self.sim.pressure = pressure

                # Apply forces
                total_force_v = f_v + self.base_wind
                self.sim.set_control_force(f_u, total_force_v, None)

                # Step
                self.sim.step()

                return (
                    self.sim.density,
                    self.sim.velocity.u_data,
                    self.sim.velocity.v_data,
                    self.sim.pressure,
                )

            # Time stepping
            for t in range(self.num_frames):
                # Inject source for first few frames
                if t < 15:
                    current_density = torch.where(
                        self.source_mask, self.source_val, current_density
                    )

                # Checkpoint the simulation step
                # We pass force_u and force_v to ensure gradients flow to them
                current_density, current_u, current_v, current_pressure = (
                    torch.utils.checkpoint.checkpoint(
                        run_simulation_step,
                        current_density,
                        current_u,
                        current_v,
                        current_pressure,
                        force_u,
                        force_v,
                        use_reentrant=False,
                    )
                )

                # Compute loss at keyframes
                if t in self.keyframes:
                    target = self.keyframes[t]
                    weight = self.weights.get(t, 1.0)
                    frame_loss = torch.nn.functional.mse_loss(current_density, target)
                    total_loss = total_loss + frame_loss * 100.0 * weight

            # L2 regularization on parameters
            reg_loss = 1e-4 * torch.sum(self.model.params**2)
            total_loss = total_loss + reg_loss

            if total_loss.requires_grad:
                total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.model.params], 1.0)

            return total_loss

        loss = self.optimizer.step(closure)
        current_loss = loss.item()
        self.loss_history.append(current_loss)
        self.epochs_at_resolution += 1

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = self.model.params.clone()

        self._check_upsampling(current_loss)

        return current_loss

    def _check_upsampling(self, current_loss: float) -> None:
        """Check if we should upsample the frequency band."""
        if self.epochs_at_resolution < self.min_epochs_per_resolution:
            return

        if self.should_grow_mask(current_loss):
            next_res = self.model.resolution * 2
            if next_res <= self.max_resolution:
                print(f"Expanding frequency band {self.model.resolution} -> {next_res}")
                self.model.set_resolution(next_res)
                self._create_optimizer()
                self.epochs_at_resolution = 0

    def should_grow_mask(self, current_loss: float) -> bool:
        """Check convergence for mask growing."""
        if self.prev_loss is None or self.prev_loss == 0:
            self.prev_loss = current_loss
            return False

        relative_change = abs(current_loss - self.prev_loss) / abs(self.prev_loss)
        self.prev_loss = current_loss

        return relative_change < self.convergence_threshold
