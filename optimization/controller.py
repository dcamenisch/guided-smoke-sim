from __future__ import annotations

from typing import Optional, Dict, List

import torch
import torch.optim as optim
import torch.utils.checkpoint
from simulation.simulator import SmokeSimulator
from .model import FrequencyModel


class FrequencyOptimizer:
    """Optimizer that manages the progressive frequency optimization strategy."""

    def __init__(
        self,
        simulator: SmokeSimulator,
        keyframes: Dict[int, torch.Tensor],
        keyframe_weights: Dict[int, float],
        num_frames: int,
        band_radii: List[int],
        phase_iters: List[int],
        force_decay: float = 0.0,
        use_checkpoint: bool = True,
        loss_scale: float = 100.0,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.sim = simulator

        initial_resolution = 2 * band_radii[0] + 1
        self.model = FrequencyModel(simulator, initial_resolution)

        # Cap max frequency at Nyquist limit
        nyquist = min(simulator.nx, simulator.ny)
        if simulator.ndim == 3:
            nyquist = min(nyquist, simulator.nz)
        self.max_resolution = nyquist

        self.keyframes = keyframes
        self.weights = keyframe_weights
        self.num_frames = num_frames
        self.band_radii = band_radii
        self.phase_iters = phase_iters
        self.force_decay = force_decay
        self.use_checkpoint = use_checkpoint
        self.loss_scale = loss_scale
        self.grad_clip_norm = grad_clip_norm

        self.epochs_at_resolution = 0
        self.loss_history: List[float] = []
        self.best_loss = float("inf")
        self.best_params = None
        self.current_phase = 0

        # Validate phased schedule if provided
        if len(self.band_radii) != len(self.phase_iters):
            raise ValueError("band_radii and phase_iters must be same length")

        self.model.reorganize_parameters(initial_resolution)
        self._create_optimizer()

    def step(self) -> float:
        """Perform one optimization step (one LBFGS step)."""
        max_keyframe = max(self.keyframes.keys()) if self.keyframes else -1

        def closure():
            self.optimizer.zero_grad()

            # Initialize state
            if self.sim.ndim == 2:
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
                current_w = None
            else:  # 3D
                current_density = torch.zeros(
                    (self.sim.nz, self.sim.ny, self.sim.nx), device=self.sim.device
                )
                current_u = torch.zeros(
                    (self.sim.nz, self.sim.ny, self.sim.nx + 1), device=self.sim.device
                )
                current_v = torch.zeros(
                    (self.sim.nz, self.sim.ny + 1, self.sim.nx), device=self.sim.device
                )
                current_w = torch.zeros(
                    (self.sim.nz + 1, self.sim.ny, self.sim.nx), device=self.sim.device
                )
                current_pressure = torch.zeros(
                    (self.sim.nz, self.sim.ny, self.sim.nx), device=self.sim.device
                )

            total_loss = torch.tensor(0.0, device=self.sim.device)

            # Get force once (single force field for all frames)
            self.model.compute_force()
            forces = self.model.get_force()
            force_u, force_v = forces[0], forces[1]
            force_w = forces[2] if self.sim.ndim == 3 else None

            def run_simulation_step(density, u, v, w, pressure, f_u, f_v, f_w):
                # Set state
                self.sim.density = density
                self.sim.velocity.u_data = u
                self.sim.velocity.v_data = v
                if w is not None:
                    self.sim.velocity.w_data = w
                self.sim.pressure = pressure

                self.sim.add_source()
                self.sim.set_control_force(f_u, f_v, f_w)
                self.sim.step()

                if self.sim.ndim == 3:
                    return (
                        self.sim.density,
                        self.sim.velocity.u_data,
                        self.sim.velocity.v_data,
                        self.sim.velocity.w_data,
                        self.sim.pressure,
                    )
                else:
                    return (
                        self.sim.density,
                        self.sim.velocity.u_data,
                        self.sim.velocity.v_data,
                        None,
                        self.sim.pressure,
                    )

            # Time stepping
            for t in range(self.num_frames):
                # Optional force decay after last keyframe
                if self.force_decay > 0.0 and t > max_keyframe:
                    decay_steps = t - max_keyframe
                    decay_factor = (1.0 - self.force_decay) ** decay_steps
                    decay_force_u = force_u * decay_factor
                    decay_force_v = force_v * decay_factor
                    decay_force_w = (
                        force_w * decay_factor if force_w is not None else None
                    )
                else:
                    decay_force_u = force_u
                    decay_force_v = force_v
                    decay_force_w = force_w

                # Checkpoint the simulation step
                if self.use_checkpoint:
                    (
                        current_density,
                        current_u,
                        current_v,
                        current_w,
                        current_pressure,
                    ) = torch.utils.checkpoint.checkpoint(
                        run_simulation_step,
                        current_density,
                        current_u,
                        current_v,
                        current_w,
                        current_pressure,
                        decay_force_u,
                        decay_force_v,
                        decay_force_w,
                        use_reentrant=False,
                    )
                else:
                    (
                        current_density,
                        current_u,
                        current_v,
                        current_w,
                        current_pressure,
                    ) = run_simulation_step(
                        current_density,
                        current_u,
                        current_v,
                        current_w,
                        current_pressure,
                        decay_force_u,
                        decay_force_v,
                        decay_force_w,
                    )

                # Compute loss at keyframes
                if t in self.keyframes:
                    target = self.keyframes[t]
                    weight = self.weights.get(t, 1.0)
                    frame_loss = torch.nn.functional.mse_loss(current_density, target)
                    total_loss = total_loss + frame_loss * self.loss_scale * weight

            if total_loss.requires_grad:
                total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.model.param], self.grad_clip_norm)

            return total_loss

        loss = self.optimizer.step(closure)
        current_loss = loss.item()
        self.loss_history.append(current_loss)
        self.epochs_at_resolution += 1

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = self.model.param.clone()

        self._maybe_advance_phase()
        return current_loss

    def _maybe_advance_phase(self) -> None:
        current_phase_len = self.phase_iters[self.current_phase]
        if self.epochs_at_resolution >= current_phase_len:
            if self.current_phase + 1 < len(self.band_radii):
                self.current_phase += 1
                self.epochs_at_resolution = 0
                next_res = 2 * self.band_radii[self.current_phase] + 1
                # Cap at Nyquist limit
                next_res = min(next_res, self.max_resolution)
                self.model.reorganize_parameters(next_res)
                self._create_optimizer()

    def _create_optimizer(self) -> None:
        self.optimizer = optim.LBFGS(
            [self.model.param],
            lr=1.0,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

    def evaluate(self, num_frames: Optional[int] = None) -> list[torch.Tensor]:
        if num_frames is None:
            num_frames = self.num_frames

        frames = []
        with torch.no_grad():
            self.sim.reset()
            self.model.compute_force()
            forces = self.model.get_force()
            force_u, force_v = forces[0], forces[1]
            force_w = forces[2] if self.sim.ndim == 3 else None

            max_keyframe = max(self.keyframes.keys()) if self.keyframes else -1

            for t in range(num_frames):
                # Optional force decay after last keyframe
                if self.force_decay > 0.0 and t > max_keyframe:
                    decay_steps = t - max_keyframe
                    decay_factor = (1.0 - self.force_decay) ** decay_steps
                    decay_force_u = force_u * decay_factor
                    decay_force_v = force_v * decay_factor
                    decay_force_w = (
                        force_w * decay_factor if force_w is not None else None
                    )
                else:
                    decay_force_u = force_u
                    decay_force_v = force_v
                    decay_force_w = force_w

                self.sim.add_source()
                self.sim.set_control_force(decay_force_u, decay_force_v, decay_force_w)
                self.sim.step()
                frames.append(self.sim.density.clone())

        return frames
