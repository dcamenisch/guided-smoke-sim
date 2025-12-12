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
        max_resolution: int,
        keyframes: Dict[int, torch.Tensor],
        keyframe_weights: Dict[int, float],
        num_frames: int,
        band_radii: Optional[List[int]] = None,
        phase_iters: Optional[List[int]] = None,
        force_decay: float = 0.0,
        use_checkpoint: bool = True,
    ) -> None:
        self.sim = simulator

        # Determine initial resolution
        if band_radii is not None:
            initial_resolution = 2 * band_radii[0] + 1
        else:
            initial_resolution = 3  # Default fallback

        self.model = FrequencyModel(simulator, initial_resolution)
        self.max_resolution = max_resolution
        self.keyframes = keyframes
        self.weights = keyframe_weights
        self.num_frames = num_frames
        self.band_radii = band_radii
        self.phase_iters = phase_iters
        self.force_decay = force_decay
        self.use_checkpoint = use_checkpoint

        self.epochs_at_resolution = 0
        self.loss_history: List[float] = []
        self.best_loss = float("inf")
        self.best_params = None
        self.current_phase = 0

        # Validate phased schedule if provided
        if self.band_radii is not None:
            if self.phase_iters is None or len(self.band_radii) != len(
                self.phase_iters
            ):
                raise ValueError("band_radii and phase_iters must be same length")
            # Map band radius (reference) to our resolution parameter (size = 2*radius+1)
            first_res = 2 * self.band_radii[0] + 1
            self.model.reorganize_parameters(first_res)

        self._create_optimizer()

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
            self.model.compute_force()
            force_u, force_v = (
                self.model.get_force()
                if self.sim.ndim == 2
                else self.model.get_force()[:2]
            )
            force_w = self.model.get_force()[2] if self.sim.ndim == 3 else None

            max_keyframe = max(self.keyframes.keys()) if self.keyframes else -1

            def run_simulation_step(density, u, v, pressure, f_u, f_v):
                # Set state
                self.sim.density = density
                self.sim.velocity.u_data = u
                self.sim.velocity.v_data = v
                self.sim.pressure = pressure

                self.sim.add_source()
                self.sim.set_control_force(f_u, f_v, None)
                self.sim.step()

                return (
                    self.sim.density,
                    self.sim.velocity.u_data,
                    self.sim.velocity.v_data,
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
                else:
                    decay_force_u = force_u
                    decay_force_v = force_v

                # Checkpoint the simulation step
                if self.use_checkpoint:
                    current_density, current_u, current_v, current_pressure = (
                        torch.utils.checkpoint.checkpoint(
                            run_simulation_step,
                            current_density,
                            current_u,
                            current_v,
                            current_pressure,
                            decay_force_u,
                            decay_force_v,
                            use_reentrant=False,
                        )
                    )
                else:
                    (
                        current_density,
                        current_u,
                        current_v,
                        current_pressure,
                    ) = run_simulation_step(
                        current_density,
                        current_u,
                        current_v,
                        current_pressure,
                        decay_force_u,
                        decay_force_v,
                    )

                # Compute loss at keyframes
                if t in self.keyframes:
                    target = self.keyframes[t]
                    weight = self.weights.get(t, 1.0)
                    frame_loss = torch.nn.functional.mse_loss(current_density, target)
                    total_loss = total_loss + frame_loss * 100.0 * weight

            if total_loss.requires_grad:
                total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.model.param], 1.0)

            return total_loss

        loss = self.optimizer.step(closure)
        current_loss = loss.item()
        self.loss_history.append(current_loss)
        self.epochs_at_resolution += 1

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = self.model.param.clone()

        if self.band_radii is not None and self.phase_iters is not None:
            self._maybe_advance_phase()

        return current_loss

    def _maybe_advance_phase(self) -> None:
        """Advance to next frequency band based on fixed phase iterations."""
        if self.phase_iters is None:
            return
        current_phase_len = self.phase_iters[self.current_phase]
        if self.epochs_at_resolution >= current_phase_len:
            if self.current_phase + 1 < len(self.band_radii):
                self.current_phase += 1
                self.epochs_at_resolution = 0
                next_res = 2 * self.band_radii[self.current_phase] + 1
                print(f"Expanding frequency band {self.model.resolution} -> {next_res}")
                self.model.reorganize_parameters(next_res)
                self._create_optimizer()

    def _create_optimizer(self) -> None:
        """Create a new optimizer after parameter reorganization."""
        self.optimizer = optim.LBFGS(
            [self.model.param],
            lr=1.0,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
