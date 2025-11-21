"""2D visualization utilities for smoke simulation."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from visualization._animation import run_animation

if TYPE_CHECKING:
    from simulation.simulator import SmokeSimulator


def _to_numpy(data: Any) -> Any:
    """Convert Torch tensors to NumPy arrays for plotting."""
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return data


def _draw_imshow(
    ax: Axes,
    data: Any,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    aspect: str = "auto",
) -> None:
    ax.clear()
    ax.imshow(
        _to_numpy(data),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
    )
    ax.set_title(title)
    ax.axis("off")


def _density_panel(ax: Axes, simulator: "SmokeSimulator") -> Callable[[int], None]:
    def update(frame: int) -> None:
        _draw_imshow(
            ax,
            simulator.density,
            title=f"Density (Frame {frame})",
            cmap="hot",
            vmin=0,
            vmax=1,
        )

    return update


def _velocity_panel(ax: Axes, simulator: "SmokeSimulator") -> Callable[[int], None]:
    def update(_frame: int) -> None:
        vel_mag = simulator.get_velocity_magnitude()
        _draw_imshow(ax, vel_mag, title="Velocity Magnitude", cmap="viridis")

    return update


def _divergence_panel(ax: Axes, simulator: "SmokeSimulator") -> Callable[[int], None]:
    def update(_frame: int) -> None:
        divergence = _to_numpy(simulator.divergence)
        div_max = np.abs(divergence).max()
        _draw_imshow(
            ax,
            divergence,
            title=f"Divergence (max={div_max:.2e})",
            cmap="RdBu_r",
            vmin=-0.01,
            vmax=0.01,
        )

    return update


def _vorticity_panel(ax: Axes, simulator: "SmokeSimulator") -> Callable[[int], None]:
    def update(_frame: int) -> None:
        _draw_imshow(ax, simulator.vorticity, title="Vorticity", cmap="RdBu_r")

    return update


def create_2d_animation(
    simulator: "SmokeSimulator", frames: int = 200, interval: int = 30
) -> FuncAnimation:
    """Create animated visualization of 2D smoke simulation."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    panels = [
        _density_panel(axes[0, 0], simulator),
        _velocity_panel(axes[0, 1], simulator),
        _divergence_panel(axes[1, 0], simulator),
        _vorticity_panel(axes[1, 1], simulator),
    ]

    plt.tight_layout()

    return run_animation(
        fig,
        simulator,
        panels,
        frames,
        interval,
        message_fn=lambda frame: f"Frame {frame}: Running simulation step...",
    )
