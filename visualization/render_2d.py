"""2D visualization utilities for smoke simulation."""

import numpy as np
import matplotlib.pyplot as plt

from visualization._animation import run_animation


def _draw_imshow(ax, data, *, title, cmap, vmin=None, vmax=None, aspect="auto"):
    ax.clear()
    ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
    )
    ax.set_title(title)
    ax.axis("off")


def _density_panel(ax, simulator):
    def update(frame):
        _draw_imshow(
            ax,
            simulator.density,
            title=f"Density (Frame {frame})",
            cmap="hot",
            vmin=0,
            vmax=1,
        )

    return update


def _velocity_panel(ax, simulator):
    def update(_frame):
        vel_mag = simulator.get_velocity_magnitude()
        _draw_imshow(ax, vel_mag, title="Velocity Magnitude", cmap="viridis")

    return update


def _divergence_panel(ax, simulator):
    def update(_frame):
        divergence = simulator.divergence
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


def _vorticity_panel(ax, simulator):
    def update(_frame):
        _draw_imshow(ax, simulator.vorticity, title="Vorticity", cmap="RdBu_r")

    return update


def create_2d_animation(simulator, frames=200, interval=30):
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
