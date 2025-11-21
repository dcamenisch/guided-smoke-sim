"""3D visualization utilities for smoke simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from visualization._animation import run_animation

if TYPE_CHECKING:
    from simulation.simulator import SmokeSimulator


def render_slice(
    simulator: "SmokeSimulator", ax: Axes, slice_type: str = "mid_z"
) -> None:
    """Render a 2D slice of the 3D volume

    Args:
        simulator: SmokeSimulator3D instance
        ax: matplotlib axes to render on
        slice_type: Type of slice ("mid_z", "mid_y", "mid_x")
    """
    ax.clear()

    if slice_type == "mid_z":
        # Show xy slice at middle z
        z_slice = simulator.nz // 2
        data = simulator.density[z_slice, :, :]
        title = f"XY Slice (z={z_slice})"
    elif slice_type == "mid_y":
        # Show xz slice at middle y
        y_slice = simulator.ny // 2
        data = simulator.density[:, y_slice, :]
        title = f"XZ Slice (y={y_slice})"
    else:
        # Show yz slice at middle x
        x_slice = simulator.nx // 2
        data = simulator.density[:, :, x_slice]
        title = f"YZ Slice (x={x_slice})"

    ax.imshow(data, cmap="hot", origin="lower", vmin=0, vmax=1, interpolation="none")
    ax.set_title(title)
    ax.axis("off")


def render_volume_projection(simulator: "SmokeSimulator", ax: Axes) -> None:
    """Render maximum intensity projection

    Args:
        simulator: SmokeSimulator3D instance
        ax: matplotlib axes to render on
    """
    ax.clear()

    # Maximum intensity projection along z-axis
    projection = np.max(simulator.density, axis=0)

    ax.imshow(
        projection,
        cmap="hot",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax.set_title("Max Projection (along Z)")
    ax.axis("off")


def create_3d_animation(
    simulator: "SmokeSimulator", frames: int = 200, interval: int = 30
) -> FuncAnimation:
    """Create animated visualization of 3D smoke simulation."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    panels = [
        lambda _frame: render_slice(simulator, axes[0, 0], "mid_z"),
        lambda _frame: render_slice(simulator, axes[0, 1], "mid_y"),
        lambda _frame: render_slice(simulator, axes[1, 0], "mid_x"),
        lambda _frame: render_volume_projection(simulator, axes[1, 1]),
    ]

    plt.tight_layout()

    return run_animation(
        fig,
        simulator,
        panels,
        frames,
        interval,
        message_fn=lambda frame: f"Frame {frame}: Advancing simulation...",
    )
