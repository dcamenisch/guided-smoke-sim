"""3D visualization utilities for smoke simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def render_slice(simulator, ax, slice_type="mid_z"):
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

    ax.imshow(
        data, cmap="hot", origin="lower", vmin=0, vmax=1, interpolation="none"
    )
    ax.set_title(title)
    ax.axis("off")


def render_volume_projection(simulator, ax):
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


def create_3d_animation(simulator, frames=200, interval=30):
    """Create animated visualization of 3D smoke simulation

    Args:
        simulator: SmokeSimulator3D instance
        frames: Number of frames to animate
        interval: Time between frames in milliseconds

    Returns:
        matplotlib FuncAnimation object
    """
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def animate(frame):
        """Animation function"""
        print(f"Frame {frame}: Advancing simulation...")
        simulator.step()

        # Render different views
        render_slice(simulator, axes[0, 0], "mid_z")
        render_slice(simulator, axes[0, 1], "mid_y")
        render_slice(simulator, axes[1, 0], "mid_x")
        render_volume_projection(simulator, axes[1, 1])

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=interval)
    plt.tight_layout()

    return anim
