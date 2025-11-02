"""2D visualization utilities for smoke simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_2d_animation(simulator, frames=200, interval=30):
    """Create animated visualization of 2D smoke simulation

    Args:
        simulator: SmokeSimulator2D instance
        frames: Number of frames to animate
        interval: Time between frames in milliseconds

    Returns:
        matplotlib FuncAnimation object
    """
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    def animate(frame):
        """Animation function"""
        print(f"Frame {frame}: Running simulation step...")
        simulator.step()

        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Density
        ax = axes[0, 0]
        ax.imshow(
            simulator.density, origin="lower", cmap="hot", vmin=0, vmax=1, aspect="auto"
        )
        ax.set_title(f"Density (Frame {frame})")
        ax.axis("off")

        # Velocity magnitude
        ax = axes[0, 1]
        vel_mag = simulator.get_velocity_magnitude()
        ax.imshow(vel_mag, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title("Velocity Magnitude")
        ax.axis("off")

        # Divergence
        ax = axes[1, 0]
        div_max = np.abs(simulator.divergence).max()
        ax.imshow(
            simulator.divergence,
            origin="lower",
            cmap="RdBu_r",
            vmin=-0.01,
            vmax=0.01,
            aspect="auto",
        )
        ax.set_title(f"Divergence (max={div_max:.2e})")
        ax.axis("off")

        # Vorticity
        ax = axes[1, 1]
        ax.imshow(simulator.vorticity, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title("Vorticity")
        ax.axis("off")

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=interval)
    plt.tight_layout()

    return anim
