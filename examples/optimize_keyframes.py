"""
Keyframe Optimization Example.

This script demonstrates how to optimize a control force field to guide
the smoke simulation to match specific keyframes (target densities) at specific times.

Based on "Honey, I Shrunk the Domain" (EUROGRAPHICS 2021).
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import SmokeSimulator
from simulation.controller import FrequencyOptimizer


def create_target_density(
    nx: int, ny: int, center_y_ratio: float, device: torch.device
) -> torch.Tensor:
    """Create a target density shape (e.g., a Gaussian blob)."""
    y = torch.arange(ny, device=device).float()
    x = torch.arange(nx, device=device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    center_y, center_x = ny * center_y_ratio, nx * 0.5
    radius = min(nx, ny) * 0.15

    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    target = torch.exp(-dist_sq / (2 * (radius / 2) ** 2))
    return target


def main():
    # Parameters
    nx, ny = 64, 96
    num_frames = 40
    dt = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")

    # Initialize Simulator with proper settings
    sim = SmokeSimulator(
        nx=nx,
        ny=ny,
        dt=dt,
        device=device,
        enable_buoyancy=False,
        max_iterations=50,
        advection_rk_order=3,
    )

    # Define Keyframes
    target_mid = create_target_density(nx, ny, 0.5, device)
    target_top = create_target_density(nx, ny, 0.8, device)

    keyframes: Dict[int, torch.Tensor] = {20: target_mid, 39: target_top}
    keyframe_weights = {20: 1.0, 39: 5.0}

    # Base upward wind to help smoke reach targets
    base_wind_v = torch.full((ny + 1, nx), 0.08, device=device)

    # Source parameters
    source_mask = create_target_density(nx, ny, 0.15, device) > 0.5
    source_val = torch.tensor(1.0, device=device)

    # Cap max frequency at Nyquist limit
    max_resolution = min(nx, ny) // 2

    # Initialize Optimizer
    optimizer = FrequencyOptimizer(
        simulator=sim,
        initial_resolution=4,
        max_resolution=max_resolution,
        keyframes=keyframes,
        keyframe_weights=keyframe_weights,
        base_wind=base_wind_v,
        num_frames=num_frames,
        source_mask=source_mask,
        source_val=source_val,
    )

    print(
        f"Starting optimization for {num_frames} frames with {len(keyframes)} keyframes..."
    )
    print(
        f"Initial resolution: {optimizer.model.resolution}, Max resolution: {max_resolution}"
    )

    # Optimization Loop
    epoch = 0
    max_epochs = 100

    while epoch < max_epochs and optimizer.model.resolution <= max_resolution:
        try:
            current_loss = optimizer.step()

            if np.isnan(current_loss):
                print("Error: Loss is NaN! Stopping optimization.")
                break

            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}: Loss = {current_loss:.6f}, Resolution = {optimizer.model.resolution}"
                )

        except Exception as e:
            print(f"Optimization failed at Epoch {epoch}: {e}")
            break

        epoch += 1

    print(
        f"Optimization finished after {epoch} epochs. Best loss: {optimizer.best_loss:.6f}"
    )

    # Restore best parameters
    if optimizer.best_params is not None:
        optimizer.model.params.data = optimizer.best_params

    # Visualize Results
    frames = []
    with torch.no_grad():
        current_density = torch.zeros((ny, nx), device=device)
        current_u = torch.zeros((ny, nx + 1), device=device)
        current_v = torch.zeros((ny + 1, nx), device=device)

        force_u, force_v, _ = optimizer.model.compute_force()

        frames.append(current_density.cpu().numpy())

        for t in range(num_frames):
            if t < 15:
                current_density = torch.where(source_mask, source_val, current_density)

            total_force_v = force_v + base_wind_v

            sim.density = current_density
            sim.u = current_u
            sim.v = current_v
            sim.set_control_force(force_u, total_force_v, None)
            sim.step()

            current_density = sim.density
            current_u = sim.u
            current_v = sim.v
            frames.append(current_density.cpu().numpy())

    # Plot comparison
    plt.figure(figsize=(15, 5))

    # Plot Targets
    for i, (t, target) in enumerate(keyframes.items()):
        plt.subplot(2, len(keyframes) + 1, i + 2)
        plt.title(f"Target Frame {t}")
        plt.imshow(target.cpu().numpy(), origin="lower", vmin=0, vmax=1)
        plt.axis("off")

    # Plot Results at Keyframes
    for i, (t, target) in enumerate(keyframes.items()):
        plt.subplot(2, len(keyframes) + 1, len(keyframes) + 1 + i + 2)
        plt.title(f"Result Frame {t}")
        plt.imshow(frames[t], origin="lower", vmin=0, vmax=1)
        plt.axis("off")

    # Plot Initial
    plt.subplot(2, len(keyframes) + 1, 1)
    plt.title("Initial")
    plt.imshow(frames[0], origin="lower", vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("keyframe_optimization.png")
    print("Result saved to keyframe_optimization.png")

    # Save animation
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], origin="lower", vmin=0, vmax=1, animated=True)
    ax.set_title("Frame 0")

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save("optimization_animation.gif", writer="pillow", fps=10)
    print("Animation saved to optimization_animation.gif")


if __name__ == "__main__":
    main()
