import sys
from pathlib import Path
from typing import Dict

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.simulator import SmokeSimulator
from optimization.controller import FrequencyOptimizer


def main():
    # Parameters
    nx, ny = 64, 96
    num_frames = 100
    dt = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load keyframe
    target_path = "output/keyframes/wind_79.npz"
    target_data_np = np.load(target_path)["density"]
    target_date = torch.from_numpy(target_data_np).to(device, dtype=torch.float32)

    keyframes: Dict[int, torch.Tensor] = {79: target_date}
    keyframe_weights = {79: 1.0}

    band_radii = [1, 3, 8]
    phase_iters = [5, 5, 5]

    print(f"--- Keyframe Optimization ---")
    print(f"Running on {device}")

    # Initialize Simulator with proper settings
    sim = SmokeSimulator(
        nx=nx,
        ny=ny,
        dt=dt,
        device=device,
    )

    # Initialize Optimizer
    optimizer = FrequencyOptimizer(
        simulator=sim,
        keyframes=keyframes,
        keyframe_weights=keyframe_weights,
        num_frames=num_frames,
        band_radii=band_radii,
        phase_iters=phase_iters,
        use_checkpoint=False,
    )

    print(
        f"Starting optimization for {num_frames} frames with {len(keyframes)} keyframes..."
    )

    # Optimization Loop
    itr = 1
    prev_loss = float("inf")
    for phase in range(len(band_radii)):
        print(
            f"--- Phase {phase + 1}: Optimizing with band radius {band_radii[phase]} for {phase_iters[phase]} epochs ---"
        )

        for _ in range(phase_iters[phase]):
            loss = optimizer.step()
            print(f"Iteration {itr}: Loss = {loss:.6f}")

            if loss < prev_loss:
                prev_loss = loss
            else:
                print("No improvement, continuing to next phase early.")
                break

            itr += 1

    print("--- Optimization Complete ---")

    # Visualize Results
    optimizer.model.compute_force()
    force_u, force_v = optimizer.model.get_force_centered()
    force_u = force_u.detach().cpu().numpy()
    force_v = force_v.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.title("Optimized Control Force Field")

    # Downsample grid for clearer quiver plot
    step = 2
    ny_f, nx_f = force_u.shape
    y, x = np.mgrid[0:ny_f:step, 0:nx_f:step]

    # Plot magnitude as background
    magnitude = np.sqrt(force_u**2 + force_v**2)
    plt.imshow(magnitude, origin="lower", cmap="Blues", alpha=0.6, vmin=0.0)
    plt.colorbar(label="Magnitude")

    # Plot vectors
    plt.quiver(x, y, force_u[::step, ::step], force_v[::step, ::step], color="red")

    plt.savefig("optimized_force_field.png")
    print("Optimized force field saved to optimized_force_field.png")

    # Run simulation with optimized force
    frames_tensors = optimizer.evaluate(num_frames)
    frames = [f.cpu().numpy() for f in frames_tensors]

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
