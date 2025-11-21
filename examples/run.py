"""
Unified Smoke Simulation - Example script

Run this script to see 2D or 3D smoke simulation in action.

Usage:
    python run.py                           # Run 2D interactive animation
    python run.py --3d                      # Run 3D interactive animation
    python run.py --export                  # Export 2D simulation states
    python run.py --3d --export             # Export 3D simulation states
    python run.py --export --frames 100     # Export custom number of frames
    python run.py --export --fps 30         # Export at 30 fps (default: 24)
    python run.py --vorticity 0.3           # Enable vorticity confinement
    python run.py --semi-lagrangian         # Use semi-Lagrangian advection
    python run.py --semi-lagrangian --rk-order 3  # Use SSPRK3 for improved accuracy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from simulation import SimulationConfig
from visualization import create_2d_animation, create_3d_animation


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke Simulation")
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="three_d",
        help="Run 3D simulation (default: 2D)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export simulation states to NPZ files instead of showing animation",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=720,
        help="Number of frames to generate (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for exported simulation states (default: output/sim_2d or output/sim_3d)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Animation interval in milliseconds (default: 30)",
    )
    parser.add_argument(
        "--semi-lagrangian",
        action="store_true",
        help="Use semi-Lagrangian advection instead of MacCormack (default: MacCormack)",
    )
    parser.add_argument(
        "--vorticity",
        type=float,
        default=0.0,
        help="Vorticity confinement strength (0.0=disabled, 0.1-0.5 typical) (default: 0.0)",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=1.0,
        help="Target CFL number for adaptive time stepping (default: 1.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Target frames per second for exported animation (default: 24.0)",
    )
    parser.add_argument(
        "--rk-order",
        type=int,
        choices=[1, 3],
        default=1,
        help="Runge-Kutta order for semi-Lagrangian backtracing (1=Euler, 3=SSPRK3) (default: 1)",
    )

    args = parser.parse_args()

    # Determine dimensionality
    ndim = 3 if args.three_d else 2

    # Set default output path if not specified
    if args.output is None:
        args.output = f"output/sim_{ndim}d"

    print(f"Starting {ndim}D Smoke Simulation...")
    if args.semi_lagrangian:
        rk_method = "Euler (RK1)" if args.rk_order == 1 else "SSPRK3"
        print(f"Using Semi-Lagrangian advection with {rk_method} backtracing")
    else:
        print(f"Using MacCormack advection")
    print(f"Adaptive time stepping enabled (CFL target={args.cfl})")
    if args.vorticity > 0.0:
        print(f"Vorticity confinement enabled (epsilon={args.vorticity})")

    # Calculate frame time and dt_max from fps
    frame_time = 1.0 / args.fps
    dt_max = frame_time  # Cap dt at frame time to prevent overshooting
    print(
        f"Target frame rate: {args.fps} fps (frame_time={frame_time:.4f}s, dt_max={dt_max:.4f}s)"
    )

    # Create simulator with appropriate dimensions
    common_config = dict(
        use_maccormack=not args.semi_lagrangian,
        advection_rk_order=args.rk_order,
        vorticity_epsilon=args.vorticity,
        cfl_target=args.cfl,
        dt_max=dt_max,
    )

    if ndim == 2:
        config = SimulationConfig(nx=128, ny=192, **common_config)
    else:
        config = SimulationConfig(nx=64, ny=128, nz=64, **common_config)

    sim = config.create_simulator()

    if args.export:
        print(f"Exporting {args.frames} simulation states to {args.output}/...")
        print(f"Each frame represents {frame_time:.4f}s of simulation time")

        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run simulation with constant frame rate
        frame_count = 0
        next_frame_time = 0.0

        while frame_count < args.frames:
            # Step simulation until we reach the next frame time
            target_time = next_frame_time + frame_time
            steps_this_frame = 0

            while sim.simulation_time < target_time:
                sim.step()
                steps_this_frame += 1

            # Export frame at target time
            filepath = output_path / f"state_{frame_count:04d}.npz"
            sim.export_to_npz(filepath, timestep=frame_count)

            print(
                f"Frame {frame_count}/{args.frames}: {steps_this_frame} steps, "
                f"t={sim.simulation_time:.4f}s (target={target_time:.4f}s), "
                f"avg_dt={sim.dt:.6f}"
            )

            next_frame_time = target_time
            frame_count += 1

        print(f"\nExport complete! {args.frames} states saved to {args.output}/")
        print(f"Total simulation time: {sim.simulation_time:.4f}s")
        print(f"Average: {args.frames / sim.simulation_time:.2f} frames/second")
    else:
        print("Starting animation...")
        if ndim == 2:
            anim = create_2d_animation(sim, frames=args.frames, interval=args.interval)
        else:
            anim = create_3d_animation(sim, frames=args.frames, interval=args.interval)
        plt.show()


if __name__ == "__main__":
    main()
