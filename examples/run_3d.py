"""
3D Smoke Simulation - Example script

This matches the C++ reference implementation exactly.
Run this script to see the 3D smoke simulation in action.

Usage:
    python run_3d.py                    # Run interactive animation with MacCormack
    python run_3d.py --semi-lagrangian  # Use semi-Lagrangian advection
    python run_3d.py --export           # Export simulation states to output/sim_3d/
    python run_3d.py --export --frames 100 --output my_output/
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from simulation import SmokeSimulator3D
from visualization import create_3d_animation


def main():
    parser = argparse.ArgumentParser(description="3D Smoke Simulation")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export simulation states to NPZ files instead of showing animation",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Number of frames to generate (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/sim_3d",
        help="Output directory for exported simulation states (default: output/sim_3d)",
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

    args = parser.parse_args()

    print("Starting 3D Smoke Simulation...")
    advection_method = "Semi-Lagrangian" if args.semi_lagrangian else "MacCormack"
    print(f"Using {advection_method} advection")
    if args.vorticity > 0.0:
        print(f"Vorticity confinement enabled (epsilon={args.vorticity})")

    # Use same resolution as original
    sim = SmokeSimulator3D(
        nx=64,
        ny=96,
        nz=64,
        use_maccormack=not args.semi_lagrangian,
        vorticity_epsilon=args.vorticity,
    )

    if args.export:
        print(f"Exporting {args.frames} simulation states to {args.output}/...")

        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run simulation and export each frame
        for frame in range(args.frames):
            print(f"Frame {frame}/{args.frames}: Running simulation step...")
            sim.step()

            # Export simulation state
            filepath = output_path / f"state_{frame:04d}.npz"
            sim.export_to_npz(filepath, timestep=frame)

        print(f"\nExport complete! {args.frames} states saved to {args.output}/")
    else:
        print("Starting animation...")
        anim = create_3d_animation(sim, frames=args.frames, interval=args.interval)
        plt.show()


if __name__ == "__main__":
    main()
