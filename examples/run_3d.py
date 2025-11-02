"""
3D Smoke Simulation - Example script

This matches the C++ reference implementation exactly.
Run this script to see the 3D smoke simulation in action.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from simulation import SmokeSimulator3D
from visualization import create_3d_animation


def main():
    print("Starting 3D Smoke Simulation...")
    print("This matches the C++ reference implementation")

    # Use same resolution as original
    sim = SmokeSimulator3D(nx=48, ny=72, nz=3)

    print("Starting animation...")
    anim = create_3d_animation(sim, frames=200, interval=30)
    plt.show()


if __name__ == "__main__":
    main()
