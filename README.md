# Smoke Simulation

A 2D and 3D smoke simulator using the MAC (Marker-and-Cell) grid and Numba JIT compilation.

## Project Structure

```bash
smoke-sim/
├── core/                   # Grid data structures
│   ├── grid_2d.py         # 2D MAC grid
│   └── grid_3d.py         # 3D MAC grid
├── kernels/               # Numba-optimized computational kernels
│   ├── interpolation.py   # Bilinear/trilinear interpolation
│   ├── poisson.py         # Poisson equation solvers (Jacobi)
│   ├── advection.py       # Semi-Lagrangian advection
│   ├── velocity.py        # Velocity correction (pressure projection)
│   └── operators.py       # Differential operators (vorticity)
├── simulation/            # Simulation engines
│   ├── base_simulator.py  # Shared simulation logic
│   ├── simulator_2d.py    # 2D smoke simulator
│   └── simulator_3d.py    # 3D smoke simulator
├── visualization/         # Rendering and animation
│   ├── render_2d.py       # 2D visualization utilities
│   └── render_3d.py       # 3D visualization utilities
├── examples/              # Example scripts
│   ├── run_2d.py         # Run 2D simulation
│   └── run_3d.py         # Run 3D simulation
└── (legacy files)         # Original files kept for reference
    ├── simulation_2d.py
    ├── simulation.py
    └── macgrid.py
```

## Quick Start

Make sure you're in the project root directory before running the examples.

### Run 2D Simulation

```bash
python examples/run_2d.py
```

### Run 3D Simulation

```bash
python examples/run_3d.py
```

**Note**: Close the matplotlib window to exit the animation.

### Programmatic Usage

```python
from simulation import SmokeSimulator2D
from visualization import create_2d_animation
import matplotlib.pyplot as plt

# Create simulator
sim = SmokeSimulator2D(nx=128, ny=192)

# Run simulation steps manually
for i in range(100):
    sim.step()

# Or create an animation
anim = create_2d_animation(sim, frames=200, interval=30)
plt.show()
```

## Implementation Details

### Simulation Pipeline

Each time step follows this sequence:

1. **Add Source**: Inject smoke density at source location
2. **Apply Forces**: Add buoyancy force proportional to density
3. **Pressure Projection**:
   - Set boundary conditions
   - Compute velocity divergence
   - Solve Poisson equation for pressure
   - Correct velocity to be divergence-free
   - Compute vorticity
4. **Advection**:
   - Advect density using semi-Lagrangian method
   - Advect velocity components

### Boundary Conditions

- **Velocity**: Mixed Dirichlet (no-slip) and Neumann (no-penetration) conditions
- **Pressure**: Neumann conditions (zero gradient at boundaries)

### Grid Layout (MAC Grid)

2D Example:

- Pressure/Density: Cell centers (ny, nx)
- u-velocity: x-faces (ny, nx+1)
- v-velocity: y-faces (ny+1, nx)

3D Example:

- Pressure/Density: Cell centers (nz, ny, nx)
- u-velocity: x-faces (nz, ny, nx+1)
- v-velocity: y-faces (nz, ny+1, nx)
- w-velocity: z-faces (nz+1, ny, nx)
