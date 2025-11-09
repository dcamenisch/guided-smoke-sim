# Smoke Simulation

A high-performance 2D/3D smoke simulator using MAC grids and Numba JIT compilation.

## Quick Start

```bash
# 2D simulation with MacCormack advection
python examples/run.py

# 3D simulation
python examples/run.py --3d

# Semi-Lagrangian with SSPRK3 (higher accuracy)
python examples/run.py --semi-lagrangian --rk-order 3

# Enable vorticity confinement
python examples/run.py --vorticity 0.3

# Export simulation states
python examples/run.py --export --frames 200 --fps 24
```

## Features

- **Multiple Advection Schemes**:
  - MacCormack (2nd order, default)
  - Semi-Lagrangian with RK1 (Euler) or RK3 (SSPRK3)
- **Adaptive Time Stepping**: CFL-based automatic dt adjustment
- **Vorticity Confinement**: Optional turbulence enhancement
- **Numba JIT**: Optimized performance with parallel execution

## Python API

```python
from simulation import SimulationConfig
from visualization import create_2d_animation
import matplotlib.pyplot as plt

# Define configuration (nz=None for 2D, nz=int for 3D)
config = SimulationConfig(
  nx=128,
  ny=192,
  nz=None,
  use_maccormack=False,
  advection_rk_order=3,  # Use SSPRK3
  vorticity_epsilon=0.0,
  cfl_target=1.0,
  dt_max=0.1,
)

# Create simulator from configuration
sim = config.create_simulator()

# Run simulation
for _ in range(100):
  sim.step()

# Or create animation
anim = create_2d_animation(sim, frames=200)
plt.show()
```

### SimulationConfig

Use `SimulationConfig` when you need to share presets (CLI examples, test fixtures, batch runs). It centralizes simulator defaults and exposes `create_simulator()` for constructing matching `SmokeSimulator` instances.

## Advection Methods

### MacCormack (Default)

- 2nd order accurate with forward/backward correction
- Reduced numerical dissipation
- Best for preserving fine details

### Semi-Lagrangian

Backward particle tracing with configurable Runge-Kutta order:

- **RK1 (Euler)**: 1st order, fastest
- **RK3 (SSPRK3)**: 3rd order, higher accuracy
  - Coefficients: (2/9, 3/9, 4/9)
  - Better particle trajectories
  - 3x computational cost

## Simulation Pipeline

1. **Add Source**: Inject density at source location
2. **Apply Forces**: Buoyancy proportional to density
3. **Pressure Projection**: Enforce incompressibility (∇·u = 0)
4. **Advection**: Transport density and velocity

## Project Structure

```bash
smoke-sim/
├── core/                   # MAC grid data structures
│   ├── grid_2d.py         # 2D staggered grid (u, v velocity components)
│   └── grid_3d.py         # 3D staggered grid (u, v, w velocity components)
├── kernels/               # Numba JIT-compiled computational kernels
│   ├── advection.py       # Semi-Lagrangian (RK1/RK3) & MacCormack advection
│   ├── poisson.py         # Red-Black Gauss-Seidel pressure solver
│   ├── velocity.py        # Pressure gradient & velocity correction
│   ├── differential.py    # Divergence & vorticity computation
│   ├── interpolation.py   # Bilinear/trilinear interpolation
│   └── grid_ops.py        # MAC grid utilities (clamping, indexing)
├── physics/               # Physical force models
│   ├── buoyancy.py        # Buoyancy force (proportional to density)
│   ├── vorticity_confinement.py  # Turbulence enhancement
│   └── gravity.py         # Gravitational acceleration
├── simulation/            # Simulation engine
│   ├── base_simulator.py  # Abstract base class with simulation loop
│   └── simulator.py       # Unified 2D/3D implementation
├── visualization/         # Rendering utilities
│   ├── render_2d.py       # 2D matplotlib visualization
│   └── render_3d.py       # 3D volumetric rendering
├── examples/              # Example scripts & usage
│   └── run.py            # Main entry point with CLI arguments
├── utils/                 # Additional utilities
│   └── npz_to_vdb.py     # Convert simulation data to OpenVDB format
└── references/            # Reference implementations
```

## Command-Line Options

```bash
--3d                    3D simulation (default: 2D)
--semi-lagrangian       Use semi-Lagrangian advection
--rk-order {1,3}        Runge-Kutta order for backtracing
--vorticity FLOAT       Vorticity confinement strength (0.0-0.5)
--cfl FLOAT             Target CFL number (default: 1.0)
--export                Export states to NPZ files
--frames INT            Number of frames (default: 200)
--fps FLOAT             Target framerate (default: 24)
```

## Technical Details

### MAC Grid Layout

- **2D**: Pressure/density at centers (ny, nx), u at x-faces (ny, nx+1), v at y-faces (ny+1, nx)
- **3D**: Pressure/density at centers (nz, ny, nx), u/v/w at corresponding faces

### Boundary Conditions

- **Bottom**: No-slip (velocity = 0)
- **Top & Sides**: Open outflow (extrapolation)
- **Pressure**: Zero-gradient (Neumann)
