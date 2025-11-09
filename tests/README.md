# Test Suite

Comprehensive tests for the smoke simulation framework.

## Quick Start

```bash
pip install pytest pytest-cov
pytest                              # Run all tests
pytest --cov=. --cov-report=html   # With coverage
pytest -k "test_2d"                # Pattern matching
```

## Test Files

- **test_core.py**: MAC grid data structures and initialization
- **test_simulator.py**: Simulation initialization, time stepping, divergence-free constraint
- **test_kernels.py**: Poisson solvers, vorticity, interpolation
- **test_physics.py**: Buoyancy, vorticity confinement
- **test_export.py**: NPZ export and data integrity
- **test_integration.py**: Full workflows, mass conservation, stability
