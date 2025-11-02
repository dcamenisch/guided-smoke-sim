"""2D MAC grid for staggered velocity storage."""

import numpy as np


class MACGrid2D:
    """2D MAC grid for staggered velocity storage

    Velocity components are stored at face centers:
    - u stored at x-faces (ny, nx+1)
    - v stored at y-faces (ny+1, nx)
    """

    def __init__(self, nx, ny, dx):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        # u stored at x-faces (ny, nx+1)
        # v stored at y-faces (ny+1, nx)
        self.u_data = np.zeros((ny, nx + 1), dtype=np.float32)
        self.v_data = np.zeros((ny + 1, nx), dtype=np.float32)

    def reset(self):
        """Reset all velocity components to zero"""
        self.u_data.fill(0.0)
        self.v_data.fill(0.0)
