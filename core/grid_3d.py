"""3D MAC grid for staggered velocity storage."""

import numpy as np
import numpy.typing as npt


class MACGrid3D:
    """3D MAC grid for staggered velocity storage

    Velocity components are stored at face centers:
    - u stored at x-faces (nz, ny, nx+1)
    - v stored at y-faces (nz, ny+1, nx)
    - w stored at z-faces (nz+1, ny, nx)
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float) -> None:
        self.dx = dx
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Standardized naming to match 2D (u_data, v_data, w_data)
        self.u_data = np.zeros((nz, ny, nx + 1), dtype=np.float32)
        self.v_data = np.zeros((nz, ny + 1, nx), dtype=np.float32)
        self.w_data = np.zeros((nz + 1, ny, nx), dtype=np.float32)

    def reset(self) -> None:
        """Reset all velocity components to zero"""
        self.u_data.fill(0.0)
        self.v_data.fill(0.0)
        self.w_data.fill(0.0)
