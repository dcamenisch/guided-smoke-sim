"""3D MAC grid for staggered velocity storage backed by Torch tensors."""

from __future__ import annotations

from typing import Optional, Union

import torch


class MACGrid3D:
    """3D MAC grid for staggered velocity storage

    Velocity components are stored at face centers:
    - u stored at x-faces (nz, ny, nx+1)
    - v stored at y-faces (nz, ny+1, nx)
    - w stored at z-faces (nz+1, ny, nx)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dx = dx
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype

        # Standardized naming to match 2D (u_data, v_data, w_data)
        self.u_data = torch.zeros(
            (nz, ny, nx + 1), dtype=self.dtype, device=self.device
        )
        self.v_data = torch.zeros(
            (nz, ny + 1, nx), dtype=self.dtype, device=self.device
        )
        self.w_data = torch.zeros(
            (nz + 1, ny, nx), dtype=self.dtype, device=self.device
        )

    def reset(self) -> None:
        """Reset all velocity components to zero (differentiable)"""
        self.u_data = torch.zeros_like(self.u_data)
        self.v_data = torch.zeros_like(self.v_data)
        self.w_data = torch.zeros_like(self.w_data)

    def to(self, device: Union[str, torch.device]) -> "MACGrid3D":
        """Move grid tensors to a new device."""
        device = torch.device(device)
        self.device = device
        self.u_data = self.u_data.to(device)
        self.v_data = self.v_data.to(device)
        self.w_data = self.w_data.to(device)
        return self
