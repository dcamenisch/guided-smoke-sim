"""2D MAC grid for staggered velocity storage backed by Torch tensors."""

from __future__ import annotations

from typing import Optional, Union

import torch


class MACGrid2D:
    """2D MAC grid for staggered velocity storage

    Velocity components are stored at face centers:
    - u stored at x-faces (ny, nx+1)
    - v stored at y-faces (ny+1, nx)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype

        # u stored at x-faces (ny, nx+1)
        # v stored at y-faces (ny+1, nx)
        self.u_data = torch.zeros((ny, nx + 1), dtype=self.dtype, device=self.device)
        self.v_data = torch.zeros((ny + 1, nx), dtype=self.dtype, device=self.device)

    def reset(self) -> None:
        """Reset all velocity components to zero (differentiable)"""
        self.u_data = torch.zeros_like(self.u_data)
        self.v_data = torch.zeros_like(self.v_data)

    def to(self, device: Union[str, torch.device]) -> "MACGrid2D":
        """Move grid data to a different device."""
        device = torch.device(device)
        self.device = device
        self.u_data = self.u_data.to(device)
        self.v_data = self.v_data.to(device)
        return self
