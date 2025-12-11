"""Differentiable Conjugate Gradient Solver for Poisson Equation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function

Tensor = torch.Tensor


class ConjugateGradientSolver:
    def __init__(self, nx: int, ny: int, nz: int = 1, dx: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.ndim = 3 if nz > 1 else 2
        self.device = torch.device("cpu")

    def solve(
        self,
        rhs: Tensor,
        active_mask: Tensor,
        p_init: Tensor,
        tol: float = 1e-5,
        max_iter: int = 5000,
    ) -> Tensor:
        """Solve Ap = rhs using Conjugate Gradient.

        Args:
            rhs: Right-hand side of Poisson equation (e.g. -div(u)/dt)
            active_mask: Boolean mask (1 for fluid, 0 for solid)
            p_init: Initial guess for pressure
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Solved pressure field
        """
        self.device = rhs.device

        # Build system matrix A (represented as sparse diagonals)
        A = self.build_matrix(active_mask)

        # Scale RHS by dx^2 because A represents the discrete Laplacian * dx^2
        # (coefficients are roughly integers like 4, -1)
        b = rhs * (self.dx**2)

        # Initialize
        p = p_init.clone()

        # Compute initial residual r = b - Ap
        Ap = self.mv(A, p)
        r = b - Ap

        # Apply preconditioner (Diagonal / Jacobi)
        # PInv = 1 / Adiag
        Adiag = A[0]
        # Avoid division by zero for non-fluid cells
        Adiag_inv = torch.where(Adiag != 0, 1.0 / Adiag, torch.zeros_like(Adiag))

        z = r * Adiag_inv
        s = z.clone()

        sigma = torch.sum(z * r)

        for i in range(max_iter):
            if torch.norm(r) < tol:
                break

            As = self.mv(A, s)

            # alpha = sigma / (s . As)
            denominator = torch.sum(s * As)
            if denominator == 0:
                break

            alpha = sigma / denominator

            p = p + alpha * s
            r = r - alpha * As

            if torch.norm(r) < tol:
                break

            z = r * Adiag_inv

            sigma_new = torch.sum(z * r)

            if sigma == 0:
                break

            beta = sigma_new / sigma
            s = z + beta * s
            sigma = sigma_new

        if i == max_iter - 1:
            print(
                f"CG Solver: Max iterations ({max_iter}) reached. Residual: {torch.norm(r).item()}"
            )

        return p

    def build_matrix(self, active_mask: Tensor) -> Tensor:
        """Build sparse matrix A for Laplacian operator.

        Returns:
            Stack of tensors [Adiag, Aplusx, Aplusy, (Aplusz)]
        """
        # active_mask: 1 for fluid, 0 for solid
        # We assume Neumann BCs at solid boundaries (flux = 0)
        # This means if neighbor is solid, the coefficient is 0.

        # Aplusx: coefficient for (i, j) -> (i, j+1) (x-direction)
        # It is -1 if both are fluid, 0 otherwise.

        # Pad active_mask to check neighbors
        # pad format: (left, right, top, bottom, front, back)

        if self.ndim == 2:
            # active_mask shape: (H, W)
            pad_x = (0, 1, 0, 0)
            pad_y = (0, 0, 0, 1)

            mask_pad_x = F.pad(active_mask, pad_x)  # (H, W+1)
            mask_pad_y = F.pad(active_mask, pad_y)  # (H+1, W)

            # Check x-neighbor (right)
            # mask_pad_x[..., 1:] is mask shifted left (aligns (i, j+1) with (i, j))
            is_fluid_x = (active_mask > 0) & (mask_pad_x[..., 1:] > 0)
            Aplusx = torch.where(
                is_fluid_x,
                -torch.ones_like(active_mask, dtype=torch.float32),
                torch.zeros_like(active_mask, dtype=torch.float32),
            )

            # Check y-neighbor (down? or up? usually y increases downwards in image, but here it's grid)
            # mask_pad_y[..., 1:, :] aligns (i+1, j) with (i, j)
            is_fluid_y = (active_mask > 0) & (mask_pad_y[..., 1:, :] > 0)
            Aplusy = torch.where(
                is_fluid_y,
                -torch.ones_like(active_mask, dtype=torch.float32),
                torch.zeros_like(active_mask, dtype=torch.float32),
            )

            # Adiag: sum of absolute values of all neighbors
            # We need Aminusx and Aminusy as well to compute diagonal
            # Aminusx[i, j] connects to (i, j-1). This is same as Aplusx[i, j-1]

            # Pad Aplusx to get left neighbor connection
            # pad (1, 0, 0, 0) -> shift right
            Aplusx_pad = F.pad(Aplusx, (1, 0, 0, 0))
            Aminusx = Aplusx_pad[..., :-1]  # (H, W)

            # Pad Aplusy to get top neighbor connection
            # pad (0, 0, 1, 0)
            Aplusy_pad = F.pad(Aplusy, (0, 0, 1, 0))
            Aminusy = Aplusy_pad[..., :-1, :]

            Adiag = -(Aplusx + Aminusx + Aplusy + Aminusy)

            # For non-fluid cells, set Adiag to 1 to avoid singularity (p=0)
            # Add small epsilon to diagonal to handle null space (pure Neumann BCs)
            Adiag = torch.where(active_mask > 0, Adiag + 1e-3, torch.ones_like(Adiag))

            return torch.stack([Adiag, Aplusx, Aplusy])

        else:
            # 3D case
            # active_mask shape: (D, H, W)
            pad_x = (0, 1, 0, 0, 0, 0)
            pad_y = (0, 0, 0, 1, 0, 0)
            pad_z = (0, 0, 0, 0, 0, 1)

            mask_pad_x = F.pad(active_mask, pad_x)
            mask_pad_y = F.pad(active_mask, pad_y)
            mask_pad_z = F.pad(active_mask, pad_z)

            is_fluid_x = (active_mask > 0) & (mask_pad_x[..., 1:] > 0)
            Aplusx = torch.where(
                is_fluid_x,
                -torch.ones_like(active_mask, dtype=torch.float32),
                torch.zeros_like(active_mask, dtype=torch.float32),
            )

            is_fluid_y = (active_mask > 0) & (mask_pad_y[..., 1:, :] > 0)
            Aplusy = torch.where(
                is_fluid_y,
                -torch.ones_like(active_mask, dtype=torch.float32),
                torch.zeros_like(active_mask, dtype=torch.float32),
            )

            is_fluid_z = (active_mask > 0) & (mask_pad_z[1:, ...] > 0)
            Aplusz = torch.where(
                is_fluid_z,
                -torch.ones_like(active_mask, dtype=torch.float32),
                torch.zeros_like(active_mask, dtype=torch.float32),
            )

            Aplusx_pad = F.pad(Aplusx, (1, 0, 0, 0, 0, 0))
            Aminusx = Aplusx_pad[..., :-1]

            Aplusy_pad = F.pad(Aplusy, (0, 0, 1, 0, 0, 0))
            Aminusy = Aplusy_pad[..., :-1, :]

            Aplusz_pad = F.pad(Aplusz, (0, 0, 0, 0, 1, 0))
            Aminusz = Aplusz_pad[:-1, ...]

            Adiag = -(Aplusx + Aminusx + Aplusy + Aminusy + Aplusz + Aminusz)
            Adiag = torch.where(active_mask > 0, Adiag + 1e-3, torch.ones_like(Adiag))

            return torch.stack([Adiag, Aplusx, Aplusy, Aplusz])

    def mv(self, A: Tensor, b: Tensor) -> Tensor:
        """Matrix-vector multiplication Ap."""
        Adiag = A[0]
        Aplusx = A[1]
        Aplusy = A[2]

        # b is (H, W) or (D, H, W)

        # Pad b to access neighbors
        if self.ndim == 2:
            pad_x = (1, 1, 0, 0)
            pad_y = (0, 0, 1, 1)

            b_pad_x = F.pad(b, pad_x)
            b_pad_y = F.pad(b, pad_y)

            # Aplusx[i, j] * b[i, j+1]
            # Aplusx corresponds to connection to right.
            # b_pad_x[..., 2:] is b shifted left (i, j+1)

            # Aminusx[i, j] * b[i, j-1]
            # Aminusx is Aplusx[i, j-1] (from previous cell's perspective)
            # Wait, Aplusx[i, j-1] connects (i, j-1) to (i, j).
            # So coeff is Aplusx[i, j-1].
            # b_pad_x[..., :-2] is b shifted right (i, j-1)

            # Pad Aplusx to get Aplusx[i, j-1]
            Aplusx_pad = F.pad(Aplusx, (1, 0, 0, 0))
            Aminusx = Aplusx_pad[..., :-1]

            # Pad Aplusy to get Aplusy[i-1, j]
            Aplusy_pad = F.pad(Aplusy, (0, 0, 1, 0))
            Aminusy = Aplusy_pad[..., :-1, :]

            Ab = (
                Adiag * b
                + Aplusx * b_pad_x[..., 2:]
                + Aminusx * b_pad_x[..., :-2]
                + Aplusy * b_pad_y[..., 2:, :]
                + Aminusy * b_pad_y[..., :-2, :]
            )
            return Ab

        else:
            Aplusz = A[3]

            pad_x = (1, 1, 0, 0, 0, 0)
            pad_y = (0, 0, 1, 1, 0, 0)
            pad_z = (0, 0, 0, 0, 1, 1)

            b_pad_x = F.pad(b, pad_x)
            b_pad_y = F.pad(b, pad_y)
            b_pad_z = F.pad(b, pad_z)

            Aplusx_pad = F.pad(Aplusx, (1, 0, 0, 0, 0, 0))
            Aminusx = Aplusx_pad[..., :-1]

            Aplusy_pad = F.pad(Aplusy, (0, 0, 1, 0, 0, 0))
            Aminusy = Aplusy_pad[..., :-1, :]

            Aplusz_pad = F.pad(Aplusz, (0, 0, 0, 0, 1, 0))
            Aminusz = Aplusz_pad[:-1, ...]

            Ab = (
                Adiag * b
                + Aplusx * b_pad_x[..., 2:]
                + Aminusx * b_pad_x[..., :-2]
                + Aplusy * b_pad_y[..., 2:, :]
                + Aminusy * b_pad_y[..., :-2, :]
                + Aplusz * b_pad_z[2:, ...]
                + Aminusz * b_pad_z[:-2, ...]
            )
            return Ab
