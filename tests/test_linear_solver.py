"""Tests for the Conjugate Gradient linear solver."""

import torch
import unittest
import sys
import os

sys.path.append(os.getcwd())

from kernels.linear_solver import ConjugateGradientSolver


class TestConjugateGradientSolver(unittest.TestCase):
    def setUp(self):
        self.nx = 32
        self.ny = 32
        self.dx = 1.0 / self.nx
        self.solver = ConjugateGradientSolver(self.nx, self.ny, dx=self.dx)
        self.device = torch.device("cpu")

    def test_solve_2d_random(self):
        """Test solving Ap = b with a random known p."""
        # Create a random pressure field
        p_true = torch.randn(self.ny, self.nx, device=self.device)

        # Create active mask (all fluid)
        active_mask = torch.ones(self.ny, self.nx, device=self.device)

        # Build matrix A
        A = self.solver.build_matrix(active_mask)

        # Compute b = A * p_true
        # Note: solve() expects rhs such that b = rhs * dx^2
        # So we can just pass b_unscaled = A * p_true / dx^2 as rhs

        b_true = self.solver.mv(A, p_true)
        rhs = b_true / (self.dx**2)

        # Solve
        p_init = torch.zeros_like(p_true)
        p_solved = self.solver.solve(rhs, active_mask, p_init, tol=1e-6, max_iter=2000)

        # Check residual
        b_solved = self.solver.mv(A, p_solved)
        residual = torch.norm(b_solved - b_true)

        print(f"Residual: {residual.item()}")
        self.assertTrue(residual < 1e-3)

        # Check solution accuracy (up to constant shift for Neumann)
        # Center the solutions
        p_true_centered = p_true - p_true.mean()
        p_solved_centered = p_solved - p_solved.mean()

        diff = torch.norm(p_solved_centered - p_true_centered) / torch.norm(
            p_true_centered
        )
        print(f"Relative Error: {diff.item()}")

        # Note: CG might not converge to exact p_true if matrix is singular (Neumann)
        # But residual should be small.
        # And if we project out the null space (constant mode), it should be close.

        self.assertTrue(diff < 5e-2)

    def test_solve_3d_random(self):
        """Test solving Ap = b with a random known p in 3D."""
        nz = 16
        solver = ConjugateGradientSolver(self.nx, self.ny, nz=nz, dx=self.dx)

        p_true = torch.randn(nz, self.ny, self.nx, device=self.device)
        active_mask = torch.ones(nz, self.ny, self.nx, device=self.device)

        A = solver.build_matrix(active_mask)
        b_true = solver.mv(A, p_true)
        rhs = b_true / (self.dx**2)

        p_init = torch.zeros_like(p_true)
        p_solved = solver.solve(rhs, active_mask, p_init, tol=1e-6, max_iter=2000)

        b_solved = solver.mv(A, p_solved)
        residual = torch.norm(b_solved - b_true)
        print(f"3D Residual: {residual.item()}")
        self.assertTrue(residual < 1e-3)

        p_true_centered = p_true - p_true.mean()
        p_solved_centered = p_solved - p_solved.mean()
        diff = torch.norm(p_solved_centered - p_true_centered) / torch.norm(
            p_true_centered
        )
        print(f"3D Relative Error: {diff.item()}")
        self.assertTrue(diff < 5e-2)

    def test_solve_with_obstacle(self):
        """Test solving with an obstacle in the center."""
        # Create active mask with a hole
        active_mask = torch.ones(self.ny, self.nx, device=self.device)
        active_mask[10:22, 10:22] = 0  # Solid block

        p_true = torch.randn(self.ny, self.nx, device=self.device)
        # Zero out pressure inside obstacle (it shouldn't matter, but for consistency)
        p_true = torch.where(active_mask > 0, p_true, torch.zeros_like(p_true))

        A = self.solver.build_matrix(active_mask)
        b_true = self.solver.mv(A, p_true)
        rhs = b_true / (self.dx**2)

        p_init = torch.zeros_like(p_true)
        p_solved = self.solver.solve(rhs, active_mask, p_init, tol=1e-6, max_iter=3000)

        # Check residual only on fluid cells
        b_solved = self.solver.mv(A, p_solved)
        residual = torch.norm((b_solved - b_true) * active_mask)
        print(f"Obstacle Residual: {residual.item()}")
        self.assertTrue(residual < 1e-3)


if __name__ == "__main__":
    unittest.main()
