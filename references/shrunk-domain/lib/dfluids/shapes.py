from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

from .grid import GridType
from . import utils


class ShapeType(Enum):
    NOTHING = 0
    BOX = 1
    BOXSET = 2
    SPHERE = 4
    BLOB = 8

    def __int__(self):
        return self.value


class Shape(object):
    def __init__(self, solver):
        self.solver = solver
        self.device = self.solver.get_device()

    def apply_to_grid(self, grid, value, center, size):
        raise NotImplementedError("Shape.apply_to_grid not implemented.")

    def compute_level_set(self, center, size):
        raise NotImplementedError("Shape.compute_level_set not implemented.")


class Box(Shape):
    def __init__(self, solver):
        """Initialize a Box shape with center position and half side size.

        Args:
            center(torch.Tensor): center position of the box.
            size(torch.Tensor): half side size in all dimensions of the box.
        """
        Shape.__init__(self, solver)
        # create reference source grid
        self.grid_ref = self.solver.create_grid(GridType.FLAG)
        if self.solver.is_2d():
            W, H = self.solver.get_grid_size()
            self.grid_ref[
                :,
                int(H / 4.0) : int(3.0 * H / 4.0) + 1,
                int(W / 4.0) : int(3.0 * W / 4.0) + 1,
            ].fill_(1)
        else:
            W, H, D = self.solver.get_grid_size()
            self.grid_ref[
                :,
                int(D / 4.0) : int(3.0 * H / 4.0) + 1,
                int(H / 4.0) : int(3.0 * H / 4.0) + 1,
                int(W / 4.0) : int(3.0 * W / 4.0) + 1,
            ].fill_(1)

    def compute_level_set(self, center, size):
        if size.numel() == 1:
            size = torch.cat((size, size, size))
        if self.solver.is_2d():
            center = center[:2]
            size = size[:2]
        center = center.to(self.device).to(self.solver.get_dtype())
        size = size.to(self.device).to(self.solver.get_dtype())
        if self.solver.is_2d():
            center = center.unsqueeze(1).unsqueeze(2)
            size = size.unsqueeze(1).unsqueeze(2)
        else:
            center = center.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            size = size.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mgrid = utils.create_mesh_grid(
            size=self.solver.get_torch_size_level(),
            device=self.device,
            dtype=self.solver.get_dtype(),
        )
        mgrid = utils.normalize_mesh_grid(mgrid)
        dist_to_edge = torch.abs(mgrid - center) - size
        dist_outside = torch.norm(
            torch.max(
                dist_to_edge, torch.Tensor([0.0]).to(self.device).to(dist_to_edge.dtype)
            ),
            dim=0,
        )
        dist_inside = torch.min(
            torch.max(dist_to_edge, dim=0, keepdim=True)[0],
            torch.Tensor([0.0]).to(self.device).to(dist_to_edge.dtype),
        )
        return dist_outside + dist_inside

    def apply_to_grid(self, grid, value, center, size):
        if size.numel() == 1:
            size = torch.cat((size, size, size))
        if self.solver.is_2d():
            center = center[:2]
            size = size[:2]
        if grid.dtype == torch.float32 or grid.dtype == torch.float64:
            return self._apply_to_real_grid(grid, value, center, size)
        elif grid.dtype == torch.uint8:
            return self._apply_to_flag_grid(grid, value, center, size)
        else:
            raise ValueError("grid does not have a standard dtype.")

    # non-differentiable
    def _apply_to_flag_grid(self, grid, value, center, size):
        grid_size = torch.Tensor(self.solver.get_grid_size(), device=self.device)
        center_on_grid = (
            (torch.Tensor(center, device=self.device) + 1.0) / 2.0 * grid_size
        )
        size_on_grid = torch.Tensor(size, device=self.device) / 2.0 * grid_size
        p1 = (center_on_grid - size_on_grid).to(torch.int)
        p2 = (center_on_grid + size_on_grid).to(torch.int)
        if self.solver.is_2d():
            grid[:, p1[1] : p2[1], p1[0] : p2[0]].fill_(value)
        else:
            grid[:, p1[2] : p2[2], p1[1] : p2[1], p1[0] : p2[0]].fill_(value)
        return grid

    def _apply_to_real_grid(self, grid, value, center, size):
        center = center.to(self.device).to(self.solver.get_dtype())
        size = size.to(self.device).to(self.solver.get_dtype())
        grid_ref = value * self.grid_ref.to(self.solver.get_dtype()).unsqueeze(0)
        # scale = 0.5 * torch.inverse(torch.diag(size))
        scale = 0.5 * torch.diag(1.0 / size)
        translation = -torch.matmul(scale, center.unsqueeze(1))
        theta = torch.cat((scale, translation), dim=1).unsqueeze(0)
        flow = F.affine_grid(theta=theta, size=grid_ref.size(), align_corners=True)
        grid_src = F.grid_sample(grid_ref, flow, align_corners=True).squeeze(0)
        grid_new = torch.clamp(grid + grid_src, min=0.0, max=1.0)
        return grid_new


class BoxSet(Shape):
    def __init__(self, solver):
        Shape.__init__(self, solver)
        self.box = self.solver.create_shape(ShapeType.BOX)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)

    def compute_level_set(self, center_list, size_list):
        if len(center_list) != len(size_list):
            raise RuntimeError(
                "center_list and size_list should have " "the same length."
            )
        phi = self.box.compute_level_set(center_list[0], size_list[0])
        for i in range(1, len(center_list)):
            phi_box = self.box.compute_level_set(center_list[i], size_list[i])
            phi = self.level_op.union(phi, phi_box)
        return phi


class Sphere(Shape):
    def __init__(self, solver):
        Shape.__init__(self, solver)
        if self.solver.is_2d():
            standard_center = torch.Tensor([0.0, 0.0])
        else:
            standard_center = torch.Tensor([0.0, 0.0, 0.0])
        # rescale size s.t. sphere is not distorted in non-cubic scenes
        shortest = min(self.solver.get_grid_size())
        ratio = (
            torch.Tensor(self.solver.get_grid_size())
            .to(self.device)
            .to(self.solver.get_dtype())
            / shortest
        )
        self.ratio = (
            ratio.view(-1, 1, 1) if self.solver.is_2d() else ratio.view(-1, 1, 1, 1)
        )
        standard_size = torch.Tensor([0.5])
        phi_standard = self.compute_level_set(standard_center, standard_size)
        level_op = self.solver.create_grid_operator(GridType.LEVEL)
        is_inside_standard_sphere = level_op.is_obs(phi_standard)
        self.grid_ref = self.solver.create_grid(GridType.REAL)
        self.grid_ref[is_inside_standard_sphere] = 1.0

    def compute_level_set(self, center, size):
        if self.solver.is_2d():
            center = center[:2]
            size = size[:2]
        center = center.to(self.device).to(self.solver.get_dtype())
        size = size.to(self.device).to(self.solver.get_dtype())
        if self.solver.is_2d():
            center = center.view(-1, 1, 1)
            size = size.view(-1, 1, 1)
        else:
            center = center.view(-1, 1, 1, 1)
            size = size.view(-1, 1, 1, 1)
        mgrid = utils.create_mesh_grid(
            size=self.solver.get_torch_size_level(),
            device=self.device,
            dtype=self.solver.dtype,
        )
        mgrid = utils.normalize_mesh_grid(mgrid)
        mgrid = mgrid * self.ratio
        dist = torch.norm(mgrid - center, dim=0) - size
        return dist

    def apply_to_grid(self, grid, value, center, size):
        if self.solver.is_2d():
            center = center[:2]
        if size.numel() == 1:
            if self.solver.is_2d():
                size = torch.cat((size, size))
            else:
                size = torch.cat((size, size, size))
        center = center.to(self.device).to(self.solver.get_dtype())
        size = size.to(self.device).to(self.solver.get_dtype())
        grid_ref = value * self.grid_ref.to(self.solver.get_dtype()).unsqueeze(0)
        # scale = 0.5 * torch.inverse(torch.diag(size))
        scale = 0.5 * torch.diag(1.0 / size)
        translation = -torch.matmul(scale, center.unsqueeze(1))
        theta = torch.cat((scale, translation), dim=1).unsqueeze(0)
        flow = F.affine_grid(theta=theta, size=grid_ref.size(), align_corners=True)
        grid_src = F.grid_sample(grid_ref, flow, align_corners=True).squeeze(0)
        grid_new = torch.clamp(grid + grid_src, min=0.0, max=1.0)
        return grid_new

    def compute_volume(self, size):
        """Computes the analytic volume of the blob."""
        if self.solver.is_2d():
            return np.pi * size ** 2
        else:
            return 4.0 / 3.0 * np.pi * size ** 3


class Blob(Shape):
    def __init__(self, solver, n_sphere=8):
        Shape.__init__(self, solver)
        self.n_sphere = n_sphere
        self.theta_set = (
            torch.Tensor([2 * np.pi / n_sphere * x for x in range(self.n_sphere)])
            .to(self.device)
            .to(self.solver.get_dtype())
        )
        self.sphere = self.solver.create_shape(ShapeType.SPHERE)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)

    def compute_level_set(self, center, size, outer_size_set):
        """Computes the level set using SDF.

        Args:
            center(torch.Tensor): center coordinate of the middle sphere
            size(torch.Tensor): radius of the middle sphere
            outer_size_set(torch.Tensor): radius set of the outer spheres
                around the middle sphere.
        """
        if outer_size_set.size(0) != self.n_sphere:
            raise ValueError(
                "Size of outer_size_set should be equal " "to number of outer spheres."
            )
        if self.solver.is_2d():
            center = center[:2]
            size = size[:2]
        center = center.to(self.device).to(self.solver.get_dtype())
        size = size.to(self.device).to(self.solver.get_dtype())
        outer_size_set = outer_size_set.to(self.device).to(self.solver.get_dtype())
        phi = self.sphere.compute_level_set(center, size)
        dx_set = size * torch.cos(self.theta_set)
        dy_set = size * torch.sin(self.theta_set)
        if self.solver.is_2d():
            outer_center_offset = torch.stack((dx_set, dy_set), dim=-1)
        else:
            outer_center_offset = torch.stack((dx_set, dy_set, 0), dim=-1)
        outer_center_set = center.unsqueeze(0) + outer_center_offset

        for i in range(self.n_sphere):
            outer_center = outer_center_set[i]
            outer_size = outer_size_set[i]
            if outer_size >= 0:
                phi_i = self.sphere.compute_level_set(outer_center, outer_size)
                phi = self.level_op.union(phi, phi_i)
            else:
                phi_i = self.sphere.compute_level_set(outer_center, -outer_size)
                phi = self.level_op.subtraction(phi, phi_i)
        return phi

    def compute_volume(self, size, outer_size_set):
        """Computes the analytic volume of the blob."""
        center_volume = self.sphere.compute_volume(size)
        outer_volume = torch.sum(
            (
                torch.sign(outer_size_set)
                * self.sphere.compute_volume(torch.abs(outer_size_set))
            )
        )
        volume = center_volume + outer_volume
        return volume


class ShapeFactory(object):
    def __init__(self, solver):
        self.solver = solver

    def create(self, shape_type, **kwargs):
        if shape_type == ShapeType.BOX:
            return Box(solver=self.solver)
        elif shape_type == ShapeType.BOXSET:
            return BoxSet(solver=self.solver)
        elif shape_type == ShapeType.SPHERE:
            return Sphere(solver=self.solver)
        elif shape_type == ShapeType.BLOB:
            return Blob(solver=self.solver, **kwargs)
        else:
            raise ValueError("Shape {} is not supported".format(shape_type))
