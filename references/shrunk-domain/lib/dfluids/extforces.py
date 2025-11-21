import torch
import torch.nn.functional as F

from .grid import CellType, GridType
from .shapes import ShapeType


class ExtForces(object):
    def __init__(self, solver):
        """
        Args:
            solver(FluidSolver): FluidSolver object in current simulation,
                to provide basic information such as dx and dt.
        """
        self.solver = solver
        self.device = self.solver.get_device()
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.level_op = self.solver.create_grid_operator(GridType.LEVEL)

    def init_flags(
        self,
        phi=None,
        boundary_width=0,
        wall="xXyYzZ",
        open="      ",
        inflow="      ",
        outflow="      ",
        nonboundary_cell_type=CellType.FLUID,
    ):
        """Initialize flag domain according to input type flags."""
        dim = 2 if self.solver.is_2d() else 3
        if (
            len(wall) < 2 * dim
            or len(open) < 2 * dim
            or len(inflow) < 2 * dim
            or len(outflow) < 2 * dim
        ):
            raise ValueError(
                "wall, open, inflow, outflow strings should have"
                "the same length as 2*dim."
            )
        nonfluid_width = boundary_width + 1
        # find out boundary types
        types = [0 for i in range(dim * 2)]
        done = [False for i in range(dim * 2)]
        type_flags = ["x", "X", "y", "Y", "z", "Z"]
        for i in range(dim * 2):
            for j in range(dim * 2):
                type_flag = type_flags[j]
                if not done[i]:
                    if open[i] == type_flag:
                        types[i] = CellType.OPEN
                        done[i] = True
                    elif inflow[i] == type_flag:
                        types[i] = CellType.INFLOW
                        done[i] = True
                    elif outflow[i] == type_flag:
                        types[i] = CellType.OUTFLOW
                        done[i] = True
                    elif wall[i] == type_flag:
                        types[i] = CellType.OBSTACLE
                        done[i] = True

        flags = self.solver.create_grid(GridType.FLAG)
        if self.solver.is_2d():
            # initialize boundaries
            flags[:, :, 0:nonfluid_width].fill_(int(types[0]))
            flags[:, :, -nonfluid_width:].fill_(int(types[1]))
            flags[:, 0:nonfluid_width, :].fill_(int(types[2]))
            flags[:, -nonfluid_width:, :].fill_(int(types[3]))
            # fill out the rest of the domain
            flags[
                :, nonfluid_width:-nonfluid_width, nonfluid_width:-nonfluid_width
            ].fill_(int(nonboundary_cell_type))
        else:
            # initialize boundaries
            flags[:, :, :, 0:nonfluid_width].fill_(int(types[0]))
            flags[:, :, :, -nonfluid_width:].fill_(int(types[1]))
            flags[:, :, 0:nonfluid_width, :].fill_(int(types[2]))
            flags[:, :, -nonfluid_width:, :].fill_(int(types[3]))
            flags[:, 0:nonfluid_width, :, :].fill_(int(types[4]))
            flags[:, -nonfluid_width:, :, :].fill_(int(types[5]))
            # fill out the rest of the domain
            flags[
                :,
                nonfluid_width:-nonfluid_width,
                nonfluid_width:-nonfluid_width,
                nonfluid_width:-nonfluid_width,
            ].fill_(int(nonboundary_cell_type))

        if phi is not None:
            # Assume the scene distance is scaled into [-1, 1]
            phi_walls = []
            box = self.solver.create_shape(ShapeType.BOX)
            if self.solver.is_2d():
                W, H = self.solver.get_grid_size()
                nonfluid_width_normed_x = (boundary_width + 1) / W * 2
                nonfluid_width_normed_y = (boundary_width + 1) / H * 2
                if types[0] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([-1, 0.0]),
                            torch.Tensor([nonfluid_width_normed_x, 10.0]),
                        )
                    )
                if types[1] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([1.0, 0.0]),
                            torch.Tensor([nonfluid_width_normed_x, 10.0]),
                        )
                    )
                if types[2] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, -1.0]),
                            torch.Tensor([10.0, nonfluid_width_normed_y]),
                        )
                    )
                if types[3] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, 1.0]),
                            torch.Tensor([10.0, nonfluid_width_normed_y]),
                        )
                    )
            else:
                W, H, D = self.solver.get_grid_size()
                nonfluid_width_normed_x = (boundary_width + 1) / W * 2
                nonfluid_width_normed_y = (boundary_width + 1) / H * 2
                nonfluid_width_normed_z = (boundary_width + 1) / D * 2
                if types[0] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([-1, 0.0, 0.0]),
                            torch.Tensor([nonfluid_width_normed_x, 10.0, 10.0]),
                        )
                    )
                if types[1] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([1.0, 0.0, 0.0]),
                            torch.Tensor([nonfluid_width_normed_x, 10.0, 10.0]),
                        )
                    )
                if types[2] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, -1.0, 0.0]),
                            torch.Tensor([10.0, nonfluid_width_normed_y, 10.0]),
                        )
                    )
                if types[3] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, 1.0, 0.0]),
                            torch.Tensor([10.0, nonfluid_width_normed_y, 10.0]),
                        )
                    )
                if types[4] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, 0.0, -1.0]),
                            torch.Tensor([10.0, 10.0, nonfluid_width_normed_z]),
                        )
                    )
                if types[5] == CellType.OBSTACLE:
                    phi_walls.append(
                        box.compute_level_set(
                            torch.Tensor([0.0, 0.0, 1.0]),
                            torch.Tensor([10.0, 10.0, nonfluid_width_normed_z]),
                        )
                    )
            if len(phi_walls) > 0:
                phi = phi_walls[0]
                for phi_wall in phi_walls:
                    phi = self.level_op.union(phi, phi_wall, alpha=1.0)
            else:
                phi = torch.ones(
                    self.solver.get_torch_size_level(),
                    device=self.device,
                    dtype=self.solver.get_dtype(),
                )
            # hot fix: set the values close to 0 to be 0
            phi = torch.where(
                phi < 1e-6,
                torch.zeros(
                    phi.size(), device=self.device, dtype=self.solver.get_dtype()
                ),
                phi,
            )
        return flags, phi

    def init_domain(self, boundary_width=0):
        # WARNING: only consider the boundaries to be solids
        box = self.solver.create_shape(ShapeType.BOX)
        nonfluid_width_normed = (boundary_width + 1) * self.solver.get_dx() * 2
        if self.solver.is_2d():
            box_size = torch.Tensor(
                [1 - nonfluid_width_normed, 1 - nonfluid_width_normed]
            )
            box_center = torch.Tensor([0.0, 0.0])
        else:
            box_size = torch.Tensor(
                [
                    1 - nonfluid_width_normed,
                    1 - nonfluid_width_normed,
                    1 - nonfluid_width_normed,
                ]
            )
            box_center = torch.Tensor([0.0, 0.0, 0.0])
        phi = -box.compute_level_set(box_center, box_size)
        return phi

    def set_inflow_bcs(self, flags, vel, vel_inflow_value):
        vel_inflow = self.solver.create_grid(GridType.MAC)
        if self.solver.is_2d():
            vel_inflow[:, ...] = vel_inflow_value.view(-1, 1, 1)
        else:
            vel_inflow[:, ...] = vel_inflow_value.view(-1, 1, 1, 1)
        # for x component
        # pad other two dimension for dummy extra size in velocity
        pad_x_neg = (1, 0, 0, 1) if self.solver.is_2d() else (1, 0, 0, 1, 0, 1)
        pad_x_pos = (0, 1, 0, 1) if self.solver.is_2d() else (0, 1, 0, 1, 0, 1)
        flags_pad_x_neg = F.pad(flags, pad=pad_x_neg)
        flags_pad_x_pos = F.pad(flags, pad=pad_x_pos)
        vel_inflow_bcs_x = torch.where(
            (
                (flags_pad_x_neg == int(CellType.INFLOW))
                | (flags_pad_x_pos == int(CellType.INFLOW))
            ),
            vel_inflow[0:1, ...],
            vel[0:1, ...],
        )
        # for y component
        pad_y_neg = (0, 1, 1, 0) if self.solver.is_2d() else (0, 1, 1, 0, 0, 1)
        pad_y_pos = (0, 1, 0, 1) if self.solver.is_2d() else (0, 1, 0, 1, 0, 1)
        flags_pad_y_neg = F.pad(flags, pad=pad_y_neg)
        flags_pad_y_pos = F.pad(flags, pad=pad_y_pos)
        vel_inflow_bcs_y = torch.where(
            (
                (flags_pad_y_neg == int(CellType.INFLOW))
                | (flags_pad_y_pos == int(CellType.INFLOW))
            ),
            vel_inflow[1:2, ...],
            vel[1:2, ...],
        )
        vel_inflow_bcs = torch.cat((vel_inflow_bcs_x, vel_inflow_bcs_y), dim=0)
        # for z component
        if self.solver.is_3d():
            pad_z_neg = (0, 1, 0, 1, 1, 0)
            pad_z_pos = (0, 1, 0, 1, 0, 1)
            flags_pad_z_neg = F.pad(flags, pad=pad_z_neg)
            flags_pad_z_pos = F.pad(flags, pad=pad_z_pos)
            vel_inflow_bcs_z = torch.where(
                (
                    (flags_pad_z_neg == int(CellType.INFLOW))
                    | (flags_pad_z_pos == int(CellType.INFLOW))
                ),
                vel_inflow[2:3, ...],
                vel[2:3, ...],
            )
            vel_inflow_bcs = torch.cat((vel_inflow_bcs, vel_inflow_bcs_z), dim=0)
        return vel_inflow_bcs

    def set_wall_bcs(self, phi_obs, grid):
        if grid.size() == self.solver.get_torch_size_real():
            grid_obs = self.solver.create_grid(GridType.REAL)
            is_obs = self.level_op.is_obs(phi_obs)
        elif grid.size() == self.solver.get_torch_size_mac():
            grid_obs = self.solver.create_grid(GridType.MAC)
            fractions = self.level_op.get_fractions(phi_obs)
            is_obs = fractions < 1e-6
        # WARNING: does not distinguish 3D MAC and 3D NODE
        elif grid.size() == self.solver.get_torch_size_node():
            grid_obs = self.solver.create_grid(GridType.NODE)
            is_obs = phi_obs < 1e-6
        else:
            raise ValueError("grid has a invalid size.")
        grid = torch.where(is_obs, grid_obs, grid)
        return grid

    def add_buoyancy(self, flags, phi_obs, vel, density, gravity):
        gravity = torch.Tensor(gravity).to(self.device).to(self.solver.get_dtype())
        dx = self.solver.get_dx()
        dt = self.solver.get_dt()
        density_mac_x = self.real_op.get_mac_x(density)
        density_mac_y = self.real_op.get_mac_y(density)
        vel_x = vel[0:1, ...] + density_mac_x * (-gravity[0]) * dt / dx
        vel_y = vel[1:2, ...] + density_mac_y * (-gravity[1]) * dt / dx
        vel_buo = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            density_mac_z = self.real_op.get_mac_z(density)
            vel_z = vel[2:3, ...] + density_mac_z * (-gravity[2]) * dt / dx
            vel_buo = torch.cat((vel_buo, vel_z), dim=0)
        vel_buo = self.set_wall_bcs(phi_obs, vel_buo)
        return vel_buo

    def add_extforces(self, flags, phi_obs, vel, force):
        if force.size() != vel.size():
            raise ValueError("force and vel must have the same size")
        dt = self.solver.get_dt()
        dx = self.solver.get_dx()
        vel = vel + force * dt / dx
        vel = self.set_wall_bcs(phi_obs, vel)
        return vel

    def extrapolate_mac_simple(self, phi_obs, vel, distance=1):
        fractions = self.level_op.get_fractions(phi_obs)
        vel_x = self._extrapolate_mac_simple_single_channel(
            fractions[0:1, ...], vel[0:1, ...], distance
        )
        vel_y = self._extrapolate_mac_simple_single_channel(
            fractions[1:2, ...], vel[1:2, ...], distance
        )
        vel = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            raise NotImplementedError(
                "ExtForces.extrapolate_mac_simple: 3D case not implemented."
            )
        return vel

    def _extrapolate_mac_simple_single_channel(self, fractions, vel, distance):
        if self.solver.is_3d():
            raise NotImplementedError(
                "ExtForces.extrapolate_mac_simple_single_channel: "
                "3D case not implemented."
            )
        for d in range(1, distance + 1):
            pad_x_pos = (0, d, 0, 0)
            fractions_pad_x_pos = torch.nn.functional.pad(fractions, pad=pad_x_pos)
            vel_pad_x_pos = torch.nn.functional.pad(vel, pad=pad_x_pos)
            vel = torch.where(
                (fractions == 0) * (fractions_pad_x_pos[..., d:] > 0),
                vel_pad_x_pos[..., d:],
                vel,
            )
            pad_x_neg = (d, 0, 0, 0)
            fractions_pad_x_neg = torch.nn.functional.pad(fractions, pad=pad_x_neg)
            vel_pad_x_neg = torch.nn.functional.pad(vel, pad=pad_x_neg)
            vel = torch.where(
                (fractions == 0) * (fractions_pad_x_neg[..., :-d] > 0),
                vel_pad_x_neg[..., :-d],
                vel,
            )
            pad_y_pos = (0, 0, 0, d)
            fractions_pad_y_pos = torch.nn.functional.pad(fractions, pad=pad_y_pos)
            vel_pad_y_pos = torch.nn.functional.pad(vel, pad=pad_y_pos)
            vel = torch.where(
                (fractions == 0) * (fractions_pad_y_pos[..., d:, :] > 0),
                vel_pad_y_pos[..., d:, :],
                vel,
            )
            pad_y_neg = (0, 0, d, 0)
            fractions_pad_y_neg = torch.nn.functional.pad(fractions, pad=pad_y_neg)
            vel_pad_y_neg = torch.nn.functional.pad(vel, pad=pad_y_neg)
            vel = torch.where(
                (fractions == 0) * (fractions_pad_y_neg[..., :-d, :] > 0),
                vel_pad_y_neg[..., :-d, :],
                vel,
            )
        return vel
