from enum import Enum
import torch
import torch.nn.functional as F

from . import utils


class GridType(Enum):
    FLAG = 0
    REAL = 1
    MAC = 2
    LEVEL = 4
    NODE = 8

    def __int__(self):
        return self.value


class CellType(Enum):
    NOTHING = 0
    FLUID = 1
    OBSTACLE = 2
    EMPTY = 4
    INFLOW = 8
    OUTFLOW = 16
    OPEN = 32
    STICK = 64

    def __int__(self):
        return self.value


class GridOperator(object):
    def __init__(self, solver):
        self.solver = solver
        self.device = self.solver.get_device()
        if self.solver.is_2d():
            self.tensor_to_npy_trans_seq = (1, 2, 0)
            self.npy_to_tensor_trans_seq = (2, 0, 1)
        else:
            self.tensor_to_npy_trans_seq = (1, 2, 3, 0)
            self.npy_to_tensor_trans_seq = (3, 0, 1, 2)

    def grid2array(self, grid, **kwargs):
        tensor = grid.permute(self.tensor_to_npy_trans_seq)
        array = tensor.detach().cpu().numpy()
        return array

    def array2grid(self, array, **kwargs):
        tensor = torch.from_numpy(array).to(self.solver.get_dtype()).to(self.device)
        return tensor.permute(self.npy_to_tensor_trans_seq)

    def save_npy(self, grid, filename, **kwargs):
        array = self.grid2array(grid)
        utils.save_npy(array, filename)

    def save_vdb(self, grid, filename, **kwargs):
        array = self.grid2array(grid)
        utils.save_vdb(array, filename)

    def show_img(self, grid, colormap="RGB", **kwargs):
        array = self.grid2array(grid)
        utils.show_img(array, colormap)

    def save_img(self, grid, filename, colormap="RGB", **kwargs):
        array = self.grid2array(grid)
        utils.save_img(array, filename, colormap)

    def load_npy(self, filename, **kwargs):
        array = utils.load_npy(filename)
        return self.array2grid(array)

    def load_npz(self, filename, key, **kwargs):
        array = utils.load_npz(filename, key)
        return self.array2grid(array)

    def get_const_grid(self, value):
        raise NotImplementedError("Grid.get_const_grid not implemented.")

    def _get_const_grid(self, grid, value):
        if value.size(0) != grid.size(0):
            raise ValueError("value must have the same dimension as the grid.")
        if self.solver.is_2d():
            grid[:, ...] = value.view(-1, 1, 1)
        else:
            grid[:, ...] = value.view(-1, 1, 1, 1)
        return grid

    def trim_boundary(self, grid, bwidth=1):
        if self.solver.is_2d():
            return grid[:, bwidth:-bwidth, bwidth:-bwidth]
        else:
            return grid[:, bwidth:-bwidth, bwidth:-bwidth, bwidth:-bwidth]


class FlagGridOperator(GridOperator):
    # overload
    def array2grid(self, array, **kwargs):
        tensor = torch.from_numpy(array).to(torch.uint8).to(self.device)
        return tensor.permute(self.npy_to_tensor_trans_seq)


class RealGridOperator(GridOperator):
    # overload
    def save_img(
        self, grid, filename, colormap="RGB", project=True, transmit=0.05, **kwargs
    ):
        array = self.grid2array(grid)
        utils.save_img(array, filename, colormap, project, transmit)

    def get_const_grid(self, value):
        grid = self.solver.create_grid(GridType.REAL)
        return self._get_const_grid(grid, value)

    def get_mac_x(self, grid):
        pad = (1, 1, 1, 1) if self.solver.is_2d() else (1, 1, 1, 1, 1, 1)
        grid_pad = F.pad(grid.unsqueeze(0), pad=pad).squeeze(0)
        if self.solver.is_2d():
            mac_x = (grid_pad[:, 1:, 1:] + grid_pad[:, 1:, :-1]) * 0.5
        else:
            mac_x = (grid_pad[:, 1:, 1:, 1:] + grid_pad[:, 1:, 1:, :-1]) * 0.5
        return mac_x

    def get_mac_y(self, grid):
        pad = (1, 1, 1, 1) if self.solver.is_2d() else (1, 1, 1, 1, 1, 1)
        grid_pad = F.pad(grid.unsqueeze(0), pad=pad).squeeze(0)
        if self.solver.is_2d():
            mac_y = (grid_pad[:, 1:, 1:] + grid_pad[:, :-1, 1:]) * 0.5
        else:
            mac_y = (grid_pad[:, 1:, 1:, 1:] + grid_pad[:, 1:, :-1, 1:]) * 0.5
        return mac_y

    def get_mac_z(self, grid):
        if self.solver.is_2d():
            raise RuntimeError("2d data does not support z direction.")
        else:
            grid_pad = F.pad(grid.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1)).squeeze(0)
            mac_z = (grid_pad[:, 1:, 1:, 1:] + grid_pad[:, :-1, 1:, 1:]) * 0.5
        return mac_z

    def get_gradient(self, grid):
        pad = (1, 1, 1, 1) if self.solver.is_2d() else (1, 1, 1, 1, 1, 1)
        grid_pad = F.pad(grid.unsqueeze(0), pad=pad, mode="replicate").squeeze(0)
        if self.solver.is_2d():
            grad_x = grid_pad[:, 1:, 1:] - grid_pad[:, 1:, :-1]
            grad_y = grid_pad[:, 1:, 1:] - grid_pad[:, :-1, 1:]
            grad = torch.cat((grad_x, grad_y), dim=0)
        else:
            grad_x = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, 1:, 1:, :-1]
            grad_y = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, 1:, :-1, 1:]
            grad_z = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, :-1, 1:, 1:]
            grad = torch.cat((grad_x, grad_y, grad_z), dim=0)
        return grad

    def get_curl(self, grid):
        if self.solver.is_3d():
            raise NotImplementedError(
                "NodeGridOperator.get_curl: " "3D case not supported yet."
            )
        grid_pad = F.pad(grid.unsqueeze(0), pad=(1, 1, 1, 1), mode="replicate").squeeze(
            0
        )
        # in 2D case, the phi is a scalar field, equivalently on z direction
        # curl(phi) = (dphi/dy, -dphi/dx)
        grad_x = grid_pad[:, 1:, 1:] - grid_pad[:, 1:, :-1]
        grad_y = grid_pad[:, 1:, 1:] - grid_pad[:, :-1, 1:]
        curl = torch.cat((grad_y, -grad_x), dim=0)
        return curl

    # only implements front view rendering
    def get_rendered(self, grid, transmit=0.05):
        """Simple renderer used in TNST.

        Args:
            grid(torch.Tensor): of size [C, D, H, W]
            transmit: transmittance constant
        """
        if self.solver.is_2d():
            raise RuntimeError(
                "RealGridOperator.get_rendered only " "works for 3D case."
            )
        density = torch.flip(grid, dims=(1,))  # view density from front to back
        tau = torch.exp(-transmit * torch.cumsum(density, dim=1))
        rendered = torch.sum(density * tau, dim=1)
        rendered = rendered / torch.max(rendered)  # remap to [0, 1]
        return rendered


class MACGridOperator(GridOperator):
    def __init__(self, solver):
        GridOperator.__init__(self, solver)
        self.ext = self.solver.create_extforces()

    # overload
    def grid2array(self, grid, trim_boundary=False, **kwargs):
        if trim_boundary:
            if self.solver.is_2d():
                tensor = grid[:, :-1, :-1]
            else:
                tensor = grid[:, :-1, :-1, :-1]
        else:
            tensor = grid
        tensor = tensor.permute(self.tensor_to_npy_trans_seq)
        array = tensor.detach().cpu().numpy()
        return array

    # overload
    def array2grid(self, array, trim_boundary=False, **kwargs):
        tensor = torch.from_numpy(array).to(self.solver.get_dtype()).to(self.device)
        tensor = tensor.permute(self.npy_to_tensor_trans_seq)
        if trim_boundary:
            if self.solver.is_2d():
                grid_size = (tensor.size(0), tensor.size(1) + 1, tensor.size(2) + 1)
                grid = torch.zeros(
                    grid_size, dtype=self.solver.get_dtype(), device=self.device
                )
                grid[:, :-1, :-1] = tensor
            else:
                grid_size = (
                    tensor.size(0),
                    tensor.size(1) + 1,
                    tensor.size(2) + 1,
                    tensor.size(3) + 1,
                )
                grid = torch.zeros(
                    grid_size, dtype=self.solver.get_dtype(), device=self.device
                )
                grid[:, :-1, :-1, :-1] = tensor
        else:
            grid = tensor
        return grid

    # overload
    def save_img(self, grid, filename, trim_boundary=False, colormap="RGB", **kwargs):
        array = self.grid2array(grid, trim_boundary)
        utils.save_img(array, filename, colormap)

    # overload
    def save_npy(
        self, grid, filename, trim_boundary=False, add_redundant_channel=False, **kwargs
    ):
        array = self.grid2array(grid, trim_boundary)
        utils.save_npy(array, filename, add_redundant_channel)

    # overload
    def load_npy(self, filename, trim_boundary=False, **kwargs):
        array = utils.load_npy(filename)
        return self.array2grid(array, trim_boundary)

    # overload
    # NOTE: trim_boundary is default to be False
    def load_npz(self, filename, key, trim_boundary=False, **kwargs):
        array = utils.load_npz(filename, key)
        return self.array2grid(array, trim_boundary)

    def get_const_grid(self, value):
        grid = self.solver.create_grid(GridType.MAC)
        return self._get_const_grid(grid, value)

    def get_uniform_noise(self, min=0.0, max=1.0):
        """Generates uniform random noise in [min, max]"""
        if max < min:
            raise ValueError("max must be greater than min")
        noise_grid = torch.rand(
            self.solver.get_torch_size_mac(),
            dtype=self.solver.get_dtype(),
            device=self.device,
        )  # [0, 1]
        noise_grid = noise_grid * (max - min) + min
        return noise_grid

    def get_taylor_vortices(
        self,
        phi_obs,
        a=0.08,
        U=0.06,
        center_list=[torch.Tensor([-0.15, 0]), torch.Tensor([0.15, 0])],
    ):
        node_op = self.solver.create_grid_operator(GridType.NODE)
        if self.solver.is_3d():
            raise NotImplementedError(
                "MACGridOperator.get_taylor_vortices:" "3D case not supported yet."
            )
        n_vortices = len(center_list)
        vort = self.solver.create_grid(GridType.NODE)
        for i in range(n_vortices):
            if center_list[0].view(-1).size() != torch.Size([2]):
                raise ValueError("center_list[{}] has invalid size.".format(i))
            center = center_list[i].to(self.device).to(self.solver.get_dtype())
            mgrid = utils.create_mesh_grid(
                self.solver.get_torch_size_node(),
                device=self.device,
                dtype=self.solver.get_dtype(),
            )
            mgrid = utils.normalize_mesh_grid(mgrid)
            dist = mgrid - center.view(-1, 1, 1)
            r2 = dist[0] ** 2 + dist[1] ** 2
            vort = vort + U / a * (2 - r2 / (a * a)) * torch.exp(
                0.5 * (1 - r2 / (a * a))
            )
        vel = node_op.get_inverse_curl(phi_obs, vort)
        return vel

    def get_curl(self, grid):
        """compute the discrete curl of velocity field.

        Args:
            grid(torch.Tensor, GridType.MAC):
                velocity field of size [C, (D), H+1, W+1].

        Returns:
            vort(torch.Tensor, GridType.NODE):
                vorticity (stored on the MAC grid corner)
                of size [C, (D+1), H+1, W+1].
        """
        u = grid[0:1, ...]
        v = grid[1:2, ...]
        padx = (1, 0, 0, 0) if self.solver.is_2d() else (1, 0, 0, 0, 0, 0)
        pady = (0, 0, 1, 0) if self.solver.is_2d() else (0, 0, 1, 0, 0, 0)
        padz = None if self.solver.is_2d() else (0, 0, 0, 0, 1, 0)
        v_padx = F.pad(v, pad=padx)
        u_pady = F.pad(u, pad=pady)
        # vort_z = dv/dx-du/dy
        vort_z = (v_padx[..., 1:] - v_padx[..., :-1]) - (
            u_pady[..., 1:, :] - u_pady[..., :-1, :]
        )
        if self.solver.is_2d():
            vort = vort_z
        else:
            w = grid[2:3, ...]
            w_pady = F.pad(w, pad=pady)
            v_padz = F.pad(v, pad=padz)
            # vort_x = dw/dy - dv/dz
            vort_x = (w_pady[..., 1:, :] - w_pady[..., :-1, :]) - (
                v_padz[:, 1:, ...] - v_padz[:, :-1, ...]
            )
            u_padz = F.pad(u, pad=padz)
            w_padx = F.pad(w, pad=padx)
            # vort_y = du/dz - dw/dx
            vort_y = (u_padz[:, 1:, ...] - u_padz[:, :-1, ...]) - (
                w_padx[..., 1:] - w_padx[..., :-1]
            )
            vort = torch.cat((vort_x, vort_y, vort_z), dim=0)
        return vort

    def get_divergence(self, grid):
        """Computes the discrete divergence of velocity field.

        Args:
            grid(torch.Tensor, GridType.MAC):
                velocity field of size [C, (D), H+1, W+1].

        Returns:
            div(torch.Tensor, GridType.REAL):
                divergence of size [1, (D), H, W].
        """
        u = grid[0:1, ...]
        v = grid[1:2, ...]
        if self.solver.is_2d():
            div = u[:, :-1, 1:] - u[:, :-1, :-1] + v[:, 1:, :-1] - v[:, :-1, :-1]
        else:
            w = grid[2:3, ...]
            div = (
                u[:, :-1, :-1, 1:]
                - u[:, :-1, :-1, :-1]
                + v[:, :-1, 1:, :-1]
                - v[:, :-1, :-1, :-1]
                + w[:, 1:, :-1, :-1]
                - w[:, :-1, :-1, :-1]
            )
        return div

    def get_jacobian(self, grid):
        """Computes the discrete Jacobian of velocity field.

        Args:
            grid(torch.Tensor, GridType.MAC):
                velocity field of size [C, (D), H+1, W+1].

        Returns:
            jacobian(torch.Tensor):
                Jacobian (stored on the center of the grid cell)
                of size [C, C, (D), H, W]

        """
        u = grid[0:1, ...]
        v = grid[1:2, ...]
        u_macy = self._grid_sample_y(u)
        v_macx = self._grid_sample_x(v)
        if self.solver.is_2d():
            du_dx = u[:, :-1, 1:] - u[:, :-1, :-1]
            dv_dx = v_macx[:, :-1, 1:] - v_macx[:, :-1, :-1]
            du_dy = u_macy[:, 1:, :-1] - u_macy[:, :-1, :-1]
            dv_dy = v[:, 1:, :-1] - v[:, :-1, :-1]
            dvel_dx = torch.cat((du_dx, dv_dx), dim=0)
            dvel_dy = torch.cat((du_dy, dv_dy), dim=0)
            jacobian = torch.cat((dvel_dx.unsqueeze(1), dvel_dy.unsqueeze(1)), dim=1)
        else:
            raise NotImplementedError(
                "MACGridOperator.get_jacobian: " "3D case not supported yet."
            )
        return jacobian

    def get_mac(self, grid):
        """interpolate centered velocity field back to MAC grid.

        Args:
            grid(torch.Tensor) of size [C, (D), H, W]

        Returns:
            grid_mac(torch.Tensor) of size [C, (D+1), H+1, W+1]
        """
        pad_x = (1, 1, 0, 0) if self.solver.is_2d() else (1, 1, 0, 0, 0, 0)
        pad_y = (0, 0, 1, 1) if self.solver.is_2d() else (0, 0, 1, 1, 0, 0)
        pad_yz = (0, 0, 0, 1) if self.solver.is_2d() else (0, 0, 0, 1, 0, 1)
        pad_xz = (0, 1, 0, 0) if self.solver.is_2d() else (0, 1, 0, 0, 0, 1)
        u_centered_padx = F.pad(grid[0:1, ...], pad=pad_x)
        u_mac = (u_centered_padx[..., :-1] + u_centered_padx[..., 1:]) * 0.5
        u_mac = F.pad(u_mac, pad=pad_yz)
        v_centered_pady = F.pad(grid[1:2, ...], pad=pad_y)
        v_mac = (v_centered_pady[..., :-1, :] + v_centered_pady[..., 1:, :]) * 0.5
        v_mac = F.pad(v_mac, pad=pad_xz)
        grid_mac = torch.cat((u_mac, v_mac), dim=0)
        if self.solver.is_3d():
            pad_z = (0, 0, 0, 0, 1, 1)
            pad_xy = (0, 1, 0, 1, 0, 0)
            w_centered_padz = F.pad(grid[2:3, ...], pad=pad_z)
            w_mac = (w_centered_padz[:, :-1, ...] + w_centered_padz[:, 1:, ...]) * 0.5
            w_mac = F.pad(w_mac, pad=pad_xy)
            grid_mac = torch.cat((grid_mac, w_mac), dim=0)
        return grid_mac

    def get_centered(self, grid):
        if self.solver.is_2d():
            u_centered = (grid[0, :-1, :-1] + grid[0, :-1, 1:]) * 0.5
            v_centered = (grid[1, :-1, :-1] + grid[1, 1:, :-1]) * 0.5
            vel_centered = torch.stack((u_centered, v_centered), dim=0)
        else:
            u_centered = (grid[0, :-1, :-1, :-1] + grid[0, :-1, :-1, 1:]) * 0.5
            v_centered = (grid[1, :-1, :-1, :-1] + grid[1, :-1, 1:, :-1]) * 0.5
            w_centered = (grid[2, :-1, :-1, :-1] + grid[2, 1:, :-1, :-1]) * 0.5
            vel_centered = torch.stack((u_centered, v_centered, w_centered), dim=0)
        return vel_centered

    def get_mac_x(self, grid):
        u_mac_x = grid[0:1, ...]

        v_mac_y = grid[1:2, ...]
        pad_x_neg = (1, 0, 0, 0) if self.solver.is_2d() else (1, 0, 0, 0, 0, 0)
        v_mac_y_pad_x = F.pad(v_mac_y, pad=pad_x_neg)
        v_centered = (v_mac_y_pad_x[..., 1:] + v_mac_y_pad_x[..., :-1]) * 0.5
        v_mac_x = (v_centered[..., 1:, :] + v_centered[..., :-1, :]) * 0.5
        pad_y_pos = (0, 0, 0, 1) if self.solver.is_2d() else (0, 0, 0, 1, 0, 0)
        v_mac_x = F.pad(v_mac_x, pad=pad_y_pos)
        vel_mac_x = torch.cat((u_mac_x, v_mac_x), dim=0)
        if self.solver.is_3d():
            w_mac_z = grid[2:3, ...]
            w_mac_z_pad_x = F.pad(w_mac_z, pad=pad_x_neg)
            w_centered = (w_mac_z_pad_x[..., 1:] + w_mac_z_pad_x[..., :-1]) * 0.5
            w_mac_x = (w_centered[:, 1:, ...] + w_centered[:, :-1, ...]) * 0.5
            pad_z_pos = (0, 0, 0, 0, 0, 1)
            w_mac_x = F.pad(w_mac_x, pad=pad_z_pos)
            vel_mac_x = torch.cat((vel_mac_x, w_mac_x), dim=0)

        return vel_mac_x

    def get_mac_y(self, grid):
        v_mac_y = grid[1:2, ...]

        u_mac_x = grid[0:1, ...]
        pad_y_neg = (0, 0, 1, 0) if self.solver.is_2d() else (0, 0, 1, 0, 0, 0)
        u_mac_x_pad_y = F.pad(u_mac_x, pad=pad_y_neg)
        u_centered = (u_mac_x_pad_y[..., 1:, :] + u_mac_x_pad_y[..., :-1, :]) * 0.5
        u_mac_y = (u_centered[..., 1:] + u_centered[..., :-1]) * 0.5
        pad_x_pos = (0, 1, 0, 0) if self.solver.is_2d() else (0, 1, 0, 0, 0, 0)
        u_mac_y = F.pad(u_mac_y, pad=pad_x_pos)
        vel_mac_y = torch.cat((u_mac_y, v_mac_y), dim=0)

        if self.solver.is_3d():
            w_mac_z = grid[2:3, ...]
            w_mac_z_pad_y = F.pad(w_mac_z, pad=pad_y_neg)
            w_centered = (w_mac_z_pad_y[..., 1:, :] + w_mac_z_pad_y[..., :-1, :]) * 0.5
            w_mac_y = (w_centered[:, 1:, ...] + w_centered[:, :-1, ...]) * 0.5
            pad_z_pos = (0, 0, 0, 0, 0, 1)
            w_mac_y = F.pad(w_mac_y, pad=pad_z_pos)
            vel_mac_y = torch.cat((vel_mac_y, w_mac_y), dim=0)

        return vel_mac_y

    def get_mac_z(self, grid):
        if self.solver.is_2d():
            raise RuntimeError("MACGridOperator.get_mac_z only supports 3D case.")
        u_mac_x = grid[0:1, ...]
        v_mac_y = grid[1:2, ...]
        pad_z_neg = (0, 0, 0, 0, 1, 0)
        u_mac_x_pad_z = F.pad(u_mac_x, pad=pad_z_neg)
        u_centered = (u_mac_x_pad_z[:, 1:, ...] + u_mac_x_pad_z[:, :-1, ...]) * 0.5
        u_mac_z = (u_centered[..., 1:] + u_centered[..., :-1]) * 0.5
        pad_x_pos = (0, 1, 0, 0, 0, 0)
        u_mac_z = F.pad(u_mac_z, pad=pad_x_pos)

        v_mac_y_pad_z = F.pad(v_mac_y, pad=pad_z_neg)
        v_centered = (v_mac_y_pad_z[:, 1:, ...] + v_mac_y_pad_z[:, :-1, ...]) * 0.5
        v_mac_z = (v_centered[..., 1:, :] + v_centered[..., :-1, :]) * 0.5
        pad_y_pos = (0, 0, 0, 1, 0, 0)
        v_mac_z = F.pad(v_mac_z, pad=pad_y_pos)

        w_mac_z = grid[2:3, ...]
        vel_mac_z = torch.cat((u_mac_z, v_mac_z, w_mac_z), dim=0)
        return vel_mac_z

    def grid_sample_center(self, grid, back_trace=None):
        """Interpolate velocity field onto cell center position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                self.solver.get_torch_size_real(),
                device=self.device,
                dtype=self.solver.dtype,
            )
        # WARNING: might be inaccurate at non-solid boundaries,
        # since one valid boundary value is cut off.
        if self.solver.is_2d():
            grid = grid[:, :-1, :-1]
        else:
            grid = grid[:, :-1, :-1, :-1]
        vel_x = utils.grid_sample(
            grid[0:1, ...],
            torch.cat(
                (
                    back_trace[0:1, ...] + 0.5,
                    back_trace[1:2, ...],
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        vel_y = utils.grid_sample(
            grid[1:2, ...],
            torch.cat(
                (
                    back_trace[0:1, ...],
                    back_trace[1:2, ...] + 0.5,
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        vel = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            vel_z = utils.grid_sample(
                grid[2:3, ...],
                torch.cat(
                    (
                        back_trace[0:1, ...],
                        back_trace[1:2, ...],
                        back_trace[2:3, ...] + 0.5,
                    ),
                    dim=0,
                ),
            )
            vel = torch.cat((vel, vel_z), dim=0)
        return vel

    def grid_sample_node(self, grid, back_trace=None):
        """Interpolate velocity field onto cell node(corner) position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                self.solver.get_torch_size_node(),
                device=self.device,
                dtype=self.solver.dtype,
            )
        # WARNING: might be inaccurate at non-solid boundaries,
        # since one valid boundary value is cut off.
        vel_x = utils.grid_sample(
            grid[0:1, ...],
            torch.cat(
                (
                    back_trace[0:1, ...],
                    back_trace[1:2, ...] - 0.5,
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        vel_y = utils.grid_sample(
            grid[1:2, ...],
            torch.cat(
                (
                    back_trace[0:1, ...] - 0.5,
                    back_trace[1:2, ...],
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        vel = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            raise NotImplementedError(
                "MACGridOperator.grid_sample_node:" "3D case not supported yet."
            )
        return vel

    def grid_sample_x(self, grid, back_trace=None):
        """Interpolate velocity field onto mac_x position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                grid.size(), device=self.device, dtype=self.solver.dtype
            )
        vel_x = utils.grid_sample(grid[0:1, ...], back_trace)
        vel_y = self._grid_sample_x(grid[1:2, ...], back_trace)
        vel = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            vel_z = utils.grid_sample(
                grid[2:3, ...],
                torch.cat(
                    (
                        back_trace[0:1, ...] - 0.5,
                        back_trace[1:2, ...],
                        back_trace[2:3, ...] + 0.5,
                    ),
                    dim=0,
                ),
            )
            vel = torch.cat((vel, vel_z), dim=0)
        return vel

    def _grid_sample_x(self, grid, back_trace=None):
        """Interpolate velocity field component onto mac_x position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(grid.size(), self.solver.dtype)
        grid_macx = utils.grid_sample(
            grid,
            torch.cat(
                (
                    back_trace[0:1, ...] - 0.5,
                    back_trace[1:2, ...] + 0.5,
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        return grid_macx

    def grid_sample_y(self, grid, back_trace=None):
        """Interpolate velocity field onto mac_y position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                grid.size(), device=self.device, dtype=self.solver.dtype
            )
        vel_x = self._grid_sample_y(grid[0:1, ...], back_trace)
        vel_y = utils.grid_sample(grid[1:2, ...], back_trace)
        vel = torch.cat((vel_x, vel_y), dim=0)
        if self.solver.is_3d():
            vel_z = utils.grid_sample(
                grid[2:3, ...],
                torch.cat(
                    (
                        back_trace[0:1, ...],
                        back_trace[1:2, ...] - 0.5,
                        back_trace[2:3, ...] + 0.5,
                    ),
                    dim=0,
                ),
            )
            vel = torch.cat((vel, vel_z), dim=0)
        return vel

    def _grid_sample_y(self, grid, back_trace=None):
        """Interpolate velocity field component onto mac_y position."""
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                grid.size(), device=self.device, dtype=self.solver.dtype
            )
        grid_macy = utils.grid_sample(
            grid,
            torch.cat(
                (
                    back_trace[0:1, ...] + 0.5,
                    back_trace[1:2, ...] - 0.5,
                    back_trace[2:3, ...],
                ),
                dim=0,
            ),
        )
        return grid_macy

    def grid_sample_z(self, grid, back_trace=None):
        if back_trace is None:
            back_trace = utils.create_mesh_grid(
                grid.size(), device=self.device, dtype=self.solver.dtype
            )
        if self.solver.is_2d():
            raise RuntimeError("2d data does not support z direction.")
        vel_x = utils.grid_sample(
            grid[0:1, ...],
            torch.cat(
                (
                    back_trace[0:1, ...] + 0.5,
                    back_trace[1:2, ...],
                    back_trace[2:3, ...] - 0.5,
                ),
                dim=0,
            ),
        )
        vel_y = utils.grid_sample(
            grid[1:2, ...],
            torch.cat(
                (
                    back_trace[0:1, ...],
                    back_trace[1:2, ...] + 0.5,
                    back_trace[2:3, ...] - 0.5,
                ),
                dim=0,
            ),
        )
        vel_z = utils.grid_sample(grid[2:3, ...], back_trace)
        vel = torch.cat((vel_x, vel_y, vel_z), dim=0)
        return vel

    def get_energy(self, grid):
        grid_center = self.grid_sample_center(grid)
        energy = grid_center[0:1, ...] ** 2 + grid_center[1:2, ...] ** 2
        return energy


class LevelGridOperator(GridOperator):
    # overload
    def save_img(self, grid, filename, colormap="SEISMIC", **kwargs):
        array = self.grid2array(grid)
        utils.save_img(array, filename, colormap)

    # overload
    def show_img(self, grid, colormap="SEISMIC", **kwargs):
        array = self.grid2array(grid)
        utils.show_img(array, colormap)

    def get_const_grid(self, value):
        grid = self.solver.create_grid(GridType.LEVEL)
        return self._get_const_grid(grid, value)

    def _get_line_fractions(self, phi, direction="z"):
        """Computes the line fraction of fluid for each cell given level set

        Args:
            phi(torch.Tensor): LevelGrid of size [C, (D+1), H+1, W+1]
            direction(string): surface normal direction to choose which
                surface to compute line fraction, should be one of 'x', 'y', 'z'
        """
        if direction == "x":
            assert self.solver.is_3d()
            left = phi[..., :-1, :]
            right = phi[..., 1:, :]
            bottom = phi[:, :-1, ...]
            up = phi[:, 1:, ...]
            pad_vertical_pos = (0, 0, 0, 0, 0, 1)
            pad_horizontal_pos = (0, 0, 0, 1, 0, 0)
        elif direction == "y":
            assert self.solver.is_3d()
            left = phi[..., :-1]
            right = phi[..., 1:]
            bottom = phi[:, :-1, ...]
            up = phi[:, 1:, ...]
            pad_vertical_pos = (0, 0, 0, 0, 0, 1)
            pad_horizontal_pos = (0, 1, 0, 0, 0, 0)
        elif direction == "z":
            left = phi[..., :-1]
            right = phi[..., 1:]
            bottom = phi[..., :-1, :]
            up = phi[..., 1:, :]
            pad_vertical_pos = (
                (0, 0, 0, 1) if self.solver.is_2d() else (0, 0, 0, 1, 0, 0)
            )
            pad_horizontal_pos = (
                (0, 1, 0, 0) if self.solver.is_2d() else (0, 1, 0, 0, 0, 0)
            )
        else:
            raise ValueError("direction should be one of x, y, z.")
        max_ub = torch.max(up, bottom)
        min_ub = torch.min(up, bottom)
        fraction_horizontal = max_ub / (max_ub - min_ub + 1e-7)
        fraction_horizontal = torch.clamp(fraction_horizontal, 0.0, 1.0)
        fraction_horizontal = F.pad(
            fraction_horizontal.unsqueeze(0), pad=pad_vertical_pos, mode="replicate"
        ).squeeze(0)
        max_lr = torch.max(left, right)
        min_lr = torch.min(left, right)
        fraction_vertical = max_lr / (max_lr - min_lr + 1e-7)
        fraction_vertical = torch.clamp(fraction_vertical, 0.0, 1.0)
        fraction_vertical = F.pad(
            fraction_vertical.unsqueeze(0), pad=pad_horizontal_pos, mode="replicate"
        ).squeeze(0)
        fraction = torch.cat((fraction_horizontal, fraction_vertical), dim=0)
        return fraction

    def _get_face_fractions(self, line_fractions, phi, direction):
        """Computes face fractions given line fractions for a plane.

        Args:
            line_fractions(torch.Tensor): MACGrid of size [C, D+1, H+1, W+1]
            phi(torch.Tensor): LevelGrid of size [C, D+1, H+1, W+1]
            direction: indicates the surface normal of the line_fractions
        """
        assert self.solver.is_3d()
        if direction == "x":
            left = line_fractions[0:1, :-1, :-1, :]
            bottom = line_fractions[1:2, :-1, :-1, :]
            right = line_fractions[0:1, :-1, 1:, :]
            up = line_fractions[1:2, 1:, :-1, :]
            phi_00 = phi[:, :-1, :-1, :]
            phi_01 = phi[:, 1:, :-1, :]
            phi_11 = phi[:, 1:, 1:, :]
            pad_to_mac = (0, 0, 0, 1, 0, 1)
        elif direction == "y":
            left = line_fractions[0:1, :-1, :, :-1]
            bottom = line_fractions[1:2, :-1, :, :-1]
            right = line_fractions[0:1, :-1, :, 1:]
            up = line_fractions[1:2, 1:, :, :-1]
            phi_00 = phi[:, :-1, :, :-1]
            phi_01 = phi[:, 1:, :, :-1]
            phi_11 = phi[:, 1:, :, 1:]
            pad_to_mac = (0, 1, 0, 0, 0, 1)
        elif direction == "z":
            left = line_fractions[0:1, :, :-1, :-1]
            bottom = line_fractions[1:2, :, :-1, :-1]
            right = line_fractions[0:1, :, :-1, 1:]
            up = line_fractions[1:2, :, 1:, :-1]
            phi_00 = phi[:, :, :-1, :-1]
            phi_01 = phi[:, :, 1:, :-1]
            phi_11 = phi[:, :, 1:, 1:]
            pad_to_mac = (0, 1, 0, 1, 0, 0)
        face_fraction = torch.where(
            phi_01 < 1e-7,
            0.5 * ((left + right) * bottom + (right - left) * up),
            torch.zeros(phi_01.size(), device=self.device, dtype=left.dtype),
        )
        face_fraction = torch.where(
            (phi_01 >= 1e-7) * (phi_11 < 1e-7),
            0.5 * ((left + right) * bottom + (left - right) * up),
            face_fraction,
        )
        face_fraction = torch.where(
            (phi_01 >= 1e-7) * (phi_11 >= 1e-7) * (phi_00 < 1e-7),
            0.5 * ((left + right) * up + (right - left) * bottom),
            face_fraction,
        )
        face_fraction = torch.where(
            (phi_01 >= 1e-7) * (phi_11 >= 1e-7) * (phi_00 >= 1e-7),
            0.5 * ((left + right) * up + (left - right) * bottom),
            face_fraction,
        )
        face_fraction = F.pad(face_fraction, pad=pad_to_mac)
        return face_fraction

    def get_fractions(self, grid):
        if self.solver.is_2d():
            return self._get_line_fractions(grid, direction="z")
        else:
            line_fractions_x = self._get_line_fractions(grid, direction="x")
            line_fractions_y = self._get_line_fractions(grid, direction="y")
            line_fractions_z = self._get_line_fractions(grid, direction="z")
            face_fraction_x = self._get_face_fractions(
                line_fractions_x, grid, direction="x"
            )
            face_fraction_y = self._get_face_fractions(
                line_fractions_y, grid, direction="y"
            )
            face_fraction_z = self._get_face_fractions(
                line_fractions_z, grid, direction="z"
            )
            face_fraction = torch.cat(
                (face_fraction_x, face_fraction_y, face_fraction_z), dim=0
            )
            return face_fraction

    def is_obs(self, grid):
        fractions = self.get_fractions(grid)
        if self.solver.is_2d():
            is_obs = (
                (fractions[0:1, :-1, :-1] < 1e-6)
                * (fractions[0:1, :-1, 1:] < 1e-6)
                * (fractions[1:2, :-1, :-1] < 1e-6)
                * (fractions[1:2, 1:, :-1] < 1e-6)
            )
        else:
            is_obs = (
                (fractions[0:1, :-1, :-1, :-1] < 1e-6)
                * (fractions[0:1, :-1, :-1, 1:] < 1e-6)
                * (fractions[1:2, :-1, :-1, :-1] < 1e-6)
                * (fractions[1:2, :-1, 1:, :-1] < 1e-6)
                * (fractions[1:2, :-1, :-1, :-1] < 1e-6)
                * (fractions[1:2, 1:, :-1, :-1] < 1e-6)
            )
        return is_obs

    def is_fluid(self, grid):
        return ~self.is_obs(grid)

    def union(self, f, g, alpha=0.0):
        """Performs union operator: f {cup} g,

        Args:
            f, g: two level set functions to be operated on.
            alpha: parameter to control blending,
                   should be in the range [0,1], alpha=1.0 is equivalent
                   to boolean blending, i.e. torch.min(f, g)
        """
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha should be in the range [0,1].")
        if alpha == 1.0:
            return torch.min(f, g)
        else:
            return (
                1.0
                / (1.0 + alpha)
                * (f + g - torch.sqrt(f * f + g * g - 2 * alpha * f * g))
            )

    def intersection(self, f, g, alpha=0.0):
        """Performs intersection operator: f {cap} g,

        Args:
            f, g: two level set functions to be operated on.
            alpha: parameter to control blending,
                   should be in the range [0,1], alpha=1.0 is equivalent
                   to boolean blending, i.e. torch.max(f, g)
        """
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha should be in the range [0,1].")
        if alpha == 1.0:
            return torch.max(f, g)
        else:
            return (
                1.0
                / (1.0 + alpha)
                * (f + g + torch.sqrt(f * f + g * g - 2 * alpha * f * g))
            )

    def subtraction(self, f, g, alpha=0.0):
        """Performs subtraction operator: f - g,"""
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha should be in the range [0,1].")
        return self.intersection(f, -g, alpha)


class NodeGridOperator(GridOperator):
    def __init__(self, solver):
        GridOperator.__init__(self, solver)
        self.ls_solver = self.solver.create_ls_solver(apply_precon=True)
        self.A = None
        self.ext = self.solver.create_extforces()

    # overload
    def show_img(self, grid, colormap="RDBU", **kwargs):
        array = self.grid2array(grid)
        utils.show_img(array, colormap)

    def get_const_grid(self, value):
        grid = self.solver.create_grid(GridType.NODE)
        return self._get_const_grid(grid, value)

    def get_gradient(self, grid):
        if self.solver.is_3d():
            raise NotImplementedError(
                "NodeGridOperator.get_gradient: " "3D case not supported yet."
            )
        grad_x = grid[..., 1:] - grid[..., :-1]
        grad_x = F.pad(grad_x, pad=(0, 1, 0, 0))  # pad a dummy column
        grad_y = grid[..., 1:, :] - grid[..., :-1, :]
        grad_y = F.pad(grad_y, pad=(0, 0, 0, 1))  # pad a dummy row
        grad = torch.cat((grad_x, grad_y), dim=0)
        return grad

    def get_gaussian(self, mu, sig, amp):
        """Creates a Gaussian stream field.

        Args:
            mu(list, np.array, torch.Tensor): size C, mean of gaussian
            sig(list, np.array, torch.Tensor): size CxC,
                standard deviation matrix of gaussian
            amp(list, np.array, torch.Tensor): size 1,
                amplitude of the gaussian field
        """
        if self.solver.is_3d():
            raise NotImplementedError(
                "NodeGridOperator.get_gaussian " "not supported yet."
            )
        mu = torch.Tensor(mu).to(self.device).to(self.solver.get_dtype())
        sig = torch.Tensor(sig).to(self.device).to(self.solver.get_dtype())
        amp = torch.Tensor(amp).to(self.device).to(self.solver.get_dtype())
        assert mu.size(0) == 2 and sig.size(0) == 2 and sig.size(1) == 2
        sig_inv = torch.inverse(sig)
        mgrid = utils.create_mesh_grid(
            self.solver.get_torch_size_node(),
            device=self.device,
            dtype=self.solver.get_dtype(),
        )
        mgrid = utils.normalize_mesh_grid(mgrid)
        x0 = mgrid[0:1, ...]
        x1 = mgrid[1:2, ...]
        mahalanobis_dist_2 = (
            sig_inv[0, 0] * (x0 - mu[0]) * (x0 - mu[0])
            + sig_inv[0, 1] * (x0 - mu[0]) * (x1 - mu[1])
            + sig_inv[1, 0] * (x1 - mu[1]) * (x0 - mu[0])
            + sig_inv[1, 1] * (x1 - mu[1]) * (x1 - mu[1])
        )
        gauss = amp * torch.exp(-0.5 * mahalanobis_dist_2)
        return gauss

    def get_curl(self, grid):
        if self.solver.is_2d():
            # in 2D case, the phi is a scalar field, equivalently on z direction
            # curl(phi) = (dphi/dy, -dphi/dx)
            grad_x = grid[..., 1:] - grid[..., :-1]
            grad_x = F.pad(grad_x, pad=(0, 1, 0, 0))  # pad a dummy column
            grad_y = grid[..., 1:, :] - grid[..., :-1, :]
            grad_y = F.pad(grad_y, pad=(0, 0, 0, 1))  # pad a dummy row
            curl = torch.cat((grad_y, -grad_x), dim=0)
        else:
            # in 3D case, the values of stream fcn are stored on the edges of the cell
            gridx = grid[0:1, ...]
            gridy = grid[1:2, ...]
            gridz = grid[2:3, ...]
            # compute finite difference gradients
            dgridx_dz = gridx[:, 1:, ...] - gridx[:, :-1, ...]
            dgridx_dy = gridx[..., 1:, :] - gridx[..., :-1, :]
            dgridy_dx = gridy[..., 1:] - gridy[..., :-1]
            dgridy_dz = gridy[:, 1:, ...] - gridy[:, :-1, ...]
            dgridz_dx = gridz[..., 1:] - gridz[..., :-1]
            dgridz_dy = gridz[..., 1:, :] - gridz[..., :-1, :]
            # abandon redundant slices
            curl_x = dgridz_dy[:, :-1, ...] - dgridy_dz[..., :-1, :]
            curl_y = dgridx_dz[..., :-1] - dgridz_dx[:, :-1, ...]
            curl_z = dgridy_dx[..., :-1, :] - dgridx_dy[..., :-1]
            # pad redundant slices back for 3D MAC grid
            curl_x = F.pad(curl_x, pad=(0, 0, 0, 1, 0, 1))
            curl_y = F.pad(curl_y, pad=(0, 1, 0, 0, 0, 1))
            curl_z = F.pad(curl_z, pad=(0, 1, 0, 1, 0, 0))
            curl = torch.cat((curl_x, curl_y, curl_z), dim=0)
        return curl

    def get_inverse_curl(self, phi_obs, vort, cg_accu=1e-10):
        if self.solver.is_3d():
            raise NotImplementedError(
                "NodeGridOperator.get_inverse_curl " "3D case not supported yet."
            )
        rhs = vort
        if self.A is None:
            self.A = self.ls_solver.build_streamfunction_matrix()
            self.ls_solver.build_preconditioner(self.A)
        psi = self.ls_solver.solve_linear_system(rhs, self.A, tol=cg_accu)
        vel = self.get_curl(phi_obs, psi)
        # _, phi_obs = self.ext.init_flags(phi=True, boundary_width=0, wall='xXyY')
        vel = self.ext.set_wall_bcs(phi_obs=phi_obs, grid=vel)
        return vel


class GridOperatorFactory(object):
    def __init__(self, solver):
        self.solver = solver

    def create(self, grid_type):
        if grid_type == GridType.FLAG:
            return FlagGridOperator(self.solver)
        elif grid_type == GridType.REAL:
            return RealGridOperator(self.solver)
        elif grid_type == GridType.MAC:
            return MACGridOperator(self.solver)
        elif grid_type == GridType.LEVEL:
            return LevelGridOperator(self.solver)
        elif grid_type == GridType.NODE:
            return NodeGridOperator(self.solver)
        else:
            raise ValueError(
                (
                    "grid_type can only be GridType.FLAG, "
                    "GridType.REAL or GridType.MAC"
                )
            )


class GridFactory(object):
    def __init__(self, solver):
        self.solver = solver
        self.device = self.solver.get_device()

    def create(self, grid_type, dtype):
        if grid_type == GridType.FLAG:
            return torch.zeros(
                self.solver.get_torch_size_flag(), dtype=torch.uint8, device=self.device
            )
        elif grid_type == GridType.REAL:
            return torch.zeros(
                self.solver.get_torch_size_real(), dtype=dtype, device=self.device
            )
        elif grid_type == GridType.MAC:
            return torch.zeros(
                self.solver.get_torch_size_mac(), dtype=dtype, device=self.device
            )
        elif grid_type == GridType.LEVEL:
            return torch.zeros(
                self.solver.get_torch_size_level(), dtype=dtype, device=self.device
            )
        elif grid_type == GridType.NODE:
            return torch.zeros(
                self.solver.get_torch_size_node(), dtype=dtype, device=self.device
            )
        else:
            raise ValueError(
                (
                    "grid_type can only be GridType.FLAG, "
                    "GridType.REAL or GridType.MAC"
                )
            )
