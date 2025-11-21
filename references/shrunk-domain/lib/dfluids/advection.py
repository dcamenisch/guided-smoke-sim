import torch
import torch.nn.functional as F

from .grid import GridType
from . import utils


class Advection(object):
    def __init__(self, solver):
        self.solver = solver
        self.device = self.solver.get_device()
        self.mac_op = self.solver.create_grid_operator(GridType.MAC)
        self.real_op = self.solver.create_grid_operator(GridType.REAL)
        self.node_op = self.solver.create_grid_operator(GridType.NODE)
        self.ext = self.solver.create_extforces()

    def advect_semi_lagrange(
        self, flags, phi_obs, vel, rho, dt=None, order=1, clamp_mode=2, rk_order=3
    ):
        if dt is None:
            dt = self.solver.get_dt()
        if order == 1:
            rho_adv = self._advect_semi_lagrange(vel, rho, dt, rk_order)
        elif order == 2:
            rho_fwd = self._advect_semi_lagrange(vel, rho, dt, rk_order)
            rho_bwd = self._advect_semi_lagrange(vel, rho_fwd, -dt, rk_order)
            rho_corr = rho_fwd + (rho - rho_bwd) * 0.5
            rho_adv = self._maccormack_clamp(
                vel, rho_corr, rho_fwd, rho, dt, clamp_mode, rk_order
            )
        else:
            raise ValueError("Advection order must be 1 or 2.")
        rho_adv = self.ext.set_wall_bcs(phi_obs, rho_adv)
        return rho_adv

    def _advect_semi_lagrange(self, vel, rho, dt, rk_order):
        if rho.size() == self.solver.get_torch_size_real():
            return self._advect_real_semi_lagrange(vel, rho, dt, rk_order)
        elif rho.size() == self.solver.get_torch_size_mac():
            return self._advect_mac_semi_lagrange(vel, rho, dt, rk_order)
        # WARNING: does not distinguish 3D MAC and 3D NODE
        elif rho.size() == self.solver.get_torch_size_node():
            return self._advect_node_semi_lagrange(vel, rho, dt, rk_order)
        else:
            raise ValueError("rho has a wrong size: {}".format(rho.size()))

    def _advect_real_semi_lagrange(self, vel, rho, dt, rk_order):
        return self._advect_semi_lagrange_single_channel(
            vel=vel, rho=rho, dt=dt, rk_order=rk_order, sample_pos="center"
        )

    def _advect_node_semi_lagrange(self, vel, rho, dt, rk_order):
        return self._advect_semi_lagrange_single_channel(
            vel=vel, rho=rho, dt=dt, rk_order=rk_order, sample_pos="node"
        )

    def _advect_mac_semi_lagrange_inaccurate(self, vel, rho, dt, rk_order):
        rho_adv_centered = self._advect_semi_lagrange_single_channel(
            vel=vel,
            rho=self.mac_op.get_centered(rho),
            dt=dt,
            rk_order=rk_order,
            sample_pos="center",
        )
        rho_adv = self.mac_op.get_mac(rho_adv_centered)
        return rho_adv

    def _advect_mac_semi_lagrange(self, vel, rho, dt, rk_order):
        # advect in x direction
        rho_adv_x = self._advect_semi_lagrange_single_channel(
            vel=vel, rho=rho[0:1, ...], dt=dt, rk_order=rk_order, sample_pos="mac_x"
        )
        # advect in y direction
        rho_adv_y = self._advect_semi_lagrange_single_channel(
            vel=vel, rho=rho[1:2, ...], dt=dt, rk_order=rk_order, sample_pos="mac_y"
        )
        # concatenate back advected values into the same tensor
        rho_adv = torch.cat((rho_adv_x, rho_adv_y), dim=0)
        # advect in z direction in 3d case
        if self.solver.is_3d():
            rho_adv_z = self._advect_semi_lagrange_single_channel(
                vel=vel, rho=rho[2:3, ...], dt=dt, rk_order=rk_order, sample_pos="mac_z"
            )
            rho_adv = torch.cat((rho_adv, rho_adv_z), dim=0)
        return rho_adv

    def _advect_semi_lagrange_single_channel(self, vel, rho, dt, rk_order, sample_pos):
        # create mesh grid
        mgrid = utils.create_mesh_grid(
            rho.size(), device=self.device, dtype=self.solver.dtype
        )
        # use runge kutta or euler method to create the back trace field
        back_trace = self._runge_kutta(vel, mgrid, dt, rk_order, sample_pos)
        # use warped grid sample function from PyTorch to interpolate the grid
        # and bring the interporlated values to the current position
        rho_adv = utils.grid_sample(rho, back_trace)
        return rho_adv

    def _euler(self, vel, mgrid, dt, sample_pos):
        if sample_pos == "node":
            raise NotImplementedError(
                "Euler step not implemented " "for node data advection."
            )
        get_pos_fcn = self._map_sample_pos(sample_pos, rk=False)
        return mgrid - get_pos_fcn(vel) * dt

    def _runge_kutta(self, vel, mgrid, dt, rk_order, sample_pos):
        grid_sample_fcn = self._map_sample_pos(sample_pos, rk=True)
        if rk_order == 1:
            # return mgrid - grid_sample_fcn(vel, mgrid)*dt
            return self._euler(vel, mgrid, dt, sample_pos)
        elif rk_order == 2:
            # find the mid point position using half of time step
            # back_trace_mid = mgrid - grid_sample_fcn(vel, mgrid)*dt*0.5
            back_trace_mid = self._euler(vel, mgrid, 0.5 * dt, sample_pos)
            # find the mid point velocity
            return mgrid - grid_sample_fcn(vel, back_trace_mid) * dt
        elif rk_order == 3:
            # k1 = grid_sample_fcn(vel, mgrid)
            k1 = (mgrid - self._euler(vel, mgrid, dt, sample_pos)) / dt
            k2 = grid_sample_fcn(vel, mgrid - 0.5 * dt * k1)
            k3 = grid_sample_fcn(vel, mgrid - 0.75 * dt * k2)
            return (
                mgrid - 2.0 / 9.0 * dt * k1 - 3.0 / 9.0 * dt * k2 - 4.0 / 9.0 * dt * k3
            )
        else:
            raise ValueError("Runge-kutta only support order 1, 2 or 3.")

    def _map_sample_pos(self, sample_pos, rk):
        if sample_pos == "center":
            return self.mac_op.grid_sample_center if rk else self.mac_op.get_centered
        elif sample_pos == "node":
            return self.mac_op.grid_sample_node
        elif sample_pos == "mac_x":
            return self.mac_op.grid_sample_x if rk else self.mac_op.get_mac_x
        elif sample_pos == "mac_y":
            return self.mac_op.grid_sample_y if rk else self.mac_op.get_mac_y
        elif sample_pos == "mac_z":
            return self.mac_op.grid_sample_z if rk else self.mac_op.get_mac_z
        else:
            raise ValueError("sample_pos {} not recognized.".format(sample_pos))

    def _maccormack_clamp(
        self, vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order
    ):
        if vel.size(0) != rho_org.size(0):  # hard code to figure out grid type
            return self._maccormack_real_clamp(
                vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order
            )
        else:
            return self._maccormack_mac_clamp(
                vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order
            )

    def _maccormack_real_clamp(
        self, vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order
    ):
        return self._maccormack_clamp_single_channel(
            vel,
            rho_corr,
            rho_fwd,
            rho_org,
            dt,
            clamp_mode,
            rk_order,
            sample_pos="center",
        )

    def _maccormack_mac_clamp(
        self, vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order
    ):
        rho_adv_x = self._maccormack_clamp_single_channel(
            vel,
            rho_corr[0:1, ...],
            rho_fwd[0:1, ...],
            rho_org[0:1, ...],
            dt,
            clamp_mode,
            rk_order,
            sample_pos="mac_x",
        )
        rho_adv_y = self._maccormack_clamp_single_channel(
            vel,
            rho_corr[1:2, ...],
            rho_fwd[1:2, ...],
            rho_org[1:2, ...],
            dt,
            clamp_mode,
            rk_order,
            sample_pos="mac_y",
        )
        rho_adv = torch.cat((rho_adv_x, rho_adv_y), dim=0)
        if self.solver.is_3d():
            rho_adv_z = self._maccormack_clamp_single_channel(
                vel,
                rho_corr[2:3, ...],
                rho_fwd[2:3, ...],
                rho_org[2:3, ...],
                dt,
                clamp_mode,
                rk_order,
                sample_pos="mac_z",
            )
            rho_adv = torch.cat((rho_adv, rho_adv_z), dim=0)
        return rho_adv

    def _maccormack_clamp_single_channel(
        self, vel, rho_corr, rho_fwd, rho_org, dt, clamp_mode, rk_order, sample_pos
    ):
        # prepare min and max
        if self.solver.is_2d():
            W, H = self.solver.get_grid_size()
            max_pool = F.max_pool2d
            pad_pos = (0, 1, 0, 1)
        else:
            W, H, D = self.solver.get_grid_size()
            max_pool = F.max_pool3d
            pad_pos = (0, 1, 0, 1, 0, 1)
        mgrid = utils.create_mesh_grid(
            rho_org.size(), device=self.device, dtype=self.solver.dtype
        )
        # find max and min at back trace positions
        back_trace = self._runge_kutta(vel, mgrid, dt, rk_order, sample_pos)
        back_trace_x = torch.clamp(back_trace[0, ...], 0, W - 1).long()
        back_trace_y = torch.clamp(back_trace[1, ...], 0, H - 1).long()
        rho_org_pad_pos = F.pad(rho_org, pad=pad_pos)
        rho_org_max = max_pool(
            rho_org_pad_pos.unsqueeze(0), kernel_size=2, stride=1
        ).squeeze(0)
        rho_org_min = -max_pool(
            -rho_org_pad_pos.unsqueeze(0), kernel_size=2, stride=1
        ).squeeze(0)
        if self.solver.is_2d():
            rho_org_max_back_trace = rho_org_max[:, back_trace_y, back_trace_x]
            rho_org_min_back_trace = rho_org_min[:, back_trace_y, back_trace_x]
        else:
            back_trace_z = torch.clamp(back_trace[0, ...], 0, D - 1).long()
            rho_org_max_back_trace = rho_org_max[
                :, back_trace_z, back_trace_y, back_trace_x
            ]
            rho_org_min_back_trace = rho_org_min[
                :, back_trace_z, back_trace_y, back_trace_x
            ]

        # find max and min at forward trace positions
        forward_trace = self._runge_kutta(vel, mgrid, -dt, rk_order, sample_pos)
        forward_trace_x = torch.clamp(forward_trace[0, ...], 0, W - 1).long()
        forward_trace_y = torch.clamp(forward_trace[1, ...], 0, H - 1).long()
        rho_org_pad_pos = F.pad(rho_org, pad=pad_pos)
        rho_org_max = max_pool(
            rho_org_pad_pos.unsqueeze(0), kernel_size=2, stride=1
        ).squeeze(0)
        rho_org_min = -max_pool(
            -rho_org_pad_pos.unsqueeze(0), kernel_size=2, stride=1
        ).squeeze(0)
        if self.solver.is_2d():
            rho_org_max_forward_trace = rho_org_max[:, forward_trace_y, forward_trace_x]
            rho_org_min_forward_trace = rho_org_min[:, forward_trace_y, forward_trace_x]
        else:
            forward_trace_z = torch.clamp(forward_trace[0, ...], 0, D - 1).long()
            rho_org_max_forward_trace = rho_org_max[
                :, forward_trace_z, forward_trace_y, forward_trace_x
            ]
            rho_org_min_forward_trace = rho_org_min[
                :, forward_trace_z, forward_trace_y, forward_trace_x
            ]

        # clamp according to min and max
        if clamp_mode == 1:
            rho_org_min_trace = torch.min(
                rho_org_min_back_trace, rho_org_min_forward_trace
            )
            rho_org_max_trace = torch.max(
                rho_org_max_back_trace, rho_org_max_forward_trace
            )
            rho_adv = torch.max(rho_org_min_trace, rho_corr)
            rho_adv = torch.min(rho_org_max_trace, rho_adv)
        elif clamp_mode == 2:
            clamp_min_back = rho_corr < rho_org_min_back_trace
            clamp_max_back = rho_corr > rho_org_max_back_trace
            rho_adv = torch.where(clamp_min_back + clamp_max_back, rho_fwd, rho_corr)
        else:
            raise ValueError("clamp_mode can only be 1 or 2.")
        return rho_adv
