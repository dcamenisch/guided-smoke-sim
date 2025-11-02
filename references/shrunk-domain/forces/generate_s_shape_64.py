import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
from lib.dfluids.solver import FluidSolver
from lib.dfluids.grid import GridType
import lib.dfluids.utils as utils

# 2d big curl force
res = 64
solver = FluidSolver(grid_size=(res, res))
ext = solver.create_extforces()
mac_op = solver.create_grid_operator(GridType.MAC)
real_op = solver.create_grid_operator(GridType.REAL)
node_op = solver.create_grid_operator(GridType.NODE)
flags, phi_obs = ext.init_flags(phi=True, boundary_width=0, wall="xXyY")

phi1 = node_op.get_gaussian(
    mu=torch.Tensor([0.0, -0.5]),
    sig=torch.Tensor([[40, 0], [0, 1]]),
    amp=torch.Tensor([-0.04]),
)

phi2 = node_op.get_gaussian(
    mu=torch.Tensor([0.0, 0.5]),
    sig=torch.Tensor([[40, 0], [0, 1]]),
    amp=torch.Tensor([0.04]),
)

curl1 = node_op.get_curl(phi1)
curl2 = node_op.get_curl(phi2)

curl = curl1
curl[:, int(res / 2) :, :] = curl2[:, int(res / 2) :, :]

utils.mkdir("forces/data/2d_64_s_shape/")
mac_op.save_img(curl, "forces/data/2d_64_s_shape/lic.png", colormap="LIC")
mac_op.save_img(curl, "forces/data/2d_64_s_shape/arrow.png", colormap="ARROW")
np.savez("forces/data/2d_64_s_shape/000.npz", force=mac_op.grid2array(curl))
