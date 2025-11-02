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

curl = 0

# large scale
phi = node_op.get_gaussian(
    mu=torch.Tensor([0.0, 0.0]),
    sig=torch.Tensor([[0.1, 0], [0, 0.1]]),
    amp=torch.Tensor([-0.015]),
)
curl = curl + node_op.get_curl(phi)

# middle scale
phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.4, -0.4]),
    sig=torch.Tensor([[0.04, 0], [0, 0.04]]),
    amp=torch.Tensor([0.01]),
)
curl = curl + node_op.get_curl(phi)

phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.35, 0.1]),
    sig=torch.Tensor([[0.01, 0], [0, 0.01]]),
    amp=torch.Tensor([0.005]),
)
curl = curl + node_op.get_curl(phi)

# small scale
phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.2, 0.2]),
    sig=torch.Tensor([[0.001, 0], [0, 0.001]]),
    amp=torch.Tensor([0.002]),
)
curl = curl + node_op.get_curl(phi)

phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.25, 0.4]),
    sig=torch.Tensor([[0.001, 0], [0, 0.001]]),
    amp=torch.Tensor([-0.002]),
)
curl = curl + node_op.get_curl(phi)

phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.5, 0.35]),
    sig=torch.Tensor([[0.001, 0], [0, 0.001]]),
    amp=torch.Tensor([0.002]),
)
curl = curl + node_op.get_curl(phi)

phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.7, 0.2]),
    sig=torch.Tensor([[0.001, 0], [0, 0.001]]),
    amp=torch.Tensor([0.002]),
)
curl = curl + node_op.get_curl(phi)

phi = node_op.get_gaussian(
    mu=torch.Tensor([-0.7, 0.0]),
    sig=torch.Tensor([[0.001, 0], [0, 0.001]]),
    amp=torch.Tensor([-0.002]),
)
curl = curl + node_op.get_curl(phi)


utils.mkdir("forces/data/2d_64_multi_vortex/")
mac_op.save_img(curl, "forces/data/2d_64_multi_vortex/lic.png", colormap="LIC")
mac_op.save_img(curl, "forces/data/2d_64_multi_vortex/arrow.png", colormap="ARROW")
np.savez("forces/data/2d_64_multi_vortex/000.npz", force=mac_op.grid2array(curl))
