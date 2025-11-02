import argparse
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.join(os.getcwd()))
from lib.dfluids.utils import mkdir


class SimulationOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--root_path",
            default="./simulations/data/",
            help="path to save simulation results",
        )  # root_path
        self.parser.add_argument(
            "--dim",
            type=int,
            default=2,
            choices=[2, 3],
            help="dimensionality of the simulation",
        )  # dim
        self.parser.add_argument(
            "--res",
            nargs="+",
            type=int,
            default=[128, 128],
            help="resolution of the scene, in shape [res_x, res_y, res_z]",
        )  # res
        self.parser.add_argument(
            "--dt", type=float, default=1.0, help="time step size of the simulation"
        )  # dt
        self.parser.add_argument(
            "--advection_order",
            type=int,
            default=1,
            choices=[1, 2],
            help="order of Semi-Lagrange advection, 1 for normal SL, 2 for MacCormack",
        )  # advection_order
        self.parser.add_argument(
            "--advection_rk_order",
            type=int,
            default=3,
            choices=[1, 2, 3],
            help="order of runge-kutta interpolation in advection",
        )  # rk_order
        self.parser.add_argument(
            "--apply_precon",
            action="store_true",
            help="apply preconditioner for pressure projection",
        )  # precondition
        self.parser.add_argument(
            "--cg_accu",
            type=float,
            default=1e-6,
            help="tolerance of residual for CG solver",
        )  # cg_accu
        self.parser.add_argument(
            "--use_obstacle", action="store_true", help="use obstacle in the scene"
        )  # use_obstacle
        self.parser.add_argument(
            "--start_step", type=int, default=0, help="start step number"
        )  # start step
        self.parser.add_argument(
            "--total_steps", type=int, default=60, help="total number of time steps"
        )  # total step
        self.parser.add_argument(
            "--experiment_name", type=str, default=None, help="name of the experiment"
        )  # experiment_name
        self.parser.add_argument(
            "--wall_bnd",
            type=str,
            default="xXyY",
            help="set which scene boundary to be solid",
        )  # wall_bnd
        self.parser.add_argument(
            "--obs_size",
            nargs="+",
            type=float,
            default=[0.2],
            help="relative obstacle size",
        )  # obs_size
        self.parser.add_argument(
            "--obs_center",
            nargs="+",
            type=float,
            default=[0.0, -0.5, 0.0],
            help="relative obstacle center",
        )  # obs_center
        self.parser.add_argument(
            "--save_vort", action="store_true", help="save the vortex of the flow"
        )  # save_vort
        self.parser.add_argument(
            "--save_div",
            action="store_true",
            help="save the divergence of the velocity",
        )  # save_div
        self.parser.add_argument(
            "--save_pressure", action="store_true", help="save the pressure of the flow"
        )  # save_pressure
        self.parser.add_argument(
            "--transmit",
            type=float,
            default=0.05,
            help="transmittance of simple renderer for density vis.",
        )  # transmit
        self.parser.add_argument(
            "--buoyancy",
            type=float,
            nargs="+",
            default=[0, -4e-3, 0],
            help="buoyancy parameter",
        )  # buoyancy
        self.parser.add_argument(
            "--source_shape",
            type=str,
            choices=["BOX", "SPHERE"],
            default="BOX",
            help="shape of the source.",
        )  # source shape
        self.parser.add_argument(
            "--source_value", type=float, default=1.0, help="source value"
        )  # source value
        self.parser.add_argument(
            "--source_size",
            nargs="+",
            type=float,
            default=[0.2],
            help="relative obstacle size",
        )  # source size
        self.parser.add_argument(
            "--source_center",
            nargs="+",
            type=float,
            default=[0.0, -1.0, 0.0],
            help="relative obstacle center",
        )  # source center
        self.parser.add_argument(
            "--source_steps",
            type=int,
            default=60,
            help="number of frames to add source",
        )  # source step
        self.parser.add_argument(
            "--wind_x", type=float, default=0, help="wind parameter in x direction"
        )  # wind_x
        self.parser.add_argument(
            "--wind_y", type=float, default=0, help="wind parameter in y direction"
        )  # wind_y
        self.parser.add_argument(
            "--wind_z", type=float, default=0, help="wind parameter in z direction"
        )  # wind_z
        self.parser.add_argument(
            "--load_npz",
            action="store_true",
            help="load from npz file to initialize force and/or vel and/or density.",
        )
        self.parser.add_argument(
            "--save_npz",
            action="store_true",
            help="save quantities as npz.",
        )  # save npz
        self.parser.add_argument(
            "--save_compressed",
            action="store_true",
            help="save compressed npz.",
        )  # save compressed
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # folder names
        if self.opt.experiment_name is None:
            self.opt.experiment_name = "dff_plume_{}d_{}_order_{}".format(
                self.opt.dim, self.opt.res, self.opt.advection_order
            )
        self.opt.save_path = os.path.join(self.opt.root_path, self.opt.experiment_name)
        # execution time
        self.opt.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # print opt
        args = vars(self.opt)
        print("---------------- Options ----------------")
        print(json.dumps(args, indent=2, sort_keys=True))
        print("-----------------------------------------")
        # save opt to the disk
        mkdir(self.opt.save_path)
        filename = os.path.join(self.opt.save_path, "opt.txt")
        with open(filename, "w") as opt_file:
            json.dump(args, opt_file, indent=2, sort_keys=True)
        return self.opt
