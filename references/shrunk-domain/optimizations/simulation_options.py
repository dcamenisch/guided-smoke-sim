import argparse
from datetime import datetime
import os
import json


class SimulationOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--root_path",
            default="./optimizations/data/",
            help="root path to save results",
        )
        self.parser.add_argument(
            "--experiment_name", type=str, required=True, help="name of the experiment"
        )
        self.parser.add_argument(
            "--load_param_label",
            type=str,
            default=None,
            help="label for loading parameters",
        )
        self.parser.add_argument(
            "--total_steps", type=int, default=51, help="number of frames to simulate"
        )
        self.parser.add_argument(
            "--force_decay",
            type=float,
            default=0.0,
            help="decay rate of force field after last keyframe, 0 for not decaying.",
        )
        self.parser.add_argument(
            "--save_force",
            action="store_true",
            help="save force fields from model parameters",
        )
        self.parser.add_argument(
            "--save_vel_lic",
            action="store_true",
            help="save LIC plot instead of quiver plot for velocity",
        )
        self.parser.add_argument(
            "--save_npz",
            action="store_true",
            help="save density npy file during simulation",
        )
        self.parser.add_argument(
            "--save_compressed",
            action="store_true",
            help="save compressed npz.",
        )
        self.parser.add_argument(
            "--save_div",
            action="store_true",
            help="save divergence in RDBU colormap for "
            "both velocity and force during simulation",
        )
        self.parser.add_argument(
            "--save_vel", action="store_true", help="save velocity in RGB colormap"
        )
        self.parser.add_argument(
            "--optimization_opt_filename",
            type=str,
            default="opt.txt",
            help="filename of option file",
        )

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # folder names
        self.opt.save_path = os.path.join(self.opt.root_path, self.opt.experiment_name)

        # execution time
        self.opt.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # optimization settings
        optimization_opt_filename = os.path.join(
            self.opt.save_path, self.opt.optimization_opt_filename
        )
        with open(optimization_opt_filename) as optimization_opt_file:
            optimization_opt = json.load(optimization_opt_file)

        # print opt
        args = optimization_opt
        args.update(vars(self.opt))

        print("---------------- Options ----------------")
        print(json.dumps(args, indent=2, sort_keys=True))
        print("-----------------------------------------")

        # save opt to the disk
        filename = os.path.join(self.opt.save_path, "sim_opt.txt")
        with open(filename, "w") as opt_file:
            json.dump(args, opt_file, indent=2)

        # convert dict to namespace for further use
        self.opt = argparse.Namespace(**args)
        self.opt.keyframe_opt = argparse.Namespace(**self.opt.keyframe_opt)

        return self.opt
