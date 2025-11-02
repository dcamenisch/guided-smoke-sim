import argparse
from datetime import datetime
import os
import sys
import json

sys.path.append(os.getcwd())
from lib.dfluids.utils import mkdir


class OptimizationOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "optimization",
            type=str,
            choices=[
                "keyframe_force",
                "keyframe_prog_force",
                "keyframe_prog_freq",
            ],
            help="select which type of learning task to work on",
        )
        # invoke optimization specific arguments
        args = self.parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.optimization):
            print("Unrecognized optimization")
            self.parser.print_help()
            exit(1)
        getattr(self, args.optimization)()

        self.parser.add_argument(
            "--dim",
            type=int,
            default=2,
            choices=[2, 3],
            help="dimensionality of the simulation",
        )
        self.parser.add_argument(
            "--res",
            type=int,
            nargs="+",
            default=[128],
            help="resolution of the scene: (D)*W*H ",
        )
        self.parser.add_argument(
            "--dt", type=float, default=1.0, help="time step size of the simulation"
        )
        self.parser.add_argument(
            "--root_path",
            default="./optimizations/data/",
            help="root path to save results",
        )
        self.parser.add_argument(
            "--experiment_name", type=str, required=True, help="name of the experiment"
        )

        # optimization parameters
        self.parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "adagrad", "rmsprop", "sgd", "pytorch_lbfgs"],
            help="optimizer used to learn parameters",
        )
        self.parser.add_argument(
            "--lbfgs_line_search_fcn",
            type=str,
            default="Wolfe",
            choices=["Wolfe", "Armijo", "None"],
            help="choose line search condition function",
        )
        self.parser.add_argument(
            "--lbfgs_history_size",
            type=int,
            default=10,
            help="history size parameter for L-BFGS",
        )
        self.parser.add_argument(
            "--lbfgs_max_ls",
            type=int,
            default=25,
            help="maximum step of line search for L-BFGS",
        )
        self.parser.add_argument(
            "--lbfgs_inplace",
            action="store_true",
            help="set inplace operation for Pytorch LBFGS to be true",
        )
        self.parser.add_argument(
            "--lr", type=float, nargs="+", default=[0.0001], help="learning rate"
        )
        self.parser.add_argument(
            "--start_itr",
            type=int,
            default=1,
            help="starting number of iterations, used for logging in tensorboard",
        )
        self.parser.add_argument(
            "--start_phase", type=int, default=0, help="starting number of phases."
        )
        self.parser.add_argument(
            "--phase_itrs",
            nargs="+",
            type=int,
            default=[200],
            help="number of iterations for each phase of the optimization",
        )
        self.parser.add_argument(
            "--save_param_freq",
            type=int,
            default=10,
            help="save model parameter frequency",
        )
        self.parser.add_argument(
            "--load_param_label",
            type=str,
            default=None,
            help="label for loading parameters",
        )
        self.parser.add_argument(
            "--objective_change_tolerance",
            type=float,
            default=0.0,
            help=(
                "tolerance for relative objective change s.t. "
                "optimization stays in current phase."
            ),
        )
        self.initialized = True

    def keyframe_force(self):
        # keyframe parameters
        self.parser.add_argument(
            "--load_path", default="./simulations/data/", help="path to load keyframes"
        )
        self.parser.add_argument(
            "--keyframes", nargs="+", type=int, default=[11], help="keyframe numbers."
        )
        self.parser.add_argument(
            "--keyframes_name", type=str, required=True, help="keyframe simulation name"
        )
        self.parser.add_argument(
            "--npy_name",
            type=str,
            default="plume_wind_2d_64_uniform_old",
            help="npy simulation name",
        )
        self.parser.add_argument(
            "--load_init_vel",
            action="store_true",
            help="load a starting velocity from current folder as initialization.",
        )
        self.parser.add_argument(
            "--load_init_vel_sim",
            action="store_true",
            help="load a starting velocity from simulation folder as initialization.",
        )
        self.parser.add_argument(
            "--load_init_vel_npy",
            action="store_true",
            help="load a starting velocity from npy as initialization.",
        )
        self.parser.add_argument(
            "--load_init_density",
            action="store_true",
            help="load a starting frame from current folder as initialization.",
        )
        self.parser.add_argument(
            "--load_init_density_sim",
            action="store_true",
            help="load a starting frame from simulation folder as initialization.",
        )
        self.parser.add_argument(
            "--load_init_density_npy",
            action="store_true",
            help="load a starting frame from npy as initialization.",
        )
        self.parser.add_argument(
            "--init_frame", type=int, default=0, help="starting frame number."
        )
        self.parser.add_argument(
            "--use_sim_wind",
            action="store_true",
            help="Use the wind parameters from simulation.",
        )
        self.parser.add_argument(
            "--distance_metric",
            type=str,
            default="l1",
            choices=["l1", "l2"],
            help="distance metric to compute pointwise objective",
        )
        self.parser.add_argument(
            "--use_tikinov_reg",
            action="store_true",
            help="use Tikinov regularization on force.",
        )
        self.parser.add_argument(
            "--lambd_tikinov_reg",
            type=float,
            default=0.0,
            help="weight for Tikinov regularization on force",
        )
        self.parser.add_argument(
            "--use_tv_reg",
            action="store_true",
            help="use total variation regularization on force",
        )
        self.parser.add_argument(
            "--lambd_tv_reg",
            type=float,
            default=0.0,
            help="weight for total variation regularization " "on force",
        )
        self.parser.add_argument(
            "--log_mse",
            action="store_true",
            help="log mse objective for each iteration",
        )
        self.parser.add_argument(
            "--log_normed_mse",
            action="store_true",
            help="log normalized (through density amount) mse objective for each iteration",
        )
        self.parser.add_argument(
            "--use_stream_fcn",
            action="store_true",
            help="use stream function to represent force.",
        )

    def keyframe_prog_freq(self):
        self.keyframe_force()
        self.parser.add_argument(
            "--band_radii",
            nargs="+",
            type=int,
            default=[1, 3, 8, 23, 64],
            help="radii of fourier bands for filtering force for each phase.",
        )

    def keyframe_prog_force(self):
        self.keyframe_force()
        self.parser.add_argument(
            "--force_radii",
            nargs="+",
            type=int,
            required=True,
            help="radii of force field for each phase.",
        )

    def parse(self, filename="opt.txt"):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # folder names
        self.opt.save_path = os.path.join(self.opt.root_path, self.opt.experiment_name)

        # execution time
        self.opt.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # keyframe settings
        self.opt.keyframes_load_path = os.path.join(
            self.opt.load_path, self.opt.keyframes_name
        )
        self.opt.npy_load_path = os.path.join(self.opt.load_path, self.opt.npy_name)
        # load keyframe options from file
        keyframe_opt_filename = os.path.join(self.opt.keyframes_load_path, "opt.txt")
        with open(keyframe_opt_filename) as keyframe_opt_file:
            keyframe_opt = json.load(keyframe_opt_file)
        self.opt.keyframe_opt = keyframe_opt

        # print opt
        args = vars(self.opt)
        print("----------------- Options ----------------")
        print(json.dumps(args, indent=2, sort_keys=True))
        print("------------------------------------------")
        # save opt to the disk
        mkdir(self.opt.save_path)
        opt_filename = os.path.join(self.opt.save_path, filename)
        with open(opt_filename, "w") as opt_file:
            json.dump(args, opt_file, indent=2)

        # convert dict to namespace for further use
        self.opt.keyframe_opt = argparse.Namespace(**keyframe_opt)

        return self.opt
