import os
import sys
import torch

sys.path.append(os.getcwd())
from optimizations.simulation_options import SimulationOptions
from optimizations.optimizations.progressive_frequency_optimization import (
    ProgressiveFrequencyOptimization,
)


@torch.no_grad()
def main():
    opt = SimulationOptions().parse()
    optimization = ProgressiveFrequencyOptimization(opt)

    # load model parameters
    if opt.load_param_label is not None:
        optimization.load(label=opt.load_param_label, format="pth")
    else:
        print("No model loaded, using initial parameters to simulate...")
    # simulation core
    optimization.simulate()


if __name__ == "__main__":
    main()
