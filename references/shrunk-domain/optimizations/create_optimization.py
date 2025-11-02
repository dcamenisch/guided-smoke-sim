import os
import sys

sys.path.append(os.getcwd())
from optimizations.optimizations.force_optimization import ForceOptimization
from optimizations.optimizations.progressive_force_optimization import (
    ProgressiveForceOptimization,
)
from optimizations.optimizations.progressive_frequency_optimization import (
    ProgressiveFrequencyOptimization,
)


def create_optimization(opt):
    if opt.optimization == "keyframe_force":
        return ForceOptimization(opt)
    elif opt.optimization == "keyframe_prog_force":
        return ProgressiveForceOptimization(opt)
    elif opt.optimization == "keyframe_prog_freq":
        return ProgressiveFrequencyOptimization(opt)
    else:
        raise NotImplementedError("Optimization not recognized from opt.optimization.")
