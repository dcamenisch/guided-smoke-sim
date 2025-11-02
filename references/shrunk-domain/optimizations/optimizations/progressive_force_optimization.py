import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from optimizations.optimizations.force_optimization import ForceOptimization
from optimizations.models.progressive_force_model import ProgressiveForceModel


class ProgressiveForceOptimization(ForceOptimization):
    def __init__(self, opt):
        super(ProgressiveForceOptimization, self).__init__(opt)
        assert len(opt.phase_itrs) == len(opt.force_radii)
        size = self.opt.force_radii[self.opt.start_phase] * 2 + 1
        self.model.reorganize_parameters_(size=size)

    def create_model(self):
        """Create 'ProgressiveForceModel' as the model."""
        size = 2 * self.opt.force_radii[self.opt.start_phase] + 1
        self.model = ProgressiveForceModel(
            self.solver,
            self.opt.keyframe_opt,
            init_size=size,
            use_stream_fcn=self.opt.use_stream_fcn,
            use_sim_wind=self.opt.use_sim_wind,
        )

    def set_optimization(self, phase: int):
        """Set optimizer and reorganize parameter size for phase.

        Args:
            phase (int): The index of phase.

        """
        size = 2 * self.opt.force_radii[phase] + 1
        self.model.reorganize_parameters_(size=size)
        super(ProgressiveForceOptimization, self).set_optimization(phase)

    def load(self, label: str, format: str = "pth"):
        """Load state dict for self.model

        Args:
            label (str): The label of iteration to be loaded.
            format (str): The format of file.
        """
        state_dict = super(ForceOptimization, self).load_state_dict(label, format)
        params_size = state_dict["param"].size()
        self.model.reorganize_parameters_(size=params_size[1])
        self.model.load_state_dict(state_dict)
