import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())
from optimizations.optimizations.force_optimization import ForceOptimization
from optimizations.models.progressive_frequency_model import ProgressiveFrequencyModel


class ProgressiveFrequencyOptimization(ForceOptimization):
    def __init__(self, opt):
        assert len(opt.band_radii) == len(opt.phase_itrs)
        super().__init__(opt)

    def create_model(self):
        """Create 'ProgressiveFrequencyModel' as the model."""
        size = 2 * self.opt.band_radii[self.opt.start_phase] + 1
        self.model = ProgressiveFrequencyModel(
            self.solver,
            self.opt.keyframe_opt,
            init_size=size,
            use_stream_fcn=self.opt.use_stream_fcn,
            use_sim_wind=self.opt.use_sim_wind,
        )

    def set_optimization(self, phase: int):
        """Set optimizer and reorganize parameter size for phase.

        Args:
            phase (int): The index ofphase.

        """
        size = 2 * self.opt.band_radii[phase] + 1
        self.model.reorganize_parameters_(size=size)
        super(ProgressiveFrequencyOptimization, self).set_optimization(phase)

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
