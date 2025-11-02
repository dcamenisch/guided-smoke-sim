# Shrunk-Domain

Official PyTorch implementation for [Honey, I Shrunk the Domain: Frequency-aware Force Field Reduction for Efficient Fluids Optimization](https://cgl.ethz.ch/publications/papers/paperTan21a.php).

### Installation
``` bash
# create a conda env
conda create -n shrunk-domain
conda activate shrunk-domain
# install pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
# install other requirements
pip install -r requirements.txt
```

### Simulation
The sequences for the benchmark tests can be generated through simulation by running the provided simulation scripts. For example, to generate 2D uniform-wind simulation:
```bash
bash scripts/simulate_2d_uniform.sh
```

### Optimization
Optimizations can be used for keyframe matching. Before running the optimization scripts, we need to put the keyframe data in `.npz` format into `simulations/data` directory. For example, the above 2D 128x128 uniform-wind simulation will create `.npz` files under `simulations/data/benchmark/2d_128_uniform`. A corresponding `opt.txt` file is also needed to specify simulation configurations as guidance to the simulations used in the optimization process.

After the data and config files are prepared, we can start to run the optimization. The provided scripts give examples to run different optimization methods: baseline+TV `bash scripts/optimize_2d_baseline.sh`, progressive upsampling `bash scripts/optimize_2d_prog_force.sh` and ours `bash scripts/optimize_2d_prog_freq.sh`.

After the optimizaiton is done, we can run a simulation by loading the saved optimized parameter to get to the target frame:

``` bash
# need to specify EXPERIMENT_NAME in this script first
bash scripts/optimize_simulate.sh
```

### Structure of the code
- `lib/dfluids` implements differentiable operations needed to run smoke simulations.
- `lib/PyTorch_LBFGS.py` is adapted from [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS), the optimization processes use this implementation of LBFGS optimizer.
- `simulations` defines argparse interfaces for users to define simulation config, and uses operations from `dfluids` to run smoke simulations.
- `optimizations/models` defines models to wrap optimization parameters and how the parameters are used in the simulation.
- `optimizations/optimizations` specifies optimization processes, e.g. procedures to compute objective function, to optimize the parameters, etc.
- `forces` contains scripts to generate force fields used for the benchmark tests.
