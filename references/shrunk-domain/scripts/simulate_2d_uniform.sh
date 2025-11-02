python simulations/simulate.py --experiment_name benchmark/2d_64_uniform \
--dim 2 --res 64 64 --cg_accu 1e-3 --wall_bnd xXyY \
--wind_x 0.005 --save_npz --save_compressed

python simulations/simulate.py --experiment_name benchmark/2d_128_uniform \
--dim 2 --res 128 128 --cg_accu 1e-3 --wall_bnd xXyY \
--wind_x 0.0015 --save_npz --save_compressed

python simulations/simulate.py --experiment_name benchmark/2d_256_uniform \
--dim 2 --res 256 256 --cg_accu 1e-3 --wall_bnd xXyY \
--wind_x 0.0005 --save_npz --save_compressed