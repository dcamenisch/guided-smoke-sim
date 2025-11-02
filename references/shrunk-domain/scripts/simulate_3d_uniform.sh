python simulations/simulate.py --experiment_name benchmark/3d_128_uniform_rk1 \
--dim 3 --res 128 128 128 --total_steps 70 --source_steps 70 --cg_accu 1e-3 --wall_bnd xXyYzZ \
--wind_x 0.003 --buoyancy 0 -0.003 0 --save_npz --save_compressed --advection_rk_order 1

python simulations/simulate.py --experiment_name benchmark/3d_128_uniform_rk3 \
--dim 3 --res 128 128 128 --total_steps 70 --source_steps 70 --cg_accu 1e-3 --wall_bnd xXyYzZ \
--wind_x 0.003 --buoyancy 0 -0.003 0 --save_npz --save_compressed --advection_rk_order 3
