res='64 128'
for RES in $res; do
    python forces/generate_s_shape_${RES}.py

    mkdir -p simulations/data/benchmark/2d_${RES}_s_shape/npz

    cp forces/data/2d_${RES}_s_shape/000.npz simulations/data/benchmark/2d_${RES}_s_shape/npz/

    python simulations/simulate.py --experiment_name benchmark/2d_${RES}_s_shape \
    --dim 2 --res ${RES} ${RES} --total_steps 70 --source_steps 70 --cg_accu 1e-3 --wall_bnd xXyY \
    --buoyancy 0 -0.004 0 --load_npz --save_npz --save_compressed
done

python forces/generate_s_shape_256.py

mkdir -p simulations/data/benchmark/2d_256_s_shape/npz

cp forces/data/2d_256_s_shape/000.npz simulations/data/benchmark/2d_256_s_shape/npz/

python simulations/simulate.py --experiment_name benchmark/2d_256_s_shape \
--dim 2 --res 256 256 --total_steps 70 --source_steps 70 --cg_accu 1e-3 --wall_bnd xXyY \
--buoyancy 0 -0.003 0 --load_npz --save_npz --save_compressed