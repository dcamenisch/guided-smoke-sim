res='64 128 256'
for RES in $res; do
    python forces/generate_multi_vortex_${RES}.py

    mkdir -p simulations/data/benchmark/2d_${RES}_multi_vortex/npz

    cp forces/data/2d_${RES}_multi_vortex/000.npz simulations/data/benchmark/2d_${RES}_multi_vortex/npz/
    
    python simulations/simulate.py --experiment_name benchmark/2d_${RES}_multi_vortex \
    --dim 2 --res ${RES} ${RES} --total_steps 70 --source_steps 70 --cg_accu 1e-3 --wall_bnd xXyY \
    --buoyancy 0 -0.003 0 --load_npz --save_npz --save_compressed
done