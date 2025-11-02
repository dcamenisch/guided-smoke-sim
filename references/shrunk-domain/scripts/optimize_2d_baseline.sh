# Baseline + TV
python optimizations/optimize.py keyframe_force --dim 2 --res 128 --save_param_freq 40 \
--experiment_name keyframe_2d_128_[uniform,51]_mse_tv_10_pytorch_lbfgs_lr_1_tol_0.00001 \
--phase_itrs 5000 --optimizer pytorch_lbfgs --lr 1 --use_tv_reg --lambd_tv_reg 10 \
--objective_change_tolerance 0.000001 --distance_metric l2 \
--keyframes 51 --keyframes_name benchmark/2d_128_uniform
