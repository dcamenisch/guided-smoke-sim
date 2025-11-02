python optimizations/optimize.py keyframe_prog_freq --dim 2 --res 128 --save_param_freq 40 \
--experiment_name keyframe_2d_128_[uniform,51]_stream_mse_filter_force_[1,3,8,23,64]_pytorch_lbfgs_lr_1_tol_0.00001 \
--phase_itrs 500 500 500 500 500 --optimizer pytorch_lbfgs --lr 1 1 1 1 1 --objective_change_tolerance 0.00001 \
--use_stream_fcn --band_radii 1 3 8 23 64 --distance_metric l2 \
--keyframes 51 --keyframes_name benchmark/2d_128_uniform
