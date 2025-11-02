EXPERIMENT_NAME=keyframe_2d_128_[uniform,51]_mse_tv_10_pytorch_lbfgs_lr_1_tol_0.00001 

python optimizations/simulate.py --total_steps 60 --load_param_label best \
--experiment_name ${EXPERIMENT_NAME} --save_vel --save_npz --save_div --save_compressed
