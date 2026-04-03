#!/bin/bash
current_dir=$(pwd)
echo $current_dir
python VLAAttacker/UPA_wrapper.py \
    --maskidx 0 \
    --lr 2e-3 \
    --server $current_dir \
    --device 0 \
    --iter 2000 \
    --accumulate 1 \
    --bs 8 \
    --warmup 20 \
    --tags "debug testrun" \
    --filterGripTrainTo1 false \
    --geometry true \
    --attack_mode "projection" \
    --patch_size "3,50,50" \
    --projection_size "3,50,50" \
    --projection_alpha 0.35 \
    --projection_alpha_jitter 0.10 \
    --projection_soft_edge 2.5 \
    --projection_angle 25 \
    --projection_shear 0.15 \
    --projection_scale_min 0.8 \
    --projection_scale_max 1.2 \
    --projection_region "desk_bottom" \
    --projector_gamma 2.2 \
    --projector_gain 1.0 \
    --projector_psf false \
    --wandb_project "false" \
    --wandb_entity "xxx" \
    --innerLoop 50 \
    --dataset "libero_spatial" # "libero_spatial" / "libero_10" / "libero_goal" / "libero_goal" / "bridge_orig"
