#!/bin/bash
current_dir=$(pwd)
echo "$current_dir"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

python3.10 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("libero") is None:
    print("ERROR: `libero` is not installed. Please install LIBERO before running online env rollout.")
    sys.exit(1)
try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    print(f"ERROR: `libero` is installed but OffScreenRenderEnv is unavailable: {exc}")
    sys.exit(1)
print("LIBERO dependency check passed.")
PY

python3.10 VLAAttacker/UADA_rollout_online_env_wrapper.py \
    --maskidx 0,1,2 \
    --use_all_joints false \
    --gripper_weight 0.5 \
    --lr 2e-3 \
    --server "$current_dir" \
    --device 3 \
    --iter 2000 \
    --accumulate 2 \
    --bs 1 \
    --warmup 20 \
    --tags "UADA_rollout_online_env" \
    --geometry true \
    --attack_mode "projection" \
    --patch_size "3,22,22" \
    --projection_size "3,22,22" \
    --projection_alpha 0.55 \
    --projection_alpha_jitter 0.00 \
    --projection_soft_edge 1.2 \
    --projection_angle 0 \
    --projection_fixed_angle true \
    --projection_shear 0.00 \
    --projection_scale_min 1.0 \
    --projection_scale_max 1.0 \
    --projection_region "lower_half_fixed" \
    --projection_lower_start 0.55 \
    --projection_width_ratio 0.90 \
    --projection_height_ratio 0.95 \
    --projection_margin_x 0.04 \
    --projection_keystone 0.22 \
    --projection_keystone_jitter 0.00 \
    --projector_gamma 1.8 \
    --projector_gain 1.35 \
    --projector_channel_gain "1.08,1.04,1.00" \
    --projector_ambient 0.08 \
    --projector_vignetting 0.08 \
    --projector_distance_falloff 0.10 \
    --projector_psf false \
    --wandb_project "projectAttack" \
    --wandb_entity "1473195970-beihang-university" \
    --dataset "libero_spatial" \
    --resize_patch false \
    --phase1_ratio 0.4 \
    --phase1_rollout 32 \
    --phase2_rollout 128 \
    --lambda_action_gap 1.0 \
    --lambda_history 0.5 \
    --lambda_ce 0.1 \
    --save_interval 50 \
    --eval_enabled true \
    --val_deterministic true \
    --val_seed 42 \
    --val_disable_lighting true \
    --lighting_aug_enabled false \
    --lighting_aug_train_only false \
    --phase1_disable_lighting true \
    --phase1_disable_projection_randomization true \
    --lighting_backend "ic_light" \
    --lighting_model_id "stabilityai/sdxl-turbo" \
    --lighting_pool_size 2 \
    --lighting_refresh_interval 100 \
    --lighting_num_inference_steps 8 \
    --lighting_guidance_scale 7.0 \
    --lighting_blend_min 0.15 \
    --lighting_blend_max 0.45 \
    --lighting_apply_prob 1.0 \
    --lighting_seed 42 \
    --ic_light_repo "/home/yxx/IC-Light" \
    --ic_light_model_path "/home/yxx/IC-Light/models/iclight_sd15_fbc.safetensors" \
    --ic_light_scope "full" \
    --ic_light_bg_control "legacy_prompt" \
    --record_online_videos true \
    --record_online_videos_last_only false \
    --record_online_train_video false \
    --record_online_val_video true \
    --record_online_video_frame_source "projected_input" \
    --record_online_video_fps 10 \
    --viz_enabled true \
    --viz_policy "milestone" \
    --viz_samples 2 \
    --viz_save_best true \
    --viz_save_last true \
    --task_suite_name "auto" \
    --online_train_tasks_per_iter 2 \
    --online_train_episodes_per_task 4 \
    --online_val_episodes 8 \
    --num_steps_wait 10 \
    --max_env_steps "auto_by_suite" \
    --env_resolution 256 \
    --online_ce_mode "pseudo_clean" \
    --env_action_source "adv" \
    --env_seed 42 \
    --auto_gpu_tune false
