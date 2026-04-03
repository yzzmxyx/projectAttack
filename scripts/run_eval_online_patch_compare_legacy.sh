#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/yxx/projectAttack"
EXP_DIR="${EXP_DIR:-/home/yxx/projectAttack/run/UADA_rollout_online_env/bde0d941-cd74-4525-baa9-40c59a8c6cdd}"
DEVICE="${DEVICE:-3}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-5}"
PATCH_A_SUBDIR="${PATCH_A_SUBDIR:-0}"
PATCH_B_SUBDIR="${PATCH_B_SUBDIR:-last}"

cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "EXP_DIR=$EXP_DIR"
echo "DEVICE=$DEVICE"
echo "TASK_SUITE_NAME=$TASK_SUITE_NAME"
echo "TRIALS_PER_TASK=$TRIALS_PER_TASK"
echo "PATCH_A_SUBDIR=$PATCH_A_SUBDIR"
echo "PATCH_B_SUBDIR=$PATCH_B_SUBDIR"

python3.10 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("libero") is None:
    print("ERROR: `libero` is not installed. Please install LIBERO before running online patch compare eval.")
    sys.exit(1)
try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    print(f"ERROR: `libero` is installed but OffScreenRenderEnv is unavailable: {exc}")
    sys.exit(1)
print("LIBERO dependency check passed.")
PY

python3.10 evaluation_tool/eval_online_patch_compare.py \
    --exp_dir "$EXP_DIR" \
    --patch_a_subdir "$PATCH_A_SUBDIR" \
    --patch_b_subdir "$PATCH_B_SUBDIR" \
    --task_suite_name "$TASK_SUITE_NAME" \
    --device "$DEVICE" \
    --trials_per_task "$TRIALS_PER_TASK" \
    --val_seed 42 \
    --seed 42 \
    --maskidx 0,1,2 \
    --use_all_joints false \
    --gripper_weight 0.5 \
    --num_steps_wait 10 \
    --max_env_steps auto_by_suite \
    --eval_rollout_steps auto_by_suite \
    --env_resolution 256 \
    --lambda_action_gap 1.0 \
    --lambda_history 0.0 \
    --lambda_ce 0.1 \
    --online_ce_mode off \
    --geometry true \
    --attack_mode projection \
    --projection_alpha 0.55 \
    --projection_alpha_jitter 0.00 \
    --projection_soft_edge 1.2 \
    --projection_angle 0 \
    --projection_fixed_angle true \
    --projection_shear 0.00 \
    --projection_scale_min 1.0 \
    --projection_scale_max 1.0 \
    --projection_region lower_half_fixed \
    --projection_lower_start 0.55 \
    --projection_width_ratio 0.90 \
    --projection_height_ratio 0.95 \
    --projection_margin_x 0.04 \
    --projection_keystone 0.22 \
    --projection_keystone_jitter 0.00 \
    --projector_gamma 1.8 \
    --projector_gain 1.35 \
    --projector_channel_gain 1.08,1.04,1.00 \
    --projector_ambient 0.08 \
    --projector_vignetting 0.08 \
    --projector_distance_falloff 0.10 \
    --projector_psf false \
    --projection_randomization_enabled false \
    --lighting_aug_enabled true \
    --lighting_backend legacy \
    --lighting_model_id stabilityai/sdxl-turbo \
    --lighting_pool_size 2 \
    --lighting_refresh_interval 100 \
    --lighting_num_inference_steps 2 \
    --lighting_guidance_scale 0.0 \
    --lighting_blend_min 0.15 \
    --lighting_blend_max 0.45 \
    --lighting_apply_prob 1.0 \
    --lighting_seed 42 \
    --val_disable_lighting false \
    --record_video true \
    --video_frame_source projected_input \
    --video_fps 10
