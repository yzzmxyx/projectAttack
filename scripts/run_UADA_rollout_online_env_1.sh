#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "${current_dir}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"

# Match the historical probe budget/task-suite defaults so GT runs stay directly comparable.
DATASET_NAME="${DATASET:-libero_spatial}"
DEVICE_ID="${DEVICE:-7}"
ACCUMULATE_STEPS="${ACCUMULATE:-1}"
WARMUP_STEPS="${WARMUP:-2}"
PHASE2_ROLLOUT_STEPS="${PHASE2_ROLLOUT:-24}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL:-50}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-projectAttack}"
WANDB_ENTITY_NAME="${WANDB_ENTITY:-1473195970-beihang-university}"
LAMBDA_ACTION_GAP="${LAMBDA_ACTION_GAP:-1.0}"
LAMBDA_HISTORY="${LAMBDA_HISTORY:-0.0}"
LAMBDA_HISTORY_LEGACY="${LAMBDA_HISTORY_LEGACY:-0.0}"
LAMBDA_CE="${LAMBDA_CE:-0.0}"
LAMBDA_CE_PHASE2="${LAMBDA_CE_PHASE2:-0.0}"
LAMBDA_CONTINUOUS_ROLLOUT="${LAMBDA_CONTINUOUS_ROLLOUT:-0.0}"
LAMBDA_WINDOW_ROLLOUT_LOSS="${LAMBDA_WINDOW_ROLLOUT_LOSS:-1.0}"
IMPULSE_ROLLOUT_METRIC_ENABLED="${IMPULSE_ROLLOUT_METRIC_ENABLED:-false}"
WINDOW_ROLLOUT_PROBE_ENABLED_VALUE="${WINDOW_ROLLOUT_PROBE_ENABLED:-true}"
WINDOW_ROLLOUT_METRIC_MODE_VALUE="${WINDOW_ROLLOUT_METRIC_MODE:-adv_gt}"
WINDOW_ROLLOUT_FUTURE_MODE_VALUE="${WINDOW_ROLLOUT_FUTURE_MODE:-drop_attack_after_window}"
WINDOW_ROLLOUT_EXP_BASE_VALUE="${WINDOW_ROLLOUT_EXP_BASE:-0.9}"
WINDOW_ROLLOUT_FUTURE_HORIZON_VALUE="${WINDOW_ROLLOUT_FUTURE_HORIZON:-8}"
WINDOW_ROLLOUT_PHASE_SCOPE_VALUE="${WINDOW_ROLLOUT_PHASE_SCOPE:-initial}"
LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-1.0}"
ONLINE_CE_MODE_NAME="${ONLINE_CE_MODE:-off}"
SIGLIP_MODEL_NAME_VALUE="${SIGLIP_MODEL_NAME:-google/siglip-so400m-patch14-384}"
SIGLIP_INPUT_SIZE_VALUE="${SIGLIP_INPUT_SIZE:-384}"
PATCH_SIZE_VALUE="${PATCH_SIZE:-3,50,50}"
PROJECTION_SIZE_VALUE="${PROJECTION_SIZE:-${PATCH_SIZE_VALUE}}"
ENV_RESOLUTION_VALUE="${ENV_RESOLUTION:-256}"
TRAIN_TASKS_PER_ITER="${ONLINE_TRAIN_TASKS_PER_ITER:-1}"
TRAIN_EPISODES_PER_TASK="${ONLINE_TRAIN_EPISODES_PER_TASK:-4}"
VAL_EPISODES="${ONLINE_VAL_EPISODES:-8}"
VAL_MAX_ENV_STEPS="${VAL_MAX_ENV_STEPS:-180}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-auto}"
VIZ_ENABLED_VALUE="${VIZ_ENABLED:-false}"
VIZ_SAVE_BEST_VALUE="${VIZ_SAVE_BEST:-false}"
VIZ_SAVE_LAST_VALUE="${VIZ_SAVE_LAST:-false}"
PHASE1_DISABLE_PROJ_RAND="${PHASE1_DISABLE_PROJECTION_RANDOMIZATION:-true}"
LEARN_PROJECTOR_GAIN="${LEARN_PROJECTOR_GAIN:-true}"
LEARN_PROJECTOR_CHANNEL_GAIN="${LEARN_PROJECTOR_CHANNEL_GAIN:-true}"
PHOTOMETRIC_LR_RATIO="${PHOTOMETRIC_LR_RATIO:-0.1}"
LOW_MEM_PRESET="${LOW_MEM_PRESET:-false}"
ULTRA_LOW_MEM_PRESET="${ULTRA_LOW_MEM_PRESET:-false}"
EVAL_ENABLED_VALUE="${EVAL_ENABLED:-true}"
RECORD_ONLINE_VIDEOS_VALUE="${RECORD_ONLINE_VIDEOS:-true}"
RECORD_ONLINE_VIDEOS_LAST_ONLY_VALUE="${RECORD_ONLINE_VIDEOS_LAST_ONLY:-false}"
RECORD_ONLINE_TRAIN_VIDEO_VALUE="${RECORD_ONLINE_TRAIN_VIDEO:-false}"
RECORD_ONLINE_VAL_VIDEO_VALUE="${RECORD_ONLINE_VAL_VIDEO:-true}"
RESUME_RUN_DIR_VALUE="${RESUME_RUN_DIR:-}"
TEXTURE_PARAM_MODE_VALUE="${TEXTURE_PARAM_MODE:-direct}"
LATENT_HW_VALUE="${LATENT_HW:-12,12}"
LAMBDA_TV_VALUE="${LAMBDA_TV:-0.0}"
TRAIN_ANCHOR_HORIZON_ITERS_VALUE="${TRAIN_ANCHOR_HORIZON_ITERS:-1}"
DETERMINISTIC_ANCHOR_SAMPLING_VALUE="${DETERMINISTIC_ANCHOR_SAMPLING:-false}"
PHASE1_ACTION_GAP_MODE_VALUE="${PHASE1_ACTION_GAP_MODE:-gt_farthest}"

if [[ -n "${RESUME_RUN_DIR_VALUE}" && -n "${INIT_PROJECTION_TEXTURE_PATH:-}" ]]; then
    echo "ERROR: RESUME_RUN_DIR and INIT_PROJECTION_TEXTURE_PATH are mutually exclusive."
    exit 1
fi

resume_args=()
if [[ -n "${RESUME_RUN_DIR_VALUE}" ]]; then
    resume_args+=(--resume_run_dir "${RESUME_RUN_DIR_VALUE}")
fi

if [[ "${LOW_MEM_PRESET}" == "true" ]]; then
    PATCH_SIZE_VALUE="${PATCH_SIZE:-3,22,22}"
    PROJECTION_SIZE_VALUE="${PROJECTION_SIZE:-${PATCH_SIZE_VALUE}}"
    # Keep SigLIP input size aligned with model positional embeddings.
    # For the default `...patch14-384` model, forcing 224 triggers shape mismatch
    # like: tensor a (256) vs tensor b (729).
    if [[ -n "${SIGLIP_INPUT_SIZE:-}" ]]; then
        SIGLIP_INPUT_SIZE_VALUE="${SIGLIP_INPUT_SIZE}"
    elif [[ "${SIGLIP_MODEL_NAME_VALUE}" == *"384"* ]]; then
        SIGLIP_INPUT_SIZE_VALUE="384"
    else
        SIGLIP_INPUT_SIZE_VALUE="224"
    fi
    TRAIN_EPISODES_PER_TASK="${ONLINE_TRAIN_EPISODES_PER_TASK:-2}"
    VAL_EPISODES="${ONLINE_VAL_EPISODES:-2}"
    PHASE2_ROLLOUT_STEPS="${PHASE2_ROLLOUT:-8}"
    VAL_MAX_ENV_STEPS="${VAL_MAX_ENV_STEPS:-80}"
    ENV_RESOLUTION_VALUE="${ENV_RESOLUTION:-224}"
fi

if [[ "${ULTRA_LOW_MEM_PRESET}" == "true" ]]; then
    PATCH_SIZE_VALUE="${PATCH_SIZE:-3,18,18}"
    PROJECTION_SIZE_VALUE="${PROJECTION_SIZE:-${PATCH_SIZE_VALUE}}"
    ENV_RESOLUTION_VALUE="${ENV_RESOLUTION:-192}"
    TRAIN_EPISODES_PER_TASK="${ONLINE_TRAIN_EPISODES_PER_TASK:-1}"
    VAL_EPISODES="${ONLINE_VAL_EPISODES:-1}"
    PHASE2_ROLLOUT_STEPS="${PHASE2_ROLLOUT:-6}"
    VAL_MAX_ENV_STEPS="${VAL_MAX_ENV_STEPS:-60}"
    LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-0.0}"
    EVAL_ENABLED_VALUE="${EVAL_ENABLED:-false}"
    RECORD_ONLINE_VIDEOS_VALUE="${RECORD_ONLINE_VIDEOS:-false}"
    RECORD_ONLINE_VIDEOS_LAST_ONLY_VALUE="${RECORD_ONLINE_VIDEOS_LAST_ONLY:-true}"
    RECORD_ONLINE_TRAIN_VIDEO_VALUE="${RECORD_ONLINE_TRAIN_VIDEO:-false}"
    RECORD_ONLINE_VAL_VIDEO_VALUE="${RECORD_ONLINE_VAL_VIDEO:-false}"
fi

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
    --device "${DEVICE_ID}" \
    --iter 5000 \
    --accumulate "${ACCUMULATE_STEPS}" \
    --bs 1 \
    --warmup "${WARMUP_STEPS}" \
    --tags "UADA_rollout_online_env" "fair_compare" \
    --geometry true \
    --attack_mode "projection" \
    --patch_size "${PATCH_SIZE_VALUE}" \
    --projection_size "${PROJECTION_SIZE_VALUE}" \
    --init_projection_texture_path "${INIT_PROJECTION_TEXTURE_PATH:-}" \
    "${resume_args[@]}" \
    --projection_alpha 0.55 \
    --projection_alpha_jitter 0.00 \
    --projection_soft_edge 1.2 \
    --projection_angle "${PROJECTION_ANGLE:-0}" \
    --projection_fixed_angle "${PROJECTION_FIXED_ANGLE:-true}" \
    --projection_shear "${PROJECTION_SHEAR:-0.0}" \
    --projection_scale_min "${PROJECTION_SCALE_MIN:-0.3}" \
    --projection_scale_max "${PROJECTION_SCALE_MAX:-0.6}" \
    --projection_region "lower_half_fixed" \
    --projection_lower_start 0.55 \
    --projection_width_ratio 0.35 \
    --projection_height_ratio 0.35 \
    --projection_margin_x 0.04 \
    --projection_keystone 0.22 \
    --projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER:-0.0}" \
    --projector_gamma 1.8 \
    --projector_gain 1.35 \
    --projector_channel_gain "1.08,1.04,1.00" \
    --learn_projector_gain "${LEARN_PROJECTOR_GAIN}" \
    --learn_projector_channel_gain "${LEARN_PROJECTOR_CHANNEL_GAIN}" \
    --photometric_lr_ratio "${PHOTOMETRIC_LR_RATIO}" \
    --projector_ambient 0.08 \
    --projector_vignetting 0.08 \
    --projector_distance_falloff 0.10 \
    --projector_psf false \
    --projection_randomization_enabled "${PROJECTION_RANDOMIZATION_ENABLED:-true}" \
    --wandb_project "${WANDB_PROJECT_NAME}" \
    --wandb_entity "${WANDB_ENTITY_NAME}" \
    --dataset "${DATASET_NAME}" \
    --resize_patch false \
    --phase1_ratio 0.4 \
    --phase1_rollout 8 \
    --phase2_rollout "${PHASE2_ROLLOUT_STEPS}" \
    --lambda_action_gap "${LAMBDA_ACTION_GAP}" \
    --lambda_history "${LAMBDA_HISTORY}" \
    --lambda_history_legacy "${LAMBDA_HISTORY_LEGACY}" \
    --lambda_ce "${LAMBDA_CE}" \
    --lambda_ce_phase2 "${LAMBDA_CE_PHASE2}" \
    --lambda_continuous_rollout "${LAMBDA_CONTINUOUS_ROLLOUT}" \
    --lambda_window_rollout_loss "${LAMBDA_WINDOW_ROLLOUT_LOSS}" \
    --impulse_rollout_metric_enabled "${IMPULSE_ROLLOUT_METRIC_ENABLED}" \
    --window_rollout_probe_enabled "${WINDOW_ROLLOUT_PROBE_ENABLED_VALUE}" \
    --window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}" \
    --window_rollout_future_mode "${WINDOW_ROLLOUT_FUTURE_MODE_VALUE}" \
    --window_rollout_exp_base "${WINDOW_ROLLOUT_EXP_BASE_VALUE}" \
    --window_rollout_future_horizon "${WINDOW_ROLLOUT_FUTURE_HORIZON_VALUE}" \
    --window_rollout_phase_scope "${WINDOW_ROLLOUT_PHASE_SCOPE_VALUE}" \
    --lambda_siglip "${LAMBDA_SIGLIP}" \
    --siglip_model_name "${SIGLIP_MODEL_NAME_VALUE}" \
    --siglip_input_size "${SIGLIP_INPUT_SIZE_VALUE}" \
    --save_interval "${SAVE_INTERVAL_STEPS}" \
    --eval_enabled "${EVAL_ENABLED_VALUE}" \
    --val_deterministic true \
    --val_seed 42 \
    --val_disable_lighting true \
    --lighting_aug_enabled false \
    --lighting_aug_train_only false \
    --phase1_disable_lighting true \
    --phase1_disable_projection_randomization "${PHASE1_DISABLE_PROJ_RAND}" \
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
    --record_online_videos "${RECORD_ONLINE_VIDEOS_VALUE}" \
    --record_online_videos_last_only "${RECORD_ONLINE_VIDEOS_LAST_ONLY_VALUE}" \
    --record_online_train_video "${RECORD_ONLINE_TRAIN_VIDEO_VALUE}" \
    --record_online_val_video "${RECORD_ONLINE_VAL_VIDEO_VALUE}" \
    --record_online_video_frame_source "projected_input" \
    --record_online_video_fps 10 \
    --viz_enabled "${VIZ_ENABLED_VALUE}" \
    --viz_policy "milestone" \
    --viz_samples 2 \
    --viz_save_best "${VIZ_SAVE_BEST_VALUE}" \
    --viz_save_last "${VIZ_SAVE_LAST_VALUE}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --online_train_tasks_per_iter "${TRAIN_TASKS_PER_ITER}" \
    --online_train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
    --online_val_episodes "${VAL_EPISODES}" \
    --num_steps_wait 10 \
    --max_env_steps "auto_by_suite" \
    --val_max_env_steps "${VAL_MAX_ENV_STEPS}" \
    --env_resolution "${ENV_RESOLUTION_VALUE}" \
    --online_ce_mode "${ONLINE_CE_MODE_NAME}" \
    --env_action_source "adv" \
    --env_seed 42 \
    --action_gap_mode "${ACTION_GAP_MODE:-gt_farthest}" \
    --phase1_action_gap_mode "${PHASE1_ACTION_GAP_MODE_VALUE}" \
    --gt_dataset_root "${GT_DATASET_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}" \
    --gt_action_bank_path "${GT_ACTION_BANK_PATH:-}" \
    --gt_softmin_tau "${GT_SOFTMIN_TAU:-0.05}" \
    --phase_state_mode "${PHASE_STATE_MODE:-phase_cycle}" \
    --phase_state_cache_path "${PHASE_STATE_CACHE_PATH:-}" \
    --texture_param_mode "${TEXTURE_PARAM_MODE_VALUE}" \
    --latent_hw "${LATENT_HW_VALUE}" \
    --lambda_tv "${LAMBDA_TV_VALUE}" \
    --train_anchor_horizon_iters "${TRAIN_ANCHOR_HORIZON_ITERS_VALUE}" \
    --deterministic_anchor_sampling "${DETERMINISTIC_ANCHOR_SAMPLING_VALUE}" \
    --auto_gpu_tune false
