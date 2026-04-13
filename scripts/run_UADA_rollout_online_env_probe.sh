#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROBE_ID="$(python3.10 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
PROBE_ROOT="${current_dir}/run/UADA_rollout_online_env_probe/${PROBE_ID}"
LOG_DIR="${PROBE_ROOT}/logs"
SUMMARY_CSV="${PROBE_ROOT}/probe_summary.csv"
mkdir -p "${LOG_DIR}"

# Historical probe baseline used for fair comparisons against newer GT runs.
DATASET_NAME="${DATASET:-libero_spatial}"
ACTION_GAP_MODE_NAME="${ACTION_GAP_MODE:-clean_adv}"
PHASE_STATE_MODE_NAME="${PHASE_STATE_MODE:-initial_only}"

cat > "${SUMMARY_CSV}" <<'EOF'
variant,exp_id,run_dir,iter,lambda_action_gap,lambda_history,lambda_history_legacy,lambda_ce,online_ce_mode,action_gap_mode_active,final_val_done_rate,final_val_episode_len,final_val_action_gap,final_val_gt_action_gap,final_val_active_action_gap,final_val_history1,final_val_history2,final_val_ce,final_val_ce_objective,final_val_rollout_score,final_val_gt_rollout_score,final_val_active_rollout_score,final_val_objective_score,final_val_gt_objective_score,final_val_active_objective_score
EOF

echo "Probe root: ${PROBE_ROOT}"

run_variant() {
    local variant="$1"
    local lambda_action_gap="$2"
    local lambda_history="$3"
    local lambda_history_legacy="$4"
    local lambda_ce="$5"
    local online_ce_mode="$6"
    local safe_variant
    safe_variant="$(echo "${variant}" | tr '+' '_' )"
    local log_path="${LOG_DIR}/${safe_variant}.log"

    python3.10 "${current_dir}/VLAAttacker/UADA_rollout_online_env_wrapper.py" \
        --maskidx 0,1,2 \
        --use_all_joints false \
        --gripper_weight 0.5 \
        --lr 2e-3 \
        --server "${current_dir}" \
        --device "${DEVICE:-6}" \
        --iter 20 \
        --accumulate 1 \
        --bs 1 \
        --warmup 2 \
        --tags "UADA_rollout_online_env_probe" "${variant}" \
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
        --wandb_project "false" \
        --dataset "${DATASET_NAME}" \
        --resize_patch false \
        --phase1_ratio 0.4 \
        --phase1_rollout 8 \
        --phase2_rollout 24 \
        --lambda_action_gap "${lambda_action_gap}" \
        --lambda_history "${lambda_history}" \
        --lambda_history_legacy "${lambda_history_legacy}" \
        --lambda_ce "${lambda_ce}" \
        --save_interval 5 \
        --eval_enabled true \
        --val_deterministic true \
        --val_seed 42 \
        --val_disable_lighting true \
        --lighting_aug_enabled false \
        --lighting_aug_train_only false \
        --phase1_disable_lighting true \
        --phase1_disable_projection_randomization false \
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
        --viz_enabled false \
        --viz_policy "milestone" \
        --viz_samples 2 \
        --viz_save_best false \
        --viz_save_last false \
        --task_suite_name "auto" \
        --online_train_tasks_per_iter 1 \
        --online_train_episodes_per_task 1 \
        --online_val_episodes 8 \
        --num_steps_wait 10 \
        --max_env_steps "auto_by_suite" \
        --env_resolution 256 \
        --online_ce_mode "${online_ce_mode}" \
        --action_gap_mode "${ACTION_GAP_MODE_NAME}" \
        --gt_dataset_root "${GT_DATASET_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}" \
        --gt_action_bank_path "${GT_ACTION_BANK_PATH:-}" \
        --gt_softmin_tau "${GT_SOFTMIN_TAU:-0.05}" \
        --phase_state_mode "${PHASE_STATE_MODE_NAME}" \
        --phase_state_cache_path "${PHASE_STATE_CACHE_PATH:-}" \
        --env_action_source "adv" \
        --env_seed 42 \
        --probe_mode true \
        --probe_variant "${variant}" \
        --auto_gpu_tune false | tee "${log_path}"

    local exp_id
    exp_id="$(grep '^exp_id:' "${log_path}" | tail -n1 | cut -d':' -f2- | tr -d '[:space:]')"
    if [[ -z "${exp_id}" ]]; then
        echo "Failed to parse exp_id for variant ${variant}" >&2
        exit 1
    fi

    local run_dir="${current_dir}/run/UADA_rollout_online_env/${exp_id}"
    local final_json="${run_dir}/probe_final_val.json"
    if [[ ! -f "${final_json}" ]]; then
        echo "Missing probe_final_val.json for variant ${variant}: ${final_json}" >&2
        exit 1
    fi

    python3.10 - "${SUMMARY_CSV}" "${final_json}" "${exp_id}" <<'PY'
import csv
import json
import sys

summary_csv, final_json, exp_id = sys.argv[1:4]
with open(final_json, "r") as file:
    data = json.load(file)

fieldnames = [
    "variant",
    "exp_id",
    "run_dir",
    "iter",
    "lambda_action_gap",
    "lambda_history",
    "lambda_history_legacy",
    "lambda_ce",
    "online_ce_mode",
    "action_gap_mode_active",
    "final_val_done_rate",
    "final_val_episode_len",
    "final_val_action_gap",
    "final_val_gt_action_gap",
    "final_val_active_action_gap",
    "final_val_history1",
    "final_val_history2",
    "final_val_ce",
    "final_val_ce_objective",
    "final_val_rollout_score",
    "final_val_gt_rollout_score",
    "final_val_active_rollout_score",
    "final_val_objective_score",
    "final_val_gt_objective_score",
    "final_val_active_objective_score",
]
row = {
    "variant": data["variant"],
    "exp_id": exp_id,
    "run_dir": data["run_dir"],
    "iter": data["iter"],
    "lambda_action_gap": data["lambda_action_gap"],
    "lambda_history": data["lambda_history"],
    "lambda_history_legacy": data["lambda_history_legacy"],
    "lambda_ce": data["lambda_ce"],
    "online_ce_mode": data["online_ce_mode"],
    "action_gap_mode_active": data["action_gap_mode_active"],
    "final_val_done_rate": data["final_val_done_rate"],
    "final_val_episode_len": data["final_val_episode_len"],
    "final_val_action_gap": data["final_val_action_gap"],
    "final_val_gt_action_gap": data["final_val_gt_action_gap"],
    "final_val_active_action_gap": data["final_val_active_action_gap"],
    "final_val_history1": data["final_val_history1"],
    "final_val_history2": data["final_val_history2"],
    "final_val_ce": data["final_val_ce"],
    "final_val_ce_objective": data["final_val_ce_objective"],
    "final_val_rollout_score": data["final_val_rollout_score"],
    "final_val_gt_rollout_score": data["final_val_gt_rollout_score"],
    "final_val_active_rollout_score": data["final_val_active_rollout_score"],
    "final_val_objective_score": data["final_val_objective_score"],
    "final_val_gt_objective_score": data["final_val_gt_objective_score"],
    "final_val_active_objective_score": data["final_val_active_objective_score"],
}
with open(summary_csv, "a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

run_variant "action" "1.0" "0.0" "0.0" "0.0" "off"
run_variant "history1" "0.0" "0.5" "0.0" "0.0" "off"
run_variant "history2" "0.0" "0.0" "0.5" "0.0" "off"
run_variant "action+history1" "1.0" "0.5" "0.0" "0.0" "off"
run_variant "action+history2" "1.0" "0.0" "0.5" "0.0" "off"
run_variant "action+history1+history2" "1.0" "0.5" "0.5" "0.0" "off"
run_variant "action+ce" "1.0" "0.0" "0.0" "0.1" "pseudo_clean"
run_variant "action+history1+history2+ce" "1.0" "0.5" "0.5" "0.1" "pseudo_clean"

echo "Probe summary written to: ${SUMMARY_CSV}"
