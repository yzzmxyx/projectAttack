#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
    echo "Python interpreter not runnable: ${PYTHON_BIN}" >&2
    exit 1
fi

PROBE_ID="$("${PYTHON_BIN}" - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
PROBE_ROOT="${current_dir}/run/UADA_rollout_online_env_probe_window_rollout_adv_gt_siglip_gt_phase_forms/${PROBE_ID}"
LOG_DIR="${PROBE_ROOT}/logs"
SUMMARY_CSV="${PROBE_ROOT}/probe_summary.csv"
SUMMARY_JSON="${PROBE_ROOT}/probe_summary.json"
BEST_JSON="${PROBE_ROOT}/best_by_total_score.json"
BEST_BY_LOSS_FAMILY_JSON="${PROBE_ROOT}/best_by_loss_family.json"
mkdir -p "${LOG_DIR}"

DATASET_NAME="${DATASET:-libero_spatial}"
WINDOW_ROLLOUT_FUTURE_HORIZON="${WINDOW_ROLLOUT_FUTURE_HORIZON:-8}"
WINDOW_ROLLOUT_EXP_BASE="${WINDOW_ROLLOUT_EXP_BASE:-0.9}"
WINDOW_ROLLOUT_FUTURE_MODE="${WINDOW_ROLLOUT_FUTURE_MODE:-drop_attack_after_window}"
WINDOW_ROLLOUT_PROBE_ENABLED_VALUE="true"
WINDOW_ROLLOUT_METRIC_MODE_VALUE="adv_gt"
TASK_SUITE_NAME_VALUE="${TASK_SUITE_NAME:-auto}"
ONLINE_TRAIN_TASKS_PER_ITER_VALUE="${ONLINE_TRAIN_TASKS_PER_ITER:-1}"
ONLINE_TRAIN_EPISODES_PER_TASK_VALUE="${ONLINE_TRAIN_EPISODES_PER_TASK:-10}"
ONLINE_VAL_EPISODES_VALUE="${ONLINE_VAL_EPISODES:-8}"
MAX_ENV_STEPS_VALUE="${MAX_ENV_STEPS:-auto_by_suite}"
VAL_MAX_ENV_STEPS_VALUE="${VAL_MAX_ENV_STEPS:-180}"
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-5}"
VAL_DETERMINISTIC_VALUE="${VAL_DETERMINISTIC:-true}"
VAL_SEED_VALUE="${VAL_SEED:-42}"
ENV_SEED_VALUE="${ENV_SEED:-42}"

PROJECTION_FIXED_ANGLE_VALUE="${PROJECTION_FIXED_ANGLE:-true}"
PROJECTION_ANGLE_VALUE="${PROJECTION_ANGLE:-0}"
PROJECTION_ALPHA_JITTER_VALUE="${PROJECTION_ALPHA_JITTER:-0.0}"
PROJECTION_KEYSTONE_JITTER_VALUE="${PROJECTION_KEYSTONE_JITTER:-0.0}"
PROJECTION_SCALE_MIN_VALUE="${PROJECTION_SCALE_MIN:-0.5}"
PROJECTION_SCALE_MAX_VALUE="${PROJECTION_SCALE_MAX:-0.5}"
PROJECTION_SHEAR_VALUE="${PROJECTION_SHEAR:-0.0}"
PHASE1_DISABLE_PROJECTION_RANDOMIZATION_VALUE="${PHASE1_DISABLE_PROJECTION_RANDOMIZATION:-true}"

LEARN_PROJECTOR_GAIN_VALUE="false"
LEARN_PROJECTOR_CHANNEL_GAIN_VALUE="false"

echo "Probe root: ${PROBE_ROOT}"
echo "  python_bin=$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
PY
)"
echo "Alignment snapshot:"
echo "  dataset=${DATASET_NAME}"
echo "  task_suite_name=${TASK_SUITE_NAME_VALUE}"
echo "  online_train_tasks_per_iter=${ONLINE_TRAIN_TASKS_PER_ITER_VALUE}"
echo "  online_train_episodes_per_task=${ONLINE_TRAIN_EPISODES_PER_TASK_VALUE}"
echo "  online_val_episodes=${ONLINE_VAL_EPISODES_VALUE}"
echo "  max_env_steps=${MAX_ENV_STEPS_VALUE}"
echo "  val_max_env_steps=${VAL_MAX_ENV_STEPS_VALUE}"
echo "  save_interval=${SAVE_INTERVAL_VALUE}"
echo "  val_deterministic=${VAL_DETERMINISTIC_VALUE}"
echo "  val_seed=${VAL_SEED_VALUE}"
echo "  env_seed=${ENV_SEED_VALUE}"
echo "  phase1_disable_projection_randomization=${PHASE1_DISABLE_PROJECTION_RANDOMIZATION_VALUE}"
echo "  projection_fixed_angle=${PROJECTION_FIXED_ANGLE_VALUE}"
echo "  projection_angle=${PROJECTION_ANGLE_VALUE}"
echo "  projection_alpha_jitter=${PROJECTION_ALPHA_JITTER_VALUE}"
echo "  projection_shear=${PROJECTION_SHEAR_VALUE}"
echo "  projection_scale_min=${PROJECTION_SCALE_MIN_VALUE}"
echo "  projection_scale_max=${PROJECTION_SCALE_MAX_VALUE}"
echo "  projection_keystone_jitter=${PROJECTION_KEYSTONE_JITTER_VALUE}"
echo "  learn_projector_gain=${LEARN_PROJECTOR_GAIN_VALUE}"
echo "  learn_projector_channel_gain=${LEARN_PROJECTOR_CHANNEL_GAIN_VALUE}"
echo "  window_rollout_probe_enabled=${WINDOW_ROLLOUT_PROBE_ENABLED_VALUE}"
echo "  window_rollout_metric_mode=${WINDOW_ROLLOUT_METRIC_MODE_VALUE}"
echo "  window_rollout_future_mode=${WINDOW_ROLLOUT_FUTURE_MODE}"
echo "  window_rollout_future_horizon=${WINDOW_ROLLOUT_FUTURE_HORIZON}"
echo "  window_rollout_exp_base=${WINDOW_ROLLOUT_EXP_BASE}"

"${PYTHON_BIN}" - "${SUMMARY_CSV}" <<'PY'
import csv
import sys

summary_csv = sys.argv[1]
fieldnames = [
    "order_idx",
    "loss_family",
    "form_name",
    "variant",
    "phase_state_mode",
    "window_rollout_phase_scope",
    "exp_id",
    "run_dir",
    "window_rollout_probe_enabled",
    "window_rollout_metric_mode",
    "window_rollout_future_mode",
    "window_phase_name",
    "window_start_step",
    "window_end_step",
    "window_rollout_future_horizon",
    "window_rollout_exp_base",
    "lambda_action_gap",
    "lambda_siglip",
    "lambda_ce",
    "lambda_window_rollout_loss",
    "online_ce_mode",
    "action_gap_mode_active",
    "final_val_gt_action_gap",
    "final_val_siglip_distance",
    "final_val_ce_objective",
    "final_val_window_rollout_metric_value",
    "final_val_window_rollout_delta_weighted",
    "final_val_window_rollout_delta_weighted_loss",
    "final_val_window_rollout_clean_gt_action_gap",
    "final_val_window_rollout_adv_gt_action_gap",
    "final_val_window_rollout_deattack_gt_action_gap",
    "final_val_window_rollout_selected_gt_action_gap",
    "final_val_total_probe_score",
    "final_val_total_rollout_score",
    "final_val_done_rate",
    "final_val_episode_len",
]
with open(summary_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
PY

run_variant() {
    local order_idx="$1"
    local loss_family="$2"
    local form_name="$3"
    local phase_state_mode="$4"
    local window_rollout_phase_scope="$5"
    local variant="$6"
    local lambda_action_gap="$7"
    local lambda_siglip="$8"
    local lambda_ce="$9"
    local online_ce_mode="${10}"
    local action_gap_mode="${11}"
    local lambda_window_rollout_loss="${12}"

    local safe_variant
    safe_variant="$(echo "${variant}" | tr '+/' '__')"
    local log_path="${LOG_DIR}/${order_idx}_${safe_variant}.log"

    "${PYTHON_BIN}" "${current_dir}/VLAAttacker/UADA_rollout_online_env_wrapper.py" \
        --maskidx 0,1,2 \
        --use_all_joints false \
        --gripper_weight 0.5 \
        --lr "${LR:-2e-3}" \
        --server "${current_dir}" \
        --device "${DEVICE:-1}" \
        --iter "${ITER:-20}" \
        --accumulate 1 \
        --bs 1 \
        --warmup "${WARMUP:-2}" \
        --tags "UADA_rollout_online_env_probe_window_rollout_adv_gt_siglip_gt_phase_forms" "${loss_family}" "${form_name}" "${variant}" \
        --geometry true \
        --attack_mode "projection" \
        --patch_size "${PATCH_SIZE:-3,50,50}" \
        --projection_size "${PROJECTION_SIZE:-3,50,50}" \
        --projection_alpha "${PROJECTION_ALPHA:-0.55}" \
        --projection_alpha_jitter "${PROJECTION_ALPHA_JITTER_VALUE}" \
        --projection_soft_edge "${PROJECTION_SOFT_EDGE:-1.2}" \
        --projection_angle "${PROJECTION_ANGLE_VALUE}" \
        --projection_fixed_angle "${PROJECTION_FIXED_ANGLE_VALUE}" \
        --projection_shear "${PROJECTION_SHEAR_VALUE}" \
        --projection_scale_min "${PROJECTION_SCALE_MIN_VALUE}" \
        --projection_scale_max "${PROJECTION_SCALE_MAX_VALUE}" \
        --projection_region "${PROJECTION_REGION:-lower_half_fixed}" \
        --projection_lower_start "${PROJECTION_LOWER_START:-0.55}" \
        --projection_width_ratio "${PROJECTION_WIDTH_RATIO:-0.90}" \
        --projection_height_ratio "${PROJECTION_HEIGHT_RATIO:-0.95}" \
        --projection_margin_x "${PROJECTION_MARGIN_X:-0.04}" \
        --projection_keystone "${PROJECTION_KEYSTONE:-0.22}" \
        --projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER_VALUE}" \
        --projector_gamma "${PROJECTOR_GAMMA:-1.8}" \
        --projector_gain "${PROJECTOR_GAIN:-1.35}" \
        --projector_channel_gain "${PROJECTOR_CHANNEL_GAIN:-1.08,1.04,1.00}" \
        --learn_projector_gain "${LEARN_PROJECTOR_GAIN_VALUE}" \
        --learn_projector_channel_gain "${LEARN_PROJECTOR_CHANNEL_GAIN_VALUE}" \
        --projector_ambient "${PROJECTOR_AMBIENT:-0.08}" \
        --projector_vignetting "${PROJECTOR_VIGNETTING:-0.08}" \
        --projector_distance_falloff "${PROJECTOR_DISTANCE_FALLOFF:-0.10}" \
        --projector_psf false \
        --wandb_project "false" \
        --dataset "${DATASET_NAME}" \
        --resize_patch false \
        --phase1_ratio "${PHASE1_RATIO:-0.4}" \
        --phase1_rollout "${PHASE1_ROLLOUT:-8}" \
        --phase2_rollout "${PHASE2_ROLLOUT:-24}" \
        --lambda_action_gap "${lambda_action_gap}" \
        --lambda_history 0.0 \
        --lambda_history_legacy 0.0 \
        --lambda_ce "${lambda_ce}" \
        --lambda_ce_phase2 0.0 \
        --lambda_continuous_rollout 0.0 \
        --lambda_window_rollout_loss "${lambda_window_rollout_loss}" \
        --impulse_rollout_metric_enabled false \
        --window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}" \
        --lambda_siglip "${lambda_siglip}" \
        --save_interval "${SAVE_INTERVAL_VALUE}" \
        --eval_enabled true \
        --val_deterministic "${VAL_DETERMINISTIC_VALUE}" \
        --val_seed "${VAL_SEED_VALUE}" \
        --val_disable_lighting true \
        --lighting_aug_enabled false \
        --lighting_aug_train_only false \
        --phase1_disable_lighting true \
        --phase1_disable_projection_randomization "${PHASE1_DISABLE_PROJECTION_RANDOMIZATION_VALUE}" \
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
        --record_online_videos_last_only true \
        --record_online_train_video false \
        --record_online_val_video true \
        --record_online_video_frame_source "projected_input" \
        --record_online_video_fps 10 \
        --viz_enabled false \
        --viz_policy "milestone" \
        --viz_samples 2 \
        --viz_save_best false \
        --viz_save_last false \
        --task_suite_name "${TASK_SUITE_NAME_VALUE}" \
        --online_train_tasks_per_iter "${ONLINE_TRAIN_TASKS_PER_ITER_VALUE}" \
        --online_train_episodes_per_task "${ONLINE_TRAIN_EPISODES_PER_TASK_VALUE}" \
        --online_val_episodes "${ONLINE_VAL_EPISODES_VALUE}" \
        --num_steps_wait "${NUM_STEPS_WAIT:-10}" \
        --max_env_steps "${MAX_ENV_STEPS_VALUE}" \
        --val_max_env_steps "${VAL_MAX_ENV_STEPS_VALUE}" \
        --env_resolution "${ENV_RESOLUTION:-256}" \
        --online_ce_mode "${online_ce_mode}" \
        --action_gap_mode "${action_gap_mode}" \
        --gt_dataset_root "${GT_DATASET_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}" \
        --gt_action_bank_path "${GT_ACTION_BANK_PATH:-}" \
        --gt_softmin_tau "${GT_SOFTMIN_TAU:-0.05}" \
        --phase_state_mode "${phase_state_mode}" \
        --phase_state_cache_path "${PHASE_STATE_CACHE_PATH:-}" \
        --env_action_source "adv" \
        --env_seed "${ENV_SEED_VALUE}" \
        --probe_mode true \
        --probe_variant "${variant}" \
        --window_rollout_probe_enabled "${WINDOW_ROLLOUT_PROBE_ENABLED_VALUE}" \
        --window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}" \
        --window_rollout_future_mode "${WINDOW_ROLLOUT_FUTURE_MODE}" \
        --window_rollout_exp_base "${WINDOW_ROLLOUT_EXP_BASE}" \
        --window_rollout_future_horizon "${WINDOW_ROLLOUT_FUTURE_HORIZON}" \
        --window_rollout_phase_scope "${window_rollout_phase_scope}" \
        --auto_gpu_tune false | tee "${log_path}"

    local exp_id
    exp_id="$(grep '^exp_id:' "${log_path}" | tail -n1 | cut -d':' -f2- | tr -d '[:space:]')"
    if [[ -z "${exp_id}" ]]; then
        echo "Failed to parse exp_id for variant=${variant}" >&2
        exit 1
    fi

    local run_dir="${current_dir}/run/UADA_rollout_online_env/${exp_id}"
    local final_json="${run_dir}/probe_final_val.json"
    if [[ ! -f "${final_json}" ]]; then
        echo "Missing probe_final_val.json for variant=${variant}: ${final_json}" >&2
        exit 1
    fi

    "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${final_json}" "${exp_id}" "${order_idx}" "${loss_family}" "${form_name}" "${phase_state_mode}" "${window_rollout_phase_scope}" <<'PY'
import csv
import json
import sys

(
    summary_csv,
    final_json,
    exp_id,
    order_idx,
    loss_family,
    form_name,
    phase_state_mode,
    window_rollout_phase_scope,
) = sys.argv[1:9]

with open(final_json, "r", encoding="utf-8") as file:
    data = json.load(file)

active_term = float(data.get("final_val_active_action_gap", data.get("final_val_action_gap", 0.0)))
siglip_term = float(data.get("final_val_siglip_distance", 0.0))
ce_term = float(data.get("final_val_ce_objective", 0.0))
window_probe_enabled = int(data.get("window_rollout_probe_enabled", 0))
window_metric_mode = str(data.get("window_rollout_metric_mode", "adv_gt"))
window_future_mode = str(data.get("window_rollout_future_mode", "keep_adv"))
window_metric_value = float(data.get("final_val_window_rollout_metric_value", 0.0)) if window_probe_enabled else 0.0
window_term = float(data.get("final_val_window_rollout_delta_weighted", 0.0)) if window_probe_enabled else 0.0
window_term_loss = (
    float(data.get("final_val_window_rollout_delta_weighted_loss", 0.0)) if window_probe_enabled else 0.0
)
lambda_action_gap = float(data.get("lambda_action_gap", 0.0))
lambda_siglip = float(data.get("lambda_siglip", 0.0))
lambda_ce = float(data.get("lambda_ce", 0.0))
lambda_window_rollout_loss = float(data.get("lambda_window_rollout_loss", 0.0))

total_probe_score = (
    (lambda_action_gap * active_term)
    + (lambda_siglip * siglip_term)
    - (lambda_ce * ce_term)
    + (lambda_window_rollout_loss * window_metric_value)
)

fieldnames = [
    "order_idx",
    "loss_family",
    "form_name",
    "variant",
    "phase_state_mode",
    "window_rollout_phase_scope",
    "exp_id",
    "run_dir",
    "window_rollout_probe_enabled",
    "window_rollout_metric_mode",
    "window_rollout_future_mode",
    "window_phase_name",
    "window_start_step",
    "window_end_step",
    "window_rollout_future_horizon",
    "window_rollout_exp_base",
    "lambda_action_gap",
    "lambda_siglip",
    "lambda_ce",
    "lambda_window_rollout_loss",
    "online_ce_mode",
    "action_gap_mode_active",
    "final_val_gt_action_gap",
    "final_val_siglip_distance",
    "final_val_ce_objective",
    "final_val_window_rollout_metric_value",
    "final_val_window_rollout_delta_weighted",
    "final_val_window_rollout_delta_weighted_loss",
    "final_val_window_rollout_clean_gt_action_gap",
    "final_val_window_rollout_adv_gt_action_gap",
    "final_val_window_rollout_deattack_gt_action_gap",
    "final_val_window_rollout_selected_gt_action_gap",
    "final_val_total_probe_score",
    "final_val_total_rollout_score",
    "final_val_done_rate",
    "final_val_episode_len",
]
row = {
    "order_idx": int(order_idx),
    "loss_family": loss_family,
    "form_name": form_name,
    "variant": data.get("variant", ""),
    "phase_state_mode": phase_state_mode,
    "window_rollout_phase_scope": window_rollout_phase_scope,
    "exp_id": exp_id,
    "run_dir": data.get("run_dir", ""),
    "window_rollout_probe_enabled": int(window_probe_enabled),
    "window_rollout_metric_mode": window_metric_mode,
    "window_rollout_future_mode": window_future_mode,
    "window_phase_name": data.get("window_phase_name", ""),
    "window_start_step": data.get("window_start_step", 0.0),
    "window_end_step": data.get("window_end_step", 0.0),
    "window_rollout_future_horizon": data.get("window_rollout_future_horizon", 0),
    "window_rollout_exp_base": data.get("window_rollout_exp_base", 0.9),
    "lambda_action_gap": lambda_action_gap,
    "lambda_siglip": lambda_siglip,
    "lambda_ce": lambda_ce,
    "lambda_window_rollout_loss": lambda_window_rollout_loss,
    "online_ce_mode": data.get("online_ce_mode", ""),
    "action_gap_mode_active": data.get("action_gap_mode_active", ""),
    "final_val_gt_action_gap": data.get("final_val_gt_action_gap", 0.0),
    "final_val_siglip_distance": siglip_term,
    "final_val_ce_objective": ce_term,
    "final_val_window_rollout_metric_value": window_metric_value,
    "final_val_window_rollout_delta_weighted": window_term,
    "final_val_window_rollout_delta_weighted_loss": window_term_loss,
    "final_val_window_rollout_clean_gt_action_gap": data.get("final_val_window_rollout_clean_gt_action_gap", 0.0),
    "final_val_window_rollout_adv_gt_action_gap": data.get("final_val_window_rollout_adv_gt_action_gap", 0.0),
    "final_val_window_rollout_deattack_gt_action_gap": data.get("final_val_window_rollout_deattack_gt_action_gap", 0.0),
    "final_val_window_rollout_selected_gt_action_gap": data.get("final_val_window_rollout_selected_gt_action_gap", 0.0),
    "final_val_total_probe_score": total_probe_score,
    "final_val_total_rollout_score": data.get("final_val_total_rollout_score", 0.0),
    "final_val_done_rate": data.get("final_val_done_rate", 0.0),
    "final_val_episode_len": data.get("final_val_episode_len", 0.0),
}
with open(summary_csv, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

run_variant 1 "gt+siglip+rollout" "phase-cycle" "phase_cycle" "initial" "gt+siglip+rollout__phase-cycle" 1 1 0 "off" "gt_farthest" 1
run_variant 2 "gt+siglip+rollout" "only-initial" "initial_only" "initial" "gt+siglip+rollout__only-initial" 1 1 0 "off" "gt_farthest" 1
run_variant 3 "gt+siglip+rollout" "only-contact-manipulate" "contact_manipulate_only" "initial" "gt+siglip+rollout__only-contact-manipulate" 1 1 0 "off" "gt_farthest" 1
run_variant 4 "gt+siglip+rollout" "only-post-contact" "post_contact_only" "initial" "gt+siglip+rollout__only-post-contact" 1 1 0 "off" "gt_farthest" 1

"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${BEST_JSON}" "${BEST_BY_LOSS_FAMILY_JSON}" <<'PY'
import csv
import json
import sys

summary_csv, summary_json, best_json, best_by_loss_family_json = sys.argv[1:5]


def coerce(value):
    if value is None:
        return value
    text = str(value).strip()
    if text == "":
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


with open(summary_csv, "r", encoding="utf-8") as file:
    rows = [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(file)]

with open(summary_json, "w", encoding="utf-8") as file:
    json.dump(rows, file, indent=2, ensure_ascii=False)

best_row = max(rows, key=lambda row: float(row.get("final_val_total_probe_score", float("-inf")))) if rows else {}
with open(best_json, "w", encoding="utf-8") as file:
    json.dump(best_row, file, indent=2, ensure_ascii=False)

best_by_loss_family = {}
for row in rows:
    loss_family = str(row.get("loss_family", "")).strip()
    if not loss_family:
        continue
    current = best_by_loss_family.get(loss_family)
    if current is None or float(row.get("final_val_total_probe_score", float("-inf"))) > float(
        current.get("final_val_total_probe_score", float("-inf"))
    ):
        best_by_loss_family[loss_family] = row

with open(best_by_loss_family_json, "w", encoding="utf-8") as file:
    json.dump(best_by_loss_family, file, indent=2, ensure_ascii=False)
PY

echo "Window rollout adv-gt SigLIP/GT phase-form probe finished: ${PROBE_ROOT}"
