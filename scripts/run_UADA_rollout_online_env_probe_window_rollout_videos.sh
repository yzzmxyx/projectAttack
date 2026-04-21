#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
current_dir="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PROBE_ROOT="${1:-${PROBE_ROOT:-}}"
if [[ -z "${INPUT_PROBE_ROOT}" ]]; then
    echo "Usage: $0 <probe_root>" >&2
    echo "Or set PROBE_ROOT=/path/to/run/UADA_rollout_online_env_probe_window_rollout/<probe_id>" >&2
    exit 1
fi

if [[ ! -d "${INPUT_PROBE_ROOT}" ]]; then
    echo "Probe root does not exist: ${INPUT_PROBE_ROOT}" >&2
    exit 1
fi

VIDEO_ID="$(python3.10 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
VIDEO_ROOT="${INPUT_PROBE_ROOT}/supplement_videos/${VIDEO_ID}"
LOG_DIR="${VIDEO_ROOT}/logs"
SELECTION_TSV="${VIDEO_ROOT}/selected_variants.tsv"
SUMMARY_CSV="${VIDEO_ROOT}/video_replay_summary.csv"
mkdir -p "${LOG_DIR}"

DATASET_NAME="${DATASET:-libero_10}"
PHASE_STATE_MODE_NAME="${PHASE_STATE_MODE:-phase_cycle}"
ONLINE_CE_MODE_DEFAULT="${ONLINE_CE_MODE_DEFAULT:-pseudo_clean}"
VIDEO_DEVICE="${DEVICE:-4}"
VIDEO_ITER="${VIDEO_ITER:-1}"
VIDEO_VAL_EPISODES="${VIDEO_VAL_EPISODES:-8}"
VIDEO_ALIGN_MODE="${VIDEO_ALIGN_MODE:-match_source}"

python3.10 - "${INPUT_PROBE_ROOT}" "${SELECTION_TSV}" <<'PY'
import csv
import os
import sys

probe_root, out_tsv = sys.argv[1:3]
variants = (
    "rollout-only",
    "gt+rollout",
    "siglip+rollout",
    "gt+siglip+rollout",
    "rollout+gt+siglip+ce",
)
phase_names = ("initial", "contact_manipulate", "post_contact")

best_rows = {}
for phase_name in phase_names:
    summary_csv = os.path.join(probe_root, phase_name, "probe_summary.csv")
    if not os.path.exists(summary_csv):
        continue
    with open(summary_csv, "r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            variant = str(row.get("variant", "")).strip()
            if variant not in variants:
                continue
            try:
                score = float(row.get("final_val_total_probe_score", "-inf"))
            except ValueError:
                score = float("-inf")
            current = best_rows.get(variant)
            if current is None or score > current["score"]:
                row = dict(row)
                row["phase_name"] = phase_name
                best_rows[variant] = {"score": score, "row": row}

with open(out_tsv, "w", encoding="utf-8") as file:
    for variant in variants:
        if variant not in best_rows:
            continue
        row = best_rows[variant]["row"]
        line = "\t".join(
            [
                variant,
                row.get("phase_name", ""),
                row.get("exp_id", ""),
                row.get("run_dir", ""),
                row.get("lambda_action_gap", "0"),
                row.get("lambda_siglip", "0"),
                row.get("lambda_ce", "0"),
                row.get("online_ce_mode", "pseudo_clean"),
                row.get("action_gap_mode_active", "clean_adv"),
                row.get("window_rollout_exp_base", "0.9"),
                row.get("window_rollout_future_horizon", "8"),
            ]
        )
        file.write(line + "\n")
PY

cat > "${SUMMARY_CSV}" <<'EOF'
variant,source_phase,source_exp_id,source_run_dir,replay_exp_id,replay_run_dir,video_manifest_path,video_path,status
EOF

echo "Video supplement root: ${VIDEO_ROOT}"

while IFS=$'\t' read -r variant phase_name source_exp_id source_run_dir lambda_action_gap lambda_siglip lambda_ce online_ce_mode action_gap_mode_active window_rollout_exp_base window_rollout_future_horizon; do
    if [[ -z "${variant}" ]]; then
        continue
    fi

    if [[ ! -d "${source_run_dir}" ]]; then
        echo "${variant},${phase_name},${source_exp_id},${source_run_dir},,,,\"\",missing_source_run_dir" >> "${SUMMARY_CSV}"
        continue
    fi

    source_texture_path="${source_run_dir}/last/projection_texture.pt"
    if [[ ! -f "${source_texture_path}" ]]; then
        source_texture_path="${source_run_dir}/last/patch.pt"
    fi
    if [[ ! -f "${source_texture_path}" ]]; then
        echo "${variant},${phase_name},${source_exp_id},${source_run_dir},,,,\"\",missing_source_texture" >> "${SUMMARY_CSV}"
        continue
    fi

    safe_variant="$(echo "${variant}" | tr '+/' '__')"
    log_path="${LOG_DIR}/${safe_variant}.log"
    replay_probe_variant="${variant}_video_replay"
    window_rollout_enabled="true"
    window_rollout_scope="${phase_name}"
    window_rollout_exp_base_value="${window_rollout_exp_base}"
    window_rollout_future_horizon_value="${window_rollout_future_horizon}"
    if [[ "${VIDEO_ALIGN_MODE}" == "full_probe" ]]; then
        window_rollout_enabled="false"
        window_rollout_scope="all"
        window_rollout_exp_base_value="0.9"
        window_rollout_future_horizon_value="8"
    fi

    python3.10 "${current_dir}/VLAAttacker/UADA_rollout_online_env_wrapper.py" \
        --maskidx 0,1,2 \
        --use_all_joints false \
        --gripper_weight 0.5 \
        --lr 0.0 \
        --server "${current_dir}" \
        --device "${VIDEO_DEVICE}" \
        --iter "${VIDEO_ITER}" \
        --accumulate 1 \
        --bs 1 \
        --warmup 0 \
        --tags "UADA_rollout_online_env_probe_window_rollout_video_replay" "${phase_name}" "${variant}" \
        --geometry true \
        --attack_mode "projection" \
        --patch_size "${PATCH_SIZE:-3,50,50}" \
        --projection_size "${PROJECTION_SIZE:-3,50,50}" \
        --init_projection_texture_path "${source_texture_path}" \
        --projection_alpha "${PROJECTION_ALPHA:-0.55}" \
        --projection_alpha_jitter "${PROJECTION_ALPHA_JITTER:-0.00}" \
        --projection_soft_edge "${PROJECTION_SOFT_EDGE:-1.2}" \
        --projection_angle "${PROJECTION_ANGLE:-25}" \
        --projection_fixed_angle "${PROJECTION_FIXED_ANGLE:-false}" \
        --projection_shear "${PROJECTION_SHEAR:-0.15}" \
        --projection_scale_min "${PROJECTION_SCALE_MIN:-0.8}" \
        --projection_scale_max "${PROJECTION_SCALE_MAX:-1.2}" \
        --projection_region "${PROJECTION_REGION:-lower_half_fixed}" \
        --projection_lower_start "${PROJECTION_LOWER_START:-0.55}" \
        --projection_width_ratio "${PROJECTION_WIDTH_RATIO:-0.90}" \
        --projection_height_ratio "${PROJECTION_HEIGHT_RATIO:-0.95}" \
        --projection_margin_x "${PROJECTION_MARGIN_X:-0.04}" \
        --projection_keystone "${PROJECTION_KEYSTONE:-0.22}" \
        --projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER:-0.03}" \
        --projector_gamma "${PROJECTOR_GAMMA:-1.8}" \
        --projector_gain "${PROJECTOR_GAIN:-1.35}" \
        --projector_channel_gain "${PROJECTOR_CHANNEL_GAIN:-1.08,1.04,1.00}" \
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
        --impulse_rollout_metric_enabled false \
        --lambda_siglip "${lambda_siglip}" \
        --save_interval 1 \
        --eval_enabled true \
        --val_deterministic true \
        --val_seed "${VAL_SEED:-42}" \
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
        --record_online_videos_last_only true \
        --record_online_train_video false \
        --record_online_val_video true \
        --record_online_video_frame_source "${VIDEO_FRAME_SOURCE:-projected_input}" \
        --record_online_video_fps "${VIDEO_FPS:-10}" \
        --viz_enabled false \
        --viz_policy "milestone" \
        --viz_samples 1 \
        --viz_save_best false \
        --viz_save_last false \
        --task_suite_name "${TASK_SUITE_NAME:-auto}" \
        --online_train_tasks_per_iter 1 \
        --online_train_episodes_per_task 10 \
        --online_val_episodes "${VIDEO_VAL_EPISODES}" \
        --num_steps_wait "${NUM_STEPS_WAIT:-10}" \
        --max_env_steps "${MAX_ENV_STEPS:-auto_by_suite}" \
        --val_max_env_steps "${VAL_MAX_ENV_STEPS:-120}" \
        --env_resolution "${ENV_RESOLUTION:-256}" \
        --online_ce_mode "${online_ce_mode:-${ONLINE_CE_MODE_DEFAULT}}" \
        --action_gap_mode "${action_gap_mode_active}" \
        --gt_dataset_root "${GT_DATASET_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}" \
        --gt_action_bank_path "${GT_ACTION_BANK_PATH:-}" \
        --gt_softmin_tau "${GT_SOFTMIN_TAU:-0.05}" \
        --phase_state_mode "${PHASE_STATE_MODE_NAME}" \
        --phase_state_cache_path "${PHASE_STATE_CACHE_PATH:-}" \
        --env_action_source "adv" \
        --env_seed "${ENV_SEED:-42}" \
        --probe_mode true \
        --probe_variant "${replay_probe_variant}" \
        --window_rollout_probe_enabled "${window_rollout_enabled}" \
        --window_rollout_exp_base "${window_rollout_exp_base_value}" \
        --window_rollout_future_horizon "${window_rollout_future_horizon_value}" \
        --window_rollout_phase_scope "${window_rollout_scope}" \
        --auto_gpu_tune false | tee "${log_path}"

    replay_exp_id="$(grep '^exp_id:' "${log_path}" | tail -n1 | cut -d':' -f2- | tr -d '[:space:]')"
    if [[ -z "${replay_exp_id}" ]]; then
        echo "${variant},${phase_name},${source_exp_id},${source_run_dir},,,,\"\",replay_exp_id_parse_failed" >> "${SUMMARY_CSV}"
        continue
    fi

    replay_run_dir="${current_dir}/run/UADA_rollout_online_env/${replay_exp_id}"
    manifest_path="${replay_run_dir}/videos/video_manifest.csv"
    video_path=""
    if [[ -f "${manifest_path}" ]]; then
        video_path="$(python3.10 - "${manifest_path}" <<'PY'
import csv
import sys

manifest_path = sys.argv[1]
rows = []
with open(manifest_path, "r", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        rows.append(row)
if not rows:
    print("")
    raise SystemExit(0)
val_rows = [row for row in rows if str(row.get("split", "")).lower() == "val"]
target = val_rows[-1] if val_rows else rows[-1]
print(str(target.get("video_path", "")))
PY
)"
    fi

    if [[ -n "${video_path}" ]]; then
        status="ok"
    else
        status="missing_video_path"
    fi
    echo "${variant},${phase_name},${source_exp_id},${source_run_dir},${replay_exp_id},${replay_run_dir},${manifest_path},${video_path},${status}" >> "${SUMMARY_CSV}"
done < "${SELECTION_TSV}"

echo "Video supplement completed: ${VIDEO_ROOT}"
echo "Summary: ${SUMMARY_CSV}"
