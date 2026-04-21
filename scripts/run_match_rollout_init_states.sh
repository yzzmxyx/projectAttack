#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

ROLLOUT_ROOT_VALUE="${ROLLOUT_ROOT:-${PROJECT_ROOT}/run/UADA_rollout_online_env}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-${ROLLOUT_ROOT_VALUE}/init_state_matches}"

cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation_tool/match_rollout_init_states.py"
    --rollout_root "${ROLLOUT_ROOT_VALUE}"
    --output_root "${OUTPUT_ROOT_VALUE}"
    --run_ids "${RUN_IDS:-all}"
    --dataset "${DATASET:-auto}"
    --task_suite_name "${TASK_SUITE_NAME:-auto}"
    --manifest_split "${MANIFEST_SPLIT:-val}"
    --match_all_val_episodes "${MATCH_ALL_VAL_EPISODES:-true}"
    --val_deterministic "${VAL_DETERMINISTIC:-true}"
    --online_val_episodes "${ONLINE_VAL_EPISODES:-8}"
    --val_seed "${VAL_SEED:-42}"
    --rlds_root "${RLDS_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}"
    --steps_parquet "${STEPS_PARQUET:-}"
    --phases_parquet "${PHASES_PARQUET:-}"
    --device "${DEVICE:-auto}"
    --num_steps_wait "${NUM_STEPS_WAIT:-10}"
    --env_resolution "${ENV_RESOLUTION:-256}"
    --window_stride "${WINDOW_STRIDE:-8}"
    --recovery_vision_backbone "${RECOVERY_VISION_BACKBONE:-dinoclip-vit-l-336px}"
    --recovery_image_resize_strategy "${RECOVERY_IMAGE_RESIZE_STRATEGY:-resize-naive}"
)

force_rebuild_text="$(printf '%s' "${FORCE_REBUILD:-false}" | tr '[:upper:]' '[:lower:]')"
if [[ "${force_rebuild_text}" == "1" || "${force_rebuild_text}" == "true" || "${force_rebuild_text}" == "yes" ]]; then
    cmd+=(--force_rebuild)
fi

printf 'Running init-state match command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

mkdir -p "${OUTPUT_ROOT_VALUE}"
{
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    printf '%q' "${cmd[@]}"
    printf '\n'
} > "${OUTPUT_ROOT_VALUE}/run_command.sh"
chmod +x "${OUTPUT_ROOT_VALUE}/run_command.sh"

"${cmd[@]}"

echo "Init-state matching outputs written to: ${OUTPUT_ROOT_VALUE}"
