#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

RUN_ID="$(${PYTHON_BIN} - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"

DATASET_VALUE="${DATASET:-libero_10}"
case "${DATASET_VALUE}" in
    *_no_noops)
        SIDECAR_DATASET="${DATASET_VALUE}"
        ;;
    libero_spatial|libero_object|libero_goal|libero_10)
        SIDECAR_DATASET="${DATASET_VALUE}_no_noops"
        ;;
    *)
        SIDECAR_DATASET="${DATASET_VALUE}"
        ;;
esac

DEFAULT_SIDECAR_ROOT="${PROJECT_ROOT}/data/libero_sidecars/${SIDECAR_DATASET}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/run/recovery_visual_smoke/${RUN_ID}}"
STEPS_PARQUET_VALUE="${STEPS_PARQUET:-${DEFAULT_SIDECAR_ROOT}/steps.parquet}"
PHASES_PARQUET_VALUE="${PHASES_PARQUET:-${DEFAULT_SIDECAR_ROOT}/phases.parquet}"
RLDS_ROOT_VALUE="${RLDS_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}"

mkdir -p "${OUTPUT_ROOT}"

cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation_tool/eval_single_state_recovery_visual_smoke.py"
    --dataset "${DATASET_VALUE}"
    --task_suite_name "${TASK_SUITE_NAME:-auto}"
    --task_id "${TASK_ID:-4}"
    --init_state_idx "${INIT_STATE_IDX:-0}"
    --rlds_root "${RLDS_ROOT_VALUE}"
    --steps_parquet "${STEPS_PARQUET_VALUE}"
    --phases_parquet "${PHASES_PARQUET_VALUE}"
    --output_root "${OUTPUT_ROOT}"
    --device "${DEVICE:-auto}"
    --num_steps_wait "${NUM_STEPS_WAIT:-10}"
    --env_resolution "${ENV_RESOLUTION:-256}"
    --window_stride "${WINDOW_STRIDE:-8}"
    --recovery_vision_backbone "${RECOVERY_VISION_BACKBONE:-pixel}"
    --recovery_image_resize_strategy "${RECOVERY_IMAGE_RESIZE_STRATEGY:-resize-naive}"
    --anchor_steps "${ANCHOR_STEPS:-auto}"
    --max_anchors "${MAX_ANCHORS:-3}"
)

force_rebuild_text="$(printf '%s' "${FORCE_REBUILD:-false}" | tr '[:upper:]' '[:lower:]')"
if [[ "${force_rebuild_text}" == "1" || "${force_rebuild_text}" == "true" || "${force_rebuild_text}" == "yes" ]]; then
    cmd+=(--force_rebuild)
fi

printf 'Running smoke command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

{
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    printf '%q' "${cmd[@]}"
    printf '\n'
} > "${OUTPUT_ROOT}/run_command.sh"
chmod +x "${OUTPUT_ROOT}/run_command.sh"

"${cmd[@]}"

echo "Visual smoke outputs written to: ${OUTPUT_ROOT}"
