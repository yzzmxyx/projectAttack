#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
MODE_NAME="$(printf '%s' "${MODE:-libero}" | tr '[:upper:]' '[:lower:]')"
CONFIG_PATH_VALUE="${RISK_WINDOW_CONFIG:-${PROJECT_ROOT}/risk_window/default_config.json}"
ASSET_ROOT_VALUE="${ASSET_ROOT:-}"
LOG_DIR_VALUE="${RISK_WINDOW_LOG_DIR:-}"
ACTION_POLICY_VALUE="${RISK_WINDOW_ACTION:-log_only}"
OVERLAY_VALUE="${RISK_WINDOW_OVERLAY:-false}"

if [[ -z "${ASSET_ROOT_VALUE}" ]]; then
    ASSET_ROOT_VALUE="$("${PYTHON_BIN}" - "${PROJECT_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).expanduser().resolve()
search_root = project_root / "run" / "vulnerability_window_search"
candidates = []
if search_root.exists():
    for child in search_root.iterdir():
        if not child.is_dir():
            continue
        top_windows = child / "top_windows.json"
        if top_windows.exists():
            candidates.append((top_windows.stat().st_mtime, child))
if not candidates:
    raise SystemExit("")
candidates.sort(reverse=True)
print(str(candidates[0][1]))
PY
)"
fi

if [[ -z "${ASSET_ROOT_VALUE}" ]]; then
    echo "ASSET_ROOT is required, or there must be at least one existing run/vulnerability_window_search/* asset root." >&2
    exit 1
fi

if [[ ! -d "${ASSET_ROOT_VALUE}" ]]; then
    echo "Asset root does not exist: ${ASSET_ROOT_VALUE}" >&2
    exit 1
fi

printf 'Validating risk_window assets at %s\n' "${ASSET_ROOT_VALUE}"
"${PYTHON_BIN}" -m risk_window.cli validate-assets --asset-root "${ASSET_ROOT_VALUE}"

mapfile -t inferred_fields < <("${PYTHON_BIN}" - "${ASSET_ROOT_VALUE}" <<'PY'
import json
import os
import sys
from pathlib import Path

asset_root = Path(sys.argv[1]).expanduser().resolve()
run_config_path = asset_root / "run_config.json"
payload = {}
if run_config_path.exists():
    with open(run_config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
suite_name = str(payload.get("task_suite_name", args.get("task_suite_name", "libero_10")))
patch_path = str(payload.get("patch_path", args.get("patch_path", "")))
task_description = str(payload.get("task_description", ""))
if patch_path and not os.path.isabs(patch_path):
    patch_path = str((asset_root / patch_path).resolve())
print(suite_name)
print(patch_path)
print(task_description)
PY
)

SUITE_NAME_VALUE="${SUITE_NAME:-${inferred_fields[0]:-libero_10}}"
PATCH_PATH_VALUE="${PATCH_PATH:-${inferred_fields[1]:-}}"
TASK_DESCRIPTION_VALUE="${TASK_DESCRIPTION:-${inferred_fields[2]:-}}"

case "${SUITE_NAME_VALUE}" in
    libero_spatial)
        PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-spatial}"
        PATCH_X_VALUE="${PATCH_X:-120}"
        PATCH_Y_VALUE="${PATCH_Y:-160}"
        PATCH_ANGLE_VALUE="${PATCH_ANGLE:-0}"
        PATCH_SHX_VALUE="${PATCH_SHX:-0}"
        PATCH_SHY_VALUE="${PATCH_SHY:-0}"
        ;;
    libero_object)
        PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-object}"
        PATCH_X_VALUE="${PATCH_X:-30}"
        PATCH_Y_VALUE="${PATCH_Y:-150}"
        PATCH_ANGLE_VALUE="${PATCH_ANGLE:-0}"
        PATCH_SHX_VALUE="${PATCH_SHX:-0}"
        PATCH_SHY_VALUE="${PATCH_SHY:-0}"
        ;;
    libero_goal)
        PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-goal}"
        PATCH_X_VALUE="${PATCH_X:-15}"
        PATCH_Y_VALUE="${PATCH_Y:-158}"
        PATCH_ANGLE_VALUE="${PATCH_ANGLE:-0}"
        PATCH_SHX_VALUE="${PATCH_SHX:-0}"
        PATCH_SHY_VALUE="${PATCH_SHY:-0}"
        ;;
    libero_10)
        PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b-finetuned-libero-10}"
        PATCH_X_VALUE="${PATCH_X:-5}"
        PATCH_Y_VALUE="${PATCH_Y:-160}"
        PATCH_ANGLE_VALUE="${PATCH_ANGLE:-0}"
        PATCH_SHX_VALUE="${PATCH_SHX:-0}"
        PATCH_SHY_VALUE="${PATCH_SHY:-0}"
        ;;
    *)
        PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b}"
        PATCH_X_VALUE="${PATCH_X:-5}"
        PATCH_Y_VALUE="${PATCH_Y:-160}"
        PATCH_ANGLE_VALUE="${PATCH_ANGLE:-0}"
        PATCH_SHX_VALUE="${PATCH_SHX:-0}"
        PATCH_SHY_VALUE="${PATCH_SHY:-0}"
        ;;
esac

if [[ -z "${LOG_DIR_VALUE}" ]]; then
    LOG_DIR_VALUE="${ASSET_ROOT_VALUE}/risk_window_trial_logs"
fi

mkdir -p "${LOG_DIR_VALUE}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

printf 'Mode: %s\n' "${MODE_NAME}"
printf 'Suite: %s\n' "${SUITE_NAME_VALUE}"
printf 'Task description: %s\n' "${TASK_DESCRIPTION_VALUE}"
printf 'Risk-window config: %s\n' "${CONFIG_PATH_VALUE}"
printf 'Risk-window log dir: %s\n' "${LOG_DIR_VALUE}"

if [[ "${MODE_NAME}" == "libero" ]]; then
    if [[ -z "${PATCH_PATH_VALUE}" ]]; then
        echo "PATCH_PATH is required for MODE=libero. It was not found in ${ASSET_ROOT_VALUE}/run_config.json." >&2
        exit 1
    fi
    if [[ ! -f "${PATCH_PATH_VALUE}" ]]; then
        echo "Patch file not found: ${PATCH_PATH_VALUE}" >&2
        exit 1
    fi

    CUDA_ID_VALUE="${CUDAID:-${DEVICE:-0}}"
    NUM_TRIALS_VALUE="${TRIALS:-1}"
    EXP_NAME_VALUE="${EXP_NAME:-risk_window_trial_${SUITE_NAME_VALUE}}"
    RUN_ID_NOTE_VALUE="${RUN_ID_NOTE:-risk_window_trial}"

    cmd=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robot/libero/run_libero_eval_args_geo_batch.py"
        --exp_name "${EXP_NAME_VALUE}"
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT_VALUE}"
        --task_suite_name "${SUITE_NAME_VALUE}"
        --num_trials_per_task "${NUM_TRIALS_VALUE}"
        --run_id_note "${RUN_ID_NOTE_VALUE}"
        --local_log_dir "${LOG_DIR_VALUE}"
        --patchroot "${PATCH_PATH_VALUE}"
        --cudaid "${CUDA_ID_VALUE}"
        --x "${PATCH_X_VALUE}"
        --y "${PATCH_Y_VALUE}"
        --angle "${PATCH_ANGLE_VALUE}"
        --shx "${PATCH_SHX_VALUE}"
        --shy "${PATCH_SHY_VALUE}"
        --risk_window_enable true
        --risk_window_config "${CONFIG_PATH_VALUE}"
        --risk_window_asset_root "${ASSET_ROOT_VALUE}"
        --risk_window_log_dir "${LOG_DIR_VALUE}"
        --risk_window_action "${ACTION_POLICY_VALUE}"
        --risk_window_overlay "${OVERLAY_VALUE}"
    )
    printf 'Executing LIBERO trial command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    exit 0
fi

if [[ "${MODE_NAME}" == "bridge" ]]; then
    PRETRAINED_CHECKPOINT_VALUE="${PRETRAINED_CHECKPOINT:-openvla/openvla-7b}"
    MAX_EPISODES_VALUE="${MAX_EPISODES:-1}"
    MAX_STEPS_VALUE="${MAX_STEPS:-60}"
    CONTROL_FREQUENCY_VALUE="${CONTROL_FREQUENCY:-5}"
    HOST_IP_VALUE="${HOST_IP:-localhost}"
    PORT_VALUE="${PORT:-5556}"

    cmd=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/robot/bridge/run_bridgev2_eval.py"
        --model_family openvla
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT_VALUE}"
        --host_ip "${HOST_IP_VALUE}"
        --port "${PORT_VALUE}"
        --max_episodes "${MAX_EPISODES_VALUE}"
        --max_steps "${MAX_STEPS_VALUE}"
        --control_frequency "${CONTROL_FREQUENCY_VALUE}"
        --risk_window_enable true
        --risk_window_config "${CONFIG_PATH_VALUE}"
        --risk_window_asset_root "${ASSET_ROOT_VALUE}"
        --risk_window_log_dir "${LOG_DIR_VALUE}"
        --risk_window_action "${ACTION_POLICY_VALUE}"
        --risk_window_overlay "${OVERLAY_VALUE}"
    )
    printf 'Executing Bridge trial command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    printf 'Bridge mode uses interactive task input from run_bridgev2_eval.py.\n'
    "${cmd[@]}"
    exit 0
fi

echo "Unsupported MODE=${MODE_NAME}. Expected one of: libero, bridge." >&2
exit 1
