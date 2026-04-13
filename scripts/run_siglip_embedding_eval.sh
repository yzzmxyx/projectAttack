#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/run/smoke_ic_light_datasetbg_10_variants}"
# IMAGE_PATHS="${IMAGE_PATHS:-}"
IMAGE_PATHS="/home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000000_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000299_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000799_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000800_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000949_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_001849_online/step_000_projected_input.png /home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_001999_online/step_000_projected_input.png"
# REFERENCE_IMAGE="${REFERENCE_IMAGE:-}"
REFERENCE_IMAGE="/home/yxx/projectAttack/run/UADA_rollout_online_env/6fae87a9-c78f-45f2-9c7b-79f2470533fc/visualization/iter_000000_online/step_000_orig.png"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
INCLUDE_EXT="${INCLUDE_EXT:-.png,.jpg,.jpeg,.webp,.bmp}"
EXCLUDE_SUBSTRINGS="${EXCLUDE_SUBSTRINGS:-contact_sheet,compare,triptych}"
MODEL_NAME="${MODEL_NAME:-google/siglip-so400m-patch14-384}"

cd "${PROJECT_ROOT}"

if [[ $# -gt 0 ]]; then
    echo "PROJECT_ROOT=${PROJECT_ROOT}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "Forwarding explicit CLI args to eval_siglip_embedding_distance.py"
    "${PYTHON_BIN}" evaluation_tool/eval_siglip_embedding_distance.py "$@"
    exit 0
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "INPUT_PATH=${INPUT_PATH}"
echo "IMAGE_PATHS=${IMAGE_PATHS}"
echo "REFERENCE_IMAGE=${REFERENCE_IMAGE}"
echo "DEVICE=${DEVICE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "INCLUDE_EXT=${INCLUDE_EXT}"
echo "EXCLUDE_SUBSTRINGS=${EXCLUDE_SUBSTRINGS}"
echo "MODEL_NAME=${MODEL_NAME}"

cmd=(
    "${PYTHON_BIN}" "evaluation_tool/eval_siglip_embedding_distance.py"
    "--device" "${DEVICE}"
    "--batch_size" "${BATCH_SIZE}"
    "--include_ext" "${INCLUDE_EXT}"
    "--exclude_substrings" "${EXCLUDE_SUBSTRINGS}"
    "--model_name" "${MODEL_NAME}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
    cmd+=("--output_dir" "${OUTPUT_DIR}")
fi

if [[ -n "${REFERENCE_IMAGE}" ]]; then
    cmd+=("--reference_image" "${REFERENCE_IMAGE}")
fi

if [[ -n "${IMAGE_PATHS}" ]]; then
    read -r -a image_paths_array <<< "${IMAGE_PATHS}"
    if [[ ${#image_paths_array[@]} -eq 0 ]]; then
        echo "ERROR: IMAGE_PATHS is set but no valid paths were parsed." >&2
        exit 1
    fi
    cmd+=("--image_paths")
    cmd+=("${image_paths_array[@]}")
else
    cmd+=("--input_path" "${INPUT_PATH}")
fi

printf 'Running command:'
for token in "${cmd[@]}"; do
    printf ' %q' "${token}"
done
printf '\n'

"${cmd[@]}"
