#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/run/auto_launch_logs"
mkdir -p "${LOG_DIR}"

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-20}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-12000}"
RUN_CMD="${RUN_CMD:-bash scripts/run_UADA_rollout_online_env.sh}"
PYTORCH_ALLOC_CONF_OVERRIDE="${PYTORCH_ALLOC_CONF_OVERRIDE:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

MONITOR_LOG="${LOG_DIR}/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
RUN_LOG="${LOG_DIR}/run_UADA_rollout_online_env_$(date +%Y%m%d_%H%M%S).log"
STATE_FILE="${LOG_DIR}/latest_auto_launch_state.txt"

log() {
  local msg="$1"
  echo "[$(date '+%F %T')] ${msg}" | tee -a "${MONITOR_LOG}"
}

pick_best_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '
      {
        idx=$1; free=$2;
        gsub(/ /, "", idx);
        gsub(/ /, "", free);
        if (NR==1 || free>best_free) {best_idx=idx; best_free=free}
      }
      END {
        if (NR==0) exit 1;
        print best_idx " " best_free
      }'
}

log "Auto-launch monitor started."
log "Rule: launch when best GPU free memory >= ${GPU_FREE_MIN_MB} MiB."
log "Command: ${RUN_CMD}"

while true; do
  best="$(pick_best_gpu || true)"
  if [[ -z "${best}" ]]; then
    log "No GPU info available from nvidia-smi. Retry in ${CHECK_INTERVAL_SEC}s."
    sleep "${CHECK_INTERVAL_SEC}"
    continue
  fi

  best_idx="$(awk '{print $1}' <<< "${best}")"
  best_free="$(awk '{print $2}' <<< "${best}")"
  log "Current best GPU=${best_idx}, free=${best_free} MiB."

  if [[ "${best_free}" =~ ^[0-9]+$ ]] && (( best_free >= GPU_FREE_MIN_MB )); then
    log "Threshold reached. Launching on GPU ${best_idx}."
    {
      echo "monitor_log=${MONITOR_LOG}"
      echo "run_log=${RUN_LOG}"
      echo "gpu_index=${best_idx}"
      echo "gpu_free_mb=${best_free}"
      echo "launch_time=$(date '+%F %T')"
    } > "${STATE_FILE}"

    (
      cd "${PROJECT_ROOT}"
      export DEVICE="${best_idx}"
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC_CONF_OVERRIDE}"
      eval "${RUN_CMD}"
    ) >> "${RUN_LOG}" 2>&1 &

    run_pid="$!"
    log "Launched PID=${run_pid}. Run log: ${RUN_LOG}"
    echo "run_pid=${run_pid}" >> "${STATE_FILE}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
