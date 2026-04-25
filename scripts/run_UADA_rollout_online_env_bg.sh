#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_SCRIPT="${TARGET_SCRIPT:-${SCRIPT_DIR}/run_UADA_rollout_online_env.sh}"
RUN_NAME="${RUN_NAME:-run_UADA_rollout_online_env}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/run/background_logs}"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"
META_FILE="${LOG_DIR}/${RUN_NAME}.latest"

mkdir -p "${LOG_DIR}"

usage() {
  cat <<EOF
Usage: bash scripts/run_UADA_rollout_online_env_bg.sh [start|status|stop|logs]

Commands:
  start   Launch ${TARGET_SCRIPT} in the background (default)
  status  Show whether the background job is still running
  stop    Stop the background job using the saved PID
  logs    Print the latest log path and tail it

Examples:
  bash scripts/run_UADA_rollout_online_env_bg.sh
  DEVICE=0 DATASET=libero_spatial bash scripts/run_UADA_rollout_online_env_bg.sh start
  bash scripts/run_UADA_rollout_online_env_bg.sh status
EOF
}

is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

read_meta_value() {
  local key="$1"
  if [[ -f "${META_FILE}" ]]; then
    awk -F'=' -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${META_FILE}"
  fi
}

current_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

start_job() {
  local existing_pid
  existing_pid="$(current_pid || true)"
  if [[ -n "${existing_pid}" ]] && is_running "${existing_pid}"; then
    echo "Background job is already running."
    echo "PID: ${existing_pid}"
    echo "Log: $(read_meta_value log_file)"
    exit 0
  fi

  if [[ -n "${existing_pid}" ]]; then
    rm -f "${PID_FILE}"
  fi

  if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Target script not found: ${TARGET_SCRIPT}" >&2
    exit 1
  fi

  local timestamp log_file child_pid
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="${LOG_DIR}/${RUN_NAME}_${timestamp}.log"

  {
    echo "[$(date '+%F %T')] Launching ${TARGET_SCRIPT}"
    echo "[$(date '+%F %T')] Working directory: ${PROJECT_ROOT}"
    echo "[$(date '+%F %T')] Log file: ${log_file}"
  } >> "${log_file}"

  (
    cd "${PROJECT_ROOT}"
    if command -v setsid >/dev/null 2>&1; then
      nohup setsid bash "${TARGET_SCRIPT}" >> "${log_file}" 2>&1 < /dev/null &
    else
      nohup bash "${TARGET_SCRIPT}" >> "${log_file}" 2>&1 < /dev/null &
    fi
    child_pid="$!"
    echo "${child_pid}" > "${PID_FILE}"
    {
      echo "pid=${child_pid}"
      echo "log_file=${log_file}"
      echo "target_script=${TARGET_SCRIPT}"
      echo "started_at=$(date '+%F %T')"
      echo "working_directory=${PROJECT_ROOT}"
    } > "${META_FILE}"
  )

  child_pid="$(current_pid)"
  sleep 1

  if ! is_running "${child_pid}"; then
    echo "Background job exited immediately. Check the log:"
    echo "${log_file}"
    exit 1
  fi

  echo "Background job started."
  echo "PID: ${child_pid}"
  echo "Log: ${log_file}"
}

status_job() {
  local pid
  pid="$(current_pid || true)"

  if [[ -z "${pid}" ]]; then
    echo "No PID file found. No managed background job is running."
    exit 1
  fi

  if is_running "${pid}"; then
    echo "Background job is running."
    echo "PID: ${pid}"
    echo "Log: $(read_meta_value log_file)"
    exit 0
  fi

  echo "PID file exists, but the process is no longer running."
  echo "Last log: $(read_meta_value log_file)"
  exit 1
}

stop_job() {
  local pid
  pid="$(current_pid || true)"

  if [[ -z "${pid}" ]]; then
    echo "No PID file found. Nothing to stop."
    exit 1
  fi

  if ! is_running "${pid}"; then
    echo "Process ${pid} is already stopped."
    rm -f "${PID_FILE}"
    exit 0
  fi

  kill "${pid}"
  echo "Sent SIGTERM to PID ${pid}."
  rm -f "${PID_FILE}"
}

logs_job() {
  local log_file
  log_file="$(read_meta_value log_file)"

  if [[ -z "${log_file}" ]]; then
    echo "No log metadata found."
    exit 1
  fi

  echo "Tailing log: ${log_file}"
  tail -f "${log_file}"
}

ACTION="${1:-start}"

case "${ACTION}" in
  start)
    start_job
    ;;
  status)
    status_job
    ;;
  stop)
    stop_job
    ;;
  logs)
    logs_job
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${ACTION}" >&2
    usage >&2
    exit 1
    ;;
esac
