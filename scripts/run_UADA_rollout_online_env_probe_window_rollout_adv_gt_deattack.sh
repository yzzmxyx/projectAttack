#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROBE_ROOT_NAME="${PROBE_ROOT_NAME:-UADA_rollout_online_env_probe_window_rollout_adv_gt_deattack}"
export PROBE_RUN_TAG="${PROBE_RUN_TAG:-UADA_rollout_online_env_probe_window_rollout_adv_gt_deattack}"
export PROBE_FINISH_LABEL="${PROBE_FINISH_LABEL:-Window rollout adv-gt deattack probe finished}"
export WINDOW_ROLLOUT_FUTURE_MODE="${WINDOW_ROLLOUT_FUTURE_MODE:-drop_attack_after_window}"

exec "${SCRIPT_DIR}/run_UADA_rollout_online_env_probe_window_rollout_adv_gt.sh"
