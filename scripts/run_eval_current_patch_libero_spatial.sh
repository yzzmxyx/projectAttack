#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [[ -x "/home/ubuntu/anaconda3/envs/roboticAttack/bin/python" ]]; then
        PYTHON_BIN="/home/ubuntu/anaconda3/envs/roboticAttack/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="python3.10"
    fi
fi

if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
    echo "Python interpreter not runnable: ${PYTHON_BIN}" >&2
    exit 1
fi

# Resolve a usable LIBERO source root and expose it to Python import path.
LIBERO_ROOT_VALUE="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT_VALUE}" ]]; then
    for candidate in "${PROJECT_ROOT}/LIBERO" "/home/ubuntu/libero_pipeline/LIBERO"; do
        if [[ -d "${candidate}" ]]; then
            LIBERO_ROOT_VALUE="${candidate}"
            break
        fi
    done
fi

if [[ -n "${LIBERO_ROOT_VALUE}" && -d "${LIBERO_ROOT_VALUE}" ]]; then
    export PYTHONPATH="${LIBERO_ROOT_VALUE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

if importlib.util.find_spec("libero") is None:
    print("ERROR: `libero` is not importable in current Python.", file=sys.stderr)
    print(f"  python={sys.executable}", file=sys.stderr)
    print(f"  LIBERO_ROOT={os.environ.get('LIBERO_ROOT', '')}", file=sys.stderr)
    print("  tip: set LIBERO_ROOT to your LIBERO repo root (directory containing `libero/`).", file=sys.stderr)
    sys.exit(1)

try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    print(f"ERROR: `libero` is visible but OffScreenRenderEnv import failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY

PYTORCH_CUDA_ALLOC_CONF_DEFAULT="max_split_size_mb:128,garbage_collection_threshold:0.8"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_OVERRIDE:-${PYTORCH_CUDA_ALLOC_CONF_DEFAULT}}"

INPUT_REF="${1:-}"
PATCH_PATH="${PATCH_PATH:-}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
SOURCE_EXP_ID="${SOURCE_EXP_ID:-}"
NO_PATCH_CLEAN="${NO_PATCH_CLEAN:-false}"

if [[ -n "${INPUT_REF}" && -z "${PATCH_PATH}" && -z "${SOURCE_RUN_DIR}" && -z "${SOURCE_EXP_ID}" ]]; then
    if [[ -f "${INPUT_REF}" ]]; then
        PATCH_PATH="${INPUT_REF}"
    elif [[ -d "${INPUT_REF}" ]]; then
        SOURCE_RUN_DIR="${INPUT_REF}"
    else
        SOURCE_EXP_ID="${INPUT_REF}"
    fi
fi

EVAL_ID="$("${PYTHON_BIN}" - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
EVAL_ROOT="${PROJECT_ROOT}/run/UADA_eval_current_patch_libero_spatial/${EVAL_ID}"
LOG_PATH="${EVAL_ROOT}/eval.log"
mkdir -p "${EVAL_ROOT}"

if [[ "${NO_PATCH_CLEAN}" != "true" ]]; then
mapfile -t PATCH_INFO < <(
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PATCH_PATH="${PATCH_PATH}" \
    SOURCE_RUN_DIR="${SOURCE_RUN_DIR}" \
    SOURCE_EXP_ID="${SOURCE_EXP_ID}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
patch_path_arg = os.environ.get("PATCH_PATH", "").strip()
source_run_dir_arg = os.environ.get("SOURCE_RUN_DIR", "").strip()
source_exp_id_arg = os.environ.get("SOURCE_EXP_ID", "").strip()
run_base = project_root / "run" / "UADA_rollout_online_env"

def locate_patch_from_run_dir(run_dir_str):
    run_dir = Path(os.path.abspath(os.path.expanduser(run_dir_str)))
    for candidate in (run_dir / "last" / "projection_texture.pt", run_dir / "last" / "patch.pt"):
        if candidate.is_file():
            return run_dir, candidate
    return run_dir, None

resolved_run_dir = None
resolved_patch_path = None

if patch_path_arg:
    resolved_patch_path = Path(os.path.abspath(os.path.expanduser(patch_path_arg)))
    if not resolved_patch_path.is_file():
        raise FileNotFoundError(f"PATCH_PATH not found: {resolved_patch_path}")
    if resolved_patch_path.parent.name == "last":
        resolved_run_dir = resolved_patch_path.parent.parent
elif source_run_dir_arg:
    resolved_run_dir, resolved_patch_path = locate_patch_from_run_dir(source_run_dir_arg)
    if resolved_patch_path is None:
        raise FileNotFoundError(f"No last/projection_texture.pt or last/patch.pt found under {resolved_run_dir}")
elif source_exp_id_arg:
    resolved_run_dir, resolved_patch_path = locate_patch_from_run_dir(str(run_base / source_exp_id_arg))
    if resolved_patch_path is None:
        raise FileNotFoundError(f"No last/projection_texture.pt or last/patch.pt found for SOURCE_EXP_ID={source_exp_id_arg}")
else:
    candidates = []
    if run_base.is_dir():
        for pattern in ("*/last/projection_texture.pt", "*/last/patch.pt"):
            for path in run_base.glob(pattern):
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                preference = 1 if path.name == "projection_texture.pt" else 0
                candidates.append((mtime, preference, path))
    if not candidates:
        raise FileNotFoundError(f"No candidate patch found under {run_base}. Set PATCH_PATH, SOURCE_RUN_DIR, or SOURCE_EXP_ID.")
    _, _, resolved_patch_path = max(candidates, key=lambda item: (item[0], item[1]))
    resolved_run_dir = resolved_patch_path.parent.parent

loaded = torch.as_tensor(torch.load(resolved_patch_path, map_location="cpu"))
shape_csv = ",".join(str(int(x)) for x in loaded.shape)
print("" if resolved_run_dir is None else str(resolved_run_dir))
print(str(resolved_patch_path))
print(shape_csv)
PY
)

if [[ "${#PATCH_INFO[@]}" -lt 3 ]]; then
    echo "Failed to resolve patch source." >&2
    exit 1
fi

RESOLVED_SOURCE_RUN_DIR="${PATCH_INFO[0]}"
RESOLVED_PATCH_PATH="${PATCH_INFO[1]}"
INFERRED_PATCH_SHAPE="${PATCH_INFO[2]}"

PROJECTION_SIZE_RAW="${PROJECTION_SIZE:-auto}"
if [[ "${PROJECTION_SIZE_RAW}" == "auto" ]]; then
    PROJECTION_SIZE_VALUE="${INFERRED_PATCH_SHAPE}"
else
    PROJECTION_SIZE_VALUE="${PROJECTION_SIZE_RAW}"
fi
ATTACK_MODE_VALUE="${ATTACK_MODE:-projection}"
else
RESOLVED_SOURCE_RUN_DIR=""
RESOLVED_PATCH_PATH=""
PROJECTION_SIZE_VALUE="${PROJECTION_SIZE:-3,70,70}"
ATTACK_MODE_VALUE="${ATTACK_MODE:-clean}"
fi

echo "Eval root: ${EVAL_ROOT}"
echo "  python_bin=$("${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
PY
)"
echo "  source_run_dir=${RESOLVED_SOURCE_RUN_DIR}"
echo "  patch_path=${RESOLVED_PATCH_PATH}"
echo "  attack_mode=${ATTACK_MODE_VALUE}"
echo "  projection_size=${PROJECTION_SIZE_VALUE}"
echo "  eval_runs=${EVAL_RUNS:-25}"
echo "  max_env_steps=${MAX_ENV_STEPS:-180}"
echo "  rollouts=${EVAL_ROOT}/rollouts"

"${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation_tool/eval_current_patch_online_env_simple.py" \
    --patch_path "${RESOLVED_PATCH_PATH}" \
    --attack_mode "${ATTACK_MODE_VALUE}" \
    --output_dir "${EVAL_ROOT}" \
    --dataset "${DATASET:-libero_spatial}" \
    --task_suite_name "${TASK_SUITE_NAME:-libero_spatial}" \
    --device "${DEVICE:-0}" \
    --eval_runs "${EVAL_RUNS:-25}" \
    --max_env_steps "${MAX_ENV_STEPS:-180}" \
    --num_steps_wait "${NUM_STEPS_WAIT:-10}" \
    --env_resolution "${ENV_RESOLUTION:-256}" \
    --val_deterministic "${VAL_DETERMINISTIC:-true}" \
    --val_seed "${VAL_SEED:-42}" \
    --seed "${SEED:-42}" \
    --projection_size "${PROJECTION_SIZE_VALUE}" \
    --projection_alpha "${PROJECTION_ALPHA:-0.55}" \
    --projection_alpha_jitter "${PROJECTION_ALPHA_JITTER:-0.0}" \
    --projection_soft_edge "${PROJECTION_SOFT_EDGE:-1.2}" \
    --projection_angle "${PROJECTION_ANGLE:-25}" \
    --projection_fixed_angle "${PROJECTION_FIXED_ANGLE:-false}" \
    --projection_shear "${PROJECTION_SHEAR:-0.15}" \
    --projection_scale_min "${PROJECTION_SCALE_MIN:-1.0}" \
    --projection_scale_max "${PROJECTION_SCALE_MAX:-1.0}" \
    --projection_region "${PROJECTION_REGION:-lower_half_fixed}" \
    --projection_lower_start "${PROJECTION_LOWER_START:-0.55}" \
    --projection_width_ratio "${PROJECTION_WIDTH_RATIO:-1.20}" \
    --projection_height_ratio "${PROJECTION_HEIGHT_RATIO:-1.00}" \
    --projection_margin_x "${PROJECTION_MARGIN_X:-0.04}" \
    --projection_keystone "${PROJECTION_KEYSTONE:-0.22}" \
    --projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER:-0.03}" \
    --projection_randomization_enabled "${PROJECTION_RANDOMIZATION_ENABLED:-false}" \
    --projector_gamma "${PROJECTOR_GAMMA:-1.8}" \
    --projector_gain "${PROJECTOR_GAIN:-1.35}" \
    --projector_channel_gain "${PROJECTOR_CHANNEL_GAIN:-1.08,1.04,1.00}" \
    --projector_ambient "${PROJECTOR_AMBIENT:-0.08}" \
    --projector_vignetting "${PROJECTOR_VIGNETTING:-0.08}" \
    --projector_distance_falloff "${PROJECTOR_DISTANCE_FALLOFF:-0.10}" \
    --projector_psf "${PROJECTOR_PSF:-false}" \
    --video_frame_source "${VIDEO_FRAME_SOURCE:-projected_input}" \
    --video_fps "${VIDEO_FPS:-10}" | tee "${LOG_PATH}"

echo "Eval completed: ${EVAL_ROOT}"
echo "Video manifest: ${EVAL_ROOT}/video_manifest.csv"
