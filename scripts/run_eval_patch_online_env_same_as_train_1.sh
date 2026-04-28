#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
ISOLATED_PYDEPS_DIR="${PROJECT_ROOT}/.isolated_pydeps/run_uada_online_env_1"

PYTHON_BIN_VALUE="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN_VALUE}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN_VALUE="${CONDA_PREFIX}/bin/python"
    elif [[ -x "/home/ubuntu/anaconda3/envs/roboticAttack/bin/python" ]]; then
        PYTHON_BIN_VALUE="/home/ubuntu/anaconda3/envs/roboticAttack/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN_VALUE="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN_VALUE="python3.10"
    fi
fi

LIBERO_ROOT_VALUE="${LIBERO_ROOT:-/home/ubuntu/libero_pipeline/LIBERO}"
mkdir -p "${ISOLATED_PYDEPS_DIR}"

if ! PYTHONNOUSERSITE=1 PYTHONPATH="${ISOLATED_PYDEPS_DIR}" "${PYTHON_BIN_VALUE}" - <<'PY'
import importlib.util
import sys

try:
    import tokenizers
    import transformers
except Exception:
    sys.exit(1)

if transformers.__version__ != "4.40.1" or tokenizers.__version__ != "0.19.1":
    sys.exit(1)

if importlib.util.find_spec("transformers.models.mega.configuration_mega") is None:
    sys.exit(1)
PY
then
    "${PYTHON_BIN_VALUE}" -m pip install \
        --target "${ISOLATED_PYDEPS_DIR}" \
        --upgrade \
        --no-deps \
        "transformers==4.40.1" \
        "tokenizers==0.19.1" \
        "regex" \
        "safetensors" \
        "mpmath" \
        "rich"
fi

EXTRA_PYTHONPATH="${ISOLATED_PYDEPS_DIR}"
if [[ -d "${LIBERO_ROOT_VALUE}" ]]; then
    EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH}:${LIBERO_ROOT_VALUE}"
fi
export PYTHONPATH="${EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

INPUT_REF="${1:-}"
PATCH_PATH_VALUE="${PATCH_PATH:-}"
PATCH_DIR_VALUE="${PATCH_DIR:-}"
SOURCE_RUN_DIR_VALUE="${SOURCE_RUN_DIR:-}"
SOURCE_EXP_ID_VALUE="${SOURCE_EXP_ID:-}"

if [[ -n "${INPUT_REF}" && -z "${PATCH_PATH_VALUE}" && -z "${PATCH_DIR_VALUE}" && -z "${SOURCE_RUN_DIR_VALUE}" && -z "${SOURCE_EXP_ID_VALUE}" ]]; then
    if [[ -f "${INPUT_REF}" ]]; then
        PATCH_PATH_VALUE="${INPUT_REF}"
    elif [[ -d "${INPUT_REF}" ]]; then
        PATCH_DIR_VALUE="${INPUT_REF}"
    else
        SOURCE_EXP_ID_VALUE="${INPUT_REF}"
    fi
fi

mapfile -t PATCH_INFO < <(
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PATCH_PATH="${PATCH_PATH_VALUE}" \
    PATCH_DIR="${PATCH_DIR_VALUE}" \
    SOURCE_RUN_DIR="${SOURCE_RUN_DIR_VALUE}" \
    SOURCE_EXP_ID="${SOURCE_EXP_ID_VALUE}" \
    "${PYTHON_BIN_VALUE}" - <<'PY'
import os
from pathlib import Path

import torch

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
patch_path_arg = os.environ.get("PATCH_PATH", "").strip()
patch_dir_arg = os.environ.get("PATCH_DIR", "").strip()
source_run_dir_arg = os.environ.get("SOURCE_RUN_DIR", "").strip()
source_exp_id_arg = os.environ.get("SOURCE_EXP_ID", "").strip()
run_base = project_root / "run" / "UADA_rollout_online_env"

sources = [bool(patch_path_arg), bool(patch_dir_arg), bool(source_run_dir_arg), bool(source_exp_id_arg)]
if sum(sources) > 1:
    raise ValueError("Only one of PATCH_PATH, PATCH_DIR, SOURCE_RUN_DIR, SOURCE_EXP_ID may be set.")

def candidate_paths(root: Path):
    return [
        root / "last" / "projection_texture.pt",
        root / "last" / "patch.pt",
        root / "projection_texture.pt",
        root / "patch.pt",
        root / "initial" / "projection_texture.pt",
        root / "initial" / "patch.pt",
    ]

def resolve_from_dir(root_str: str):
    root = Path(os.path.abspath(os.path.expanduser(root_str)))
    for candidate in candidate_paths(root):
        if candidate.is_file():
            return root, candidate
    raise FileNotFoundError(
        f"No patch found under {root}. Checked: "
        "last/projection_texture.pt, last/patch.pt, projection_texture.pt, patch.pt, "
        "initial/projection_texture.pt, initial/patch.pt"
    )

resolved_run_dir = None
resolved_patch_path = None

if patch_path_arg:
    resolved_patch_path = Path(os.path.abspath(os.path.expanduser(patch_path_arg)))
    if not resolved_patch_path.is_file():
        raise FileNotFoundError(f"PATCH_PATH not found: {resolved_patch_path}")
    if resolved_patch_path.parent.name == "last":
        resolved_run_dir = resolved_patch_path.parent.parent
    else:
        resolved_run_dir = resolved_patch_path.parent
elif patch_dir_arg:
    resolved_run_dir, resolved_patch_path = resolve_from_dir(patch_dir_arg)
elif source_run_dir_arg:
    resolved_run_dir, resolved_patch_path = resolve_from_dir(source_run_dir_arg)
elif source_exp_id_arg:
    resolved_run_dir, resolved_patch_path = resolve_from_dir(str(run_base / source_exp_id_arg))
else:
    raise FileNotFoundError(
        "No patch source provided. Pass a patch file or run dir as argv[1], "
        "or set PATCH_PATH / PATCH_DIR / SOURCE_RUN_DIR / SOURCE_EXP_ID."
    )

loaded = torch.as_tensor(torch.load(resolved_patch_path, map_location="cpu"))
shape_csv = ",".join(str(int(x)) for x in loaded.shape)
print(str(resolved_run_dir))
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

EVAL_RUN_UUID="$("${PYTHON_BIN_VALUE}" - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/run/UADA_eval_patch_online_env_same_as_train_1}"
OUTPUT_DIR_VALUE="${OUTPUT_DIR:-${EVAL_OUTPUT_ROOT}/${EVAL_RUN_UUID}}"
mkdir -p "${OUTPUT_DIR_VALUE}"

PATCH_SIZE_VALUE="${PATCH_SIZE:-${INFERRED_PATCH_SHAPE}}"
PROJECTION_SIZE_VALUE="${PROJECTION_SIZE:-${INFERRED_PATCH_SHAPE}}"

DATASET_NAME="${DATASET:-libero_spatial}"
DEVICE_ID="${DEVICE_ID:-${DEVICE:-7}}"
GPU_HEALTHCHECK_ENABLED="${GPU_HEALTHCHECK_ENABLED:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-false}"
WANDB_ENTITY_NAME="${WANDB_ENTITY:-1473195970-beihang-university}"
LAMBDA_ACTION_GAP="${LAMBDA_ACTION_GAP:-1.5}"
LAMBDA_ANTI_GT="${LAMBDA_ANTI_GT:-0.02}"
LAMBDA_LOGIT_MARGIN="${LAMBDA_LOGIT_MARGIN:-0.02}"
LAMBDA_HISTORY="${LAMBDA_HISTORY:-0.0}"
LAMBDA_HISTORY_LEGACY="${LAMBDA_HISTORY_LEGACY:-0.0}"
LAMBDA_CE="${LAMBDA_CE:-0.0}"
LAMBDA_CE_PHASE2="${LAMBDA_CE_PHASE2:-0.0}"
LAMBDA_CONTINUOUS_ROLLOUT="${LAMBDA_CONTINUOUS_ROLLOUT:-0.0}"
LAMBDA_WINDOW_ROLLOUT_LOSS="${LAMBDA_WINDOW_ROLLOUT_LOSS:-0.2}"
IMPULSE_ROLLOUT_METRIC_ENABLED="${IMPULSE_ROLLOUT_METRIC_ENABLED:-false}"
WINDOW_ROLLOUT_PROBE_ENABLED_VALUE="${WINDOW_ROLLOUT_PROBE_ENABLED:-true}"
WINDOW_ROLLOUT_METRIC_MODE_VALUE="${WINDOW_ROLLOUT_METRIC_MODE:-adv_gt}"
WINDOW_ROLLOUT_FUTURE_MODE_VALUE="${WINDOW_ROLLOUT_FUTURE_MODE:-drop_attack_after_window}"
WINDOW_ROLLOUT_EXP_BASE_VALUE="${WINDOW_ROLLOUT_EXP_BASE:-0.9}"
WINDOW_ROLLOUT_FUTURE_HORIZON_VALUE="${WINDOW_ROLLOUT_FUTURE_HORIZON:-8}"
WINDOW_ROLLOUT_PHASE_SCOPE_VALUE="${WINDOW_ROLLOUT_PHASE_SCOPE:-initial}"
LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-0.5}"
ONLINE_CE_MODE_NAME="${ONLINE_CE_MODE:-off}"
SIGLIP_MODEL_NAME_VALUE="${SIGLIP_MODEL_NAME:-google/siglip-so400m-patch14-384}"
SIGLIP_INPUT_SIZE_VALUE="${SIGLIP_INPUT_SIZE:-384}"
ENV_RESOLUTION_VALUE="${ENV_RESOLUTION:-256}"
VAL_EPISODES="${ONLINE_VAL_EPISODES:-8}"
VAL_MAX_ENV_STEPS="${VAL_MAX_ENV_STEPS:-180}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-auto}"
PHASE1_DISABLE_PROJ_RAND="${PHASE1_DISABLE_PROJECTION_RANDOMIZATION:-true}"
LEARN_PROJECTOR_GAIN="${LEARN_PROJECTOR_GAIN:-false}"
LEARN_PROJECTOR_CHANNEL_GAIN="${LEARN_PROJECTOR_CHANNEL_GAIN:-false}"
PHOTOMETRIC_LR_RATIO="${PHOTOMETRIC_LR_RATIO:-0.1}"
RECORD_ONLINE_VIDEOS_VALUE="${RECORD_ONLINE_VIDEOS:-true}"
RECORD_ONLINE_VIDEOS_LAST_ONLY_VALUE="${RECORD_ONLINE_VIDEOS_LAST_ONLY:-false}"
RECORD_ONLINE_TRAIN_VIDEO_VALUE="${RECORD_ONLINE_TRAIN_VIDEO:-false}"
RECORD_ONLINE_VAL_VIDEO_VALUE="${RECORD_ONLINE_VAL_VIDEO:-true}"
TEXTURE_PARAM_MODE_VALUE="${TEXTURE_PARAM_MODE:-direct}"
LATENT_HW_VALUE="${LATENT_HW:-12,12}"
LAMBDA_TV_VALUE="${LAMBDA_TV:-0.001}"
TRAIN_ANCHOR_HORIZON_ITERS_VALUE="${TRAIN_ANCHOR_HORIZON_ITERS:-1}"
DETERMINISTIC_ANCHOR_SAMPLING_VALUE="${DETERMINISTIC_ANCHOR_SAMPLING:-false}"
PHASE1_ACTION_GAP_MODE_VALUE="${PHASE1_ACTION_GAP_MODE:-gt_farthest}"

if [[ "${GPU_HEALTHCHECK_ENABLED}" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    resolved_device="$("${PYTHON_BIN_VALUE}" - "${DEVICE_ID}" <<'PY'
import csv
import subprocess
import sys

requested = str(sys.argv[1]).strip()
query = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu,ecc.errors.uncorrected.aggregate.total",
    "--format=csv,noheader,nounits",
]

try:
    output = subprocess.check_output(query, text=True)
except Exception:
    print(requested)
    raise SystemExit(0)

rows = []
for raw_line in output.strip().splitlines():
    line = [part.strip() for part in next(csv.reader([raw_line]))]
    if len(line) != 4:
        continue
    try:
        rows.append(
            {
                "index": line[0],
                "memory_used": int(line[1]),
                "utilization": int(line[2]),
                "ecc_uncorrectable": int(line[3]),
            }
        )
    except ValueError:
        continue

selected = next((row for row in rows if row["index"] == requested), None)
if selected is None:
    print(requested)
    raise SystemExit(0)

if selected["ecc_uncorrectable"] <= 0:
    print(requested)
    raise SystemExit(0)

candidates = [row for row in rows if row["ecc_uncorrectable"] <= 0]
if not candidates:
    print(requested)
    raise SystemExit(0)

candidates.sort(key=lambda row: (row["memory_used"], row["utilization"], int(row["index"])))
print(candidates[0]["index"])
PY
)"

    if [[ "${resolved_device}" != "${DEVICE_ID}" ]]; then
        echo "Selected GPU ${DEVICE_ID} has uncorrectable ECC errors; switching eval script to healthy GPU ${resolved_device}."
        DEVICE_ID="${resolved_device}"
    fi
fi

echo "Resolved patch source:"
echo "  source_run_dir=${RESOLVED_SOURCE_RUN_DIR}"
echo "  patch_path=${RESOLVED_PATCH_PATH}"
echo "  patch_size=${PATCH_SIZE_VALUE}"
echo "  projection_size=${PROJECTION_SIZE_VALUE}"
echo "  output_dir=${OUTPUT_DIR_VALUE}"
echo "  note=Runs one no-op train iteration (lr=0) to trigger the exact same online val evaluation path."

"${PYTHON_BIN_VALUE}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("libero") is None:
    print("ERROR: `libero` is not installed. Please install LIBERO before running online env eval.")
    sys.exit(1)
try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    print(f"ERROR: `libero` is installed but OffScreenRenderEnv is unavailable: {exc}")
    sys.exit(1)
print("LIBERO dependency check passed.")
PY

"${PYTHON_BIN_VALUE}" VLAAttacker/UADA_rollout_online_env_wrapper.py \
    --maskidx 0,1,2 \
    --use_all_joints false \
    --gripper_weight 0.5 \
    --lr 0.0 \
    --server "${PROJECT_ROOT}" \
    --device "${DEVICE_ID}" \
    --iter 1 \
    --accumulate 1 \
    --bs 1 \
    --warmup 1 \
    --tags "UADA_rollout_online_env" "eval_only_same_as_train_1" \
    --geometry true \
    --attack_mode "projection" \
    --patch_size "${PATCH_SIZE_VALUE}" \
    --projection_size "${PROJECTION_SIZE_VALUE}" \
    --init_projection_texture_path "${RESOLVED_PATCH_PATH}" \
    --output_dir "${OUTPUT_DIR_VALUE}" \
    --projection_alpha 0.55 \
    --projection_alpha_jitter 0.00 \
    --projection_soft_edge 1.2 \
    --projection_angle "${PROJECTION_ANGLE:-0}" \
    --projection_fixed_angle "${PROJECTION_FIXED_ANGLE:-true}" \
    --projection_shear "${PROJECTION_SHEAR:-0.0}" \
    --projection_scale_min "${PROJECTION_SCALE_MIN:-1.0}" \
    --projection_scale_max "${PROJECTION_SCALE_MAX:-1.0}" \
    --projection_region "lower_half_fixed" \
    --projection_lower_start 0.55 \
    --projection_width_ratio 0.35 \
    --projection_height_ratio 0.35 \
    --projection_margin_x 0.04 \
    --projection_keystone 0.22 \
    --projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER:-0.0}" \
    --projector_gamma 1.8 \
    --projector_gain 1.35 \
    --projector_channel_gain "1.08,1.04,1.00" \
    --learn_projector_gain "${LEARN_PROJECTOR_GAIN}" \
    --learn_projector_channel_gain "${LEARN_PROJECTOR_CHANNEL_GAIN}" \
    --photometric_lr_ratio "${PHOTOMETRIC_LR_RATIO}" \
    --projector_ambient 0.08 \
    --projector_vignetting 0.08 \
    --projector_distance_falloff 0.10 \
    --projector_psf false \
    --projection_randomization_enabled "${PROJECTION_RANDOMIZATION_ENABLED:-true}" \
    --wandb_project "${WANDB_PROJECT_NAME}" \
    --wandb_entity "${WANDB_ENTITY_NAME}" \
    --dataset "${DATASET_NAME}" \
    --resize_patch false \
    --phase1_ratio 0.4 \
    --phase1_rollout 4 \
    --phase2_rollout "${PHASE2_ROLLOUT:-8}" \
    --lambda_action_gap "${LAMBDA_ACTION_GAP}" \
    --lambda_anti_gt "${LAMBDA_ANTI_GT}" \
    --lambda_logit_margin "${LAMBDA_LOGIT_MARGIN}" \
    --lambda_history "${LAMBDA_HISTORY}" \
    --lambda_history_legacy "${LAMBDA_HISTORY_LEGACY}" \
    --lambda_ce "${LAMBDA_CE}" \
    --lambda_ce_phase2 "${LAMBDA_CE_PHASE2}" \
    --lambda_continuous_rollout "${LAMBDA_CONTINUOUS_ROLLOUT}" \
    --lambda_window_rollout_loss "${LAMBDA_WINDOW_ROLLOUT_LOSS}" \
    --impulse_rollout_metric_enabled "${IMPULSE_ROLLOUT_METRIC_ENABLED}" \
    --window_rollout_probe_enabled "${WINDOW_ROLLOUT_PROBE_ENABLED_VALUE}" \
    --window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}" \
    --window_rollout_future_mode "${WINDOW_ROLLOUT_FUTURE_MODE_VALUE}" \
    --window_rollout_exp_base "${WINDOW_ROLLOUT_EXP_BASE_VALUE}" \
    --window_rollout_future_horizon "${WINDOW_ROLLOUT_FUTURE_HORIZON_VALUE}" \
    --window_rollout_phase_scope "${WINDOW_ROLLOUT_PHASE_SCOPE_VALUE}" \
    --lambda_siglip "${LAMBDA_SIGLIP}" \
    --siglip_model_name "${SIGLIP_MODEL_NAME_VALUE}" \
    --siglip_input_size "${SIGLIP_INPUT_SIZE_VALUE}" \
    --save_interval 1 \
    --eval_enabled true \
    --val_deterministic true \
    --val_seed 42 \
    --val_disable_lighting true \
    --lighting_aug_enabled false \
    --lighting_aug_train_only false \
    --phase1_disable_lighting true \
    --phase1_disable_projection_randomization "${PHASE1_DISABLE_PROJ_RAND}" \
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
    --record_online_videos "${RECORD_ONLINE_VIDEOS_VALUE}" \
    --record_online_videos_last_only "${RECORD_ONLINE_VIDEOS_LAST_ONLY_VALUE}" \
    --record_online_train_video "${RECORD_ONLINE_TRAIN_VIDEO_VALUE}" \
    --record_online_val_video "${RECORD_ONLINE_VAL_VIDEO_VALUE}" \
    --record_online_video_frame_source "projected_input" \
    --record_online_video_fps 10 \
    --viz_enabled false \
    --viz_policy "milestone" \
    --viz_samples 2 \
    --viz_save_best false \
    --viz_save_last false \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --online_train_tasks_per_iter 1 \
    --online_train_episodes_per_task 1 \
    --online_val_episodes "${VAL_EPISODES}" \
    --num_steps_wait 10 \
    --max_env_steps "auto_by_suite" \
    --val_max_env_steps "${VAL_MAX_ENV_STEPS}" \
    --env_resolution "${ENV_RESOLUTION_VALUE}" \
    --online_ce_mode "${ONLINE_CE_MODE_NAME}" \
    --env_action_source "adv" \
    --env_seed 42 \
    --action_gap_mode "${ACTION_GAP_MODE:-gt_farthest}" \
    --phase1_action_gap_mode "${PHASE1_ACTION_GAP_MODE_VALUE}" \
    --gt_dataset_root "${GT_DATASET_ROOT:-/home/yxx/roboticAttack/openvla-main/dataset}" \
    --gt_action_bank_path "${GT_ACTION_BANK_PATH:-}" \
    --gt_softmin_tau "${GT_SOFTMIN_TAU:-0.05}" \
    --phase_state_mode "${PHASE_STATE_MODE:-contact_manipulate_only}" \
    --phase_state_cache_path "${PHASE_STATE_CACHE_PATH:-}" \
    --texture_param_mode "${TEXTURE_PARAM_MODE_VALUE}" \
    --latent_hw "${LATENT_HW_VALUE}" \
    --lambda_tv "${LAMBDA_TV_VALUE}" \
    --train_anchor_horizon_iters "${TRAIN_ANCHOR_HORIZON_ITERS_VALUE}" \
    --deterministic_anchor_sampling "${DETERMINISTIC_ANCHOR_SAMPLING_VALUE}" \
    --auto_gpu_tune false
