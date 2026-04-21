import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "VLAAttacker" / "UADA_rollout_online_env_wrapper.py"
RUN_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_UADA_rollout_online_env.sh"


def _wrapper_argument_defaults():
    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[first_arg.value] = ast.literal_eval(keyword.value)
    return defaults


def test_wrapper_loss_defaults_match_phase_split_policy():
    defaults = _wrapper_argument_defaults()
    assert defaults["--lambda_action_gap"] == 1.0
    assert defaults["--lambda_history"] == 0.0
    assert defaults["--lambda_history_legacy"] == 0.0
    assert defaults["--lambda_ce"] == 0.02
    assert defaults["--lambda_ce_phase2"] == 0.0
    assert defaults["--lambda_continuous_rollout"] == 0.0
    assert defaults["--impulse_rollout_metric_enabled"] is False
    assert defaults["--window_rollout_probe_enabled"] is False
    assert defaults["--window_rollout_metric_mode"] == "delta_weighted"
    assert defaults["--window_rollout_exp_base"] == 0.9
    assert defaults["--window_rollout_future_horizon"] == 8
    assert defaults["--window_rollout_phase_scope"] == "all"
    assert defaults["--init_projection_texture_path"] == ""
    assert defaults["--lambda_siglip"] == 0.15
    assert defaults["--val_max_env_steps"] == 120
    assert defaults["--learn_projector_gain"] is False
    assert defaults["--learn_projector_channel_gain"] is False
    assert defaults["--photometric_lr_ratio"] == 0.1


def test_run_script_env_defaults_match_loss_defaults():
    contents = RUN_SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'LAMBDA_ACTION_GAP="${LAMBDA_ACTION_GAP:-1.0}"',
        'LAMBDA_HISTORY="${LAMBDA_HISTORY:-0.0}"',
        'LAMBDA_HISTORY_LEGACY="${LAMBDA_HISTORY_LEGACY:-0.0}"',
        'LAMBDA_CE="${LAMBDA_CE:-0.02}"',
        'LAMBDA_CE_PHASE2="${LAMBDA_CE_PHASE2:-0.0}"',
        'LAMBDA_CONTINUOUS_ROLLOUT="${LAMBDA_CONTINUOUS_ROLLOUT:-0.0}"',
        'IMPULSE_ROLLOUT_METRIC_ENABLED="${IMPULSE_ROLLOUT_METRIC_ENABLED:-false}"',
        'LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-0.15}"',
        'VAL_MAX_ENV_STEPS="${VAL_MAX_ENV_STEPS:-120}"',
        'LEARN_PROJECTOR_GAIN="${LEARN_PROJECTOR_GAIN:-true}"',
        'LEARN_PROJECTOR_CHANNEL_GAIN="${LEARN_PROJECTOR_CHANNEL_GAIN:-true}"',
        'PHOTOMETRIC_LR_RATIO="${PHOTOMETRIC_LR_RATIO:-0.1}"',
        '--lambda_ce_phase2 "${LAMBDA_CE_PHASE2}"',
        '--lambda_continuous_rollout "${LAMBDA_CONTINUOUS_ROLLOUT}"',
        '--impulse_rollout_metric_enabled "${IMPULSE_ROLLOUT_METRIC_ENABLED}"',
        '--val_max_env_steps "${VAL_MAX_ENV_STEPS}"',
        '--projection_angle "${PROJECTION_ANGLE:-25}"',
        '--projection_fixed_angle "${PROJECTION_FIXED_ANGLE:-false}"',
        '--projection_shear "${PROJECTION_SHEAR:-0.15}"',
        '--projection_scale_min "${PROJECTION_SCALE_MIN:-0.8}"',
        '--projection_scale_max "${PROJECTION_SCALE_MAX:-1.2}"',
        '--projection_keystone_jitter "${PROJECTION_KEYSTONE_JITTER:-0.03}"',
        '--learn_projector_gain "${LEARN_PROJECTOR_GAIN}"',
        '--learn_projector_channel_gain "${LEARN_PROJECTOR_CHANNEL_GAIN}"',
        '--photometric_lr_ratio "${PHOTOMETRIC_LR_RATIO}"',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {RUN_SCRIPT_PATH}"
