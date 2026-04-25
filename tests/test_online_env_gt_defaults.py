import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "VLAAttacker" / "UADA_rollout_online_env_wrapper.py"
METADATA_HELPER_PATH = REPO_ROOT / "experiments" / "robot" / "libero" / "phase_state_cache_metadata.py"
SCRIPT_PATHS = [
    REPO_ROOT / "scripts" / "run_UADA_rollout_online_env.sh",
    REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe.sh",
    REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_round2.sh",
    REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_siglip.sh",
]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrapper_argument_defaults():
    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if len(node.args) == 0:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[first_arg.value] = ast.literal_eval(keyword.value)
    return defaults


def test_wrapper_defaults_enable_gt_farthest_phase_cycle_for_libero_10():
    defaults = _wrapper_argument_defaults()
    assert defaults["--dataset"] == "libero_10"
    assert defaults["--action_gap_mode"] == "gt_farthest"
    assert defaults["--phase_state_mode"] == "phase_cycle"
    assert defaults["--window_rollout_future_mode"] == "keep_adv"


def test_launch_scripts_default_to_new_dataset_and_modes():
    expected_snippets_by_script = {
        REPO_ROOT / "scripts" / "run_UADA_rollout_online_env.sh": (
            'DATASET_NAME="${DATASET:-libero_spatial}"',
            'ONLINE_CE_MODE_NAME="${ONLINE_CE_MODE:-off}"',
            'LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-1.0}"',
            'WINDOW_ROLLOUT_FUTURE_MODE_VALUE="${WINDOW_ROLLOUT_FUTURE_MODE:-drop_attack_after_window}"',
            '--action_gap_mode "${ACTION_GAP_MODE:-gt_farthest}"',
            '--phase_state_mode "${PHASE_STATE_MODE:-contact_manipulate_only}"',
        ),
        REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe.sh": (
            'DATASET_NAME="${DATASET:-libero_spatial}"',
            'ACTION_GAP_MODE_NAME="${ACTION_GAP_MODE:-clean_adv}"',
            'PHASE_STATE_MODE_NAME="${PHASE_STATE_MODE:-initial_only}"',
        ),
        REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_round2.sh": (
            '--dataset "${DATASET:-libero_10}"',
            '--action_gap_mode "${ACTION_GAP_MODE:-gt_farthest}"',
            '--phase_state_mode "${PHASE_STATE_MODE:-phase_cycle}"',
        ),
        REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_siglip.sh": (
            'DATASET_NAME="${DATASET:-libero_spatial}"',
            'ACTION_GAP_MODE_NAME="${ACTION_GAP_MODE:-clean_adv}"',
            'PHASE_STATE_MODE_NAME="${PHASE_STATE_MODE:-initial_only}"',
        ),
    }
    for script_path in SCRIPT_PATHS:
        contents = script_path.read_text(encoding="utf-8")
        for snippet in expected_snippets_by_script[script_path]:
            assert snippet in contents, f"Missing `{snippet}` in {script_path}"


def test_phase_state_cache_metadata_helper_preserves_summary_fields():
    module = _load_module(METADATA_HELPER_PATH, "test_phase_state_cache_metadata_module")
    payload = {
        "schema_version": 1,
        "dataset": "libero_10",
        "suite_name": "libero_10",
        "phase_parquet": "/tmp/phases.parquet",
        "phase_starts": ["initial", "contact_manipulate", "post_contact"],
        "alignment": "ordinal_modulo",
        "num_steps_wait": 10,
        "env_resolution": 128,
        "metadata": {
            "num_states": 30,
            "num_records": 30,
            "num_tasks": 10,
            "task_ids": [0, 1, 2],
            "max_tasks": None,
            "max_init_states_per_task": None,
            "state_shape_examples": {"(0, 0, 'initial')": [123]},
        },
    }
    metadata = module.build_phase_state_cache_metadata(payload, "/tmp/phase_state_cache.pt")
    assert metadata["phase_state_cache_pt"] == "/tmp/phase_state_cache.pt"
    assert metadata["dataset"] == "libero_10"
    assert metadata["suite_name"] == "libero_10"
    assert metadata["alignment"] == "ordinal_modulo"
    assert metadata["num_states"] == 30
    assert metadata["num_records"] == 30
    assert metadata["num_tasks"] == 10
    assert metadata["task_ids"] == [0, 1, 2]
    assert metadata["phase_starts"] == ["initial", "contact_manipulate", "post_contact"]
    assert metadata["state_shape_examples"] == {"(0, 0, 'initial')": [123]}
