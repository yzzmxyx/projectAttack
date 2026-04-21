import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "evaluation_tool" / "match_rollout_init_states.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_match_rollout_init_states.sh"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argument_defaults(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
                try:
                    defaults[first_arg.value] = ast.literal_eval(keyword.value)
                except Exception:
                    continue
    return defaults


def test_match_rollout_init_state_cli_defaults_match_current_probe_assumptions():
    defaults = _argument_defaults(TOOL_PATH)
    assert defaults["--manifest_split"] == "val"
    assert defaults["--match_all_val_episodes"] is True
    assert defaults["--val_deterministic"] is True
    assert defaults["--online_val_episodes"] == 8
    assert defaults["--val_seed"] == 42


def test_build_val_schedule_specs_wraps_tasks_and_init_state_indices():
    tool = _load_module(TOOL_PATH, "test_match_rollout_init_states_schedule")
    specs = tool.build_val_schedule_specs(
        task_descriptions=["task_a", "task_b"],
        init_state_counts=[3, 5],
        online_val_episodes=5,
        val_seed=42,
    )
    assert [(row["ep_idx"], row["task_id"], row["init_state_idx"]) for row in specs] == [
        (0, 0, 0),
        (1, 1, 3),
        (2, 0, 2),
        (3, 1, 0),
        (4, 0, 1),
    ]


def test_select_latest_manifest_row_prefers_requested_split_and_highest_iter():
    tool = _load_module(TOOL_PATH, "test_match_rollout_init_states_manifest")
    row = tool.select_latest_manifest_row(
        rows=[
            {"split": "train", "iter_idx": "8", "task_id": "3"},
            {"split": "val", "iter_idx": "4", "task_id": "0"},
            {"split": "val", "iter_idx": "9", "task_id": "1"},
        ],
        split="val",
    )
    assert row is not None
    assert row["iter_idx"] == "9"
    assert row["task_id"] == "1"


def test_infer_recorded_episode_spec_uses_matching_task_id_and_description():
    tool = _load_module(TOOL_PATH, "test_match_rollout_init_states_recorded")
    spec = tool.infer_recorded_episode_spec(
        manifest_row={
            "split": "val",
            "iter_idx": "19",
            "task_id": "1",
            "task_description": "Task B",
            "video_path": "/tmp/fake.mp4",
        },
        schedule_specs=[
            {"ep_idx": 0, "task_id": 0, "task_description": "task a", "init_state_idx": 2},
            {"ep_idx": 1, "task_id": 1, "task_description": "task b", "init_state_idx": 4},
        ],
    )
    assert spec is not None
    assert spec["ep_idx"] == 1
    assert spec["init_state_idx"] == 4
    assert spec["match_basis"] == "latest_recorded_manifest"


def test_run_script_defaults_to_batch_matching_mode():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '--run_ids "${RUN_IDS:-all}"' in contents
    assert '--match_all_val_episodes "${MATCH_ALL_VAL_EPISODES:-true}"' in contents
    assert '--val_seed "${VAL_SEED:-42}"' in contents
