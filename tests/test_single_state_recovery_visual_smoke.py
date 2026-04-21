import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval_single_state_recovery_visual_smoke.sh"
TOOL_PATH = REPO_ROOT / "evaluation_tool" / "eval_single_state_recovery_visual_smoke.py"


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


def test_visual_smoke_cli_defaults_match_smoke_expectations():
    defaults = _argument_defaults(TOOL_PATH)
    assert defaults["--num_steps_wait"] == 10
    assert defaults["--env_resolution"] == 256
    assert defaults["--window_stride"] == 8
    assert defaults["--recovery_vision_backbone"] == "pixel"
    assert defaults["--recovery_image_resize_strategy"] == "resize-naive"
    assert defaults["--anchor_steps"] == "auto"
    assert defaults["--max_anchors"] == 3
    assert "--phase_state_cache_path" not in defaults


def test_parse_anchor_steps_and_auto_mode():
    tool = _load_module(TOOL_PATH, "test_visual_smoke_parse_steps")
    assert tool.parse_anchor_steps("auto") is None
    assert tool.parse_anchor_steps("0, 16, 8, 16, -3") == [0, 8, 16]


def test_select_visual_anchor_steps_prefers_first_available_subset():
    tool = _load_module(TOOL_PATH, "test_visual_smoke_select_steps")
    assert tool.select_visual_anchor_steps(None, [0, 8, 16, 24], 3) == [0, 8, 16]
    assert tool.select_visual_anchor_steps([16, 8, 16], [0, 8, 16, 24], 3) == [8, 16]


def test_visual_smoke_run_script_defaults_to_lightweight_recovery_settings():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '--task_id "${TASK_ID:-4}"' in contents
    assert '--recovery_vision_backbone "${RECOVERY_VISION_BACKBONE:-pixel}"' in contents
    assert '--anchor_steps "${ANCHOR_STEPS:-auto}"' in contents
    assert "phase_state_cache" not in contents
