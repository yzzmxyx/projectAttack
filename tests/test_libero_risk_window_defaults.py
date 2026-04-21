import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_PATH = REPO_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval_args_geo_batch.py"


def _argument_defaults():
    tree = ast.parse(LIBERO_PATH.read_text(encoding="utf-8"))
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


def test_libero_defaults_keep_risk_window_disabled():
    defaults = _argument_defaults()
    assert defaults["--risk_window_enable"] is False
    assert defaults["--risk_window_config"] == ""
    assert defaults["--risk_window_asset_root"] == ""
    assert defaults["--risk_window_log_dir"] == ""
    assert defaults["--risk_window_action"] == "log_only"
    assert defaults["--risk_window_overlay"] is False
