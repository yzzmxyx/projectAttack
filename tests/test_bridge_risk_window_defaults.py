import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "experiments" / "robot" / "bridge" / "run_bridgev2_eval.py"


def _dataclass_defaults():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    defaults = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "GenerateConfig":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                try:
                    defaults[item.target.id] = ast.literal_eval(item.value)
                except Exception:
                    continue
    return defaults


def test_bridge_defaults_keep_risk_window_disabled():
    defaults = _dataclass_defaults()
    assert defaults["risk_window_enable"] is False
    assert defaults["risk_window_config"] == ""
    assert defaults["risk_window_asset_root"] == ""
    assert defaults["risk_window_log_dir"] == ""
    assert defaults["risk_window_action"] == "log_only"
    assert defaults["risk_window_overlay"] is False
