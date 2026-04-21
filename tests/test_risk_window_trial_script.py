from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_risk_window_trial.sh"


def test_risk_window_trial_script_exposes_both_libero_and_bridge_modes():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'MODE_NAME="$(printf \'%s\' "${MODE:-libero}"',
        'python3.10',
        'risk_window.cli validate-assets',
        '--risk_window_enable true',
        '--risk_window_asset_root "${ASSET_ROOT_VALUE}"',
        'run_libero_eval_args_geo_batch.py',
        'run_bridgev2_eval.py',
        'Bridge mode uses interactive task input',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {SCRIPT_PATH}"
