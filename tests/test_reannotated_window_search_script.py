from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_reannotated_vulnerability_window_search.sh"


def test_reannotated_window_search_script_chains_match_and_search():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'ROLLOUT_RUN_ID is required' in contents
    assert 'match_rollout_init_states.py' in contents
    assert 'run_eval_vulnerability_window_search.sh' in contents
    assert 'PATCH_PATH="${PATCH_PATH_VALUE}"' in contents
    assert '--match_all_val_episodes false' in contents
