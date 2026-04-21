from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_window_rollout_videos.sh"


def test_video_supplement_script_replays_from_existing_probe_runs():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'INPUT_PROBE_ROOT="${1:-${PROBE_ROOT:-}}"',
        'source_texture_path="${source_run_dir}/last/projection_texture.pt"',
        '--init_projection_texture_path "${source_texture_path}"',
        '--record_online_videos true',
        '--record_online_val_video true',
        '--online_val_episodes "${VIDEO_VAL_EPISODES}"',
        'status="ok"',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {SCRIPT_PATH}"


def test_video_supplement_script_covers_five_probe_variants():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    expected_variants = (
        '"rollout-only"',
        '"gt+rollout"',
        '"siglip+rollout"',
        '"gt+siglip+rollout"',
        '"rollout+gt+siglip+ce"',
    )
    for variant in expected_variants:
        assert variant in contents, f"Missing variant `{variant}` in {SCRIPT_PATH}"
