import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_window_rollout.sh"
ADV_GT_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_window_rollout_adv_gt.sh"
SIGLIP_GT_PHASE_FORMS_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_UADA_rollout_online_env_probe_window_rollout_siglip_gt_phase_forms.sh"
)
HELPER_PATH = REPO_ROOT / "VLAAttacker" / "white_patch" / "window_rollout_probe_utils.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_window_rollout_probe_script_includes_phase_and_horizon_defaults():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'DATASET_NAME="${DATASET:-libero_spatial}"',
        'PHASE_STATE_MODE_NAME="${PHASE_STATE_MODE:-phase_cycle}"',
        'WINDOW_ROLLOUT_FUTURE_HORIZON="${WINDOW_ROLLOUT_FUTURE_HORIZON:-8}"',
        'WINDOW_ROLLOUT_EXP_BASE="${WINDOW_ROLLOUT_EXP_BASE:-0.9}"',
        'WINDOW_ROLLOUT_PROBE_ENABLED_VALUE="true"',
        'ONLINE_VAL_EPISODES_VALUE="${ONLINE_VAL_EPISODES:-8}"',
        'VAL_MAX_ENV_STEPS_VALUE="${VAL_MAX_ENV_STEPS:-120}"',
        'SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-5}"',
        'PHASE1_DISABLE_PROJECTION_RANDOMIZATION_VALUE="${PHASE1_DISABLE_PROJECTION_RANDOMIZATION:-true}"',
        'PROJECTION_FIXED_ANGLE_VALUE="${PROJECTION_FIXED_ANGLE:-true}"',
        'PROJECTION_SCALE_MIN_VALUE="${PROJECTION_SCALE_MIN:-1.0}"',
        'PROJECTION_SCALE_MAX_VALUE="${PROJECTION_SCALE_MAX:-1.0}"',
        'LEARN_PROJECTOR_GAIN_VALUE="false"',
        'LEARN_PROJECTOR_CHANNEL_GAIN_VALUE="false"',
        '--record_online_videos true',
        '--record_online_videos_last_only true',
        '--record_online_val_video true',
        '--window_rollout_probe_enabled "${WINDOW_ROLLOUT_PROBE_ENABLED_VALUE}"',
        '--window_rollout_exp_base "${WINDOW_ROLLOUT_EXP_BASE}"',
        '--window_rollout_future_horizon "${WINDOW_ROLLOUT_FUTURE_HORIZON}"',
        '--window_rollout_phase_scope "${WINDOW_ROLLOUT_PHASE_SCOPE_VALUE}"',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {SCRIPT_PATH}"


def test_window_rollout_probe_script_lists_all_probe_variants():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    expected_variants = (
        'run_variant 1 "rollout-only" 0 0 0',
        'run_variant 2 "gt+rollout" 1 0 0',
        'run_variant 3 "siglip+rollout" 0 1 0',
        'run_variant 4 "gt+siglip+rollout" 1 1 0',
        'run_variant 5 "gt+siglip+rollout+ce" 1 1 0.1',
    )
    for snippet in expected_variants:
        assert snippet in contents, f"Missing variant snippet `{snippet}` in {SCRIPT_PATH}"


def test_window_rollout_adv_gt_probe_script_includes_metric_mode_and_summary_fields():
    contents = ADV_GT_SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'PROBE_ROOT="${current_dir}/run/UADA_rollout_online_env_probe_window_rollout_adv_gt/${PROBE_ID}"',
        'WINDOW_ROLLOUT_METRIC_MODE_VALUE="adv_gt"',
        '--window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}"',
        'window_rollout_metric_mode',
        'final_val_window_rollout_metric_value',
        '"gt+siglip+rollout+ce" 1 1 0.1',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {ADV_GT_SCRIPT_PATH}"


def test_window_rollout_siglip_gt_phase_forms_script_has_expected_defaults_and_outputs():
    contents = SIGLIP_GT_PHASE_FORMS_SCRIPT_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        'PROBE_ROOT="${current_dir}/run/UADA_rollout_online_env_probe_window_rollout_siglip_gt_phase_forms/${PROBE_ID}"',
        'WINDOW_ROLLOUT_METRIC_MODE_VALUE="delta_weighted"',
        'PROJECTION_SCALE_MIN_VALUE="${PROJECTION_SCALE_MIN:-0.5}"',
        'PROJECTION_SCALE_MAX_VALUE="${PROJECTION_SCALE_MAX:-0.5}"',
        '--window_rollout_metric_mode "${WINDOW_ROLLOUT_METRIC_MODE_VALUE}"',
        'BEST_BY_LOSS_FAMILY_JSON="${PROBE_ROOT}/best_by_loss_family.json"',
        'loss_family,form_name,variant,phase_state_mode,window_rollout_phase_scope',
        'final_val_window_rollout_metric_value',
        'best_by_loss_family_json',
    )
    for snippet in expected_snippets:
        assert snippet in contents, f"Missing `{snippet}` in {SIGLIP_GT_PHASE_FORMS_SCRIPT_PATH}"


def test_window_rollout_siglip_gt_phase_forms_script_lists_all_six_variants():
    contents = SIGLIP_GT_PHASE_FORMS_SCRIPT_PATH.read_text(encoding="utf-8")
    expected_variants = (
        'run_variant 1 "rollout+siglip" "phase-cycle" "phase_cycle" "all" "rollout+siglip__phase-cycle" 0 1 0 "off" "clean_adv" 1',
        'run_variant 2 "rollout+siglip" "only-initial" "initial_only" "initial" "rollout+siglip__only-initial" 0 1 0 "off" "clean_adv" 1',
        'run_variant 3 "rollout+siglip" "only-contact-manipulate" "phase_cycle" "contact_manipulate" "rollout+siglip__only-contact-manipulate" 0 1 0 "off" "clean_adv" 1',
        'run_variant 4 "gt+siglip" "phase-cycle" "phase_cycle" "all" "gt+siglip__phase-cycle" 1 1 0 "off" "gt_farthest" 0',
        'run_variant 5 "gt+siglip" "only-initial" "initial_only" "initial" "gt+siglip__only-initial" 1 1 0 "off" "gt_farthest" 0',
        'run_variant 6 "gt+siglip" "only-contact-manipulate" "phase_cycle" "contact_manipulate" "gt+siglip__only-contact-manipulate" 1 1 0 "off" "gt_farthest" 0',
    )
    for snippet in expected_variants:
        assert snippet in contents, f"Missing variant snippet `{snippet}` in {SIGLIP_GT_PHASE_FORMS_SCRIPT_PATH}"

    assert contents.count('run_variant ') == 6


def test_window_rollout_helper_resolves_windows_and_metric_selection():
    module = _load_module(HELPER_PATH, "test_window_rollout_probe_utils")

    assert module.normalize_window_rollout_phase_scope("contact") == "contact_manipulate"
    assert module.normalize_window_rollout_metric_mode("adv") == "adv_gt"
    assert module.normalize_window_rollout_metric_mode("delta") == "delta_weighted"
    initial_window = module.resolve_phase_window(
        source_T=100,
        contact_step=30,
        post_step=70,
        phase_name="initial",
    )
    assert initial_window["window_start_step"] == 0
    assert initial_window["window_end_step"] == 29

    contact_window = module.resolve_phase_window(
        source_T=100,
        contact_step=30,
        post_step=70,
        phase_name="contact_manipulate",
    )
    assert contact_window["window_start_step"] == 30
    assert contact_window["window_end_step"] == 69

    assert module.infer_phase_name_from_boundaries(10, 100, 30, 70) == "pre_contact"
    assert module.infer_phase_name_from_boundaries(30, 100, 30, 70) == "contact_manipulate"
    assert module.infer_phase_name_from_boundaries(75, 100, 30, 70) == "post_contact"

    weighted = module.compute_weighted_window_rollout_delta([1.0, 2.0, -1.0], exp_base=0.9)
    assert abs(weighted - (1.0 + (2.0 * 0.9) + (-1.0 * 0.81))) < 1e-9
    assert module.select_window_rollout_metric_value("delta_weighted", weighted, 7.5) == weighted
    assert module.select_window_rollout_metric_value("adv_gt", weighted, 7.5) == 7.5
