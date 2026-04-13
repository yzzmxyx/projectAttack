import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "VLAAttacker" / "white_patch" / "gt_phase_schedule.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_gt_phase_schedule_module", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_start_progresses_across_all_phases():
    module = _load_module()
    phases = [
        module.infer_gt_phase_for_step(
            step_idx=step_idx,
            horizon=5,
            phase_start_name="initial",
            contact_ratio=0.3,
            post_ratio=0.7,
        )
        for step_idx in range(5)
    ]
    assert phases == [
        "pre_contact",
        "pre_contact",
        "contact_manipulate",
        "post_contact",
        "post_contact",
    ]


def test_contact_start_can_advance_into_post_contact():
    module = _load_module()
    phases = [
        module.infer_gt_phase_for_step(
            step_idx=step_idx,
            horizon=5,
            phase_start_name="contact_manipulate",
            contact_ratio=0.3,
            post_ratio=0.7,
        )
        for step_idx in range(5)
    ]
    assert phases == [
        "contact_manipulate",
        "contact_manipulate",
        "contact_manipulate",
        "post_contact",
        "post_contact",
    ]


def test_post_start_stays_in_post_contact():
    module = _load_module()
    phases = [
        module.infer_gt_phase_for_step(
            step_idx=step_idx,
            horizon=5,
            phase_start_name="post_contact",
            contact_ratio=0.3,
            post_ratio=0.7,
        )
        for step_idx in range(5)
    ]
    assert phases == ["post_contact"] * 5


def test_boundary_ratios_are_clamped_into_a_valid_pair():
    module = _load_module()
    contact_ratio, post_ratio = module.clamp_phase_boundary_ratios(contact_ratio=1.2, post_ratio=0.1)
    assert contact_ratio == 1.0
    assert post_ratio == 1.0
