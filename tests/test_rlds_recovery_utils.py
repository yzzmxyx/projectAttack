import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "evaluation_tool" / "rlds_recovery_utils.py"
BUILDER_PATH = REPO_ROOT / "evaluation_tool" / "build_single_state_rlds_recovery_asset.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval_vulnerability_window_search.sh"


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
                defaults[first_arg.value] = ast.literal_eval(keyword.value)
    return defaults


def test_recovery_builder_cli_defaults_match_plan():
    defaults = _argument_defaults(BUILDER_PATH)
    assert defaults["--window_stride"] == 8
    assert defaults["--num_steps_wait"] == 10
    assert defaults["--env_resolution"] == 256
    assert defaults["--recovery_vision_backbone"] == "dinoclip-vit-l-336px"
    assert defaults["--recovery_image_resize_strategy"] == "resize-naive"


def test_build_recovery_anchor_steps_merges_window_stride_and_phase_boundaries():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_anchor_steps")
    anchors = utils.build_recovery_anchor_steps(
        num_steps=25,
        window_stride=8,
        phase_boundaries={"contact_manipulate": 9, "post_contact": 17},
    )
    assert anchors == [0, 8, 9, 16, 17, 24]


def test_resolve_recovery_anchor_step_handles_exact_and_fallback_matches():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_anchor_lookup")
    assert utils.resolve_recovery_anchor_step(16, [0, 8, 16, 24]) == (16, True)
    assert utils.resolve_recovery_anchor_step(19, [0, 8, 16, 24]) == (16, False)
    assert utils.resolve_recovery_anchor_step(0, []) == (None, False)


def test_compute_robot_state_distance_sums_joint_gripper_and_eef_terms():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_robot_distance")
    reference = {
        "joint_state": np.zeros((3,), dtype=np.float32),
        "gripper_state": np.zeros((2,), dtype=np.float32),
        "eef_state": np.zeros((6,), dtype=np.float32),
    }
    candidate = {
        "joint_state": np.asarray([3.0, 4.0, 0.0], dtype=np.float32),
        "gripper_state": np.asarray([0.0, 2.0], dtype=np.float32),
        "eef_state": np.asarray([1.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    metrics = utils.compute_robot_state_distance(reference, candidate)
    assert metrics["joint_l2"] == 5.0
    assert metrics["gripper_l2"] == 2.0
    assert metrics["eef_l2"] == 3.0
    assert metrics["total"] == 10.0


def test_sort_candidate_summaries_prioritizes_robot_then_image_distances():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_candidate_sort")
    sorted_rows = utils.sort_candidate_summaries(
        [
            {
                "source_episode_key": "episode_c",
                "robot_distance_total": 1.0,
                "agentview_feature_distance": 0.5,
                "wrist_feature_distance": 0.5,
            },
            {
                "source_episode_key": "episode_b",
                "robot_distance_total": 0.5,
                "agentview_feature_distance": 0.7,
                "wrist_feature_distance": 0.4,
            },
            {
                "source_episode_key": "episode_a",
                "robot_distance_total": 0.5,
                "agentview_feature_distance": 0.2,
                "wrist_feature_distance": 0.9,
            },
        ]
    )
    assert [row["source_episode_key"] for row in sorted_rows] == ["episode_a", "episode_b", "episode_c"]


def test_window_search_script_includes_recovery_builder_and_asset_flags():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "build_single_state_rlds_recovery_asset.py" in contents
    assert '--recovery_asset_path "${RECOVERY_ASSET_PATH_VALUE}"' in contents
    assert 'USE_RLDS_RECOVERY_VALUE' in contents


def test_unwrap_object_env_prefers_inner_domain_env_over_outer_wrapper():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_object_env")

    class InnerEnv:
        def __init__(self):
            self.sim = object()
            self.robots = []

        def get_object(self, name):
            return name

    class OuterWrapper:
        def __init__(self, inner):
            self.sim = object()
            self.robots = []
            self.env = inner

    inner = InnerEnv()
    outer = OuterWrapper(inner)
    resolved = utils._unwrap_object_env(outer)
    assert resolved is inner


def test_collect_object_specs_returns_empty_with_reason_when_object_env_is_missing():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_collect_missing")

    class WrapperOnlyEnv:
        def __init__(self):
            self.sim = object()
            self.robots = []

    specs, reason = utils._collect_object_specs(WrapperOnlyEnv())
    assert specs == []
    assert "get_object" in str(reason)


def test_collect_object_specs_filters_invalid_entries_without_crashing():
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_collect_filtering")

    class FakeObject:
        def __init__(self, joints):
            self.joints = joints

    class FakeModel:
        def __init__(self):
            self.addr_map = {
                "good_joint": (0, 6),
                "scalar_joint": 7,
            }

        def get_joint_qpos_addr(self, joint_name):
            if joint_name not in self.addr_map:
                raise KeyError(joint_name)
            return self.addr_map[joint_name]

    class FakeSim:
        def __init__(self):
            self.model = FakeModel()

    class InnerEnv:
        def __init__(self):
            self.sim = FakeSim()
            self.obj_of_interest = [None, "good", "good", "boom", "missing"]
            self.objects_dict = {"scalar": object()}
            self.fixtures_dict = {"fixture": object()}

        def get_object(self, name):
            if name == "boom":
                raise RuntimeError("bad object lookup")
            mapping = {
                "good": FakeObject(["good_joint"]),
                "scalar": FakeObject(["scalar_joint"]),
                "fixture": FakeObject([]),
            }
            return mapping.get(name)

    class OuterWrapper:
        def __init__(self, inner):
            self.sim = object()
            self.robots = []
            self.env = inner

    specs, reason = utils._collect_object_specs(OuterWrapper(InnerEnv()))
    assert reason is None
    assert [spec["name"] for spec in specs] == ["good", "scalar"]
    assert [spec["kind"] for spec in specs] == ["free_joint", "scalar_joint"]
    assert [spec["joint_name"] for spec in specs] == ["good_joint", "scalar_joint"]
    assert [(spec["q_slice"].start, spec["q_slice"].stop) for spec in specs] == [(0, 7), (7, 8)]


def test_run_local_alignment_search_skips_object_search_when_specs_are_unavailable(monkeypatch):
    utils = _load_module(UTILS_PATH, "test_rlds_recovery_utils_local_search_skip")

    class FakeEnv:
        def get_sim_state(self):
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    baseline_eval = {
        "robot_metrics": {"total": 0.0},
        "image_metrics": {"agentview_l1": 0.0},
        "alignment_score": 1.25,
        "env_views": {"agentview": None, "wrist": None},
    }

    monkeypatch.setattr(utils, "_evaluate_alignment_state", lambda **kwargs: dict(baseline_eval))
    monkeypatch.setattr(utils, "_collect_object_specs", lambda env: ([], "no_object_env"))

    obs = {"marker": "obs"}
    resolved_obs, payload = utils.run_local_alignment_search(
        env=FakeEnv(),
        obs=obs,
        reference_robot_state={},
        reference_views={},
        feature_extractor=None,
        resize_size=64,
    )

    assert resolved_obs is obs
    assert np.allclose(payload["sim_state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert payload["best_eval"]["alignment_score"] == pytest.approx(1.25)
    assert payload["trace"]["object_search_skipped"] is True
    assert payload["trace"]["skip_reason"] == "no_object_env"
    assert payload["trace"]["accepted_object_updates"] == []
