"""Helpers for RLDS-assisted single-state recovery assets.

This module keeps the heavy runtime logic for:
1. selecting the best RLDS episode for a specific LIBERO task/init_state,
2. restoring robot joint / gripper state from RLDS,
3. building aligned simulator-state anchors for vulnerability-window search.

Most pure helpers are intentionally kept dependency-light so unit tests can
exercise them without TensorFlow / MuJoCo.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from vulnerability_window_utils import enumerate_window_starts  # noqa: E402


DEFAULT_RLDS_ROOT = "/home/yxx/roboticAttack/openvla-main/dataset"
SUITE_TO_RLDS_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
    "libero_90": "libero_90",
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if torch.is_tensor(value):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(dict(payload)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def normalize_instruction_key(text: Any) -> str:
    normalized = str(text).replace("\n", " ").strip().lower()
    normalized = " ".join(normalized.split())
    while normalized.endswith(".") or normalized.endswith("?"):
        normalized = normalized[:-1].rstrip()
    return normalized


def resolve_task_suite_name(dataset: str, task_suite_name: str) -> str:
    requested = normalize_instruction_key(task_suite_name)
    if requested not in ("", "auto", "none", "null"):
        return str(task_suite_name)
    dataset_key = normalize_instruction_key(dataset)
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if suite_name in dataset_key:
            return suite_name
    return "libero_spatial"


def resolve_rlds_dataset_name(dataset: str, task_suite_name: str = "auto") -> str:
    dataset_key = normalize_instruction_key(dataset)
    if dataset_key.endswith("_no_noops"):
        return str(dataset).strip()
    suite_name = resolve_task_suite_name(dataset=dataset, task_suite_name=task_suite_name)
    if suite_name not in SUITE_TO_RLDS_DATASET:
        raise ValueError(f"Unsupported LIBERO suite for RLDS recovery: `{dataset}` / `{task_suite_name}`.")
    return SUITE_TO_RLDS_DATASET[suite_name]


def build_recovery_anchor_steps(
    num_steps: int,
    window_stride: int,
    phase_boundaries: Mapping[str, int | None] | None = None,
) -> list[int]:
    anchors = set(enumerate_window_starts(num_gt_steps=int(num_steps), window_stride=int(window_stride)))
    anchors.add(0)
    for value in (phase_boundaries or {}).values():
        if value is None:
            continue
        boundary = int(value)
        if 0 <= boundary < int(num_steps):
            anchors.add(boundary)
    return sorted(anchors)


def resolve_recovery_anchor_step(requested_step: int, available_steps: Iterable[int]) -> tuple[int | None, bool]:
    requested = max(0, int(requested_step))
    unique_steps = sorted(set(int(step) for step in available_steps if int(step) <= requested))
    if not unique_steps:
        return None, False
    exact = requested in unique_steps
    return int(unique_steps[-1]), bool(exact)


def compute_vector_l2(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float32).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Mismatched vector shapes: {left_arr.shape} vs {right_arr.shape}")
    return float(np.linalg.norm(left_arr - right_arr))


def compute_robot_state_distance(reference: Mapping[str, Sequence[float]], candidate: Mapping[str, Sequence[float]]) -> dict[str, float]:
    joint_l2 = compute_vector_l2(reference["joint_state"], candidate["joint_state"])
    gripper_l2 = compute_vector_l2(reference["gripper_state"], candidate["gripper_state"])
    eef_l2 = compute_vector_l2(reference["eef_state"], candidate["eef_state"])
    return {
        "joint_l2": float(joint_l2),
        "gripper_l2": float(gripper_l2),
        "eef_l2": float(eef_l2),
        "total": float(joint_l2 + gripper_l2 + eef_l2),
    }


def candidate_sort_key(summary: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(summary.get("robot_distance_total", float("inf"))),
        float(summary.get("agentview_feature_distance", float("inf"))),
        float(summary.get("wrist_feature_distance", float("inf"))),
        str(summary.get("source_episode_key", "")),
    )


def sort_candidate_summaries(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(candidate) for candidate in sorted(candidates, key=candidate_sort_key)]


def qpos_addr_to_slice(addr: Any) -> slice:
    if isinstance(addr, tuple):
        if len(addr) == 2:
            return slice(int(addr[0]), int(addr[1]) + 1)
        if len(addr) == 1:
            return slice(int(addr[0]), int(addr[0]) + 1)
    return slice(int(addr), int(addr) + 1)


def _unwrap_env(env: object, max_depth: int = 8) -> object:
    current = env
    visited: set[int] = set()
    for _ in range(max(1, int(max_depth))):
        if current is None:
            break
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)
        if hasattr(current, "robots") and hasattr(current, "sim"):
            return current
        current = getattr(current, "env", None)
    raise AttributeError("Could not unwrap LIBERO environment to an object exposing `.robots` and `.sim`.")


def _unwrap_object_env(env: object, max_depth: int = 8) -> object:
    current = env
    visited: set[int] = set()
    fallback = None
    for _ in range(max(1, int(max_depth))):
        if current is None:
            break
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)
        if hasattr(current, "sim"):
            fallback = current
        if hasattr(current, "sim") and hasattr(current, "get_object"):
            return current
        current = getattr(current, "env", None)

    if fallback is not None:
        raise AttributeError(
            "Could not unwrap LIBERO environment to an object exposing `.sim` and `.get_object()`; "
            f"closest candidate type was `{type(fallback).__name__}`."
        )
    raise AttributeError("Could not unwrap LIBERO environment to an object exposing `.sim` and `.get_object()`.")


def extract_robot_state_from_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    from experiments.robot.libero.libero_utils import quat2axisangle

    joint_state = np.asarray(obs.get("robot0_joint_pos", np.zeros((7,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    gripper_state = np.asarray(
        obs.get("robot0_gripper_qpos", np.zeros((2,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)), dtype=np.float32).reshape(-1)
    eef_state = np.concatenate([eef_pos, quat2axisangle(eef_quat.copy())], axis=0).astype(np.float32)
    return {
        "joint_state": joint_state.astype(np.float32),
        "gripper_state": gripper_state.astype(np.float32),
        "eef_state": eef_state.astype(np.float32),
    }


def apply_robot_state_writeback(
    env: object,
    joint_state: Sequence[float],
    gripper_state: Sequence[float],
) -> dict[str, list[list[float]]]:
    inner_env = _unwrap_env(env)
    robot = inner_env.robots[0]
    joint_values = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    gripper_values = np.asarray(gripper_state, dtype=np.float32).reshape(-1)

    if joint_values.shape[0] != len(robot.robot_model.joints):
        raise ValueError(
            f"Expected {len(robot.robot_model.joints)} arm joints, got {joint_values.shape[0]}"
        )

    gripper_joint_names = list(getattr(robot.gripper, "joints", []))
    if gripper_values.shape[0] == 1 and len(gripper_joint_names) == 2:
        gripper_values = np.repeat(gripper_values, 2)
    if gripper_joint_names and gripper_values.shape[0] != len(gripper_joint_names):
        raise ValueError(
            f"Expected {len(gripper_joint_names)} gripper joints, got {gripper_values.shape[0]}"
        )

    arm_writes = []
    for joint_name, joint_value in zip(robot.robot_model.joints, joint_values):
        inner_env.sim.data.set_joint_qpos(joint_name, float(joint_value))
        arm_writes.append([joint_name, float(joint_value)])

    gripper_writes = []
    for joint_name, joint_value in zip(gripper_joint_names, gripper_values):
        inner_env.sim.data.set_joint_qpos(joint_name, float(joint_value))
        gripper_writes.append([joint_name, float(joint_value)])

    inner_env.sim.forward()
    if hasattr(env, "check_success"):
        env.check_success()
    if hasattr(env, "_post_process"):
        env._post_process()
    if hasattr(env, "_update_observables"):
        env._update_observables(force=True)
    return {
        "arm_joint_writes": arm_writes,
        "gripper_joint_writes": gripper_writes,
    }


def refresh_env_observation(env: object) -> Mapping[str, Any]:
    inner_env = getattr(env, "env", env)
    if hasattr(inner_env, "_get_observations"):
        return inner_env._get_observations()
    raise AttributeError("Environment does not expose `_get_observations`; cannot refresh observation.")


def resolve_torch_device(device: str | int | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    text = str(device).strip().lower()
    if text in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text.isdigit():
        idx = int(text)
        if idx < 0 or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(f"cuda:{idx}")
    if text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(device))


def resize_uint8_image(image: np.ndarray, resize_size: int | tuple[int, int]) -> np.ndarray:
    from experiments.robot.libero.libero_utils import resize_image

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    return resize_image(np.asarray(image, dtype=np.uint8), resize_size)


def extract_env_view_images(env_obs: Mapping[str, Any], resize_size: int | tuple[int, int]) -> dict[str, np.ndarray | None]:
    from experiments.robot.libero.libero_utils import get_libero_image

    if isinstance(resize_size, int):
        resize_tuple = (resize_size, resize_size)
    else:
        resize_tuple = tuple(resize_size)

    agentview = get_libero_image(env_obs, resize_tuple)
    wrist_raw = env_obs.get("robot0_eye_in_hand_image")
    wrist = None
    if wrist_raw is not None:
        wrist = resize_uint8_image(np.asarray(wrist_raw, dtype=np.uint8), resize_tuple)
    return {"agentview": np.asarray(agentview, dtype=np.uint8), "wrist": None if wrist is None else np.asarray(wrist, dtype=np.uint8)}


def pack_candidate_step0(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    eef_gripper = np.asarray(record["eef_gripper_state"], dtype=np.float32)
    joint_state = np.asarray(record["joint_state"], dtype=np.float32)
    return {
        "joint_state": joint_state[0].astype(np.float32),
        "gripper_state": eef_gripper[0, -2:].astype(np.float32),
        "eef_state": eef_gripper[0, :6].astype(np.float32),
    }


def mean_abs_image_error(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Image shapes must match, got {left_arr.shape} and {right_arr.shape}")
    return float(np.mean(np.abs(left_arr - right_arr)) / 255.0)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float32).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Feature shapes must match, got {left_arr.shape} and {right_arr.shape}")
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    cosine = float(np.dot(left_arr, right_arr) / (left_norm * right_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(1.0 - cosine)


class VisionFeatureExtractor:
    def __init__(
        self,
        backbone_id: str,
        image_resize_strategy: str,
        device: torch.device,
        fallback_image_size: int = 64,
    ) -> None:
        self.backbone_id = str(backbone_id)
        self.image_resize_strategy = str(image_resize_strategy)
        self.device = device
        self.fallback_image_size = int(fallback_image_size)
        self.mode = "backbone"
        self._backbone = None
        self._image_transform = None
        self._warning = None
        if str(self.backbone_id).strip().lower() in ("", "none", "pixel"):
            self.mode = "pixel"
            return
        try:
            from prismatic.models.materialize import get_vision_backbone_and_transform

            backbone, image_transform = get_vision_backbone_and_transform(
                vision_backbone_id=str(self.backbone_id),
                image_resize_strategy=str(self.image_resize_strategy),
            )
            backbone = backbone.to(self.device)
            backbone.eval()
            self._backbone = backbone
            self._image_transform = image_transform
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.mode = "pixel"
            self._warning = f"Falling back to pixel features because `{self.backbone_id}` failed to load: {exc}"

    @property
    def warning(self) -> str | None:
        return self._warning

    @property
    def backend_name(self) -> str:
        if self.mode == "pixel":
            return "pixel_fallback"
        return str(self.backbone_id)

    def _encode_pixel(self, image: np.ndarray) -> np.ndarray:
        resized = resize_uint8_image(np.asarray(image, dtype=np.uint8), (self.fallback_image_size, self.fallback_image_size))
        feature = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
        norm = float(np.linalg.norm(feature))
        if norm > 0.0:
            feature = feature / norm
        return feature.astype(np.float32)

    def encode(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        if self.mode == "pixel":
            return self._encode_pixel(image)

        from PIL import Image

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        transformed = self._image_transform(pil_image)
        with torch.no_grad():
            if isinstance(transformed, dict):
                batch = {key: value.unsqueeze(0).to(self.device) for key, value in transformed.items()}
                features = self._backbone(batch)
            else:
                features = self._backbone(transformed.unsqueeze(0).to(self.device))
        pooled = features.mean(dim=1).detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(pooled))
        if norm > 0.0:
            pooled = pooled / norm
        return pooled


def compute_image_alignment_metrics(
    env_views: Mapping[str, np.ndarray | None],
    ref_views: Mapping[str, np.ndarray | None],
    feature_extractor: VisionFeatureExtractor,
) -> dict[str, float | None]:
    env_agent = env_views.get("agentview")
    ref_agent = ref_views.get("agentview")
    env_wrist = env_views.get("wrist")
    ref_wrist = ref_views.get("wrist")

    agent_feat_dist = None
    wrist_feat_dist = None
    agent_feat_env = feature_extractor.encode(env_agent)
    agent_feat_ref = feature_extractor.encode(ref_agent)
    if agent_feat_env is not None and agent_feat_ref is not None:
        agent_feat_dist = cosine_distance(agent_feat_env, agent_feat_ref)

    wrist_feat_env = feature_extractor.encode(env_wrist)
    wrist_feat_ref = feature_extractor.encode(ref_wrist)
    if wrist_feat_env is not None and wrist_feat_ref is not None:
        wrist_feat_dist = cosine_distance(wrist_feat_env, wrist_feat_ref)

    agent_l1 = mean_abs_image_error(env_agent, ref_agent)
    wrist_l1 = mean_abs_image_error(env_wrist, ref_wrist)
    return {
        "agentview_feature_distance": agent_feat_dist,
        "wrist_feature_distance": wrist_feat_dist,
        "agentview_l1": agent_l1,
        "wrist_l1": wrist_l1,
    }


def _optional_float(value: Any) -> float:
    if value is None:
        return 0.0
    scalar = float(value)
    if math.isnan(scalar) or math.isinf(scalar):
        return 0.0
    return scalar


def total_alignment_score(
    robot_metrics: Mapping[str, float],
    image_metrics: Mapping[str, float | None],
    regularizer: float = 0.0,
) -> float:
    return float(
        robot_metrics["total"]
        + (0.5 * _optional_float(image_metrics.get("agentview_feature_distance")))
        + (0.25 * _optional_float(image_metrics.get("wrist_feature_distance")))
        + (0.5 * _optional_float(image_metrics.get("agentview_l1")))
        + (0.25 * _optional_float(image_metrics.get("wrist_l1")))
        + (0.05 * float(regularizer))
    )


def _phase_boundaries_for_episode(phases_df, source_episode_key: str) -> dict[str, int]:
    episode_df = phases_df.loc[phases_df["episode_key"] == source_episode_key]
    if len(episode_df) == 0:
        return {}
    boundaries = {}
    for phase_name in ("contact_manipulate", "post_contact"):
        phase_rows = episode_df.loc[episode_df["phase"] == phase_name]
        if len(phase_rows) == 0:
            continue
        boundaries[phase_name] = int(phase_rows["phase_start_t"].iloc[0])
    return boundaries


def _load_sidecar_episode(steps_df, source_episode_key: str) -> dict[str, np.ndarray]:
    episode_df = steps_df.loc[steps_df["episode_key"] == source_episode_key].sort_values("t")
    if len(episode_df) == 0:
        raise RuntimeError(f"No sidecar rows found for source_episode_key `{source_episode_key}`.")
    return {
        "raw_actions": np.stack(episode_df["raw_action"].tolist(), axis=0).astype(np.float32),
        "normalized_actions": np.stack(episode_df["normalized_action"].tolist(), axis=0).astype(np.float32),
        "eef_states": np.stack(episode_df["eef_state"].tolist(), axis=0).astype(np.float32),
        "gripper_states": np.stack(episode_df["gripper_state"].tolist(), axis=0).astype(np.float32),
        "joint_states": np.stack(episode_df["joint_state"].tolist(), axis=0).astype(np.float32),
    }


def _validate_sidecar_rlds_episode_pair(
    sidecar_episode: Mapping[str, np.ndarray],
    rlds_record: Mapping[str, Any],
    source_episode_key: str,
) -> None:
    sidecar_raw_actions = np.asarray(sidecar_episode["raw_actions"], dtype=np.float32)
    rlds_actions = np.asarray(rlds_record["actions"], dtype=np.float32)
    if sidecar_raw_actions.shape != rlds_actions.shape:
        raise RuntimeError(
            f"Sidecar / RLDS action shape mismatch for `{source_episode_key}`: "
            f"{sidecar_raw_actions.shape} vs {rlds_actions.shape}"
        )
    if not np.allclose(sidecar_raw_actions, rlds_actions):
        max_abs_diff = float(np.max(np.abs(sidecar_raw_actions - rlds_actions)))
        raise RuntimeError(
            f"Sidecar / RLDS raw actions differ for `{source_episode_key}`; "
            f"max_abs_diff={max_abs_diff:.8f}"
        )

    rlds_joint_state = np.asarray(rlds_record["joint_state"], dtype=np.float32)
    rlds_eef_gripper_state = np.asarray(rlds_record["eef_gripper_state"], dtype=np.float32)

    sidecar_step0_joint = np.asarray(sidecar_episode["joint_states"][0], dtype=np.float32)
    sidecar_step0_eef = np.asarray(sidecar_episode["eef_states"][0], dtype=np.float32)
    sidecar_step0_gripper = np.asarray(sidecar_episode["gripper_states"][0], dtype=np.float32)

    rlds_step0_joint = np.asarray(rlds_joint_state[0], dtype=np.float32)
    rlds_step0_eef = np.asarray(rlds_eef_gripper_state[0, :6], dtype=np.float32)
    rlds_step0_gripper = np.asarray(rlds_eef_gripper_state[0, -2:], dtype=np.float32)

    checks = (
        ("joint_state", sidecar_step0_joint, rlds_step0_joint),
        ("eef_state", sidecar_step0_eef, rlds_step0_eef),
        ("gripper_state", sidecar_step0_gripper, rlds_step0_gripper),
    )
    for field_name, sidecar_value, rlds_value in checks:
        if sidecar_value.shape != rlds_value.shape:
            raise RuntimeError(
                f"Sidecar / RLDS step-0 `{field_name}` shape mismatch for `{source_episode_key}`: "
                f"{sidecar_value.shape} vs {rlds_value.shape}"
            )
        if not np.allclose(sidecar_value, rlds_value):
            max_abs_diff = float(np.max(np.abs(sidecar_value - rlds_value)))
            raise RuntimeError(
                f"Sidecar / RLDS step-0 `{field_name}` differs for `{source_episode_key}`; "
                f"max_abs_diff={max_abs_diff:.8f}"
            )


def _collect_object_specs(env: object) -> tuple[list[dict[str, Any]], str | None]:
    import robosuite.utils.transform_utils as T

    del T  # silence lint for runtime-only import use in yaw helper below.

    try:
        object_env = _unwrap_object_env(env)
    except AttributeError as exc:
        return [], str(exc)

    object_names = []
    seen_names: set[str] = set()
    for candidate in list(getattr(object_env, "obj_of_interest", [])):
        if candidate is None:
            continue
        name = str(candidate)
        if name in seen_names:
            continue
        seen_names.add(name)
        object_names.append(name)
    for candidate in list(getattr(object_env, "objects_dict", {}).keys()):
        if candidate is None:
            continue
        name = str(candidate)
        if name in seen_names:
            continue
        seen_names.add(name)
        object_names.append(name)
    for candidate in list(getattr(object_env, "fixtures_dict", {}).keys()):
        if candidate is None:
            continue
        name = str(candidate)
        if name in seen_names:
            continue
        seen_names.add(name)
        object_names.append(name)

    specs = []
    for name in object_names:
        try:
            obj = object_env.get_object(name)
        except Exception:
            continue
        if obj is None or not hasattr(obj, "joints") or len(obj.joints) == 0:
            continue
        joint_name = obj.joints[-1]
        if joint_name is None:
            continue
        try:
            q_slice = qpos_addr_to_slice(object_env.sim.model.get_joint_qpos_addr(joint_name))
        except Exception:
            continue
        q_dim = int(q_slice.stop - q_slice.start)
        if q_dim == 7:
            specs.append({"name": str(name), "kind": "free_joint", "joint_name": str(joint_name), "q_slice": q_slice})
        elif q_dim == 1:
            specs.append({"name": str(name), "kind": "scalar_joint", "joint_name": str(joint_name), "q_slice": q_slice})
    return specs, None


def _read_joint_qpos(env: object, joint_name: str) -> np.ndarray:
    inner_env = _unwrap_env(env)
    q_slice = qpos_addr_to_slice(inner_env.sim.model.get_joint_qpos_addr(joint_name))
    return np.asarray(inner_env.sim.data.qpos[q_slice], dtype=np.float32).copy()


def _write_joint_qpos(env: object, joint_name: str, qpos: Sequence[float]) -> None:
    inner_env = _unwrap_env(env)
    values = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if values.shape[0] == 1:
        inner_env.sim.data.set_joint_qpos(joint_name, float(values[0]))
    else:
        inner_env.sim.data.set_joint_qpos(joint_name, values.astype(np.float32))
    inner_env.sim.forward()


def _apply_yaw_to_free_joint(qpos: np.ndarray, delta_degrees: float) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    updated = np.asarray(qpos, dtype=np.float32).copy()
    quat_wxyz = updated[3:7]
    quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
    euler = np.asarray(T.mat2euler(T.quat2mat(quat_xyzw)), dtype=np.float32)
    euler[2] += np.deg2rad(float(delta_degrees))
    quat_xyzw_new = np.asarray(T.mat2quat(T.euler2mat(euler)), dtype=np.float32)
    updated[3:7] = np.asarray(T.convert_quat(quat_xyzw_new, to="wxyz"), dtype=np.float32)
    return updated


def _evaluate_alignment_state(
    env: object,
    obs: Mapping[str, Any],
    reference_robot_state: Mapping[str, Sequence[float]],
    reference_views: Mapping[str, np.ndarray | None],
    feature_extractor: VisionFeatureExtractor,
    resize_size: int,
    regularizer: float = 0.0,
) -> dict[str, Any]:
    current_robot_state = extract_robot_state_from_obs(obs)
    robot_metrics = compute_robot_state_distance(current_robot_state, reference_robot_state)
    env_views = extract_env_view_images(obs, resize_size=resize_size)
    image_metrics = compute_image_alignment_metrics(env_views=env_views, ref_views=reference_views, feature_extractor=feature_extractor)
    total = total_alignment_score(robot_metrics=robot_metrics, image_metrics=image_metrics, regularizer=regularizer)
    return {
        "robot_metrics": to_jsonable(robot_metrics),
        "image_metrics": to_jsonable(image_metrics),
        "alignment_score": float(total),
        "env_views": env_views,
    }


def run_local_alignment_search(
    env: object,
    obs: Mapping[str, Any],
    reference_robot_state: Mapping[str, Sequence[float]],
    reference_views: Mapping[str, np.ndarray | None],
    feature_extractor: VisionFeatureExtractor,
    resize_size: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    base_sim_state = np.asarray(env.get_sim_state(), dtype=np.float32).copy()
    best_eval = _evaluate_alignment_state(
        env=env,
        obs=obs,
        reference_robot_state=reference_robot_state,
        reference_views=reference_views,
        feature_extractor=feature_extractor,
        resize_size=resize_size,
    )
    initial_alignment_score = float(best_eval["alignment_score"])
    object_specs, object_spec_error = _collect_object_specs(env)
    if object_spec_error is not None or len(object_specs) == 0:
        skip_reason = object_spec_error or "no_object_specs_found"
        trace = {
            "initial_alignment_score": float(initial_alignment_score),
            "final_alignment_score": float(best_eval["alignment_score"]),
            "accepted_object_updates": [],
            "object_search_skipped": True,
            "skip_reason": str(skip_reason),
            "searched_object_count": int(len(object_specs)),
        }
        return obs, {"sim_state": base_sim_state, "best_eval": best_eval, "trace": trace}

    accepted_updates: list[dict[str, Any]] = []

    xy_steps = (0.02, 0.01, 0.005)
    yaw_steps = (15.0, 7.5, 3.0)
    scalar_steps = (0.10, 0.05, 0.02)

    for spec in object_specs:
        current_best_qpos = _read_joint_qpos(env, spec["joint_name"])
        object_updates = []

        if spec["kind"] == "free_joint":
            for xy_step, yaw_step in zip(xy_steps, yaw_steps):
                proposals = []
                for dx in (-xy_step, 0.0, xy_step):
                    for dy in (-xy_step, 0.0, xy_step):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        proposal = current_best_qpos.copy()
                        proposal[0] += dx
                        proposal[1] += dy
                        proposals.append(("xy", dx, dy, proposal))
                for delta_yaw in (-yaw_step, yaw_step):
                    proposals.append(("yaw", delta_yaw, 0.0, _apply_yaw_to_free_joint(current_best_qpos, delta_yaw)))

                improved = False
                for move_type, delta_a, delta_b, proposal_qpos in proposals:
                    _write_joint_qpos(env, spec["joint_name"], proposal_qpos)
                    refreshed_obs = refresh_env_observation(env)
                    regularizer = float(np.linalg.norm(proposal_qpos - current_best_qpos))
                    proposal_eval = _evaluate_alignment_state(
                        env=env,
                        obs=refreshed_obs,
                        reference_robot_state=reference_robot_state,
                        reference_views=reference_views,
                        feature_extractor=feature_extractor,
                        resize_size=resize_size,
                        regularizer=regularizer,
                    )
                    if float(proposal_eval["alignment_score"]) + 1e-6 < float(best_eval["alignment_score"]):
                        best_eval = proposal_eval
                        current_best_qpos = proposal_qpos.copy()
                        obs = refreshed_obs
                        object_updates.append(
                            {
                                "move_type": move_type,
                                "delta_a": float(delta_a),
                                "delta_b": float(delta_b),
                                "alignment_score": float(best_eval["alignment_score"]),
                            }
                        )
                        improved = True
                    else:
                        _write_joint_qpos(env, spec["joint_name"], current_best_qpos)
                if not improved:
                    _write_joint_qpos(env, spec["joint_name"], current_best_qpos)

        elif spec["kind"] == "scalar_joint":
            low_value = float(current_best_qpos[0])
            for step_size in scalar_steps:
                improved = False
                for delta in (-step_size, step_size):
                    proposal_qpos = np.asarray([low_value + float(delta)], dtype=np.float32)
                    _write_joint_qpos(env, spec["joint_name"], proposal_qpos)
                    refreshed_obs = refresh_env_observation(env)
                    regularizer = float(np.linalg.norm(proposal_qpos - current_best_qpos))
                    proposal_eval = _evaluate_alignment_state(
                        env=env,
                        obs=refreshed_obs,
                        reference_robot_state=reference_robot_state,
                        reference_views=reference_views,
                        feature_extractor=feature_extractor,
                        resize_size=resize_size,
                        regularizer=regularizer,
                    )
                    if float(proposal_eval["alignment_score"]) + 1e-6 < float(best_eval["alignment_score"]):
                        best_eval = proposal_eval
                        current_best_qpos = proposal_qpos.copy()
                        low_value = float(proposal_qpos[0])
                        obs = refreshed_obs
                        object_updates.append(
                            {
                                "move_type": "scalar_joint",
                                "delta_a": float(delta),
                                "delta_b": 0.0,
                                "alignment_score": float(best_eval["alignment_score"]),
                            }
                        )
                        improved = True
                    else:
                        _write_joint_qpos(env, spec["joint_name"], current_best_qpos)
                if not improved:
                    _write_joint_qpos(env, spec["joint_name"], current_best_qpos)

        if object_updates:
            accepted_updates.append({"object_name": spec["name"], "updates": object_updates})

    final_state = np.asarray(env.get_sim_state(), dtype=np.float32).copy()
    trace = {
        "initial_alignment_score": float(initial_alignment_score),
        "final_alignment_score": float(best_eval["alignment_score"]),
        "accepted_object_updates": accepted_updates,
        "object_search_skipped": False,
        "skip_reason": None,
        "searched_object_count": int(len(object_specs)),
    }
    return obs, {"sim_state": final_state, "best_eval": best_eval, "trace": trace}


def _load_runtime_dependencies():
    import pandas as pd

    from experiments.robot.libero.libero_sidecar_utils import canonical_instruction, make_episode_key
    from experiments.robot.libero.reconstruct_hdf5_from_rlds import _require_runtime_dependencies, iter_rlds_episodes

    h5py, np_module, tfds, transform_utils = _require_runtime_dependencies()
    del h5py, transform_utils
    return pd, np_module, tfds, canonical_instruction, make_episode_key, iter_rlds_episodes


def _render_env_initial_reference(
    env: object,
    initial_state: np.ndarray,
    num_steps_wait: int,
    resize_size: int,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, np.ndarray | None]]:
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    env.reset()
    obs = env.set_init_state(np.asarray(initial_state, dtype=np.float32).copy())
    dummy_action = get_libero_dummy_action("openvla")
    for _ in range(max(0, int(num_steps_wait))):
        obs, _reward, done, _info = env.step(dummy_action)
        if done:
            break
    robot_state = extract_robot_state_from_obs(obs)
    views = extract_env_view_images(obs, resize_size=resize_size)
    return obs, robot_state, views


def _iter_instruction_candidates(
    dataset_name: str,
    rlds_root: str,
    instruction_key: str,
    feature_extractor: VisionFeatureExtractor,
    env_initial_robot: Mapping[str, Sequence[float]],
    env_initial_views: Mapping[str, np.ndarray | None],
) -> list[dict[str, Any]]:
    pd, np_module, tfds, canonical_instruction, make_episode_key, iter_rlds_episodes = _load_runtime_dependencies()
    del pd
    summaries: list[dict[str, Any]] = []
    for record in iter_rlds_episodes(
        np_module=np_module,
        tfds=tfds,
        dataset_name=str(dataset_name),
        data_root=Path(rlds_root),
        split="train",
        include_images=True,
    ):
        if normalize_instruction_key(record["instruction"]) != instruction_key:
            continue
        source_episode_key = make_episode_key(
            str(dataset_name),
            int(record["rlds_episode_index"]),
            str(record["rlds_file_path"]),
            canonical_instruction(record["instruction"]),
        )
        candidate_robot = pack_candidate_step0(record)
        robot_metrics = compute_robot_state_distance(env_initial_robot, candidate_robot)
        candidate_views = {
            "agentview": resize_uint8_image(np.asarray(record["agentview_rgb"][0], dtype=np.uint8), env_initial_views["agentview"].shape[:2]),
            "wrist": None
            if record.get("eye_in_hand_rgb") is None
            else resize_uint8_image(np.asarray(record["eye_in_hand_rgb"][0], dtype=np.uint8), env_initial_views["agentview"].shape[:2]),
        }
        image_metrics = compute_image_alignment_metrics(
            env_views=env_initial_views,
            ref_views=candidate_views,
            feature_extractor=feature_extractor,
        )
        summaries.append(
            {
                "source_episode_key": str(source_episode_key),
                "rlds_episode_index": int(record["rlds_episode_index"]),
                "rlds_file_path": str(record["rlds_file_path"]),
                "instruction": str(record["instruction"]),
                "num_steps": int(record["num_steps"]),
                "robot_distance_total": float(robot_metrics["total"]),
                "robot_distance_joint_l2": float(robot_metrics["joint_l2"]),
                "robot_distance_gripper_l2": float(robot_metrics["gripper_l2"]),
                "robot_distance_eef_l2": float(robot_metrics["eef_l2"]),
                "agentview_feature_distance": image_metrics["agentview_feature_distance"],
                "wrist_feature_distance": image_metrics["wrist_feature_distance"],
                "agentview_l1": image_metrics["agentview_l1"],
                "wrist_l1": image_metrics["wrist_l1"],
            }
        )
    return sort_candidate_summaries(summaries)


def _load_selected_rlds_record(
    dataset_name: str,
    rlds_root: str,
    source_episode_key: str,
) -> dict[str, Any]:
    pd, np_module, tfds, canonical_instruction, make_episode_key, iter_rlds_episodes = _load_runtime_dependencies()
    del pd
    for record in iter_rlds_episodes(
        np_module=np_module,
        tfds=tfds,
        dataset_name=str(dataset_name),
        data_root=Path(rlds_root),
        split="train",
        include_images=True,
    ):
        current_key = make_episode_key(
            str(dataset_name),
            int(record["rlds_episode_index"]),
            str(record["rlds_file_path"]),
            canonical_instruction(record["instruction"]),
        )
        if current_key == source_episode_key:
            return record
    raise RuntimeError(f"Selected RLDS episode `{source_episode_key}` was not found under `{rlds_root}`.")


def build_single_state_recovery_asset(
    dataset: str,
    task_suite_name: str,
    task_id: int,
    init_state_idx: int,
    rlds_root: str,
    steps_parquet: str,
    phases_parquet: str,
    output_root: str,
    device: str | int | torch.device = "auto",
    num_steps_wait: int = 10,
    env_resolution: int = 256,
    window_stride: int = 8,
    recovery_vision_backbone: str = "dinoclip-vit-l-336px",
    recovery_image_resize_strategy: str = "resize-naive",
    force_rebuild: bool = False,
) -> dict[str, Any]:
    import pandas as pd

    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
    from libero.libero import benchmark

    output_dir = Path(output_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_asset_path = output_dir / "recovery_asset.json"
    matched_episode_cache_path = output_dir / "matched_episode_cache.npz"
    aligned_state_cache_path = output_dir / "aligned_state_cache.pt"
    alignment_trace_path = output_dir / "alignment_trace.json"

    if recovery_asset_path.exists() and matched_episode_cache_path.exists() and aligned_state_cache_path.exists() and not force_rebuild:
        with recovery_asset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["recovery_asset_path"] = str(recovery_asset_path)
        payload["matched_episode_cache_path"] = str(matched_episode_cache_path)
        payload["aligned_state_cache_path"] = str(aligned_state_cache_path)
        payload["alignment_trace_path"] = str(alignment_trace_path)
        return payload

    suite_name = resolve_task_suite_name(dataset=dataset, task_suite_name=task_suite_name)
    rlds_dataset_name = resolve_rlds_dataset_name(dataset=dataset, task_suite_name=suite_name)
    resolved_device = resolve_torch_device(device)
    feature_extractor = VisionFeatureExtractor(
        backbone_id=str(recovery_vision_backbone),
        image_resize_strategy=str(recovery_image_resize_strategy),
        device=resolved_device,
    )

    steps_df = pd.read_parquet(
        steps_parquet,
        columns=["episode_key", "t", "raw_action", "normalized_action", "eef_state", "gripper_state", "joint_state"],
    )
    phases_df = pd.read_parquet(
        phases_parquet,
        columns=["episode_key", "phase", "phase_start_t"],
    )

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(int(task_id))
    init_states = task_suite.get_task_init_states(int(task_id))
    if int(init_state_idx) < 0 or int(init_state_idx) >= len(init_states):
        raise ValueError(f"init_state_idx {init_state_idx} is out of range [0, {len(init_states)})")
    initial_state = np.asarray(init_states[int(init_state_idx)], dtype=np.float32).copy()
    env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
    dummy_action = get_libero_dummy_action("openvla")

    try:
        env_obs0, env_robot0, env_views0 = _render_env_initial_reference(
            env=env,
            initial_state=initial_state,
            num_steps_wait=int(num_steps_wait),
            resize_size=max(64, int(env_resolution)),
        )
        instruction_key = normalize_instruction_key(task_description)
        candidate_summaries = _iter_instruction_candidates(
            dataset_name=rlds_dataset_name,
            rlds_root=rlds_root,
            instruction_key=instruction_key,
            feature_extractor=feature_extractor,
            env_initial_robot=env_robot0,
            env_initial_views=env_views0,
        )
        if not candidate_summaries:
            raise RuntimeError(
                f"No RLDS candidates found for task_id={task_id}, instruction={task_description!r}, dataset={rlds_dataset_name}"
            )

        selected_summary = dict(candidate_summaries[0])
        source_episode_key = str(selected_summary["source_episode_key"])
        selected_rlds_record = _load_selected_rlds_record(
            dataset_name=rlds_dataset_name,
            rlds_root=rlds_root,
            source_episode_key=source_episode_key,
        )
        sidecar_episode = _load_sidecar_episode(steps_df=steps_df, source_episode_key=source_episode_key)
        _validate_sidecar_rlds_episode_pair(
            sidecar_episode=sidecar_episode,
            rlds_record=selected_rlds_record,
            source_episode_key=source_episode_key,
        )

        matched_episode_cache = {
            "raw_actions": sidecar_episode["raw_actions"],
            "normalized_actions": sidecar_episode["normalized_actions"],
            "eef_states": sidecar_episode["eef_states"],
            "gripper_states": sidecar_episode["gripper_states"],
            "joint_states": sidecar_episode["joint_states"],
            "agentview_rgb": np.asarray(selected_rlds_record["agentview_rgb"], dtype=np.uint8),
            "wrist_rgb": np.asarray(selected_rlds_record["eye_in_hand_rgb"], dtype=np.uint8),
        }
        np.savez_compressed(matched_episode_cache_path, **matched_episode_cache)

        phase_boundaries = _phase_boundaries_for_episode(phases_df=phases_df, source_episode_key=source_episode_key)
        anchor_steps = build_recovery_anchor_steps(
            num_steps=int(sidecar_episode["raw_actions"].shape[0]),
            window_stride=int(window_stride),
            phase_boundaries=phase_boundaries,
        )

        env.reset()
        obs = env.set_init_state(initial_state.copy())
        for _ in range(max(0, int(num_steps_wait))):
            obs, _reward, done_wait, _info = env.step(dummy_action)
            if done_wait:
                break

        anchor_states: dict[int, torch.Tensor] = {}
        anchor_records: dict[int, dict[str, Any]] = {}
        alignment_trace_records: list[dict[str, Any]] = []

        previous_anchor = 0
        for anchor_step in anchor_steps:
            while previous_anchor < int(anchor_step):
                action = np.asarray(sidecar_episode["raw_actions"][previous_anchor], dtype=np.float32)
                obs, _reward, done_replay, _info = env.step(action.tolist())
                previous_anchor += 1
                if done_replay:
                    break

            robot_writes = apply_robot_state_writeback(
                env=env,
                joint_state=sidecar_episode["joint_states"][anchor_step],
                gripper_state=sidecar_episode["gripper_states"][anchor_step],
            )
            obs = refresh_env_observation(env)
            reference_robot_state = {
                "joint_state": sidecar_episode["joint_states"][anchor_step],
                "gripper_state": sidecar_episode["gripper_states"][anchor_step],
                "eef_state": sidecar_episode["eef_states"][anchor_step],
            }
            reference_views = {
                "agentview": resize_uint8_image(matched_episode_cache["agentview_rgb"][anchor_step], max(64, int(env_resolution))),
                "wrist": resize_uint8_image(matched_episode_cache["wrist_rgb"][anchor_step], max(64, int(env_resolution))),
            }
            pre_eval = _evaluate_alignment_state(
                env=env,
                obs=obs,
                reference_robot_state=reference_robot_state,
                reference_views=reference_views,
                feature_extractor=feature_extractor,
                resize_size=max(64, int(env_resolution)),
            )
            obs, aligned_payload = run_local_alignment_search(
                env=env,
                obs=obs,
                reference_robot_state=reference_robot_state,
                reference_views=reference_views,
                feature_extractor=feature_extractor,
                resize_size=max(64, int(env_resolution)),
            )
            sim_state = np.asarray(aligned_payload["sim_state"], dtype=np.float32).copy()
            anchor_states[int(anchor_step)] = torch.from_numpy(sim_state.astype(np.float32))

            labels = []
            if int(anchor_step) == 0:
                labels.append("initial")
            if int(anchor_step) in set(int(step) for step in enumerate_window_starts(sidecar_episode["raw_actions"].shape[0], int(window_stride))):
                labels.append("window_anchor")
            for phase_name, phase_step in phase_boundaries.items():
                if int(phase_step) == int(anchor_step):
                    labels.append(f"phase:{phase_name}")
            anchor_record = {
                "step": int(anchor_step),
                "anchor_labels": labels,
                "robot_writes": robot_writes,
                "pre_alignment_score": float(pre_eval["alignment_score"]),
                "post_alignment_score": float(aligned_payload["best_eval"]["alignment_score"]),
                "robot_metrics": pre_eval["robot_metrics"],
                "image_metrics": pre_eval["image_metrics"],
                "post_robot_metrics": aligned_payload["best_eval"]["robot_metrics"],
                "post_image_metrics": aligned_payload["best_eval"]["image_metrics"],
                "accepted_object_updates": aligned_payload["trace"]["accepted_object_updates"],
                "object_search_skipped": bool(aligned_payload["trace"].get("object_search_skipped", False)),
                "object_search_skip_reason": aligned_payload["trace"].get("skip_reason"),
                "searched_object_count": int(aligned_payload["trace"].get("searched_object_count", 0)),
            }
            anchor_records[int(anchor_step)] = anchor_record
            alignment_trace_records.append(anchor_record)
            obs = refresh_env_observation(env)

        torch.save(
            {
                "schema_version": 1,
                "dataset": str(dataset),
                "task_suite_name": str(suite_name),
                "task_id": int(task_id),
                "init_state_idx": int(init_state_idx),
                "source_episode_key": str(source_episode_key),
                "step_states": anchor_states,
                "step_records": anchor_records,
                "phase_boundaries": phase_boundaries,
                "window_stride": int(window_stride),
                "num_steps_wait": int(num_steps_wait),
                "env_resolution": int(env_resolution),
            },
            aligned_state_cache_path,
        )

        alignment_trace = {
            "task_id": int(task_id),
            "init_state_idx": int(init_state_idx),
            "task_description": str(task_description),
            "source_episode_key": str(source_episode_key),
            "phase_boundaries": phase_boundaries,
            "candidate_summaries": candidate_summaries,
            "anchor_steps": anchor_steps,
            "anchor_records": alignment_trace_records,
        }
        write_json(alignment_trace_path, alignment_trace)

        recovery_asset = {
            "schema_version": 1,
            "dataset": str(dataset),
            "task_suite_name": str(suite_name),
            "task_id": int(task_id),
            "init_state_idx": int(init_state_idx),
            "task_description": str(task_description),
            "recovery_status": "matched",
            "source_episode_key": str(source_episode_key),
            "selected_rlds_episode_index": int(selected_summary["rlds_episode_index"]),
            "candidate_count": int(len(candidate_summaries)),
            "candidate_summaries_preview": candidate_summaries[: min(10, len(candidate_summaries))],
            "robot_init_residual": {
                "joint_l2": float(selected_summary["robot_distance_joint_l2"]),
                "gripper_l2": float(selected_summary["robot_distance_gripper_l2"]),
                "eef_l2": float(selected_summary["robot_distance_eef_l2"]),
                "total": float(selected_summary["robot_distance_total"]),
            },
            "image_alignment_metrics": {
                "agentview_feature_distance": selected_summary["agentview_feature_distance"],
                "wrist_feature_distance": selected_summary["wrist_feature_distance"],
                "agentview_l1": selected_summary["agentview_l1"],
                "wrist_l1": selected_summary["wrist_l1"],
            },
            "phase_boundaries": phase_boundaries,
            "anchor_steps": anchor_steps,
            "vision_backend": str(feature_extractor.backend_name),
            "vision_backend_warning": feature_extractor.warning,
            "matched_episode_cache_path": str(matched_episode_cache_path),
            "aligned_state_cache_path": str(aligned_state_cache_path),
            "alignment_trace_path": str(alignment_trace_path),
            "recovery_asset_path": str(recovery_asset_path),
            "rlds_root": str(Path(rlds_root).expanduser().resolve()),
            "steps_parquet": str(Path(steps_parquet).expanduser().resolve()),
            "phases_parquet": str(Path(phases_parquet).expanduser().resolve()),
            "num_steps_wait": int(num_steps_wait),
            "env_resolution": int(env_resolution),
            "window_stride": int(window_stride),
        }
        write_json(recovery_asset_path, recovery_asset)
        return recovery_asset
    finally:
        if hasattr(env, "close"):
            env.close()


def load_recovery_asset(path: str | os.PathLike[str]) -> dict[str, Any]:
    asset_path = Path(path).expanduser().resolve()
    with asset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["recovery_asset_path"] = str(asset_path)
    return payload


def load_matched_episode_cache(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as payload:
        return {str(key): np.asarray(payload[key]) for key in payload.files}
