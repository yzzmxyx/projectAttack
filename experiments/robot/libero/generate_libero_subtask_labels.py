"""
Generate precise LIBERO subtask timing labels from expert demonstrations.

This script replays raw LIBERO HDF5 demos in simulation, re-evaluates every goal
predicate at each rollout step, and exports sidecar label files that capture:

- per-predicate truth traces
- first-true / toggle event steps
- node-state transitions induced by the full predicate vector
- predicate-specific environment snapshots at event steps

It can also build a join manifest against the existing no-noops RLDS datasets so
later training jobs can look up labels by episode.

Example:
    /home/yxx/miniconda3/envs/robo_env/bin/python \
        /home/yxx/projectAttack/experiments/robot/libero/generate_libero_subtask_labels.py \
        --libero_task_suite libero_spatial \
        --libero_hdf5_data_dir /path/to/regenerated--no_noops/libero_spatial \
        --output_root /path/to/subtask_labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _require_runtime_dependencies():
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        from libero.libero import benchmark, get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "Missing LIBERO labeling dependencies. "
            "Run this script inside a LIBERO/OpenVLA environment such as "
            "`/home/yxx/miniconda3/envs/robo_env/bin/python`."
        ) from exc
    return h5py, np, benchmark, get_libero_path, OffScreenRenderEnv


SUITE_TO_RLDS_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

DEFAULT_RLDS_ROOT = Path("/home/yxx/roboticAttack/openvla-main/dataset")
SUPPORTED_SUITES = tuple(SUITE_TO_RLDS_DATASET.keys())
DEFAULT_LOCAL_HDF5_SEARCH_ROOTS = (
    Path("/home/yxx/roboticAttack/LIBERO/libero/datasets"),
    Path("/home/yxx/projectAttack/LIBERO/libero/datasets"),
    Path("/home/yxx/robot_ori/LIBERO/libero/datasets"),
    Path("/mnt/data2/yxx/roboticAttack/LIBERO/libero/datasets"),
    Path("/mnt/data2/yxx/projectAttack/LIBERO/libero/datasets"),
)


def is_noop(action, prev_action=None, threshold=1e-4):
    """Mirror the no-op filtering used by regenerate_libero_dataset.py."""
    if prev_action is None:
        return float((action[:-1] ** 2).sum()) ** 0.5 < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return float((action[:-1] ** 2).sum()) ** 0.5 < threshold and gripper_action == prev_gripper_action


def normalize_instruction(text: str) -> str:
    text = text.replace("\n", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".? ")


def predicate_key(state: Sequence[str]) -> str:
    predicate_name = str(state[0]).lower()
    arguments = ", ".join(str(x) for x in state[1:])
    return f"{predicate_name}({arguments})"


def predicate_record(state: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": str(state[0]).lower(),
        "arguments": [str(x) for x in state[1:]],
        "arity": max(0, len(state) - 1),
        "key": predicate_key(state),
    }


def node_state_key(truth_values: Sequence[bool]) -> str:
    return "".join("1" if value else "0" for value in truth_values)


def parse_task_ids(task_ids: Optional[str]) -> Optional[List[int]]:
    if not task_ids:
        return None
    values = []
    for item in task_ids.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def sha256_array(np_module, array) -> str:
    contiguous = np_module.ascontiguousarray(array, dtype=np_module.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def decode_maybe_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def make_env(OffScreenRenderEnv, get_libero_path, task, resolution: int):
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def get_dummy_action() -> List[float]:
    return [0, 0, 0, 0, 0, 0, -1]


def local_hdf5_dir_candidates(base_root: Path, suite_name: str) -> List[Path]:
    return [
        base_root,
        base_root / suite_name,
        base_root / f"{suite_name}_no_noops",
        base_root / "regenerated--no_noops" / suite_name,
        base_root / "LIBERO" / "libero" / "datasets" / suite_name,
        base_root / "LIBERO" / "libero" / "datasets" / f"{suite_name}_no_noops",
        base_root / "LIBERO" / "libero" / "datasets" / "regenerated--no_noops" / suite_name,
        base_root / "libero" / "datasets" / suite_name,
        base_root / "libero" / "datasets" / f"{suite_name}_no_noops",
        base_root / "libero" / "datasets" / "regenerated--no_noops" / suite_name,
    ]


def discover_hdf5_data_dir(
    suite_name: str,
    task_names: Sequence[str],
    requested_root: Optional[Path] = None,
) -> Optional[Path]:
    candidates = []
    if requested_root is not None:
        candidates.extend(local_hdf5_dir_candidates(requested_root, suite_name))
    for root in DEFAULT_LOCAL_HDF5_SEARCH_ROOTS:
        candidates.extend(local_hdf5_dir_candidates(root, suite_name))

    seen = set()
    scored_candidates = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        score = sum(int((candidate / f"{task_name}_demo.hdf5").exists()) for task_name in task_names)
        if score > 0:
            scored_candidates.append((score, candidate))

    if not scored_candidates:
        return None
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates[0][1]


def detect_hdf5_source_kind(demo_data) -> str:
    """Infer whether an HDF5 demo is original raw LIBERO or regenerated no-noops."""
    keys = set(demo_data.keys())
    if "obs" in keys and {"dones", "rewards", "robot_states"}.issubset(keys):
        return "regenerated"
    return "raw"


def extract_object_descriptor(obj_state) -> Dict[str, Any]:
    descriptor = {
        "name": getattr(obj_state, "object_name", None),
        "state_type": getattr(obj_state, "object_state_type", None),
        "parent_name": getattr(obj_state, "parent_name", None),
    }
    try:
        geom_state = obj_state.get_geom_state()
        descriptor["geom"] = {
            "pos": to_jsonable(geom_state.get("pos")),
            "quat": to_jsonable(geom_state.get("quat")),
        }
    except Exception:
        descriptor["geom"] = None

    try:
        descriptor["joint_state"] = to_jsonable(obj_state.get_joint_state())
    except Exception:
        descriptor["joint_state"] = None
    return descriptor


def snapshot_for_predicate(np_module, env, state: Sequence[str], result: Optional[bool] = None) -> Dict[str, Any]:
    record = predicate_record(state)
    predicate_name = record["name"]
    if result is None:
        result = bool(env._eval_predicate(state))
    snapshot: Dict[str, Any] = {
        "predicate": record,
        "result": bool(result),
    }

    if predicate_name in {"on", "in"} and len(state) == 3:
        object_state = env.object_states_dict[state[1]]
        target_state = env.object_states_dict[state[2]]
        object_desc = extract_object_descriptor(object_state)
        target_desc = extract_object_descriptor(target_state)
        object_pos = object_desc.get("geom", {}).get("pos") if object_desc.get("geom") else None
        target_pos = target_desc.get("geom", {}).get("pos") if target_desc.get("geom") else None
        xy_distance = None
        z_delta = None
        if object_pos is not None and target_pos is not None:
            object_pos_np = np_module.asarray(object_pos, dtype=np_module.float32)
            target_pos_np = np_module.asarray(target_pos, dtype=np_module.float32)
            xy_distance = float(np_module.linalg.norm(object_pos_np[:2] - target_pos_np[:2]))
            z_delta = float(object_pos_np[2] - target_pos_np[2])

        metrics = {
            "contact": bool(target_state.check_contact(object_state)),
            "xy_distance": xy_distance,
            "z_delta": z_delta,
        }
        if predicate_name == "on":
            metrics["ontop"] = bool(target_state.check_ontop(object_state))
        if predicate_name == "in":
            metrics["contain"] = bool(target_state.check_contain(object_state))
        snapshot.update({"object": object_desc, "target": target_desc, "metrics": metrics})
        return snapshot

    if predicate_name in {"open", "close", "turnon", "turnoff", "up"} and len(state) == 2:
        obj_state = env.object_states_dict[state[1]]
        obj_desc = extract_object_descriptor(obj_state)
        metrics = {
            "joint_state": obj_desc.get("joint_state"),
        }
        if predicate_name in {"open", "close"}:
            metrics["is_open"] = bool(obj_state.is_open())
            metrics["is_close"] = bool(obj_state.is_close())
        if predicate_name in {"turnon", "turnoff"}:
            metrics["turn_on"] = bool(obj_state.turn_on())
            metrics["turn_off"] = bool(obj_state.turn_off())
        if predicate_name == "up":
            geom = obj_desc.get("geom") or {}
            pos = geom.get("pos")
            metrics["z"] = float(pos[2]) if pos is not None else None
            metrics["threshold"] = 1.0
        snapshot.update({"object": obj_desc, "metrics": metrics})
        return snapshot

    snapshot["details"] = "Unsupported predicate snapshot shape; stored predicate result only."
    return snapshot


def evaluate_goal_truth(env, goal_states: Sequence[Sequence[str]]) -> List[bool]:
    return [bool(env._eval_predicate(state)) for state in goal_states]


def filter_actions(np_module, raw_actions) -> Dict[str, Any]:
    kept_actions = []
    kept_indices = []
    for raw_idx, action in enumerate(raw_actions):
        action_np = np_module.asarray(action, dtype=np_module.float32)
        prev_kept = kept_actions[-1] if kept_actions else None
        if is_noop(action_np, prev_kept):
            continue
        kept_actions.append(action_np)
        kept_indices.append(raw_idx)

    if kept_actions:
        filtered_actions = np_module.stack(kept_actions, axis=0)
    else:
        action_dim = int(raw_actions.shape[-1]) if len(raw_actions.shape) > 1 else 0
        filtered_actions = np_module.zeros((0, action_dim), dtype=np_module.float32)

    return {
        "actions": filtered_actions,
        "kept_raw_indices": kept_indices,
        "num_noops_filtered": int(raw_actions.shape[0] - filtered_actions.shape[0]),
    }


def prepare_source_actions(np_module, source_actions, source_kind: str) -> Dict[str, Any]:
    source_actions = np_module.asarray(source_actions, dtype=np_module.float32)
    if source_kind == "raw":
        prepared = filter_actions(np_module, source_actions)
        prepared["source_kind"] = source_kind
        return prepared
    if source_kind == "regenerated":
        prepared = {
            "actions": source_actions,
            "kept_raw_indices": list(range(int(source_actions.shape[0]))),
            "num_noops_filtered": 0,
            "source_kind": source_kind,
        }
        return prepared
    raise ValueError(f"Unsupported HDF5 source kind: {source_kind}")


def label_episode(
    np_module,
    env,
    goal_states: Sequence[Sequence[str]],
    filtered_actions,
    initial_state,
    num_steps_wait: int,
) -> Dict[str, Any]:
    env.reset()
    env.set_init_state(initial_state)
    obs = None
    reward = 0.0
    done = False
    info = {}

    dummy_action = get_dummy_action()
    for _ in range(num_steps_wait):
        obs, reward, done, info = env.step(dummy_action)

    initial_truth = evaluate_goal_truth(env, goal_states)
    predicate_entries = []
    total_predicate_events = 0

    node_events = [
        {
            "step": -1,
            "state_key": node_state_key(initial_truth),
            "truth": list(initial_truth),
            "event_types": ["initial_state", "first_enter"],
            "changed_predicates": [],
            "predicate_snapshots": {},
        }
    ]
    node_first_enter_step = {node_state_key(initial_truth): -1}

    for predicate_index, state in enumerate(goal_states):
        predicate_events = []
        if initial_truth[predicate_index]:
            predicate_events.append(
                {
                    "step": -1,
                    "truth": True,
                    "previous_truth": None,
                    "event_types": ["initial_true", "first_true"],
                    "snapshot": snapshot_for_predicate(np_module, env, state, result=True),
                }
            )
            total_predicate_events += 1

        predicate_entries.append(
            {
                "index": predicate_index,
                **predicate_record(state),
                "initial_truth": bool(initial_truth[predicate_index]),
                "truth": [],
                "first_true_step": -1 if initial_truth[predicate_index] else None,
                "toggle_steps": [],
                "event_records": predicate_events,
            }
        )

    previous_truth = list(initial_truth)
    first_success_step = -1 if all(initial_truth) else None

    for step_idx, action in enumerate(filtered_actions):
        obs, reward, done, info = env.step(action.tolist())
        current_truth = evaluate_goal_truth(env, goal_states)
        changed_predicate_keys = []
        predicate_snapshots = {}

        for predicate_index, state in enumerate(goal_states):
            current_value = bool(current_truth[predicate_index])
            predicate_entry = predicate_entries[predicate_index]
            predicate_entry["truth"].append(current_value)

            if current_value and predicate_entry["first_true_step"] is None:
                predicate_entry["first_true_step"] = step_idx

            if current_value != previous_truth[predicate_index]:
                predicate_entry["toggle_steps"].append(step_idx)
                event_types = ["toggle"]
                if current_value and predicate_entry["first_true_step"] == step_idx:
                    event_types.append("first_true")

                snapshot = snapshot_for_predicate(np_module, env, state, result=current_value)
                predicate_entry["event_records"].append(
                    {
                        "step": step_idx,
                        "truth": current_value,
                        "previous_truth": bool(previous_truth[predicate_index]),
                        "event_types": event_types,
                        "snapshot": snapshot,
                    }
                )
                total_predicate_events += 1
                predicate_snapshots[predicate_entry["key"]] = snapshot
                changed_predicate_keys.append(predicate_entry["key"])

        state_key = node_state_key(current_truth)
        if changed_predicate_keys:
            node_event_types = ["transition"]
            if state_key not in node_first_enter_step:
                node_first_enter_step[state_key] = step_idx
                node_event_types.append("first_enter")

            node_events.append(
                {
                    "step": step_idx,
                    "state_key": state_key,
                    "truth": list(bool(x) for x in current_truth),
                    "event_types": node_event_types,
                    "changed_predicates": changed_predicate_keys,
                    "predicate_snapshots": predicate_snapshots,
                }
            )

        if first_success_step is None and all(current_truth):
            first_success_step = step_idx

        previous_truth = list(bool(x) for x in current_truth)

    final_truth = list(previous_truth)
    final_all_true = bool(all(final_truth))
    success = bool(done)
    success_matches_goal_state = success == final_all_true

    if not success_matches_goal_state:
        raise RuntimeError(
            "Replay success mismatch: env.step(done) disagrees with explicit goal predicate evaluation."
        )

    empty_event_rollout = total_predicate_events == 0
    if len(filtered_actions) == 0:
        final_reward = 0.0
    else:
        final_reward = float(reward)

    return {
        "predicates": predicate_entries,
        "node_states": {
            "initial_state": list(initial_truth),
            "initial_state_key": node_state_key(initial_truth),
            "first_enter_step_by_key": node_first_enter_step,
            "events": node_events,
        },
        "rollout_summary": {
            "num_predicates": len(goal_states),
            "num_filtered_actions": int(len(filtered_actions)),
            "first_success_step": first_success_step,
            "final_truth": final_truth,
            "final_all_true": final_all_true,
            "success": success,
            "success_matches_goal_state": success_matches_goal_state,
            "empty_event_rollout": empty_event_rollout,
            "final_reward": final_reward,
        },
        "last_obs_present": obs is not None,
        "last_info": to_jsonable(info),
    }


def maybe_import_tfds():  # pragma: no cover - runtime-only helper
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "RLDS join requested, but tensorflow/tensorflow_datasets are unavailable. "
            "Run this script in `/home/yxx/miniconda3/envs/robo_env/bin/python` or pass `--skip_rlds_join`."
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    return tfds


def iter_rlds_episode_records(
    np_module,
    dataset_name: str,
    data_root: Path,
    split: str,
) -> Iterable[Dict[str, Any]]:
    tfds = maybe_import_tfds()
    builder = tfds.builder(dataset_name, data_dir=str(data_root))
    decoders = {
        "steps": {
            "observation": {
                "image": tfds.decode.SkipDecoding(),
                "wrist_image": tfds.decode.SkipDecoding(),
            }
        }
    }
    dataset = builder.as_dataset(split=split, shuffle_files=False, decoders=decoders)

    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = list(episode["steps"])
        if not steps:
            continue
        actions = np_module.stack(
            [np_module.asarray(step["action"], dtype=np_module.float32) for step in steps],
            axis=0,
        )
        instruction = decode_maybe_bytes(steps[0]["language_instruction"])
        file_path = decode_maybe_bytes(episode["episode_metadata"]["file_path"])
        yield {
            "rlds_split": split,
            "rlds_episode_index": episode_index,
            "rlds_file_path": file_path,
            "rlds_file_basename": os.path.basename(file_path),
            "instruction": instruction,
            "instruction_normalized": normalize_instruction(instruction),
            "num_steps": int(actions.shape[0]),
            "action_sha256": sha256_array(np_module, actions),
        }


def sample_rlds_source_hints(
    np_module,
    dataset_name: str,
    data_root: Path,
    split: str = "train",
    max_samples: int = 4,
) -> List[Dict[str, Any]]:
    hints = []
    for record in iter_rlds_episode_records(np_module, dataset_name, data_root, split):
        hints.append(
            {
                "rlds_file_path": record["rlds_file_path"],
                "rlds_source_dir": os.path.dirname(record["rlds_file_path"]),
                "rlds_file_basename": record["rlds_file_basename"],
                "instruction_normalized": record["instruction_normalized"],
                "num_steps": record["num_steps"],
            }
        )
        if len(hints) >= max_samples:
            break
    return hints


def build_rlds_join_manifest(
    np_module,
    label_manifest_entries: Sequence[Dict[str, Any]],
    dataset_name: str,
    rlds_data_root: Path,
    split: str = "train",
    require_full_rlds_coverage: bool = True,
) -> Dict[str, Any]:
    rlds_records = list(iter_rlds_episode_records(np_module, dataset_name, rlds_data_root, split))
    indexed_records = defaultdict(list)
    for record in rlds_records:
        key = (
            record["rlds_file_basename"],
            int(record["num_steps"]),
            record["action_sha256"],
        )
        indexed_records[key].append(record)

    matches = []
    unmatched_source_demo_keys = []
    matched_rlds_keys = set()

    for entry in label_manifest_entries:
        key = (
            entry["raw_hdf5_basename"],
            int(entry["num_filtered_actions"]),
            entry["filtered_action_sha256"],
        )
        candidates = indexed_records.get(key, [])
        if len(candidates) > 1:
            normalized_instruction = entry["instruction_normalized"]
            candidates = [c for c in candidates if c["instruction_normalized"] == normalized_instruction]

        if len(candidates) != 1:
            unmatched_source_demo_keys.append(entry["source_demo_key"])
            continue

        record = candidates[0]
        matches.append(
            {
                "source_demo_key": entry["source_demo_key"],
                "label_path": entry["label_path"],
                "rlds_split": record["rlds_split"],
                "rlds_episode_index": record["rlds_episode_index"],
                "rlds_file_path": record["rlds_file_path"],
                "rlds_instruction": record["instruction"],
                "num_steps": record["num_steps"],
                "action_sha256": record["action_sha256"],
            }
        )
        matched_rlds_keys.add((record["rlds_split"], record["rlds_episode_index"]))

    unmatched_rlds_records = [
        record
        for record in rlds_records
        if (record["rlds_split"], record["rlds_episode_index"]) not in matched_rlds_keys
    ]

    if unmatched_source_demo_keys or (require_full_rlds_coverage and unmatched_rlds_records):
        raise RuntimeError(
            "RLDS join failed. "
            f"Unmatched labels: {len(unmatched_source_demo_keys)}, "
            f"unmatched RLDS episodes: {len(unmatched_rlds_records)}."
        )

    return {
        "rlds_dataset_name": dataset_name,
        "rlds_data_root": str(rlds_data_root),
        "rlds_split": split,
        "match_count": len(matches),
        "matches": matches,
        "unmatched_source_demo_keys": unmatched_source_demo_keys,
        "unmatched_rlds_records": unmatched_rlds_records,
    }


def summarize_suite(label_records: Sequence[Dict[str, Any]], episode_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    predicate_name_counter = Counter()
    predicate_key_counter = Counter()
    predicate_first_true_hist = defaultdict(Counter)
    node_state_counts = []
    empty_event_rollouts = 0
    replay_success_count = 0
    failure_skipped_count = 0

    for episode_record in episode_records:
        if episode_record["rollout_summary"]["success"]:
            replay_success_count += 1
        else:
            failure_skipped_count += 1

    for label in label_records:
        if label["rollout_summary"]["empty_event_rollout"]:
            empty_event_rollouts += 1
        node_state_counts.append(len(label["node_states"]["first_enter_step_by_key"]))

        for predicate in label["predicates"]:
            predicate_name_counter[predicate["name"]] += 1
            predicate_key_counter[predicate["key"]] += 1
            if predicate["first_true_step"] is not None:
                predicate_first_true_hist[predicate["key"]][str(predicate["first_true_step"])] += 1

    node_stats = {
        "count": len(node_state_counts),
        "min": min(node_state_counts) if node_state_counts else 0,
        "max": max(node_state_counts) if node_state_counts else 0,
        "mean": (sum(node_state_counts) / len(node_state_counts)) if node_state_counts else 0.0,
    }

    return {
        "num_episode_labels": len(label_records),
        "num_episode_records": len(episode_records),
        "num_replay_successes": replay_success_count,
        "num_failed_replays": failure_skipped_count,
        "num_empty_event_rollouts": empty_event_rollouts,
        "predicate_occurrences_by_name": dict(sorted(predicate_name_counter.items())),
        "predicate_occurrences_by_key": dict(sorted(predicate_key_counter.items())),
        "predicate_first_true_step_histogram_by_key": {
            key: dict(sorted(counter.items(), key=lambda item: int(item[0])))
            for key, counter in sorted(predicate_first_true_hist.items())
        },
        "node_state_count_stats": node_stats,
    }


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=True, separators=(",", ":"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate precise LIBERO subtask timing sidecar labels.")
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=SUPPORTED_SUITES,
        required=True,
        help="LIBERO task suite to process.",
    )
    parser.add_argument(
        "--libero_hdf5_data_dir",
        type=str,
        default=None,
        help=(
            "Directory containing LIBERO HDF5 demos. For roboticAttack/OpenVLA GT, this should usually be "
            "the regenerated--no_noops/<suite> directory that was used to build modified_libero_rlds."
        ),
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --libero_hdf5_data_dir.",
    )
    parser.add_argument(
        "--hdf5_source_kind",
        type=str,
        choices=["auto", "raw", "regenerated"],
        default="auto",
        help=(
            "HDF5 source type. `raw` applies the regenerate_libero_dataset no-op filter; "
            "`regenerated` uses actions as-is; `auto` infers from HDF5 group keys."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory where sidecar labels and manifests will be written.",
    )
    parser.add_argument(
        "--rlds_data_root",
        type=str,
        default=str(DEFAULT_RLDS_ROOT),
        help="Root directory containing RLDS no-noops datasets.",
    )
    parser.add_argument(
        "--rlds_dataset_name",
        type=str,
        default=None,
        help="Override RLDS dataset name. Defaults to the suite-specific *_no_noops dataset.",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Optional comma-separated task IDs to process.",
    )
    parser.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help="Optional cap on the number of demos processed per task.",
    )
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Number of dummy steps to wait after restoring the initial state.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Environment camera resolution for the replay env.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite the existing suite output directory if present.",
    )
    parser.add_argument(
        "--skip_rlds_join",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the RLDS join manifest stage.",
    )
    parser.add_argument(
        "--keep_failed_replays",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write labels for replay-failed demos too. Disabled by default to align with no-noops RLDS.",
    )
    return parser


def main(args):
    h5py, np, benchmark, get_libero_path, OffScreenRenderEnv = _require_runtime_dependencies()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()

    selected_task_ids = parse_task_ids(args.task_ids)
    if selected_task_ids is None:
        selected_task_ids = list(range(task_suite.n_tasks))

    task_names = [task_suite.get_task(task_id).name for task_id in selected_task_ids]
    requested_hdf5_root_arg = args.libero_hdf5_data_dir or args.libero_raw_data_dir
    if args.libero_hdf5_data_dir and args.libero_raw_data_dir:
        hdf5_dir = Path(args.libero_hdf5_data_dir).expanduser().resolve()
        raw_dir_alias = Path(args.libero_raw_data_dir).expanduser().resolve()
        if hdf5_dir != raw_dir_alias:
            raise ValueError("--libero_hdf5_data_dir and --libero_raw_data_dir point to different paths.")

    requested_hdf5_root = (
        Path(requested_hdf5_root_arg).expanduser().resolve()
        if requested_hdf5_root_arg
        else None
    )
    hdf5_root = discover_hdf5_data_dir(args.libero_task_suite, task_names, requested_hdf5_root)
    if hdf5_root is None:
        rlds_dataset_name = args.rlds_dataset_name or SUITE_TO_RLDS_DATASET[args.libero_task_suite]
        rlds_hints = []
        try:
            rlds_hints = sample_rlds_source_hints(
                np,
                rlds_dataset_name,
                Path(args.rlds_data_root).expanduser().resolve(),
                max_samples=4,
            )
        except Exception as exc:
            rlds_hints = [{"error": f"Could not read RLDS source hints: {exc}"}]
        raise FileNotFoundError(
            "Could not find a local LIBERO HDF5 source directory. "
            "roboticAttack/OpenVLA GT uses modified_libero_rlds whose metadata points to "
            "`regenerated--no_noops/<suite>/*_demo.hdf5`, not to raw TFRecord files. "
            f"Pass a local copy with --libero_hdf5_data_dir. RLDS source hints: {rlds_hints}"
        )

    output_suite_dir = Path(args.output_root).expanduser().resolve() / args.libero_task_suite
    if output_suite_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_suite_dir}. Pass `--overwrite` to replace it."
            )
        shutil.rmtree(output_suite_dir)
    output_suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []
    written_label_records: List[Dict[str, Any]] = []
    episode_records: List[Dict[str, Any]] = []

    metadata_payload = {
        "suite_name": args.libero_task_suite,
        "resolved_hdf5_data_dir": str(hdf5_root),
        "requested_hdf5_data_dir": str(requested_hdf5_root) if requested_hdf5_root else None,
        "hdf5_source_kind": args.hdf5_source_kind,
        "output_dir": str(output_suite_dir),
        "rlds_data_root": str(Path(args.rlds_data_root).expanduser().resolve()),
        "rlds_dataset_name": args.rlds_dataset_name or SUITE_TO_RLDS_DATASET[args.libero_task_suite],
        "num_steps_wait": int(args.num_steps_wait),
        "resolution": int(args.resolution),
        "keep_failed_replays": bool(args.keep_failed_replays),
        "skip_rlds_join": bool(args.skip_rlds_join),
        "task_ids": selected_task_ids,
    }
    write_json(output_suite_dir / "run_metadata.json", metadata_payload)

    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)
        env = make_env(OffScreenRenderEnv, get_libero_path, task, resolution=int(args.resolution))
        goal_states = [list(state) for state in env.parsed_problem["goal_state"]]
        source_hdf5_path = hdf5_root / f"{task.name}_demo.hdf5"
        if not source_hdf5_path.exists():
            raise FileNotFoundError(f"Missing LIBERO HDF5 demo file: {source_hdf5_path}")

        print(f"[task {task_id}] {task.name}")
        with h5py.File(source_hdf5_path, "r") as source_file:
            demo_group = source_file["data"]
            demo_keys = sorted(demo_group.keys(), key=lambda key: int(key.split("_")[-1]))
            if args.max_episodes_per_task is not None:
                demo_keys = demo_keys[: int(args.max_episodes_per_task)]

            for demo_key in demo_keys:
                demo_data = demo_group[demo_key]
                if demo_data.attrs.get("source_format", "") == "rlds_pseudo_hdf5" or "states" not in demo_data:
                    raise RuntimeError(
                        "This demo does not contain a `states` dataset and cannot be used for exact "
                        "predicate-event labeling. It is likely a pseudo-HDF5 reconstructed from RLDS. "
                        "Use a real raw/regenerated LIBERO HDF5 file for this script, or use the pseudo-HDF5 "
                        "only for action/image/state alignment."
                    )
                source_actions = np.asarray(demo_data["actions"][()], dtype=np.float32)
                source_states = np.asarray(demo_data["states"][()])
                detected_source_kind = detect_hdf5_source_kind(demo_data)
                source_kind = detected_source_kind if args.hdf5_source_kind == "auto" else args.hdf5_source_kind
                filtered = prepare_source_actions(np, source_actions, source_kind)
                filtered_actions = filtered["actions"]

                episode_payload = {
                    "schema_version": 1,
                    "suite_name": args.libero_task_suite,
                    "task_id": int(task_id),
                    "task_name": task.name,
                    "task_description": task.language,
                    "task_description_normalized": normalize_instruction(task.language),
                    "task_bddl_file": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                    "goal_state": [predicate_record(state) for state in goal_states],
                    "source": {
                        "source_demo_key": f"{task.name}#{demo_key}",
                        "task_name": task.name,
                        "demo_key": demo_key,
                        "hdf5_path": str(source_hdf5_path),
                        "hdf5_basename": source_hdf5_path.name,
                        "hdf5_source_kind": source_kind,
                        "hdf5_source_kind_detected": detected_source_kind,
                        "raw_hdf5_path": str(source_hdf5_path),
                        "raw_hdf5_basename": source_hdf5_path.name,
                        "source_action_sha256": sha256_array(np, source_actions),
                        "raw_action_sha256": sha256_array(np, source_actions),
                        "filtered_action_sha256": sha256_array(np, filtered_actions),
                        "num_source_actions": int(source_actions.shape[0]),
                        "num_raw_actions": int(source_actions.shape[0]),
                        "num_filtered_actions": int(filtered_actions.shape[0]),
                        "num_noops_filtered": int(filtered["num_noops_filtered"]),
                        "kept_raw_action_indices": filtered["kept_raw_indices"],
                    },
                }

                label_payload = label_episode(
                    np,
                    env,
                    goal_states,
                    filtered_actions,
                    source_states[0],
                    num_steps_wait=int(args.num_steps_wait),
                )
                episode_payload.update(label_payload)

                episode_records.append(episode_payload)
                replay_success = bool(episode_payload["rollout_summary"]["success"])
                if not replay_success and not args.keep_failed_replays:
                    print(f"  - {demo_key}: replay failed, skipped label write")
                    continue

                label_path = output_suite_dir / "episodes" / task.name / f"{demo_key}.json"
                write_json(label_path, episode_payload)
                print(
                    f"  - {demo_key}: wrote {label_path.name} "
                    f"(steps={episode_payload['source']['num_filtered_actions']}, "
                    f"success={replay_success})"
                )
                written_label_records.append(episode_payload)

                manifest_entries.append(
                    {
                        "source_demo_key": episode_payload["source"]["source_demo_key"],
                        "task_id": int(task_id),
                        "task_name": task.name,
                        "demo_key": demo_key,
                        "instruction": task.language,
                        "instruction_normalized": normalize_instruction(task.language),
                        "label_path": str(label_path.relative_to(output_suite_dir)),
                        "hdf5_path": str(source_hdf5_path),
                        "hdf5_basename": source_hdf5_path.name,
                        "hdf5_source_kind": source_kind,
                        "raw_hdf5_path": str(source_hdf5_path),
                        "raw_hdf5_basename": source_hdf5_path.name,
                        "num_source_actions": int(source_actions.shape[0]),
                        "num_raw_actions": int(source_actions.shape[0]),
                        "num_filtered_actions": int(filtered_actions.shape[0]),
                        "raw_action_sha256": episode_payload["source"]["raw_action_sha256"],
                        "source_action_sha256": episode_payload["source"]["source_action_sha256"],
                        "filtered_action_sha256": episode_payload["source"]["filtered_action_sha256"],
                        "success": replay_success,
                        "first_success_step": episode_payload["rollout_summary"]["first_success_step"],
                    }
                )

        env.close()

    label_manifest_payload = {
        "suite_name": args.libero_task_suite,
        "count": len(manifest_entries),
        "entries": manifest_entries,
    }
    write_json(output_suite_dir / "label_manifest.json", label_manifest_payload)

    summary_payload = summarize_suite(written_label_records, episode_records)
    write_json(output_suite_dir / "summary.json", summary_payload)

    if not args.skip_rlds_join:
        join_candidates = [entry for entry in manifest_entries if entry["success"]]
        require_full_rlds_coverage = (
            len(selected_task_ids) == task_suite.n_tasks
            and sorted(selected_task_ids) == list(range(task_suite.n_tasks))
            and args.max_episodes_per_task is None
        )
        rlds_join_payload = build_rlds_join_manifest(
            np,
            join_candidates,
            dataset_name=args.rlds_dataset_name or SUITE_TO_RLDS_DATASET[args.libero_task_suite],
            rlds_data_root=Path(args.rlds_data_root).expanduser().resolve(),
            split="train",
            require_full_rlds_coverage=require_full_rlds_coverage,
        )
        write_json(output_suite_dir / "rlds_join_manifest.json", rlds_join_payload)

    print(f"Finished writing LIBERO subtask labels to: {output_suite_dir}")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
