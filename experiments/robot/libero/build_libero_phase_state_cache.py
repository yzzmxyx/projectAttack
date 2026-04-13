#!/usr/bin/env python3
"""Build coarse phase start-state cache for LIBERO online rollouts.

RLDS sidecars do not contain full MuJoCo simulator states. This script builds a
minimal cache by replaying sidecar raw expert actions from LIBERO init states to
coarse phase boundaries, using ordinal modulo alignment within each task.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from libero_sidecar_utils import PHASES, STEP_VECTOR_DIMS, read_parquet, stack_vector_column, update_metadata  # noqa: E402
from phase_state_cache_metadata import build_phase_state_cache_metadata  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402


PHASE_STARTS = ("initial", "contact_manipulate", "post_contact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="libero_spatial", help="LIBERO dataset/suite, e.g. libero_spatial.")
    parser.add_argument("--phase_parquet", required=True, help="Input phases.parquet from the sidecar pipeline.")
    parser.add_argument("--output_pt", default=None, help="Output phase_state_cache.pt.")
    parser.add_argument("--metadata_json", default=None, help="Metadata JSON to update.")
    parser.add_argument("--task_ids", default=None, help="Optional comma-separated task ids to build.")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="No-op settle steps before replaying expert actions.")
    parser.add_argument("--alignment", default="ordinal_modulo", choices=["ordinal_modulo"])
    parser.add_argument("--env_resolution", type=int, default=128)
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional smoke-test task cap.")
    parser.add_argument("--max_init_states_per_task", type=int, default=None, help="Optional smoke-test init-state cap.")
    return parser.parse_args()


def normalize_instruction(instruction: Any) -> str:
    text = str(instruction).lower().replace("\n", " ").strip()
    text = " ".join(text.split())
    while text.endswith(".") or text.endswith("?"):
        text = text[:-1].rstrip()
    return text


def resolve_suite_name(dataset: str) -> str:
    dataset_key = str(dataset).lower().strip()
    if dataset_key.endswith("_no_noops"):
        dataset_key = dataset_key[: -len("_no_noops")]
    aliases = {
        "libero_spatial": "libero_spatial",
        "libero_object": "libero_object",
        "libero_goal": "libero_goal",
        "libero_10": "libero_10",
    }
    if dataset_key not in aliases:
        raise ValueError(f"Unsupported LIBERO dataset for phase-state cache: {dataset}")
    return aliases[dataset_key]


def default_output_path(phase_parquet: str | Path) -> Path:
    return Path(phase_parquet).resolve().with_name("phase_state_cache.pt")


def parse_task_ids(value: str | None, max_task_id: int) -> list[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    task_ids = []
    for item in text.split(","):
        item = item.strip()
        if item == "":
            continue
        task_id = int(item)
        if task_id < 0 or task_id >= max_task_id:
            raise ValueError(f"task_id {task_id} is out of range [0, {max_task_id})")
        task_ids.append(task_id)
    if not task_ids:
        raise ValueError("--task_ids did not contain any valid task id")
    return sorted(set(task_ids))


def collect_episode_records(phase_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    records_by_instruction: dict[str, list[dict[str, Any]]] = {}
    for episode_key, group in phase_df.groupby("episode_key", sort=False):
        group = group.sort_values("t")
        instruction = normalize_instruction(group["instruction"].iloc[0])
        actions = stack_vector_column(group["raw_action"], 7, "raw_action")
        boundaries: dict[str, int] = {}
        for phase_name in PHASES:
            phase_rows = group[group["phase"] == phase_name]
            if len(phase_rows) == 0:
                continue
            boundaries[phase_name] = int(phase_rows["phase_start_t"].iloc[0])
        if "contact_manipulate" not in boundaries or "post_contact" not in boundaries:
            continue
        records_by_instruction.setdefault(instruction, []).append(
            {
                "episode_key": str(episode_key),
                "actions": actions.astype(np.float32),
                "boundaries": boundaries,
                "T": int(group["T"].iloc[0]),
            }
        )
    return records_by_instruction


def replay_to_boundary(env, init_state: np.ndarray, raw_actions: np.ndarray, boundary_t: int, num_steps_wait: int) -> np.ndarray:
    env.reset()
    env.set_init_state(init_state)
    for _ in range(max(0, int(num_steps_wait))):
        _obs, _reward, done, _info = env.step(get_libero_dummy_action("openvla"))
        if done:
            break
    for action in raw_actions[: max(0, int(boundary_t))]:
        _obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float32).tolist())
        if done:
            break
    return np.asarray(env.get_sim_state(), dtype=np.float32).copy()


def replay_to_phase_boundaries(
    env,
    init_state: np.ndarray,
    raw_actions: np.ndarray,
    boundary_ts: dict[str, int],
    num_steps_wait: int,
) -> dict[str, np.ndarray]:
    """Replay once and snapshot multiple coarse phase boundaries.

    Boundary semantics match `replay_to_boundary`: the cached state for
    `boundary_t=k` is the simulator state after replaying exactly `k` expert
    actions from the settled init state.
    """
    requested = {str(name): max(0, int(t)) for name, t in boundary_ts.items()}
    if not requested:
        return {}

    env.reset()
    env.set_init_state(init_state)
    for _ in range(max(0, int(num_steps_wait))):
        _obs, _reward, done, _info = env.step(get_libero_dummy_action("openvla"))
        if done:
            break

    snapshots: dict[str, np.ndarray] = {}
    current_state = np.asarray(env.get_sim_state(), dtype=np.float32).copy()
    for phase_name, boundary_t in requested.items():
        if boundary_t == 0:
            snapshots[phase_name] = current_state.copy()

    max_boundary = max(requested.values())
    for step_count, action in enumerate(raw_actions[:max_boundary], start=1):
        _obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float32).tolist())
        current_state = np.asarray(env.get_sim_state(), dtype=np.float32).copy()
        for phase_name, boundary_t in requested.items():
            if phase_name not in snapshots and step_count >= boundary_t:
                snapshots[phase_name] = current_state.copy()
        if done:
            break

    # If replay terminated early, preserve previous behavior by using the final
    # reached simulator state for any later boundary request.
    for phase_name in requested:
        if phase_name not in snapshots:
            snapshots[phase_name] = current_state.copy()
    return snapshots


def main() -> None:
    args = parse_args()
    phase_path = Path(args.phase_parquet).resolve()
    output_path = Path(args.output_pt).resolve() if args.output_pt else default_output_path(phase_path)
    metadata_path = Path(args.metadata_json).resolve() if args.metadata_json else output_path.with_name("metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    suite_name = resolve_suite_name(args.dataset)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    phase_df = read_parquet(phase_path, STEP_VECTOR_DIMS)
    records_by_instruction = collect_episode_records(phase_df)

    states: dict[tuple[int, int, str], torch.Tensor] = {}
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    total_task_count = int(task_suite.n_tasks)
    selected_task_ids = parse_task_ids(args.task_ids, max_task_id=total_task_count)
    if selected_task_ids is None:
        task_count = total_task_count
        if args.max_tasks is not None:
            task_count = min(task_count, max(0, int(args.max_tasks)))
        selected_task_ids = list(range(task_count))
    print(f"building phase-state cache for task_ids={selected_task_ids}", flush=True)

    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)
        instruction = normalize_instruction(getattr(task, "language", ""))
        episode_records = records_by_instruction.get(instruction, [])
        if not episode_records:
            raise RuntimeError(f"No sidecar episode records found for task_id={task_id} instruction={instruction}")

        init_states = task_suite.get_task_init_states(task_id)
        init_count = len(init_states)
        if args.max_init_states_per_task is not None:
            init_count = min(init_count, max(0, int(args.max_init_states_per_task)))

        env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(args.env_resolution)))
        try:
            for init_state_idx in range(init_count):
                init_state = np.asarray(init_states[init_state_idx], dtype=np.float32).copy()
                source_record = episode_records[init_state_idx % len(episode_records)]

                key = (int(task_id), int(init_state_idx), "initial")
                states[key] = torch.from_numpy(init_state.astype(np.float32))
                records[key] = {
                    "task_id": int(task_id),
                    "init_state_idx": int(init_state_idx),
                    "phase": "initial",
                    "instruction": instruction,
                    "task_description": normalize_instruction(task_description),
                    "source_episode_key": str(source_record["episode_key"]),
                    "source_boundary_t": 0,
                    "source_T": int(source_record["T"]),
                    "alignment": str(args.alignment),
                }

                phase_states = replay_to_phase_boundaries(
                    env=env,
                    init_state=init_state,
                    raw_actions=source_record["actions"],
                    boundary_ts={
                        "contact_manipulate": int(source_record["boundaries"]["contact_manipulate"]),
                        "post_contact": int(source_record["boundaries"]["post_contact"]),
                    },
                    num_steps_wait=int(args.num_steps_wait),
                )

                for phase_name in ("contact_manipulate", "post_contact"):
                    boundary_t = int(source_record["boundaries"][phase_name])
                    state = phase_states[phase_name]
                    key = (int(task_id), int(init_state_idx), phase_name)
                    states[key] = torch.from_numpy(state.astype(np.float32))
                    records[key] = {
                        "task_id": int(task_id),
                        "init_state_idx": int(init_state_idx),
                        "phase": phase_name,
                        "instruction": instruction,
                        "task_description": normalize_instruction(task_description),
                        "source_episode_key": str(source_record["episode_key"]),
                        "source_boundary_t": int(boundary_t),
                        "source_T": int(source_record["T"]),
                        "alignment": str(args.alignment),
                    }
        finally:
            if hasattr(env, "close"):
                env.close()

        print(
            f"cached task_id={task_id} init_states={init_count} "
            f"instruction={instruction[:80]}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "suite_name": suite_name,
        "phase_parquet": str(phase_path),
        "phase_starts": list(PHASE_STARTS),
        "alignment": str(args.alignment),
        "num_steps_wait": int(args.num_steps_wait),
        "env_resolution": int(args.env_resolution),
        "states": states,
        "records": records,
        "metadata": {
            "num_states": int(len(states)),
            "num_records": int(len(records)),
            "num_tasks": int(len(selected_task_ids)),
            "task_ids": list(selected_task_ids),
            "max_tasks": args.max_tasks,
            "max_init_states_per_task": args.max_init_states_per_task,
            "state_shape_examples": {
                str(key): list(value.shape) for key, value in list(states.items())[:5]
            },
        },
    }
    torch.save(payload, output_path)
    update_metadata(metadata_path, "phase_state_cache", build_phase_state_cache_metadata(payload, output_path))
    print(f"wrote {output_path} states={len(states)} tasks={len(selected_task_ids)}")


if __name__ == "__main__":
    main()
