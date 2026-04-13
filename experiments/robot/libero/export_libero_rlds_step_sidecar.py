#!/usr/bin/env python3
"""Export a minimal step-level LIBERO RLDS sidecar.

This script reads TFDS/RLDS directly and writes scalar/vector metadata only.
It intentionally skips images and does not depend on the training dataloader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from libero_sidecar_utils import (
    STEP_VECTOR_DIMS,
    canonical_instruction,
    compute_action_stats,
    decode_text,
    ensure_float_list,
    make_episode_key,
    normalize_action_bounds_q99,
    transform_libero_gripper_action,
    update_metadata,
    write_parquet,
)


DEFAULT_RLDS_ROOT = "/home/yxx/roboticAttack/openvla-main/dataset"
DEFAULT_DATASET = "libero_10_no_noops"
DEFAULT_OUTPUT_DIR = "/home/yxx/projectAttack/data/libero_sidecars/libero_10_no_noops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds_root", default=DEFAULT_RLDS_ROOT, help="TFDS data_dir containing LIBERO datasets.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="TFDS dataset name, e.g. libero_10_no_noops.")
    parser.add_argument("--split", default="train", help="TFDS split to export.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for steps.parquet and metadata.json.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--progress_every", type=int, default=25, help="Print progress every N episodes.")
    return parser.parse_args()


def _extract_episode_rows(dataset: str, episode_index: int, episode: dict) -> tuple[list[dict], list[np.ndarray]]:
    source_file_path = decode_text(episode["episode_metadata"]["file_path"])
    episode_rows: list[dict] = []
    transformed_actions: list[np.ndarray] = []
    episode_key = None
    instruction = None

    for t, step in enumerate(episode["steps"]):
        if instruction is None:
            instruction = canonical_instruction(step["language_instruction"])
            episode_key = make_episode_key(dataset, episode_index, source_file_path, instruction)

        raw_action = np.asarray(step["action"], dtype=np.float32).reshape(-1)
        state = np.asarray(step["observation"]["state"], dtype=np.float32).reshape(-1)
        joint_state = np.asarray(step["observation"]["joint_state"], dtype=np.float32).reshape(-1)
        transformed_action = transform_libero_gripper_action(raw_action)

        episode_rows.append(
            {
                "episode_key": episode_key,
                "episode_index": int(episode_index),
                "t": int(t),
                "T": -1,
                "instruction": instruction,
                "source_file_path": source_file_path,
                "raw_action": ensure_float_list(raw_action, 7, "raw_action"),
                "normalized_action": None,
                "eef_state": ensure_float_list(state[:6], 6, "eef_state"),
                "gripper_state": ensure_float_list(state[-2:], 2, "gripper_state"),
                "joint_state": ensure_float_list(joint_state, 7, "joint_state"),
                "reward": float(np.asarray(step["reward"], dtype=np.float32)),
                "is_last": bool(step["is_last"]),
                "is_terminal": bool(step["is_terminal"]),
            }
        )
        transformed_actions.append(transformed_action)

    T = len(episode_rows)
    for row in episode_rows:
        row["T"] = int(T)

    return episode_rows, transformed_actions


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    output_dir = Path(args.output_dir)
    steps_path = output_dir / "steps.parquet"
    metadata_path = output_dir / "metadata.json"

    builder = tfds.builder(args.dataset, data_dir=args.rlds_root)
    ds = tfds.as_numpy(builder.as_dataset(split=args.split, shuffle_files=False))

    rows: list[dict] = []
    all_transformed_actions: list[np.ndarray] = []
    instruction_set: set[str] = set()
    episode_lengths: list[int] = []

    for episode_index, episode in enumerate(ds):
        if args.max_episodes is not None and episode_index >= args.max_episodes:
            break
        episode_rows, transformed_actions = _extract_episode_rows(args.dataset, episode_index, episode)
        if not episode_rows:
            continue
        rows.extend(episode_rows)
        all_transformed_actions.extend(transformed_actions)
        instruction_set.add(episode_rows[0]["instruction"])
        episode_lengths.append(episode_rows[0]["T"])

        if args.progress_every > 0 and (episode_index + 1) % args.progress_every == 0:
            print(f"exported episodes={episode_index + 1} steps={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No rows exported. Check --rlds_root, --dataset, and --split.")

    transformed_action_array = np.asarray(all_transformed_actions, dtype=np.float32)
    action_stats = compute_action_stats(transformed_action_array)
    q01 = action_stats["q01"]
    q99 = action_stats["q99"]

    for row, transformed_action in zip(rows, transformed_action_array):
        normalized_action = normalize_action_bounds_q99(transformed_action, q01=q01, q99=q99)
        row["normalized_action"] = ensure_float_list(normalized_action, 7, "normalized_action")

    df = pd.DataFrame(rows)
    write_parquet(df, steps_path, STEP_VECTOR_DIMS)

    metadata = {
        "dataset": args.dataset,
        "rlds_root": str(args.rlds_root),
        "split": args.split,
        "max_episodes": args.max_episodes,
        "num_episodes": len(episode_lengths),
        "num_steps": len(df),
        "num_instructions": len(instruction_set),
        "episode_length_min": int(min(episode_lengths)),
        "episode_length_max": int(max(episode_lengths)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "steps_parquet": str(steps_path),
        "vector_columns": STEP_VECTOR_DIMS,
        "normalization": {
            "type": "bounds_q99_first_6_dims_gripper_unscaled",
            "gripper_transform": "1 - clip(raw_gripper, 0, 1)",
            "action_stats": action_stats,
        },
    }
    update_metadata(metadata_path, "step_sidecar", metadata)

    print(
        "wrote "
        f"{steps_path} episodes={metadata['num_episodes']} steps={metadata['num_steps']} "
        f"instructions={metadata['num_instructions']}"
    )


if __name__ == "__main__":
    main()
