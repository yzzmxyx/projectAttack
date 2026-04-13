"""
Export the current LIBERO RLDS episodes to pseudo-HDF5 files.

This is a compatibility/export utility for situations where the local
regenerated LIBERO HDF5 files are unavailable but the modified_libero_rlds
TFRecord datasets are present.

Important:
    The exported files are *not* lossless regenerated LIBERO HDF5 files. RLDS
    contains actions, RGB observations, EEF/gripper/joint states, language, and
    source file metadata, but it does not contain MuJoCo simulator states,
    object poses, contacts, or the exact per-demo initial simulator state.

    Therefore these pseudo-HDF5 files are useful for action/image/state
    alignment and downstream data plumbing, but they are not sufficient for
    exact predicate-event labeling unless combined with a separate replay
    procedure that restores and validates simulator states.

Example:
    /home/yxx/miniconda3/envs/robo_env/bin/python \
        /home/yxx/projectAttack/experiments/robot/libero/reconstruct_hdf5_from_rlds.py \
        --libero_task_suite libero_spatial \
        --rlds_data_root /home/yxx/roboticAttack/openvla-main/dataset \
        --output_root /tmp/libero_pseudo_hdf5 \
        --max_episodes 2 \
        --overwrite

To include RGB observations, add `--include_images`. By default images are
omitted to keep smoke tests and metadata-only exports small.
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
from typing import Any, Dict, Iterable, List, Optional


SUITE_TO_RLDS_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
SUPPORTED_SUITES = tuple(SUITE_TO_RLDS_DATASET.keys())
DEFAULT_RLDS_ROOT = Path("/home/yxx/roboticAttack/openvla-main/dataset")


def _require_runtime_dependencies():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
        import robosuite.utils.transform_utils as T  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "Missing RLDS/HDF5 export dependencies. Run this script inside the "
            "LIBERO/OpenVLA environment, for example "
            "`/home/yxx/miniconda3/envs/robo_env/bin/python`."
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    return h5py, np, tfds, T


def decode_maybe_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_instruction(text: str) -> str:
    text = text.replace("\n", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".? ")


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


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=True, separators=(",", ":"))


def sha256_array(np_module, array) -> str:
    contiguous = np_module.ascontiguousarray(array, dtype=np_module.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def hdf5_string_attr(value: Any) -> str:
    return "" if value is None else str(value)


def compression_kwargs(args) -> Dict[str, Any]:
    if args.compression == "none":
        return {}
    kwargs: Dict[str, Any] = {"compression": args.compression}
    if args.compression == "gzip" and args.compression_opts is not None:
        kwargs["compression_opts"] = int(args.compression_opts)
    return kwargs


def make_decoders(tfds, include_images: bool) -> Optional[Dict[str, Any]]:
    if include_images:
        return None
    return {
        "steps": {
            "observation": {
                "image": tfds.decode.SkipDecoding(),
                "wrist_image": tfds.decode.SkipDecoding(),
            }
        }
    }


def stack_optional_step_field(np_module, steps: List[Dict[str, Any]], field_path: List[str], dtype):
    values = []
    for step in steps:
        value: Any = step
        for key in field_path:
            value = value[key]
        values.append(np_module.asarray(value, dtype=dtype))
    return np_module.stack(values, axis=0)


def iter_rlds_episodes(
    np_module,
    tfds,
    dataset_name: str,
    data_root: Path,
    split: str,
    include_images: bool,
) -> Iterable[Dict[str, Any]]:
    builder = tfds.builder(dataset_name, data_dir=str(data_root))
    dataset = builder.as_dataset(
        split=split,
        shuffle_files=False,
        decoders=make_decoders(tfds, include_images),
    )

    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = list(episode["steps"])
        if not steps:
            continue

        actions = stack_optional_step_field(np_module, steps, ["action"], np_module.float32)
        eef_gripper_state = stack_optional_step_field(
            np_module, steps, ["observation", "state"], np_module.float32
        )
        joint_state = stack_optional_step_field(
            np_module, steps, ["observation", "joint_state"], np_module.float32
        )
        rewards = stack_optional_step_field(np_module, steps, ["reward"], np_module.float32).reshape(-1)
        is_last = stack_optional_step_field(np_module, steps, ["is_last"], np_module.bool_).reshape(-1)
        is_terminal = stack_optional_step_field(np_module, steps, ["is_terminal"], np_module.bool_).reshape(-1)

        instruction = decode_maybe_bytes(steps[0]["language_instruction"])
        file_path = decode_maybe_bytes(episode["episode_metadata"]["file_path"])
        hdf5_basename = os.path.basename(file_path) or f"rlds_episode_{episode_index}_demo.hdf5"

        record: Dict[str, Any] = {
            "rlds_episode_index": int(episode_index),
            "rlds_split": split,
            "rlds_file_path": file_path,
            "rlds_file_basename": hdf5_basename,
            "instruction": instruction,
            "instruction_normalized": normalize_instruction(instruction),
            "actions": actions,
            "eef_gripper_state": eef_gripper_state,
            "joint_state": joint_state,
            "rewards": rewards,
            "dones": is_last.astype(np_module.uint8),
            "is_terminal": is_terminal.astype(np_module.uint8),
            "num_steps": int(actions.shape[0]),
            "action_sha256": sha256_array(np_module, actions),
        }
        if include_images:
            record["agentview_rgb"] = stack_optional_step_field(
                np_module, steps, ["observation", "image"], np_module.uint8
            )
            record["eye_in_hand_rgb"] = stack_optional_step_field(
                np_module, steps, ["observation", "wrist_image"], np_module.uint8
            )
        yield record


def axisangle_batch_to_quat(np_module, transform_utils, axisangle):
    quats = []
    for value in np_module.asarray(axisangle, dtype=np_module.float32):
        quats.append(np_module.asarray(transform_utils.axisangle2quat(value), dtype=np_module.float32))
    return np_module.stack(quats, axis=0)


def create_dataset(group, name: str, data, dataset_kwargs: Dict[str, Any]):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **dataset_kwargs)


def write_pseudo_hdf5_episode(
    h5py,
    np_module,
    transform_utils,
    output_path: Path,
    demo_key: str,
    record: Dict[str, Any],
    include_images: bool,
    dataset_kwargs: Dict[str, Any],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "a") as h5_file:
        h5_file.attrs["source_format"] = "rlds_pseudo_hdf5"
        h5_file.attrs["sim_states_available"] = False
        h5_file.attrs["warning"] = (
            "Pseudo-HDF5 reconstructed from RLDS. It lacks MuJoCo simulator "
            "states and is not suitable for exact predicate replay."
        )
        h5_file.attrs["rlds_source_file_basename"] = hdf5_string_attr(record["rlds_file_basename"])
        h5_file.attrs["rlds_split"] = hdf5_string_attr(record["rlds_split"])

        data_group = h5_file.require_group("data")
        if demo_key in data_group:
            del data_group[demo_key]
        episode_group = data_group.create_group(demo_key)
        obs_group = episode_group.create_group("obs")

        eef_gripper_state = np_module.asarray(record["eef_gripper_state"], dtype=np_module.float32)
        ee_states = eef_gripper_state[:, :6]
        ee_pos = ee_states[:, :3]
        ee_ori_axisangle = ee_states[:, 3:6]
        gripper_states = eef_gripper_state[:, -2:]
        ee_quat = axisangle_batch_to_quat(np_module, transform_utils, ee_ori_axisangle)
        robot_states = np_module.concatenate([gripper_states, ee_pos, ee_quat], axis=1).astype(np_module.float32)

        create_dataset(obs_group, "gripper_states", gripper_states.astype(np_module.float32), dataset_kwargs)
        create_dataset(obs_group, "joint_states", record["joint_state"].astype(np_module.float32), dataset_kwargs)
        create_dataset(obs_group, "ee_states", ee_states.astype(np_module.float32), dataset_kwargs)
        create_dataset(obs_group, "ee_pos", ee_pos.astype(np_module.float32), dataset_kwargs)
        create_dataset(obs_group, "ee_ori", ee_ori_axisangle.astype(np_module.float32), dataset_kwargs)
        if include_images:
            create_dataset(obs_group, "agentview_rgb", record["agentview_rgb"], dataset_kwargs)
            create_dataset(obs_group, "eye_in_hand_rgb", record["eye_in_hand_rgb"], dataset_kwargs)

        create_dataset(episode_group, "actions", record["actions"].astype(np_module.float32), dataset_kwargs)
        create_dataset(episode_group, "robot_states", robot_states, dataset_kwargs)
        create_dataset(episode_group, "rewards", record["rewards"].astype(np_module.float32), dataset_kwargs)
        create_dataset(episode_group, "dones", record["dones"].astype(np_module.uint8), dataset_kwargs)
        create_dataset(episode_group, "is_terminal", record["is_terminal"].astype(np_module.uint8), dataset_kwargs)

        episode_group.attrs["source_format"] = "rlds_pseudo_hdf5"
        episode_group.attrs["sim_states_available"] = False
        episode_group.attrs["states_dataset_present"] = False
        episode_group.attrs["rlds_episode_index"] = int(record["rlds_episode_index"])
        episode_group.attrs["rlds_split"] = hdf5_string_attr(record["rlds_split"])
        episode_group.attrs["rlds_file_path"] = hdf5_string_attr(record["rlds_file_path"])
        episode_group.attrs["rlds_file_basename"] = hdf5_string_attr(record["rlds_file_basename"])
        episode_group.attrs["language_instruction"] = hdf5_string_attr(record["instruction"])
        episode_group.attrs["language_instruction_normalized"] = hdf5_string_attr(
            record["instruction_normalized"]
        )
        episode_group.attrs["num_steps"] = int(record["num_steps"])
        episode_group.attrs["action_sha256"] = hdf5_string_attr(record["action_sha256"])
        episode_group.attrs["warning"] = (
            "No `states` dataset was written because RLDS does not contain full "
            "MuJoCo simulator states."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct pseudo-HDF5 files from LIBERO RLDS data.")
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=SUPPORTED_SUITES,
        required=True,
        help="LIBERO task suite to export.",
    )
    parser.add_argument(
        "--rlds_data_root",
        type=str,
        default=str(DEFAULT_RLDS_ROOT),
        help="Root directory containing modified_libero_rlds datasets.",
    )
    parser.add_argument(
        "--rlds_dataset_name",
        type=str,
        default=None,
        help="Override RLDS dataset name. Defaults to the suite-specific *_no_noops dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="RLDS split to export.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory where pseudo-HDF5 files and manifests will be written.",
    )
    parser.add_argument(
        "--include_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write agentview/wrist RGB datasets. Disabled by default to keep exports small.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on total exported RLDS episodes.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "gzip", "lzf"],
        default="gzip",
        help="HDF5 dataset compression.",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        help="gzip compression level when --compression=gzip.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite the existing suite output directory if present.",
    )
    return parser


def main(args):
    h5py, np, tfds, transform_utils = _require_runtime_dependencies()

    dataset_name = args.rlds_dataset_name or SUITE_TO_RLDS_DATASET[args.libero_task_suite]
    rlds_data_root = Path(args.rlds_data_root).expanduser().resolve()
    output_suite_dir = Path(args.output_root).expanduser().resolve() / args.libero_task_suite

    if output_suite_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_suite_dir}. Pass `--overwrite` to replace it."
            )
        shutil.rmtree(output_suite_dir)
    output_suite_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = compression_kwargs(args)
    demo_counts_by_basename: Dict[str, int] = defaultdict(int)
    episode_counts_by_basename: Counter = Counter()
    manifest_entries: List[Dict[str, Any]] = []

    metadata_payload = {
        "schema_version": 1,
        "suite_name": args.libero_task_suite,
        "rlds_dataset_name": dataset_name,
        "rlds_data_root": str(rlds_data_root),
        "rlds_split": args.split,
        "output_dir": str(output_suite_dir),
        "source_format": "rlds_pseudo_hdf5",
        "sim_states_available": False,
        "include_images": bool(args.include_images),
        "compression": args.compression,
        "compression_opts": args.compression_opts if args.compression == "gzip" else None,
        "warning": (
            "These files are reconstructed from RLDS and do not contain full MuJoCo simulator "
            "states. They cannot replace regenerated--no_noops HDF5 for exact predicate labels."
        ),
    }
    write_json(output_suite_dir / "run_metadata.json", metadata_payload)

    for record in iter_rlds_episodes(
        np,
        tfds,
        dataset_name=dataset_name,
        data_root=rlds_data_root,
        split=args.split,
        include_images=bool(args.include_images),
    ):
        if args.max_episodes is not None and len(manifest_entries) >= int(args.max_episodes):
            break

        basename = record["rlds_file_basename"]
        pseudo_hdf5_path = output_suite_dir / "hdf5" / basename
        demo_index = demo_counts_by_basename[basename]
        demo_key = f"demo_{demo_index}"
        demo_counts_by_basename[basename] += 1
        episode_counts_by_basename[basename] += 1

        write_pseudo_hdf5_episode(
            h5py,
            np,
            transform_utils,
            pseudo_hdf5_path,
            demo_key,
            record,
            include_images=bool(args.include_images),
            dataset_kwargs=dataset_kwargs,
        )

        manifest_entries.append(
            {
                "source_format": "rlds_pseudo_hdf5",
                "sim_states_available": False,
                "pseudo_hdf5_path": str(pseudo_hdf5_path.relative_to(output_suite_dir)),
                "pseudo_demo_key": demo_key,
                "rlds_dataset_name": dataset_name,
                "rlds_split": record["rlds_split"],
                "rlds_episode_index": record["rlds_episode_index"],
                "rlds_file_path": record["rlds_file_path"],
                "rlds_file_basename": record["rlds_file_basename"],
                "instruction": record["instruction"],
                "instruction_normalized": record["instruction_normalized"],
                "num_steps": record["num_steps"],
                "action_sha256": record["action_sha256"],
            }
        )
        print(
            f"[{len(manifest_entries)}] {basename}#{demo_key} "
            f"steps={record['num_steps']} images={bool(args.include_images)}"
        )

    manifest_payload = {
        "schema_version": 1,
        "suite_name": args.libero_task_suite,
        "rlds_dataset_name": dataset_name,
        "rlds_data_root": str(rlds_data_root),
        "rlds_split": args.split,
        "source_format": "rlds_pseudo_hdf5",
        "sim_states_available": False,
        "count": len(manifest_entries),
        "entries": manifest_entries,
    }
    write_json(output_suite_dir / "pseudo_hdf5_manifest.json", manifest_payload)

    summary_payload = {
        "schema_version": 1,
        "suite_name": args.libero_task_suite,
        "rlds_dataset_name": dataset_name,
        "rlds_split": args.split,
        "num_exported_episodes": len(manifest_entries),
        "num_output_hdf5_files": len(episode_counts_by_basename),
        "episodes_by_hdf5_basename": dict(sorted(episode_counts_by_basename.items())),
        "include_images": bool(args.include_images),
        "sim_states_available": False,
    }
    write_json(output_suite_dir / "summary.json", summary_payload)
    print(f"Finished writing pseudo-HDF5 export to: {output_suite_dir}")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
