#!/usr/bin/env python3
"""Batch-match rollout runs to deterministic LIBERO init states and build recovery assets."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from rlds_recovery_utils import (  # noqa: E402
    DEFAULT_RLDS_ROOT,
    build_single_state_recovery_asset,
    normalize_instruction_key,
    resolve_rlds_dataset_name,
    resolve_task_suite_name,
    to_jsonable,
    write_json,
)


KNOWN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "t", "y", "1"):
        return True
    if text in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_run_ids(value: str) -> list[str] | None:
    text = str(value).strip().lower()
    if text in ("", "all", "auto", "none", "null"):
        return None
    run_ids = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            run_ids.append(token)
    return sorted(set(run_ids))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout_root",
        type=str,
        default=str(REPO_ROOT / "run" / "UADA_rollout_online_env"),
    )
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--run_ids", type=str, default="all")
    parser.add_argument("--dataset", type=str, default="auto")
    parser.add_argument("--task_suite_name", type=str, default="auto")
    parser.add_argument("--manifest_split", type=str, default="val")
    parser.add_argument("--match_all_val_episodes", type=str2bool, default=True)
    parser.add_argument("--val_deterministic", type=str2bool, default=True)
    parser.add_argument("--online_val_episodes", type=int, default=8)
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--rlds_root", type=str, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--steps_parquet", type=str, default="")
    parser.add_argument("--phases_parquet", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--env_resolution", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--recovery_vision_backbone", type=str, default="dinoclip-vit-l-336px")
    parser.add_argument("--recovery_image_resize_strategy", type=str, default="resize-naive")
    parser.add_argument("--force_rebuild", action="store_true")
    return parser


def discover_rollout_run_dirs(rollout_root: str | os.PathLike[str], requested_run_ids: Sequence[str] | None) -> list[Path]:
    root = Path(rollout_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Rollout root does not exist: {root}")
    wanted = None if requested_run_ids is None else {str(item) for item in requested_run_ids}
    run_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if wanted is not None and child.name not in wanted:
            continue
        manifest_path = child / "videos" / "video_manifest.csv"
        probe_final_path = child / "probe_final_val.json"
        if manifest_path.exists() or probe_final_path.exists():
            run_dirs.append(child)
    return run_dirs


def load_manifest_rows(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def select_latest_manifest_row(rows: Sequence[Mapping[str, str]], split: str) -> dict[str, str] | None:
    split_key = normalize_instruction_key(split)
    candidates = [
        dict(row)
        for row in rows
        if normalize_instruction_key(row.get("split", "")) == split_key
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: int(row.get("iter_idx", 0)))


@lru_cache(maxsize=None)
def load_suite_task_catalog(suite_name: str) -> dict[str, Any]:
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[str(suite_name)]()
    tasks = []
    for task_id in range(int(task_suite.n_tasks)):
        task = task_suite.get_task(int(task_id))
        init_states = task_suite.get_task_init_states(int(task_id))
        tasks.append(
            {
                "task_id": int(task_id),
                "task_description": str(task.language),
                "task_description_key": normalize_instruction_key(task.language),
                "num_init_states": int(len(init_states)),
            }
        )
    return {
        "suite_name": str(suite_name),
        "n_tasks": int(task_suite.n_tasks),
        "tasks": tasks,
    }


def infer_suite_name(
    task_id: int,
    task_description: str,
    dataset: str,
    task_suite_name: str,
) -> str:
    requested_suite = resolve_task_suite_name(dataset=dataset, task_suite_name=task_suite_name)
    requested_suite_key = normalize_instruction_key(task_suite_name)
    requested_dataset_key = normalize_instruction_key(dataset)
    if requested_suite_key not in ("", "auto", "none", "null") or requested_dataset_key not in ("", "auto", "none", "null"):
        return str(requested_suite)

    normalized_desc = normalize_instruction_key(task_description)
    matches = []
    for suite_name in KNOWN_SUITES:
        catalog = load_suite_task_catalog(suite_name)
        if not (0 <= int(task_id) < int(catalog["n_tasks"])):
            continue
        task_meta = catalog["tasks"][int(task_id)]
        if str(task_meta["task_description_key"]) == normalized_desc:
            matches.append(str(suite_name))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) == 0:
        raise RuntimeError(
            f"Could not infer suite for task_id={task_id} task_description={task_description!r}; "
            "please pass --dataset or --task_suite_name explicitly."
        )
    raise RuntimeError(
        f"Ambiguous suite inference for task_id={task_id} task_description={task_description!r}: {matches}. "
        "Please pass --dataset or --task_suite_name explicitly."
    )


def build_val_schedule_specs(
    task_descriptions: Sequence[str],
    init_state_counts: Sequence[int],
    online_val_episodes: int,
    val_seed: int,
) -> list[dict[str, Any]]:
    if len(task_descriptions) != len(init_state_counts):
        raise ValueError("task_descriptions and init_state_counts must have the same length.")
    n_tasks = int(len(task_descriptions))
    if n_tasks <= 0:
        return []
    specs = []
    for ep_idx in range(max(0, int(online_val_episodes))):
        task_id = int(ep_idx % n_tasks)
        init_count = int(init_state_counts[task_id])
        if init_count <= 0:
            raise ValueError(f"Task {task_id} has no init states.")
        init_state_idx = int((int(val_seed) + int(ep_idx)) % init_count)
        specs.append(
            {
                "ep_idx": int(ep_idx),
                "task_id": int(task_id),
                "task_description": str(task_descriptions[task_id]),
                "task_description_key": normalize_instruction_key(task_descriptions[task_id]),
                "init_state_idx": int(init_state_idx),
                "match_basis": "deterministic_val_schedule",
            }
        )
    return specs


def infer_recorded_episode_spec(
    manifest_row: Mapping[str, str] | None,
    schedule_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if manifest_row is None:
        return None
    manifest_task_id = int(manifest_row.get("task_id", -1))
    manifest_desc_key = normalize_instruction_key(manifest_row.get("task_description", ""))
    for spec in schedule_specs:
        if int(spec.get("task_id", -1)) != manifest_task_id:
            continue
        if normalize_instruction_key(spec.get("task_description", "")) != manifest_desc_key:
            continue
        payload = dict(spec)
        payload["match_basis"] = "latest_recorded_manifest"
        payload["manifest_iter_idx"] = int(manifest_row.get("iter_idx", 0))
        payload["manifest_split"] = str(manifest_row.get("split", ""))
        payload["manifest_video_path"] = str(manifest_row.get("video_path", ""))
        return payload
    return None


def default_sidecar_paths_for_suite(suite_name: str) -> tuple[str, str]:
    sidecar_dataset = resolve_rlds_dataset_name(dataset=str(suite_name), task_suite_name=str(suite_name))
    sidecar_root = REPO_ROOT / "data" / "libero_sidecars" / sidecar_dataset
    return str(sidecar_root / "steps.parquet"), str(sidecar_root / "phases.parquet")


def ensure_recovery_asset_for_spec(
    suite_name: str,
    dataset: str,
    task_id: int,
    init_state_idx: int,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if str(args.steps_parquet).strip():
        steps_parquet = str(Path(args.steps_parquet).expanduser().resolve())
    else:
        steps_parquet, _ = default_sidecar_paths_for_suite(suite_name)
    if str(args.phases_parquet).strip():
        phases_parquet = str(Path(args.phases_parquet).expanduser().resolve())
    else:
        _, phases_parquet = default_sidecar_paths_for_suite(suite_name)

    return build_single_state_recovery_asset(
        dataset=str(dataset),
        task_suite_name=str(suite_name),
        task_id=int(task_id),
        init_state_idx=int(init_state_idx),
        rlds_root=str(args.rlds_root),
        steps_parquet=str(steps_parquet),
        phases_parquet=str(phases_parquet),
        output_root=str(output_root),
        device=str(args.device),
        num_steps_wait=int(args.num_steps_wait),
        env_resolution=int(args.env_resolution),
        window_stride=int(args.window_stride),
        recovery_vision_backbone=str(args.recovery_vision_backbone),
        recovery_image_resize_strategy=str(args.recovery_image_resize_strategy),
        force_rebuild=bool(args.force_rebuild),
    )


def write_csv(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_list = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in row_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(to_jsonable(row))


def main() -> None:
    args = build_arg_parser().parse_args()
    if not bool(args.val_deterministic):
        raise ValueError("Exact init-state matching requires --val_deterministic true.")

    rollout_root = Path(args.rollout_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root).strip()
        else (rollout_root / "init_state_matches")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_ids = parse_run_ids(str(args.run_ids))
    run_dirs = discover_rollout_run_dirs(rollout_root=rollout_root, requested_run_ids=run_ids)
    if not run_dirs:
        raise RuntimeError(f"No rollout runs found under {rollout_root}")

    shared_root = output_root / "shared_recovery_assets"
    runs_root = output_root / "runs"
    batch_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        manifest_rows = load_manifest_rows(run_dir / "videos" / "video_manifest.csv")
        latest_row = select_latest_manifest_row(rows=manifest_rows, split=str(args.manifest_split))
        if latest_row is None:
            run_summary = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "status": "skipped",
                "skip_reason": f"Missing manifest rows for split={args.manifest_split}",
            }
            write_json(runs_root / run_dir.name / "match_summary.json", run_summary)
            run_summaries.append(run_summary)
            continue

        inferred_suite = infer_suite_name(
            task_id=int(latest_row["task_id"]),
            task_description=str(latest_row["task_description"]),
            dataset=str(args.dataset),
            task_suite_name=str(args.task_suite_name),
        )
        suite_catalog = load_suite_task_catalog(inferred_suite)
        task_descriptions = [str(item["task_description"]) for item in suite_catalog["tasks"]]
        init_state_counts = [int(item["num_init_states"]) for item in suite_catalog["tasks"]]
        schedule_specs = build_val_schedule_specs(
            task_descriptions=task_descriptions,
            init_state_counts=init_state_counts,
            online_val_episodes=int(args.online_val_episodes),
            val_seed=int(args.val_seed),
        )
        recorded_spec = infer_recorded_episode_spec(manifest_row=latest_row, schedule_specs=schedule_specs)
        selected_specs: Iterable[Mapping[str, Any]]
        if bool(args.match_all_val_episodes):
            selected_specs = schedule_specs
        else:
            if recorded_spec is None:
                selected_specs = []
            else:
                selected_specs = [recorded_spec]

        matched_specs = []
        for spec in selected_specs:
            task_id = int(spec["task_id"])
            init_state_idx = int(spec["init_state_idx"])
            asset_output_root = shared_root / inferred_suite / f"task_{task_id:03d}" / f"init_{init_state_idx:03d}"
            asset_payload = ensure_recovery_asset_for_spec(
                suite_name=inferred_suite,
                dataset=inferred_suite,
                task_id=task_id,
                init_state_idx=init_state_idx,
                output_root=asset_output_root,
                args=args,
            )
            matched_row = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "suite_name": str(inferred_suite),
                "ep_idx": int(spec["ep_idx"]),
                "task_id": int(task_id),
                "task_description": str(spec["task_description"]),
                "init_state_idx": int(init_state_idx),
                "match_basis": str(spec.get("match_basis", "deterministic_val_schedule")),
                "source_episode_key": str(asset_payload["source_episode_key"]),
                "candidate_count": int(asset_payload["candidate_count"]),
                "recovery_status": str(asset_payload.get("recovery_status", "matched")),
                "recovery_asset_path": str(asset_payload["recovery_asset_path"]),
                "aligned_state_cache_path": str(asset_payload["aligned_state_cache_path"]),
            }
            batch_rows.append(matched_row)
            matched_specs.append(matched_row)

        run_summary = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "matched",
            "suite_name": str(inferred_suite),
            "manifest_split": str(args.manifest_split),
            "manifest_row": latest_row,
            "recorded_match": recorded_spec,
            "matched_specs": matched_specs,
        }
        write_json(runs_root / run_dir.name / "match_summary.json", run_summary)
        write_csv(runs_root / run_dir.name / "match_summary.csv", matched_specs)
        run_summaries.append(run_summary)

    write_csv(output_root / "matched_init_states.csv", batch_rows)
    write_json(
        output_root / "summary.json",
        {
            "rollout_root": str(rollout_root),
            "output_root": str(output_root),
            "manifest_split": str(args.manifest_split),
            "match_all_val_episodes": bool(args.match_all_val_episodes),
            "online_val_episodes": int(args.online_val_episodes),
            "val_seed": int(args.val_seed),
            "run_count": int(len(run_dirs)),
            "matched_row_count": int(len(batch_rows)),
            "run_summaries": run_summaries,
        },
    )
    print(f"[InitStateMatch] rollout_root={rollout_root}")
    print(f"[InitStateMatch] output_root={output_root}")
    print(f"[InitStateMatch] matched_runs={len(run_dirs)} matched_rows={len(batch_rows)}")


if __name__ == "__main__":
    main()
