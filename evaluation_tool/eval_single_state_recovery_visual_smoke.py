#!/usr/bin/env python3
"""Visual smoke validation for single-state RLDS recovery pairing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from rlds_recovery_utils import (  # noqa: E402
    DEFAULT_RLDS_ROOT,
    VisionFeatureExtractor,
    _load_selected_rlds_record,
    build_single_state_recovery_asset,
    compute_image_alignment_metrics,
    compute_robot_state_distance,
    extract_env_view_images,
    extract_robot_state_from_obs,
    load_matched_episode_cache,
    resolve_rlds_dataset_name,
    resolve_task_suite_name,
    resolve_torch_device,
    resize_uint8_image,
    to_jsonable,
    total_alignment_score,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="libero_10")
    parser.add_argument("--task_suite_name", type=str, default="auto")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--init_state_idx", type=int, required=True)
    parser.add_argument("--rlds_root", type=str, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--steps_parquet", type=str, required=True)
    parser.add_argument("--phases_parquet", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--env_resolution", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--recovery_vision_backbone", type=str, default="pixel")
    parser.add_argument("--recovery_image_resize_strategy", type=str, default="resize-naive")
    parser.add_argument("--anchor_steps", type=str, default="auto")
    parser.add_argument("--max_anchors", type=int, default=3)
    parser.add_argument("--force_rebuild", action="store_true")
    return parser


def parse_anchor_steps(value: str) -> list[int] | None:
    text = str(value).strip().lower()
    if text in ("", "auto", "none", "default"):
        return None
    steps: list[int] = []
    for token in str(value).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        step = int(stripped)
        if step < 0:
            continue
        steps.append(step)
    return sorted(set(steps))


def select_visual_anchor_steps(
    requested_steps: Sequence[int] | None,
    available_steps: Sequence[int],
    max_anchors: int,
) -> list[int]:
    if requested_steps is not None:
        return sorted(set(int(step) for step in requested_steps if int(step) >= 0))
    sorted_available = sorted(set(int(step) for step in available_steps if int(step) >= 0))
    limit = max(1, int(max_anchors))
    if not sorted_available:
        return [0]
    return sorted_available[:limit]


def render_text_sheet(
    panel_groups: Sequence[dict[str, Any]],
    title: str,
    output_path: str | os.PathLike[str],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    prepared = []
    max_width = 0
    max_height = 0
    for group in panel_groups:
        image = Image.fromarray(np.asarray(group["image"], dtype=np.uint8))
        prepared.append((group, image))
        max_width = max(max_width, image.width)
        max_height = max(max_height, image.height)

    if not prepared:
        raise ValueError("render_text_sheet requires at least one panel.")

    font = ImageFont.load_default()
    title_height = 28
    label_height = 92
    gutter = 16
    width = (len(prepared) * max_width) + ((len(prepared) + 1) * gutter)
    height = title_height + max_height + label_height + (2 * gutter)

    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 8), str(title), fill=(12, 12, 12), font=font)

    for idx, (group, image) in enumerate(prepared):
        x0 = gutter + idx * (max_width + gutter)
        y0 = title_height + gutter
        if image.size != (max_width, max_height):
            padded = Image.new("RGB", (max_width, max_height), color=(255, 255, 255))
            padded.paste(image, ((max_width - image.width) // 2, (max_height - image.height) // 2))
            image = padded
        canvas.paste(image, (x0, y0))
        label_lines = [str(group.get("label", ""))]
        label_lines.extend(str(line) for line in group.get("lines", []))
        label_text = "\n".join(line for line in label_lines if line)
        draw.multiline_text((x0, y0 + max_height + 8), label_text, fill=(32, 32, 32), font=font, spacing=2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def save_rgb(path: str | os.PathLike[str], image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    from PIL import Image

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(output)
    return str(output)


def short_episode_key(source_episode_key: str) -> str:
    parts = str(source_episode_key).split(":")
    if len(parts) >= 3:
        return f"{parts[1]}:{parts[2][:8]}"
    return str(source_episode_key)


def build_reference_views(record: Mapping[str, Any], step: int, resize_size: int) -> dict[str, np.ndarray | None]:
    max_step = max(0, min(int(step), int(record["agentview_rgb"].shape[0]) - 1))
    wrist_rgb = record.get("eye_in_hand_rgb")
    wrist = None
    if wrist_rgb is not None:
        wrist = resize_uint8_image(np.asarray(wrist_rgb[max_step], dtype=np.uint8), resize_size)
    return {
        "agentview": resize_uint8_image(np.asarray(record["agentview_rgb"][max_step], dtype=np.uint8), resize_size),
        "wrist": wrist,
    }


def format_metric_lines(prefix: str, metrics: Mapping[str, Any]) -> list[str]:
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.4f}"

    return [
        f"{prefix}align={_fmt(metrics.get('total_alignment_score'))}",
        f"{prefix}img_l1={_fmt(metrics.get('agentview_l1'))}",
        f"{prefix}img_feat={_fmt(metrics.get('agentview_feature_distance'))}",
        f"{prefix}robot={_fmt(metrics.get('robot_distance_total'))}",
    ]


def compare_env_to_reference(
    env_obs: Mapping[str, Any],
    reference_views: Mapping[str, np.ndarray | None],
    reference_robot_state: Mapping[str, Sequence[float]],
    feature_extractor: VisionFeatureExtractor,
    resize_size: int,
) -> dict[str, Any]:
    env_views = extract_env_view_images(env_obs, resize_size=resize_size)
    env_robot = extract_robot_state_from_obs(env_obs)
    robot_metrics = compute_robot_state_distance(env_robot, reference_robot_state)
    image_metrics = compute_image_alignment_metrics(
        env_views=env_views,
        ref_views=reference_views,
        feature_extractor=feature_extractor,
    )
    alignment_score = total_alignment_score(robot_metrics=robot_metrics, image_metrics=image_metrics)
    return {
        "env_views": env_views,
        "robot_metrics": to_jsonable(robot_metrics),
        "image_metrics": to_jsonable(image_metrics),
        "total_alignment_score": float(alignment_score),
        "robot_distance_total": float(robot_metrics["total"]),
        "robot_distance_joint_l2": float(robot_metrics["joint_l2"]),
        "robot_distance_gripper_l2": float(robot_metrics["gripper_l2"]),
        "robot_distance_eef_l2": float(robot_metrics["eef_l2"]),
        "agentview_feature_distance": image_metrics["agentview_feature_distance"],
        "wrist_feature_distance": image_metrics["wrist_feature_distance"],
        "agentview_l1": image_metrics["agentview_l1"],
        "wrist_l1": image_metrics["wrist_l1"],
    }


def restore_env_with_init_and_actions(
    env: object,
    initial_state: np.ndarray,
    dummy_action: Sequence[float],
    raw_actions: np.ndarray,
    step: int,
    num_steps_wait: int,
) -> Mapping[str, Any]:
    env.reset()
    obs = env.set_init_state(np.asarray(initial_state, dtype=np.float32).copy())
    for _ in range(max(0, int(num_steps_wait))):
        obs, _reward, done_wait, _info = env.step(list(dummy_action))
        if done_wait:
            break
    for action in raw_actions[: max(0, int(step))]:
        obs, _reward, done_replay, _info = env.step(np.asarray(action, dtype=np.float32).tolist())
        if done_replay:
            break
    return obs


def restore_env_from_sim_state(
    env: object,
    sim_state: np.ndarray | torch.Tensor,
) -> Mapping[str, Any]:
    env.reset()
    if torch.is_tensor(sim_state):
        resolved_state = sim_state.detach().cpu().numpy().astype(np.float32, copy=True)
    else:
        resolved_state = np.asarray(sim_state, dtype=np.float32).copy()
    return env.set_init_state(resolved_state)


def flatten_anchor_metric_row(
    step: int,
    method: str,
    source_episode_key: str,
    metrics: Mapping[str, Any],
    anchor_image_path: str | None,
    reference_image_path: str | None,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "method": str(method),
        "source_episode_key": str(source_episode_key),
        "agentview_feature_distance": metrics.get("agentview_feature_distance"),
        "wrist_feature_distance": metrics.get("wrist_feature_distance"),
        "agentview_l1": metrics.get("agentview_l1"),
        "wrist_l1": metrics.get("wrist_l1"),
        "robot_distance_total": metrics.get("robot_distance_total"),
        "robot_distance_joint_l2": metrics.get("robot_distance_joint_l2"),
        "robot_distance_gripper_l2": metrics.get("robot_distance_gripper_l2"),
        "robot_distance_eef_l2": metrics.get("robot_distance_eef_l2"),
        "total_alignment_score": metrics.get("total_alignment_score"),
        "env_image_path": anchor_image_path,
        "reference_image_path": reference_image_path,
    }


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
            writer.writerow(row)


def main() -> None:
    args = build_arg_parser().parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    recovery_root = output_root / "recovery_asset"

    recovery_asset = build_single_state_recovery_asset(
        dataset=str(args.dataset),
        task_suite_name=str(args.task_suite_name),
        task_id=int(args.task_id),
        init_state_idx=int(args.init_state_idx),
        rlds_root=str(args.rlds_root),
        steps_parquet=str(args.steps_parquet),
        phases_parquet=str(args.phases_parquet),
        output_root=str(recovery_root),
        device=str(args.device),
        num_steps_wait=int(args.num_steps_wait),
        env_resolution=int(args.env_resolution),
        window_stride=int(args.window_stride),
        recovery_vision_backbone=str(args.recovery_vision_backbone),
        recovery_image_resize_strategy=str(args.recovery_image_resize_strategy),
        force_rebuild=bool(args.force_rebuild),
    )

    recovered_source_episode_key = str(recovery_asset["source_episode_key"])
    matched_episode_cache = load_matched_episode_cache(recovery_asset["matched_episode_cache_path"])
    aligned_state_payload = torch.load(recovery_asset["aligned_state_cache_path"], map_location="cpu")
    if not isinstance(aligned_state_payload, dict):
        raise TypeError("Aligned state cache must be a dict payload.")
    aligned_states = aligned_state_payload.get("step_states", {})
    if not isinstance(aligned_states, dict):
        raise TypeError("Aligned state cache does not contain dict `step_states`.")
    if 0 not in aligned_states:
        raise KeyError("Aligned state cache is missing step 0; cannot run recovery-only visual smoke.")

    anchor_steps = select_visual_anchor_steps(
        requested_steps=parse_anchor_steps(args.anchor_steps),
        available_steps=recovery_asset.get("anchor_steps", sorted(aligned_states.keys())),
        max_anchors=int(args.max_anchors),
    )
    recovered_max_steps = int(matched_episode_cache["raw_actions"].shape[0])
    anchor_steps = [
        int(step)
        for step in anchor_steps
        if int(step) in aligned_states and 0 <= int(step) < recovered_max_steps
    ]
    if not anchor_steps:
        raise RuntimeError("No valid recovered anchor steps remain after intersecting with aligned_state_cache.")

    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

    suite_name = resolve_task_suite_name(dataset=str(args.dataset), task_suite_name=str(args.task_suite_name))
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(int(args.task_id))
    init_states = task_suite.get_task_init_states(int(args.task_id))
    if int(args.init_state_idx) < 0 or int(args.init_state_idx) >= len(init_states):
        raise ValueError(f"init_state_idx {args.init_state_idx} is out of range [0, {len(init_states)})")
    initial_state = np.asarray(init_states[int(args.init_state_idx)], dtype=np.float32).copy()
    env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(args.env_resolution)))
    dummy_action = get_libero_dummy_action("openvla")
    feature_extractor = VisionFeatureExtractor(
        backbone_id=str(args.recovery_vision_backbone),
        image_resize_strategy=str(args.recovery_image_resize_strategy),
        device=resolve_torch_device(args.device),
    )
    rlds_dataset_name = resolve_rlds_dataset_name(dataset=str(args.dataset), task_suite_name=str(suite_name))

    try:
        recovered_rlds_record = _load_selected_rlds_record(
            dataset_name=str(rlds_dataset_name),
            rlds_root=str(args.rlds_root),
            source_episode_key=recovered_source_episode_key,
        )

        target_obs = restore_env_with_init_and_actions(
            env=env,
            initial_state=initial_state,
            dummy_action=dummy_action,
            raw_actions=np.zeros((0, 7), dtype=np.float32),
            step=0,
            num_steps_wait=int(args.num_steps_wait),
        )
        target_views = extract_env_view_images(target_obs, resize_size=max(64, int(args.env_resolution)))
        target_dir = output_root / "target_initial"
        target_agent_path = save_rgb(target_dir / "agentview.png", target_views["agentview"])
        target_wrist_path = save_rgb(target_dir / "wrist.png", target_views["wrist"])

        initial_recovered_ref = build_reference_views(
            record=recovered_rlds_record,
            step=0,
            resize_size=max(64, int(args.env_resolution)),
        )
        recovered_initial_obs = restore_env_from_sim_state(env=env, sim_state=aligned_states[0])
        recovered_initial_metrics = compare_env_to_reference(
            env_obs=recovered_initial_obs,
            reference_views=initial_recovered_ref,
            reference_robot_state={
                "joint_state": matched_episode_cache["joint_states"][0],
                "gripper_state": matched_episode_cache["gripper_states"][0],
                "eef_state": matched_episode_cache["eef_states"][0],
            },
            feature_extractor=feature_extractor,
            resize_size=max(64, int(args.env_resolution)),
        )
        target_vs_recovered = compute_image_alignment_metrics(
            env_views=target_views,
            ref_views=initial_recovered_ref,
            feature_extractor=feature_extractor,
        )
        recovered_initial_env_path = save_rgb(
            target_dir / "recovered_env_agentview.png",
            recovered_initial_metrics["env_views"]["agentview"],
        )
        save_rgb(target_dir / "recovered_rlds_ref_agentview.png", initial_recovered_ref["agentview"])
        save_rgb(target_dir / "recovered_env_wrist.png", recovered_initial_metrics["env_views"]["wrist"])
        save_rgb(target_dir / "recovered_rlds_ref_wrist.png", initial_recovered_ref["wrist"])

        render_text_sheet(
            panel_groups=[
                {
                    "image": target_views["agentview"],
                    "label": "Target Init Env",
                    "lines": [
                        f"task={args.task_id} init={args.init_state_idx}",
                        f"wait={int(args.num_steps_wait)}",
                    ],
                },
                {
                    "image": initial_recovered_ref["agentview"],
                    "label": "Recovered RLDS Ref",
                    "lines": [
                        short_episode_key(recovered_source_episode_key),
                        f"img_l1={target_vs_recovered['agentview_l1']:.4f}" if target_vs_recovered["agentview_l1"] is not None else "img_l1=n/a",
                        f"img_feat={target_vs_recovered['agentview_feature_distance']:.4f}" if target_vs_recovered["agentview_feature_distance"] is not None else "img_feat=n/a",
                    ],
                },
                {
                    "image": recovered_initial_metrics["env_views"]["agentview"],
                    "label": "Recovered Env",
                    "lines": format_metric_lines("", recovered_initial_metrics),
                },
            ],
            title=f"Single-State Recovery Smoke | task={args.task_id} init={args.init_state_idx}",
            output_path=output_root / "target_vs_reference_agentview.png",
        )

        metric_rows: list[dict[str, Any]] = []
        anchor_results: list[dict[str, Any]] = []

        for step in anchor_steps:
            anchor_dir = output_root / f"anchor_{int(step):03d}"
            recovered_obs = restore_env_from_sim_state(env=env, sim_state=aligned_states[int(step)])
            recovered_ref = {
                "agentview": resize_uint8_image(
                    np.asarray(matched_episode_cache["agentview_rgb"][int(step)], dtype=np.uint8),
                    max(64, int(args.env_resolution)),
                ),
                "wrist": resize_uint8_image(
                    np.asarray(matched_episode_cache["wrist_rgb"][int(step)], dtype=np.uint8),
                    max(64, int(args.env_resolution)),
                ),
            }
            recovered_metrics = compare_env_to_reference(
                env_obs=recovered_obs,
                reference_views=recovered_ref,
                reference_robot_state={
                    "joint_state": matched_episode_cache["joint_states"][int(step)],
                    "gripper_state": matched_episode_cache["gripper_states"][int(step)],
                    "eef_state": matched_episode_cache["eef_states"][int(step)],
                },
                feature_extractor=feature_extractor,
                resize_size=max(64, int(args.env_resolution)),
            )

            recovered_ref_agent_path = save_rgb(anchor_dir / "recovered_rlds_ref_agentview.png", recovered_ref["agentview"])
            recovered_env_agent_path = save_rgb(anchor_dir / "recovered_env_agentview.png", recovered_metrics["env_views"]["agentview"])
            recovered_ref_wrist_path = save_rgb(anchor_dir / "recovered_rlds_ref_wrist.png", recovered_ref["wrist"])
            recovered_env_wrist_path = save_rgb(anchor_dir / "recovered_env_wrist.png", recovered_metrics["env_views"]["wrist"])

            render_text_sheet(
                panel_groups=[
                    {
                        "image": recovered_ref["agentview"],
                        "label": "Recovered RLDS Ref",
                        "lines": [short_episode_key(recovered_source_episode_key)],
                    },
                    {
                        "image": recovered_metrics["env_views"]["agentview"],
                        "label": "Recovered Env",
                        "lines": format_metric_lines("", recovered_metrics),
                    },
                ],
                title=f"Anchor Step {int(step)} Agentview Comparison",
                output_path=anchor_dir / "agentview_comparison.png",
            )

            if recovered_ref["wrist"] is not None and recovered_metrics["env_views"]["wrist"] is not None:
                render_text_sheet(
                    panel_groups=[
                        {
                            "image": recovered_ref["wrist"],
                            "label": "Recovered RLDS Wrist",
                            "lines": [short_episode_key(recovered_source_episode_key)],
                        },
                        {
                            "image": recovered_metrics["env_views"]["wrist"],
                            "label": "Recovered Env Wrist",
                            "lines": [
                                f"wrist_l1={recovered_metrics['wrist_l1']:.4f}" if recovered_metrics["wrist_l1"] is not None else "wrist_l1=n/a",
                                f"wrist_feat={recovered_metrics['wrist_feature_distance']:.4f}" if recovered_metrics["wrist_feature_distance"] is not None else "wrist_feat=n/a",
                            ],
                        },
                    ],
                    title=f"Anchor Step {int(step)} Wrist Comparison",
                    output_path=anchor_dir / "wrist_comparison.png",
                )

            recovered_row = flatten_anchor_metric_row(
                step=int(step),
                method="recovered",
                source_episode_key=recovered_source_episode_key,
                metrics=recovered_metrics,
                anchor_image_path=recovered_env_agent_path,
                reference_image_path=recovered_ref_agent_path,
            )
            metric_rows.append(recovered_row)

            anchor_summary = {
                "step": int(step),
                "source_episode_key": recovered_source_episode_key,
                "metrics": {key: value for key, value in recovered_metrics.items() if key != "env_views"},
                "agentview_paths": {
                    "reference": recovered_ref_agent_path,
                    "env": recovered_env_agent_path,
                },
                "wrist_paths": {
                    "reference": recovered_ref_wrist_path,
                    "env": recovered_env_wrist_path,
                },
            }
            write_json(anchor_dir / "summary.json", anchor_summary)
            anchor_results.append(anchor_summary)

        write_csv(output_root / "metrics.csv", metric_rows)

        summary = {
            "dataset": str(args.dataset),
            "task_suite_name": str(suite_name),
            "task_id": int(args.task_id),
            "init_state_idx": int(args.init_state_idx),
            "task_description": str(task_description),
            "source_episode_key": recovered_source_episode_key,
            "target_initial_paths": {
                "agentview": target_agent_path,
                "wrist": target_wrist_path,
                "recovered_env_agentview": recovered_initial_env_path,
            },
            "target_initial_reference_comparison": {
                "recovered": to_jsonable(target_vs_recovered),
                "recovered_env_alignment": to_jsonable(
                    {key: value for key, value in recovered_initial_metrics.items() if key != "env_views"}
                ),
            },
            "anchor_steps": [int(step) for step in anchor_steps],
            "vision_backend": str(feature_extractor.backend_name),
            "vision_backend_warning": feature_extractor.warning,
            "recovery_asset_path": str(recovery_root / "recovery_asset.json"),
            "anchor_results": anchor_results,
        }
        write_json(output_root / "summary.json", summary)

        print(f"[VisualSmoke] output_root={output_root}")
        print(f"[VisualSmoke] task_id={int(args.task_id)} init_state_idx={int(args.init_state_idx)}")
        print(f"[VisualSmoke] source_episode_key={recovered_source_episode_key}")
        print(f"[VisualSmoke] anchor_steps={json.dumps([int(step) for step in anchor_steps])}")
        print(f"[VisualSmoke] summary_path={output_root / 'summary.json'}")
        print(f"[VisualSmoke] metrics_path={output_root / 'metrics.csv'}")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
