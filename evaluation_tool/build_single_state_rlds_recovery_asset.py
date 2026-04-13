#!/usr/bin/env python3
"""Build a single-task single-init-state RLDS recovery asset for window search."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from rlds_recovery_utils import (  # noqa: E402
    build_single_state_recovery_asset,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="libero_10")
    parser.add_argument("--task_suite_name", type=str, default="auto")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--init_state_idx", type=int, required=True)
    parser.add_argument("--rlds_root", type=str, default="/home/yxx/roboticAttack/openvla-main/dataset")
    parser.add_argument("--steps_parquet", type=str, required=True)
    parser.add_argument("--phases_parquet", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--env_resolution", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=8)
    parser.add_argument("--recovery_vision_backbone", type=str, default="dinoclip-vit-l-336px")
    parser.add_argument("--recovery_image_resize_strategy", type=str, default="resize-naive")
    parser.add_argument("--force_rebuild", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = os.path.abspath(os.path.expanduser(str(args.output_root)))
    payload = build_single_state_recovery_asset(
        dataset=str(args.dataset),
        task_suite_name=str(args.task_suite_name),
        task_id=int(args.task_id),
        init_state_idx=int(args.init_state_idx),
        rlds_root=str(args.rlds_root),
        steps_parquet=str(args.steps_parquet),
        phases_parquet=str(args.phases_parquet),
        output_root=output_root,
        device=str(args.device),
        num_steps_wait=int(args.num_steps_wait),
        env_resolution=int(args.env_resolution),
        window_stride=int(args.window_stride),
        recovery_vision_backbone=str(args.recovery_vision_backbone),
        recovery_image_resize_strategy=str(args.recovery_image_resize_strategy),
        force_rebuild=bool(args.force_rebuild),
    )
    print(f"[RecoveryAsset] output_root={output_root}")
    print(f"[RecoveryAsset] source_episode_key={payload['source_episode_key']}")
    print(f"[RecoveryAsset] candidate_count={int(payload['candidate_count'])}")
    print(f"[RecoveryAsset] recovery_asset_path={payload['recovery_asset_path']}")
    print(f"[RecoveryAsset] aligned_state_cache_path={payload['aligned_state_cache_path']}")


if __name__ == "__main__":
    main()
