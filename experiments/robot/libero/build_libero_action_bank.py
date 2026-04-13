#!/usr/bin/env python3
"""Build a minimal instruction/phase/progress action memory bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from libero_sidecar_utils import (
    BANK_VECTOR_DIMS,
    PHASES,
    STEP_VECTOR_DIMS,
    read_parquet,
    stack_vector_column,
    update_metadata,
    write_parquet,
)


DEFAULT_PHASE_PARQUET = "/home/yxx/projectAttack/data/libero_sidecars/libero_10_no_noops/phases.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_parquet", default=DEFAULT_PHASE_PARQUET, help="Input phase sidecar parquet.")
    parser.add_argument("--output_parquet", default=None, help="Output flat action bank parquet.")
    parser.add_argument("--output_pt", default=None, help="Output torch action bank path.")
    parser.add_argument("--metadata_json", default=None, help="Metadata JSON to update.")
    return parser.parse_args()


def build_bank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"instruction", "phase", "progress_bin", "normalized_action", "eef_state", "gripper_state"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.loc[:, ["instruction", "phase", "progress_bin", "normalized_action", "eef_state", "gripper_state"]].copy()
    out["progress_bin"] = out["progress_bin"].astype(int)
    out["bank_key"] = (
        out["instruction"].astype(str)
        + "||"
        + out["phase"].astype(str)
        + "||"
        + out["progress_bin"].astype(str)
    )
    out = out.loc[:, ["bank_key", "instruction", "phase", "progress_bin", "normalized_action", "eef_state", "gripper_state"]]

    if not set(out["phase"].unique()).issubset(set(PHASES)):
        raise ValueError(f"Unexpected phases: {sorted(out['phase'].unique())}")
    if out["progress_bin"].min() < 0 or out["progress_bin"].max() > 7:
        raise ValueError("progress_bin must be in [0, 7]")
    return out


def build_torch_bank(bank_df: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, torch.Tensor | int]]:
    bank: dict[tuple[str, str, int], dict[str, torch.Tensor | int]] = {}
    for key, group in bank_df.groupby(["instruction", "phase", "progress_bin"], sort=True):
        instruction, phase, progress_bin = key
        actions = stack_vector_column(group["normalized_action"], 7, "normalized_action")
        eef = stack_vector_column(group["eef_state"], 6, "eef_state")
        gripper = stack_vector_column(group["gripper_state"], 2, "gripper_state")
        states = np.concatenate([eef, gripper], axis=1).astype(np.float32)
        bank[(str(instruction), str(phase), int(progress_bin))] = {
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "states": torch.from_numpy(states),
            "num_samples": int(len(group)),
        }
    return bank


def main() -> None:
    args = parse_args()
    phase_path = Path(args.phase_parquet)
    output_parquet = Path(args.output_parquet) if args.output_parquet else phase_path.with_name("action_bank.parquet")
    output_pt = Path(args.output_pt) if args.output_pt else phase_path.with_name("action_bank.pt")
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_parquet.with_name("metadata.json")

    phase_df = read_parquet(phase_path, STEP_VECTOR_DIMS)
    bank_df = build_bank_dataframe(phase_df)
    write_parquet(bank_df, output_parquet, BANK_VECTOR_DIMS)

    torch_bank = build_torch_bank(bank_df)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_bank, output_pt)

    group_sizes = [int(value["num_samples"]) for value in torch_bank.values()]
    metadata = {
        "phase_parquet": str(phase_path),
        "action_bank_parquet": str(output_parquet),
        "action_bank_pt": str(output_pt),
        "num_rows": int(len(bank_df)),
        "num_groups": int(len(torch_bank)),
        "num_instructions": int(bank_df["instruction"].nunique()),
        "min_group_size": int(min(group_sizes)) if group_sizes else 0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "bank_key": "(instruction, phase, progress_bin)",
        "torch_bank_value_schema": {"actions": "[N, 7]", "states": "[N, 8]", "num_samples": "int"},
    }
    update_metadata(metadata_path, "action_bank", metadata)

    print(
        "wrote "
        f"{output_parquet} and {output_pt} rows={metadata['num_rows']} "
        f"groups={metadata['num_groups']} min_group={metadata['min_group_size']}"
    )


if __name__ == "__main__":
    main()
