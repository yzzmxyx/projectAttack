#!/usr/bin/env python3
"""Add coarse phase/progress labels to a LIBERO step sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from libero_sidecar_utils import (
    PHASES,
    STEP_VECTOR_DIMS,
    articulation_instruction,
    fallback_time_edges,
    grasp_instruction,
    gripper_event_edges,
    phase_for_t,
    phase_progress,
    progress_bin,
    read_parquet,
    stack_vector_column,
    update_metadata,
    write_parquet,
)


DEFAULT_STEPS_PARQUET = "/home/yxx/projectAttack/data/libero_sidecars/libero_10_no_noops/steps.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps_parquet", default=DEFAULT_STEPS_PARQUET, help="Input step sidecar parquet.")
    parser.add_argument("--output_parquet", default=None, help="Output phase sidecar parquet.")
    parser.add_argument("--metadata_json", default=None, help="Metadata JSON to update.")
    return parser.parse_args()


def _choose_phase_edges(group: pd.DataFrame) -> tuple[int, int, str]:
    instruction = str(group["instruction"].iloc[0])
    T = int(group["T"].iloc[0])

    if articulation_instruction(instruction):
        first, second = fallback_time_edges(T)
        return first, second, "time_thirds_articulation"

    if grasp_instruction(instruction):
        actions = stack_vector_column(group["normalized_action"], 7, "normalized_action")
        event_edges = gripper_event_edges(actions[:, -1], T)
        if event_edges is not None:
            first, second = event_edges
            return first, second, "gripper_events"

    first, second = fallback_time_edges(T)
    return first, second, "time_30_70_fallback"


def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"episode_key", "t", "T", "instruction", "normalized_action"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("episode_key", sort=False):
        group = group.sort_values("t").copy()
        T_values = group["T"].unique()
        if len(T_values) != 1:
            raise ValueError(f"Episode {group['episode_key'].iloc[0]} has inconsistent T values: {T_values}")
        T = int(T_values[0])
        first_edge, second_edge, rule = _choose_phase_edges(group)

        phases: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        progresses: list[float] = []
        bins: list[int] = []

        for t_value in group["t"]:
            t = int(t_value)
            phase, start, end = phase_for_t(t, first_edge, second_edge)
            if end < 0:
                end = T
            progress = phase_progress(t, start, end, T)
            phases.append(phase)
            starts.append(int(start))
            ends.append(int(end))
            progresses.append(float(progress))
            bins.append(progress_bin(progress))

        group["phase"] = phases
        group["phase_start_t"] = starts
        group["phase_end_t"] = ends
        group["progress"] = progresses
        group["progress_bin"] = bins
        group["phase_rule"] = rule
        out_parts.append(group)

    out = pd.concat(out_parts, ignore_index=True)
    if not set(out["phase"].unique()).issubset(set(PHASES)):
        raise ValueError(f"Unexpected phases: {sorted(out['phase'].unique())}")
    if out["progress_bin"].min() < 0 or out["progress_bin"].max() > 7:
        raise ValueError("progress_bin must be in [0, 7]")
    if not np.isfinite(out["progress"].to_numpy(dtype=np.float32)).all():
        raise ValueError("progress contains NaN or Inf")
    return out


def main() -> None:
    args = parse_args()
    steps_path = Path(args.steps_parquet)
    output_path = Path(args.output_parquet) if args.output_parquet else steps_path.with_name("phases.parquet")
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_path.with_name("metadata.json")

    df = read_parquet(steps_path, STEP_VECTOR_DIMS)
    out = add_phase_columns(df)
    write_parquet(out, output_path, STEP_VECTOR_DIMS)

    phase_counts = {str(k): int(v) for k, v in out["phase"].value_counts().sort_index().items()}
    rule_counts = {str(k): int(v) for k, v in out["phase_rule"].value_counts().sort_index().items()}
    bin_counts = {str(k): int(v) for k, v in out["progress_bin"].value_counts().sort_index().items()}
    metadata = {
        "steps_parquet": str(steps_path),
        "phase_parquet": str(output_path),
        "num_rows": int(len(out)),
        "num_episodes": int(out["episode_key"].nunique()),
        "phase_counts": phase_counts,
        "phase_rule_counts": rule_counts,
        "progress_bin_counts": bin_counts,
        "phase_end_t_semantics": "exclusive",
    }
    update_metadata(metadata_path, "phase_sidecar", metadata)

    print(
        "wrote "
        f"{output_path} rows={metadata['num_rows']} episodes={metadata['num_episodes']} "
        f"rules={rule_counts}"
    )


if __name__ == "__main__":
    main()
