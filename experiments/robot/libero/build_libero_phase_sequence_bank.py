#!/usr/bin/env python3
"""Build an instruction/phase/horizon sequence action bank from phases.parquet."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from libero_sidecar_utils import PHASES, STEP_VECTOR_DIMS, read_parquet, update_metadata


DEFAULT_PHASE_PARQUET = "/home/yxx/projectAttack/data/libero_sidecars/libero_10_no_noops/phases.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_parquet", default=DEFAULT_PHASE_PARQUET, help="Input phase sidecar parquet.")
    parser.add_argument("--target_phase", default="contact_manipulate", choices=PHASES, help="Phase to extract.")
    parser.add_argument("--horizon", type=int, default=4, help="Fixed sequence horizon H.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument(
        "--short_policy",
        default="drop",
        choices=("drop", "pad_last", "pad_zero"),
        help="How to handle phase segments shorter than horizon.",
    )
    parser.add_argument("--dedup", action="store_true", help="Enable dedup by rounded sequence hash.")
    parser.add_argument("--dedup_round", type=int, default=6, help="Rounding decimals before hashing.")
    parser.add_argument(
        "--max_per_key",
        type=int,
        default=0,
        help="Optional truncation per (instruction, phase, horizon); <=0 means unlimited.",
    )
    parser.add_argument("--output_parquet", default=None, help="Output flat sequence bank parquet.")
    parser.add_argument("--output_pt", default=None, help="Output torch sequence bank path.")
    parser.add_argument("--metadata_json", default=None, help="Metadata JSON to update.")
    return parser.parse_args()


def canonicalize_instruction(text: object) -> str:
    value = str(text or "").replace("\n", " ").strip().lower()
    return " ".join(value.split())


def _ensure_seq_array(value: object, horizon: int, action_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [H, D], got shape {arr.shape}")
    if arr.shape[0] != int(horizon):
        raise ValueError(f"{name} must have H={horizon}, got H={arr.shape[0]}")
    if arr.shape[1] != int(action_dim):
        raise ValueError(f"{name} must have D={action_dim}, got D={arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _to_float_seq(value: object, horizon: int, action_dim: int, name: str) -> list[list[float]]:
    arr = _ensure_seq_array(value, horizon=horizon, action_dim=action_dim, name=name)
    return [[float(x) for x in row] for row in arr]


def _iter_contiguous_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    seq = [int(x) for x in indices]
    if len(seq) <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = seq[0]
    prev = seq[0]
    for idx in seq[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = idx
        prev = idx
    ranges.append((start, prev + 1))
    return ranges


def _sequence_hash(sequence: np.ndarray, round_decimals: int) -> str:
    rounded = np.round(sequence.astype(np.float32), int(round_decimals))
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def _materialize_sequences_from_segment(
    actions: np.ndarray,
    horizon: int,
    stride: int,
    short_policy: str,
) -> list[np.ndarray]:
    length = int(actions.shape[0])
    if length >= int(horizon):
        out: list[np.ndarray] = []
        for start in range(0, length - int(horizon) + 1, int(stride)):
            out.append(actions[start : start + int(horizon)].copy())
        return out

    if short_policy == "drop":
        return []

    if short_policy == "pad_last":
        pad_count = int(horizon) - length
        tail = np.repeat(actions[-1:, :], repeats=pad_count, axis=0)
        return [np.concatenate([actions, tail], axis=0).astype(np.float32)]

    if short_policy == "pad_zero":
        pad_count = int(horizon) - length
        tail = np.zeros((pad_count, int(actions.shape[1])), dtype=np.float32)
        return [np.concatenate([actions, tail], axis=0).astype(np.float32)]

    raise ValueError(f"Unsupported short_policy: {short_policy}")


def build_phase_sequence_bank_dataframe(
    df: pd.DataFrame,
    target_phase: str,
    horizon: int,
    stride: int,
    short_policy: str,
    dedup: bool,
    dedup_round: int,
    max_per_key: int,
) -> pd.DataFrame:
    required = {"episode_key", "t", "T", "instruction", "phase", "normalized_action"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    phase_name = str(target_phase).strip().lower()
    if phase_name not in set(PHASES):
        raise ValueError(f"Unsupported phase `{target_phase}`, expected one of {PHASES}.")
    if int(horizon) <= 0:
        raise ValueError("--horizon must be positive.")
    if int(stride) <= 0:
        raise ValueError("--stride must be positive.")

    rows: list[dict] = []
    for _, group in df.groupby("episode_key", sort=False):
        group = group.sort_values("t").copy()
        phase_mask = (group["phase"].astype(str).str.strip().str.lower() == phase_name).to_numpy(dtype=bool)
        if not bool(np.any(phase_mask)):
            continue
        action_mat = np.asarray(group["normalized_action"].tolist(), dtype=np.float32)
        if action_mat.ndim != 2 or action_mat.shape[1] != 7:
            raise ValueError(f"normalized_action must have shape [N, 7], got {action_mat.shape}")

        phase_indices = np.flatnonzero(phase_mask).tolist()
        contiguous_ranges = _iter_contiguous_ranges(phase_indices)
        for local_start, local_end in contiguous_ranges:
            segment_actions = action_mat[local_start:local_end]
            segment_times = group.iloc[local_start:local_end]["t"].astype(int).to_numpy()
            if segment_actions.shape[0] <= 0:
                continue

            seq_list = _materialize_sequences_from_segment(
                actions=segment_actions,
                horizon=int(horizon),
                stride=int(stride),
                short_policy=str(short_policy),
            )
            for seq_idx, seq in enumerate(seq_list):
                if seq.shape[0] > int(segment_times.shape[0]):
                    start_t = int(segment_times[0])
                else:
                    start_offset = int(seq_idx * int(stride))
                    start_t = int(segment_times[start_offset])
                end_t_exclusive = int(start_t + int(horizon))
                instruction = str(group["instruction"].iloc[0])
                instruction_key = canonicalize_instruction(instruction)
                row = {
                    "instruction": instruction,
                    "instruction_key": instruction_key,
                    "phase": phase_name,
                    "horizon": int(horizon),
                    "episode_key": str(group["episode_key"].iloc[0]),
                    "source_file_path": str(group["source_file_path"].iloc[0]) if "source_file_path" in group.columns else "",
                    "start_t": int(start_t),
                    "end_t_exclusive": int(end_t_exclusive),
                    "normalized_action_seq": _to_float_seq(seq, horizon=int(horizon), action_dim=7, name="normalized_action_seq"),
                    "sequence_hash": _sequence_hash(seq, round_decimals=int(dedup_round)),
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) <= 0:
        return out

    out = out.sort_values(["instruction_key", "phase", "horizon", "episode_key", "start_t"], kind="mergesort").reset_index(drop=True)

    if bool(dedup):
        out = out.drop_duplicates(subset=["instruction_key", "phase", "horizon", "sequence_hash"], keep="first")

    if int(max_per_key) > 0:
        out = (
            out.groupby(["instruction_key", "phase", "horizon"], sort=False, group_keys=False)
            .head(int(max_per_key))
            .reset_index(drop=True)
        )

    return out


def build_torch_phase_sequence_bank(seq_df: pd.DataFrame) -> dict[tuple[str, str, int], torch.Tensor]:
    bank: dict[tuple[str, str, int], torch.Tensor] = {}
    if len(seq_df) <= 0:
        return bank
    for key, group in seq_df.groupby(["instruction_key", "phase", "horizon"], sort=True):
        instruction_key, phase, horizon = key
        seq_list = []
        for value in group["normalized_action_seq"]:
            seq = _ensure_seq_array(value, horizon=int(horizon), action_dim=7, name="normalized_action_seq")
            seq_list.append(seq)
        stacked = np.stack(seq_list, axis=0).astype(np.float32)
        bank[(str(instruction_key), str(phase), int(horizon))] = torch.from_numpy(stacked)
    return bank


def write_sequence_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) > 0:
        horizon_values = sorted(set(int(x) for x in df["horizon"].tolist()))
        if len(horizon_values) != 1:
            raise ValueError(f"Expected one fixed horizon per file, got {horizon_values}")
        horizon = int(horizon_values[0])
        df = df.copy()
        df["normalized_action_seq"] = [
            _to_float_seq(value, horizon=horizon, action_dim=7, name="normalized_action_seq")
            for value in df["normalized_action_seq"]
        ]
    df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    phase_path = Path(args.phase_parquet)
    output_parquet = Path(args.output_parquet) if args.output_parquet else phase_path.with_name(
        f"phase_sequence_bank_h{int(args.horizon)}.parquet"
    )
    output_pt = Path(args.output_pt) if args.output_pt else phase_path.with_name(
        f"phase_sequence_bank_h{int(args.horizon)}.pt"
    )
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_parquet.with_name("metadata.json")

    phase_df = read_parquet(phase_path, STEP_VECTOR_DIMS)
    bank_df = build_phase_sequence_bank_dataframe(
        df=phase_df,
        target_phase=args.target_phase,
        horizon=int(args.horizon),
        stride=int(args.stride),
        short_policy=str(args.short_policy),
        dedup=bool(args.dedup),
        dedup_round=int(args.dedup_round),
        max_per_key=int(args.max_per_key),
    )
    write_sequence_parquet(bank_df, output_parquet)

    torch_bank = build_torch_phase_sequence_bank(bank_df)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_bank, output_pt)

    group_sizes = [int(value.shape[0]) for value in torch_bank.values()]
    metadata = {
        "phase_parquet": str(phase_path),
        "phase_sequence_bank_parquet": str(output_parquet),
        "phase_sequence_bank_pt": str(output_pt),
        "target_phase": str(args.target_phase),
        "horizon": int(args.horizon),
        "stride": int(args.stride),
        "short_policy": str(args.short_policy),
        "dedup": bool(args.dedup),
        "dedup_round": int(args.dedup_round),
        "max_per_key": int(args.max_per_key),
        "num_rows": int(len(bank_df)),
        "num_groups": int(len(torch_bank)),
        "num_instructions": int(bank_df["instruction_key"].nunique()) if len(bank_df) > 0 else 0,
        "min_group_size": int(min(group_sizes)) if group_sizes else 0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "bank_key": "(instruction_key, phase, horizon)",
        "parquet_value_schema": {"normalized_action_seq": "[H, 7]"},
        "torch_value_schema": "[K, H, 7]",
    }
    update_metadata(metadata_path, "phase_sequence_bank", metadata)

    print(
        "wrote "
        f"{output_parquet} and {output_pt} rows={metadata['num_rows']} "
        f"groups={metadata['num_groups']} horizon={metadata['horizon']}"
    )


if __name__ == "__main__":
    main()
