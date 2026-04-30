from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def canonicalize_instruction(text: str) -> str:
    value = str(text or "").replace("\n", " ").strip().lower()
    return " ".join(value.split())


def normalize_phase_name(phase_name: str) -> str:
    name = str(phase_name or "").strip().lower()
    aliases = {
        "contact": "contact_manipulate",
        "contact_manipulate": "contact_manipulate",
        "contact-manipulate": "contact_manipulate",
        "pre": "pre_contact",
        "pre_contact": "pre_contact",
        "post": "post_contact",
        "post_contact": "post_contact",
        "all": "all",
    }
    if name not in aliases:
        raise ValueError(
            "Unsupported phase name. Expected one of "
            "{all, pre_contact, contact_manipulate, post_contact}."
        )
    return aliases[name]


@dataclass
class OfflinePhaseSelector:
    target_phase: str = "contact_manipulate"
    fallback_enabled: bool = True
    ready: bool = False
    exact_phase_by_key: Dict[Tuple[str, str, int], str] = field(default_factory=dict)
    exact_phase_by_basename_key: Dict[Tuple[str, str, int], str] = field(default_factory=dict)
    phase_ratio_by_instruction: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    median_length_by_instruction: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, int] = field(
        default_factory=lambda: {
            "total": 0,
            "kept": 0,
            "dropped": 0,
            "exact_hits": 0,
            "exact_hits_basename": 0,
            "fallback_hits": 0,
            "unknown": 0,
        }
    )

    @classmethod
    def from_phase_parquet(
        cls,
        phase_parquet_path: str,
        target_phase: str = "contact_manipulate",
        fallback_enabled: bool = True,
    ) -> "OfflinePhaseSelector":
        selector = cls(
            target_phase=normalize_phase_name(target_phase),
            fallback_enabled=bool(fallback_enabled),
        )
        phase_path = Path(phase_parquet_path)
        if not phase_path.exists():
            return selector
        try:
            import pandas as pd
        except Exception:
            return selector

        df = pd.read_parquet(phase_path)
        required_cols = {"instruction", "phase", "t"}
        if not required_cols.issubset(set(df.columns)):
            return selector

        has_source_path = "source_file_path" in df.columns
        has_episode_len = "T" in df.columns

        for row in df.itertuples(index=False):
            instruction_key = canonicalize_instruction(getattr(row, "instruction", ""))
            phase_name = normalize_phase_name(getattr(row, "phase", "all"))
            t_value = int(getattr(row, "t", -1))
            if instruction_key == "" or t_value < 0:
                continue
            if has_source_path:
                source_path = str(getattr(row, "source_file_path", "") or "")
                if source_path != "":
                    selector.exact_phase_by_key[(instruction_key, source_path, t_value)] = phase_name
                    selector.exact_phase_by_basename_key[
                        (instruction_key, Path(source_path).name, t_value)
                    ] = phase_name
            if has_episode_len:
                total_steps = int(getattr(row, "T", -1))
                if total_steps > 1:
                    selector.median_length_by_instruction.setdefault(instruction_key, [])
                    selector.median_length_by_instruction[instruction_key].append(float(total_steps))

        if has_episode_len and ("phase_start_t" in df.columns):
            grouped = df.groupby("instruction", sort=False)
            for instruction, group in grouped:
                instruction_key = canonicalize_instruction(instruction)
                if instruction_key == "":
                    continue
                contact_rows = group[group["phase"] == "contact_manipulate"]
                post_rows = group[group["phase"] == "post_contact"]
                if len(contact_rows) == 0 or len(post_rows) == 0:
                    continue
                t_values = group["T"].astype(np.float32).to_numpy()
                valid = t_values > 1.0
                if not np.any(valid):
                    continue
                contact_ratio = np.median(
                    (contact_rows["phase_start_t"].astype(np.float32) / contact_rows["T"].astype(np.float32)).to_numpy()
                )
                post_ratio = np.median(
                    (post_rows["phase_start_t"].astype(np.float32) / post_rows["T"].astype(np.float32)).to_numpy()
                )
                contact_ratio = float(np.clip(contact_ratio, 0.0, 1.0))
                post_ratio = float(np.clip(post_ratio, contact_ratio, 1.0))
                selector.phase_ratio_by_instruction[instruction_key] = (contact_ratio, post_ratio)

        # Convert provisional lists into medians.
        for key, values in list(selector.median_length_by_instruction.items()):
            if isinstance(values, list):
                if len(values) <= 0:
                    selector.median_length_by_instruction.pop(key, None)
                else:
                    selector.median_length_by_instruction[key] = float(np.median(np.asarray(values, dtype=np.float32)))

        selector.ready = True
        return selector

    @staticmethod
    def _phase_from_ratio(progress: float, contact_ratio: float, post_ratio: float) -> str:
        value = float(np.clip(progress, 0.0, 1.0))
        if value < float(contact_ratio):
            return "pre_contact"
        if value < float(post_ratio):
            return "contact_manipulate"
        return "post_contact"

    def _resolve_phase_exact(self, instruction_key: str, source_file_path: str, timestep: int) -> Optional[Tuple[str, str]]:
        if instruction_key == "" or timestep < 0:
            return None
        source_path = str(source_file_path or "")
        if source_path != "":
            key = (instruction_key, source_path, int(timestep))
            if key in self.exact_phase_by_key:
                return self.exact_phase_by_key[key], "exact"
            basename_key = (instruction_key, Path(source_path).name, int(timestep))
            if basename_key in self.exact_phase_by_basename_key:
                return self.exact_phase_by_basename_key[basename_key], "exact_basename"
        return None

    def _resolve_phase_fallback(
        self,
        instruction_key: str,
        timestep: int,
        episode_length: int,
    ) -> Optional[str]:
        if not self.fallback_enabled or instruction_key == "" or timestep < 0:
            return None
        ratios = self.phase_ratio_by_instruction.get(instruction_key)
        if ratios is None:
            return None

        total_steps = int(episode_length)
        if total_steps <= 1:
            median_length = self.median_length_by_instruction.get(instruction_key, 0.0)
            total_steps = int(round(float(median_length)))
        if total_steps <= 1:
            return None

        progress = float(timestep) / float(max(1, total_steps - 1))
        return self._phase_from_ratio(progress, contact_ratio=ratios[0], post_ratio=ratios[1])

    def select_mask(
        self,
        instructions: Sequence[str],
        timesteps: Sequence[int],
        source_file_paths: Optional[Sequence[str]] = None,
        episode_lengths: Optional[Sequence[int]] = None,
    ) -> Tuple[List[bool], Dict[str, int]]:
        if len(instructions) != len(timesteps):
            raise ValueError("instructions and timesteps must have the same length.")
        size = len(instructions)
        if source_file_paths is None:
            source_file_paths = ["" for _ in range(size)]
        if episode_lengths is None:
            episode_lengths = [-1 for _ in range(size)]

        kept_mask: List[bool] = []
        for idx in range(size):
            instruction_key = canonicalize_instruction(instructions[idx])
            timestep = int(timesteps[idx])
            source_file_path = str(source_file_paths[idx] if idx < len(source_file_paths) else "")
            episode_length = int(episode_lengths[idx] if idx < len(episode_lengths) else -1)

            resolved_phase = None
            resolved_source = "unknown"
            exact_match = self._resolve_phase_exact(
                instruction_key=instruction_key,
                source_file_path=source_file_path,
                timestep=timestep,
            )
            if exact_match is not None:
                resolved_phase, resolved_source = exact_match
            else:
                resolved_phase = self._resolve_phase_fallback(
                    instruction_key=instruction_key,
                    timestep=timestep,
                    episode_length=episode_length,
                )
                if resolved_phase is not None:
                    resolved_source = "fallback"

            keep = False
            if normalize_phase_name(self.target_phase) == "all":
                keep = resolved_phase is not None
            elif resolved_phase is not None:
                keep = normalize_phase_name(resolved_phase) == normalize_phase_name(self.target_phase)

            self.stats["total"] += 1
            if keep:
                self.stats["kept"] += 1
            else:
                self.stats["dropped"] += 1

            if resolved_source == "exact":
                self.stats["exact_hits"] += 1
            elif resolved_source == "exact_basename":
                self.stats["exact_hits_basename"] += 1
            elif resolved_source == "fallback":
                self.stats["fallback_hits"] += 1
            else:
                self.stats["unknown"] += 1

            kept_mask.append(bool(keep))

        return kept_mask, dict(self.stats)


@dataclass
class InnerLoopBatchController:
    inner_loop: int = 50

    def iter_batches_for_outer(self, batch):
        repeat = max(1, int(self.inner_loop))
        for _ in range(repeat):
            yield batch


def combine_rollout_objective(
    action_gap_loss: torch.Tensor,
    siglip_distance: torch.Tensor,
    lambda_action_gap: float,
    lambda_siglip: float,
) -> torch.Tensor:
    return -(float(lambda_action_gap) * action_gap_loss + float(lambda_siglip) * siglip_distance)
