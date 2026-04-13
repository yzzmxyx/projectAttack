"""Small stdlib-only helpers for phase-state cache metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def build_phase_state_cache_metadata(payload: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    output_path = Path(output_path).expanduser().resolve()
    summary = dict(payload.get("metadata", {}))
    return {
        "phase_state_cache_pt": str(output_path),
        "dataset": str(payload.get("dataset", "")),
        "suite_name": str(payload.get("suite_name", "")),
        "phase_parquet": str(payload.get("phase_parquet", "")),
        "phase_starts": list(payload.get("phase_starts", [])),
        "alignment": str(payload.get("alignment", "")),
        "schema_version": int(payload.get("schema_version", 0)),
        "num_steps_wait": int(payload.get("num_steps_wait", 0)),
        "env_resolution": int(payload.get("env_resolution", 0)),
        "num_states": int(summary.get("num_states", 0)),
        "num_records": int(summary.get("num_records", 0)),
        "num_tasks": int(summary.get("num_tasks", 0)),
        "task_ids": list(summary.get("task_ids", [])),
        "max_tasks": summary.get("max_tasks"),
        "max_init_states_per_task": summary.get("max_init_states_per_task"),
        "state_shape_examples": dict(summary.get("state_shape_examples", {})),
    }
