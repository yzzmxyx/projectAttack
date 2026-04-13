"""Small utilities for LIBERO RLDS sidecar generation.

The sidecar path is intentionally separate from the training RLDS pipeline. It
keeps only scalar/vector supervision needed for coarse action alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


PHASES = ("pre_contact", "contact_manipulate", "post_contact")
PROGRESS_BINS = 8


STEP_VECTOR_DIMS = {
    "raw_action": 7,
    "normalized_action": 7,
    "eef_state": 6,
    "gripper_state": 2,
    "joint_state": 7,
}

BANK_VECTOR_DIMS = {
    "normalized_action": 7,
    "eef_state": 6,
    "gripper_state": 2,
}


def decode_text(value: object) -> str:
    """Decode TFDS bytes/np scalar strings into a normal Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return decode_text(value.item())
        except ValueError:
            pass
    return str(value)


def canonical_instruction(value: object) -> str:
    return " ".join(decode_text(value).strip().split())


def make_episode_key(dataset: str, episode_index: int, source_file_path: str, instruction: str) -> str:
    digest = hashlib.sha1(f"{source_file_path}|{instruction}".encode("utf-8")).hexdigest()[:12]
    return f"{dataset}:{episode_index:06d}:{digest}"


def ensure_float_list(value: object, dim: int, name: str) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} must have length {dim}, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return [float(x) for x in arr]


def stack_vector_column(series: Iterable[object], dim: int, name: str) -> np.ndarray:
    rows = [ensure_float_list(value, dim, name) for value in series]
    if not rows:
        return np.empty((0, dim), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def validate_vector_columns(df: pd.DataFrame, vector_dims: Mapping[str, int]) -> None:
    missing = [name for name in vector_dims if name not in df.columns]
    if missing:
        raise ValueError(f"Missing vector columns: {missing}")
    for name, dim in vector_dims.items():
        stack_vector_column(df[name], dim, name)


def write_parquet(df: pd.DataFrame, path: str | Path, vector_dims: Mapping[str, int] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if vector_dims:
        validate_vector_columns(df, vector_dims)
        for name, dim in vector_dims.items():
            df[name] = [ensure_float_list(value, dim, name) for value in df[name]]
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path, vector_dims: Mapping[str, int] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if vector_dims:
        validate_vector_columns(df, vector_dims)
        for name, dim in vector_dims.items():
            df[name] = [ensure_float_list(value, dim, name) for value in df[name]]
    return df


def update_metadata(path: str | Path, section: str, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata[section] = dict(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def transform_libero_gripper_action(raw_action: np.ndarray) -> np.ndarray:
    """Match OpenVLA's LIBERO action convention for the gripper.

    LIBERO raw action uses roughly -1=open and +1=close. OpenVLA clips to
    [0, 1] and inverts so 1=open and 0=close.
    """
    action = np.asarray(raw_action, dtype=np.float32).copy()
    action[-1] = 1.0 - np.clip(action[-1], 0.0, 1.0)
    return action


def compute_action_stats(transformed_actions: np.ndarray) -> dict[str, list[float]]:
    if transformed_actions.ndim != 2 or transformed_actions.shape[1] != 7:
        raise ValueError(f"Expected transformed_actions with shape [N, 7], got {transformed_actions.shape}")
    return {
        "q01": np.quantile(transformed_actions, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(transformed_actions, 0.99, axis=0).astype(np.float32).tolist(),
        "min": np.min(transformed_actions, axis=0).astype(np.float32).tolist(),
        "max": np.max(transformed_actions, axis=0).astype(np.float32).tolist(),
        "mean": np.mean(transformed_actions, axis=0).astype(np.float32).tolist(),
        "std": np.std(transformed_actions, axis=0).astype(np.float32).tolist(),
    }


def normalize_action_bounds_q99(transformed_action: np.ndarray, q01: Iterable[float], q99: Iterable[float]) -> np.ndarray:
    """Normalize action dims 0:6 with bounds-q99; keep gripper dim unchanged."""
    action = np.asarray(transformed_action, dtype=np.float32).copy()
    low = np.asarray(list(q01), dtype=np.float32)
    high = np.asarray(list(q99), dtype=np.float32)
    denom = high[:6] - low[:6]
    norm = 2.0 * (action[:6] - low[:6]) / (denom + 1e-8) - 1.0
    action[:6] = np.clip(norm, -1.0, 1.0)
    action[:6] = np.where(np.isclose(denom, 0.0), 0.0, action[:6])
    return action


def fallback_time_edges(T: int) -> tuple[int, int]:
    """Return half-open split edges for [0, 0.3T), [0.3T, 0.7T), [0.7T, T]."""
    if T <= 0:
        raise ValueError("T must be positive")
    first = int(math.floor(0.3 * T))
    second = int(math.floor(0.7 * T))
    if T >= 3:
        first = min(max(first, 1), T - 2)
        second = min(max(second, first + 1), T - 1)
    else:
        first = min(max(first, 1), T)
        second = min(max(second, first), T)
    return first, second


def articulation_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    articulation_objects = ("door", "drawer", "microwave", "cabinet", "stove", "knob")
    articulation_verbs = ("open", "close", "turn on", "turn off", "turn")
    return any(obj in lowered for obj in articulation_objects) and any(verb in lowered for verb in articulation_verbs)


def grasp_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    grasp_words = (
        "pick",
        "place",
        "put",
        "move",
        "transfer",
        "stack",
        "mug",
        "bowl",
        "plate",
        "book",
        "box",
        "basket",
        "caddy",
    )
    return any(word in lowered for word in grasp_words)


def gripper_event_edges(gripper_open_action: np.ndarray, T: int) -> tuple[int, int] | None:
    """Split by first close event and last later open event when available.

    The action is assumed to use 1=open and 0=close. Edges are half-open:
    pre=[0, first), contact=[first, second), post=[second, T].
    """
    if T <= 2:
        return None
    grip = np.asarray(gripper_open_action, dtype=np.float32).reshape(-1)
    if grip.shape[0] != T:
        raise ValueError(f"Expected {T} gripper actions, got {grip.shape[0]}")

    close_events = np.flatnonzero((grip[:-1] > 0.5) & (grip[1:] <= 0.5)) + 1
    if close_events.size == 0:
        close_events = np.flatnonzero(grip <= 0.5)
    if close_events.size == 0:
        return None

    first_close = int(close_events[0])
    open_events = np.flatnonzero((grip[:-1] <= 0.5) & (grip[1:] > 0.5)) + 1
    open_events = open_events[open_events > first_close]

    if open_events.size > 0:
        # Include the release/open command in the contact phase.
        second = min(T - 1, int(open_events[-1]) + 1)
    else:
        _, fallback_second = fallback_time_edges(T)
        second = fallback_second

    if first_close <= 0:
        first_close = 1
    if second <= first_close:
        second = min(T - 1, first_close + 1)
    if second >= T:
        second = T - 1
    if not (0 < first_close < second < T):
        return None
    return first_close, second


def phase_for_t(t: int, first_edge: int, second_edge: int) -> tuple[str, int, int]:
    if t < first_edge:
        return "pre_contact", 0, first_edge
    if t < second_edge:
        return "contact_manipulate", first_edge, second_edge
    return "post_contact", second_edge, -1


def phase_progress(t: int, phase_start_t: int, phase_end_t: int, T: int) -> float:
    end = T if phase_end_t < 0 else phase_end_t
    denom = max(1, end - phase_start_t - 1)
    return float(np.clip((t - phase_start_t) / denom, 0.0, 1.0))


def progress_bin(progress: float) -> int:
    return int(min(PROGRESS_BINS - 1, math.floor(float(progress) * PROGRESS_BINS)))
