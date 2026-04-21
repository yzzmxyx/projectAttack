"""Shared types for the risk_window package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class DetectorStatus:
    """Canonical detector status values."""

    OK = "ok"
    VISIBILITY_LOW = "visibility_low"
    UNKNOWN_PHASE = "unknown_phase"
    RECALIBRATION_REQUIRED = "recalibration_required"


@dataclass
class PhaseWindowLabel:
    """A single vulnerable window projected into coarse phase/progress space."""

    task_id: str
    init_state_idx: Optional[int]
    phase_id: str
    phase_progress_start: float
    phase_progress_end: float
    window_start_step: int
    window_end_step: int
    suite_name: str = ""
    task_description: str = ""
    rank: Optional[int] = None
    success_rate: Optional[float] = None
    mean_future20_action_l2: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferencePrototype:
    """Optional reference sequence for progress anchoring."""

    task_id: str
    init_state_idx: Optional[int]
    phase_id: str
    progress_points: List[float]
    feature_vectors: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetBundle:
    """Loaded detector assets from an existing vulnerability-search output root."""

    asset_root: str
    suite_name: str
    task_description: str
    nominal_steps: Optional[int]
    labels: List[PhaseWindowLabel] = field(default_factory=list)
    prototypes: List[ReferencePrototype] = field(default_factory=list)
    roi_config: Dict[str, Any] = field(default_factory=dict)
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    source_kind: str = "unknown"
    top_windows_path: str = ""
    run_config_path: str = ""
    steps_parquet_path: str = ""
    phases_parquet_path: str = ""


@dataclass
class DetectorOutput:
    """Per-frame detector output."""

    phase_id: str
    progress: float
    risk_score: float
    in_window: bool
    status: str
    anchor_progress: Optional[float]
    visibility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
