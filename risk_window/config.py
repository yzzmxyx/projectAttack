"""Configuration helpers for the risk_window package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CameraConfig:
    roi: Optional[Dict[str, Any]] = None
    resize: Optional[list[int]] = None
    rotation: int = 0
    perspective_correction: Optional[Dict[str, Any]] = None


@dataclass
class FeatureConfig:
    frame_diff_threshold: float = 12.0
    edge_density_threshold: float = 18.0
    min_visibility_score: float = 0.08
    min_brightness: float = 0.04
    max_brightness: float = 0.98


@dataclass
class MatchingConfig:
    prototype_topk: int = 3
    dtw_band: int = 4
    cosine_weight: float = 0.7


@dataclass
class RiskConfig:
    allowed_phases: list[str] = field(default_factory=list)
    enter_threshold: float = 0.70
    enter_consecutive: int = 3
    exit_threshold: float = 0.45
    exit_consecutive: int = 5
    ema_alpha: float = 0.35
    contact_ratio: float = 0.30
    post_ratio: float = 0.70
    window_margin: float = 0.08


@dataclass
class RuntimeConfig:
    update_interval_ms: int = 150
    temporal_window: int = 12
    log_dir: str = ""
    action_policy: str = "log_only"


@dataclass
class RiskWindowConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_payload(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Risk window config not found: {config_path}")

    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML config requested for risk_window, but PyYAML is not installed. "
                "Use JSON or install `pyyaml`."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported risk window config format: {config_path.suffix}")

    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config payload at {config_path}, got {type(payload).__name__}")
    return payload


def load_risk_window_config(path: str = "") -> RiskWindowConfig:
    payload = {} if str(path).strip() == "" else _load_payload(path)
    return RiskWindowConfig(
        camera=CameraConfig(**payload.get("camera", {})),
        features=FeatureConfig(**payload.get("features", {})),
        matching=MatchingConfig(**payload.get("matching", {})),
        risk=RiskConfig(**payload.get("risk", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
    )
