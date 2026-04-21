"""Main detector implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .assets import load_asset_bundle, phase_name_for_progress, task_matches
from .config import RiskWindowConfig, load_risk_window_config
from .features import apply_camera_transforms, extract_visual_features
from .logging import RiskWindowLogger
from .matcher import match_reference_prototypes
from .runtime import HysteresisState, update_hysteresis
from .types import AssetBundle, DetectorOutput, DetectorStatus, PhaseWindowLabel, ReferencePrototype


class RiskWindowDetector:
    """Video-only detector for online high-risk vulnerability windows."""

    def __init__(self, config: RiskWindowConfig, assets: AssetBundle, logger: Optional[RiskWindowLogger] = None):
        self.config = config
        self.assets = assets
        self.logger = logger
        self.current_task_id: Optional[str] = None
        self.current_episode_id: Optional[Any] = None
        self.current_init_state_idx: Optional[int] = None
        self._feature_history: deque[np.ndarray] = deque(maxlen=max(1, int(self.config.runtime.temporal_window)))
        self._frame_history: deque[np.ndarray] = deque(maxlen=max(1, int(self.config.runtime.temporal_window)))
        self._timestamp_history: deque[float] = deque(maxlen=max(1, int(self.config.runtime.temporal_window)))
        self._labels: List[PhaseWindowLabel] = []
        self._prototypes: List[ReferencePrototype] = []
        self._nominal_steps = self.assets.nominal_steps or 100
        self._risk_ema: Optional[float] = None
        self._baseline_frame: Optional[np.ndarray] = None
        self._last_output: Optional[DetectorOutput] = None
        self._hysteresis = HysteresisState()

    @classmethod
    def from_config(cls, config_path: str, asset_root: str, log_dir: str = "") -> "RiskWindowDetector":
        config = load_risk_window_config(config_path)
        assets = load_asset_bundle(asset_root)
        if config.camera.roi is None and isinstance(assets.roi_config, dict):
            roi = assets.roi_config.get("roi")
            if isinstance(roi, dict):
                config.camera.roi = roi
        if log_dir.strip():
            config.runtime.log_dir = log_dir
        logger = RiskWindowLogger(config.runtime.log_dir) if str(config.runtime.log_dir).strip() else None
        return cls(config=config, assets=assets, logger=logger)

    @property
    def last_output(self) -> Optional[DetectorOutput]:
        return self._last_output

    def reset(self, task_id, episode_id=None, init_state_idx=None) -> None:
        self.current_task_id = str(task_id)
        self.current_episode_id = episode_id
        self.current_init_state_idx = None if init_state_idx is None else int(init_state_idx)
        self._feature_history.clear()
        self._frame_history.clear()
        self._timestamp_history.clear()
        self._baseline_frame = None
        self._risk_ema = None
        self._hysteresis = HysteresisState()
        self._last_output = None

        candidate_labels = [label for label in self.assets.labels if task_matches(task_id, label.task_id, label.task_description)]
        if self.current_init_state_idx is not None:
            init_specific = [label for label in candidate_labels if label.init_state_idx == self.current_init_state_idx]
            if init_specific:
                candidate_labels = init_specific
        self._labels = candidate_labels

        candidate_prototypes = [
            prototype for prototype in self.assets.prototypes if task_matches(task_id, prototype.task_id, self.assets.task_description)
        ]
        if self.current_init_state_idx is not None:
            init_specific_protos = [
                prototype for prototype in candidate_prototypes if prototype.init_state_idx == self.current_init_state_idx
            ]
            if init_specific_protos:
                candidate_prototypes = init_specific_protos
        self._prototypes = candidate_prototypes

        if self._labels:
            max_end = max(label.window_end_step for label in self._labels)
            self._nominal_steps = max(self.assets.nominal_steps or 0, max_end + 1)
        else:
            self._nominal_steps = self.assets.nominal_steps or 100

        self.record_event(
            "reset",
            {
                "task_id": self.current_task_id,
                "episode_id": self.current_episode_id,
                "init_state_idx": self.current_init_state_idx,
                "label_count": len(self._labels),
                "prototype_count": len(self._prototypes),
            },
        )

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.logger is None:
            return
        self.logger.log_event(event_type, payload)

    def _resolve_status(self, metadata: Dict[str, float], frame: np.ndarray) -> str:
        if frame.size == 0:
            return DetectorStatus.RECALIBRATION_REQUIRED
        brightness = float(metadata.get("brightness", 0.0))
        visibility = float(metadata.get("visibility_score", 0.0))
        if brightness < float(self.config.features.min_brightness) or brightness > float(self.config.features.max_brightness):
            return DetectorStatus.VISIBILITY_LOW
        if visibility < float(self.config.features.min_visibility_score):
            return DetectorStatus.VISIBILITY_LOW
        if not self._labels:
            return DetectorStatus.UNKNOWN_PHASE
        return DetectorStatus.OK

    def _estimate_progress(self) -> tuple[float, Optional[float], Dict[str, Any]]:
        learned_progress = float(
            min(1.0, max(0.0, (len(self._feature_history) - 1) / float(max(1, int(self._nominal_steps) - 1))))
        )
        match = match_reference_prototypes(
            feature_history=list(self._feature_history),
            prototypes=self._prototypes,
            topk=int(self.config.matching.prototype_topk),
            band=int(self.config.matching.dtw_band),
            cosine_weight=float(self.config.matching.cosine_weight),
        )
        if match is None:
            return learned_progress, None, {}
        anchor_progress = float(match["anchor_progress"])
        blended = float(np.clip((0.65 * learned_progress) + (0.35 * anchor_progress), 0.0, 1.0))
        return blended, anchor_progress, dict(match)

    def _compute_raw_risk(self, progress: float, phase_id: str, metadata: Dict[str, float], status: str) -> tuple[float, Dict[str, Any]]:
        best_score = 0.0
        best_label = None
        max_l2 = max(
            [float(label.mean_future20_action_l2 or 0.0) for label in self._labels] + [1e-6]
        )
        for label in self._labels:
            start = float(label.phase_progress_start)
            end = float(label.phase_progress_end)
            if start <= progress <= end:
                proximity = 1.0
            else:
                distance = min(abs(progress - start), abs(progress - end))
                proximity = max(0.0, 1.0 - (distance / float(self.config.risk.window_margin)))

            severity = float(label.mean_future20_action_l2 or 0.0) / float(max_l2)
            phase_factor = 1.0 if label.phase_id == phase_id else 0.65
            score = proximity * ((0.60 + (0.40 * severity)) * phase_factor)
            if score > best_score:
                best_score = float(score)
                best_label = label

        motion_factor = min(1.0, float(metadata.get("motion_energy", 0.0)) / 0.15)
        visibility_score = float(metadata.get("visibility_score", 0.0))
        raw_risk = float(np.clip((0.85 * best_score) + (0.15 * motion_factor), 0.0, 1.0))
        raw_risk *= visibility_score
        if status != DetectorStatus.OK:
            raw_risk *= 0.25
        return raw_risk, {"best_label": None if best_label is None else asdict(best_label)}

    def predict(self, frame, timestamp) -> DetectorOutput:
        if self.current_task_id is None:
            self.reset(task_id="default")

        transformed = apply_camera_transforms(
            frame=frame,
            camera_cfg=self.config.camera,
            roi_override=self.assets.roi_config.get("roi") if isinstance(self.assets.roi_config, dict) else None,
        )
        prev_frame = self._frame_history[-1] if self._frame_history else None
        if self._baseline_frame is None:
            self._baseline_frame = transformed
        feature_vector, feature_metadata = extract_visual_features(
            current_frame=transformed,
            prev_frame=prev_frame,
            baseline_frame=self._baseline_frame,
            feature_cfg=self.config.features,
        )

        self._frame_history.append(transformed)
        self._feature_history.append(feature_vector)
        self._timestamp_history.append(float(timestamp))

        status = self._resolve_status(feature_metadata, transformed)
        progress, anchor_progress, match_meta = self._estimate_progress()
        phase_id = phase_name_for_progress(
            progress=progress,
            contact_ratio=float(self.config.risk.contact_ratio),
            post_ratio=float(self.config.risk.post_ratio),
        )
        raw_risk, risk_meta = self._compute_raw_risk(progress=progress, phase_id=phase_id, metadata=feature_metadata, status=status)
        if self._risk_ema is None:
            self._risk_ema = raw_risk
        else:
            alpha = float(self.config.risk.ema_alpha)
            self._risk_ema = ((1.0 - alpha) * float(self._risk_ema)) + (alpha * raw_risk)
        risk_score = float(np.clip(self._risk_ema, 0.0, 1.0))

        allowed_phases = set(str(item) for item in self.config.risk.allowed_phases)
        eligible = status == DetectorStatus.OK and (not allowed_phases or phase_id in allowed_phases)
        event = update_hysteresis(
            state=self._hysteresis,
            score=risk_score,
            eligible=eligible,
            enter_threshold=float(self.config.risk.enter_threshold),
            enter_consecutive=int(self.config.risk.enter_consecutive),
            exit_threshold=float(self.config.risk.exit_threshold),
            exit_consecutive=int(self.config.risk.exit_consecutive),
        )

        output = DetectorOutput(
            phase_id=phase_id,
            progress=float(progress),
            risk_score=risk_score,
            in_window=bool(self._hysteresis.in_window),
            status=status,
            anchor_progress=None if anchor_progress is None else float(anchor_progress),
            visibility_score=float(feature_metadata["visibility_score"]),
            metadata={
                **feature_metadata,
                "match": match_meta,
                "risk": risk_meta,
                "task_id": self.current_task_id,
                "episode_id": self.current_episode_id,
                "init_state_idx": self.current_init_state_idx,
            },
        )
        self._last_output = output

        if self.logger is not None:
            self.logger.log_frame(
                {
                    "timestamp": float(timestamp),
                    "task_id": self.current_task_id,
                    "episode_id": self.current_episode_id,
                    "phase_id": output.phase_id,
                    "progress": output.progress,
                    "anchor_progress": output.anchor_progress,
                    "risk_score": output.risk_score,
                    "in_window": int(output.in_window),
                    "status": output.status,
                    "visibility_score": output.visibility_score,
                }
            )
            if event is not None:
                self.logger.log_event(
                    event,
                    {
                        "timestamp": float(timestamp),
                        "task_id": self.current_task_id,
                        "episode_id": self.current_episode_id,
                        "phase_id": output.phase_id,
                        "progress": output.progress,
                        "risk_score": output.risk_score,
                        "status": output.status,
                    },
                )
        return output

    def flush(self) -> Dict[str, Any]:
        payload = {
            "task_id": self.current_task_id,
            "episode_id": self.current_episode_id,
            "init_state_idx": self.current_init_state_idx,
            "label_count": len(self._labels),
            "prototype_count": len(self._prototypes),
            "last_output": None if self._last_output is None else asdict(self._last_output),
        }
        if self.logger is not None:
            payload["logger"] = self.logger.summary()
        return payload
