"""LIBERO runtime adapter for risk_window."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from ..model import RiskWindowDetector


class LiberoRiskWindowAdapter:
    """Thin adapter between the detector and the LIBERO simulation eval loop."""

    def __init__(self, detector: RiskWindowDetector, action_policy: str = "log_only", overlay: bool = False):
        self.detector = detector
        self.action_policy = str(action_policy).strip().lower() or "log_only"
        self.overlay = bool(overlay)
        self._last_safe_action = None

    @classmethod
    def from_runtime_args(cls, config_path: str, asset_root: str, log_dir: str = "", action_policy: str = "log_only", overlay: bool = False):
        default_log_dir = log_dir if str(log_dir).strip() else os.path.join(".", "rollouts", "risk_window", "libero")
        detector = RiskWindowDetector.from_config(config_path=config_path, asset_root=asset_root, log_dir=default_log_dir)
        return cls(detector=detector, action_policy=action_policy, overlay=overlay)

    def reset(self, task_id, episode_id=None, init_state_idx=None) -> None:
        self._last_safe_action = None
        self.detector.reset(task_id=task_id, episode_id=episode_id, init_state_idx=init_state_idx)

    def inspect(self, frame, timestamp):
        return self.detector.predict(frame=frame, timestamp=timestamp)

    def apply_action_policy(self, proposed_action, dummy_wait_action=None) -> Dict[str, Any]:
        output = self.detector.last_output
        action = np.asarray(proposed_action, dtype=np.float32)
        policy_action = "log_only"
        abort_episode = False

        if output is None or (not output.in_window) or self.action_policy == "log_only":
            self._last_safe_action = action.copy()
        elif self.action_policy == "hold_last_action" and self._last_safe_action is not None:
            action = self._last_safe_action.copy()
            policy_action = "hold_last_action"
        elif self.action_policy == "dummy_wait" and dummy_wait_action is not None:
            action = np.asarray(dummy_wait_action, dtype=np.float32).copy()
            policy_action = "dummy_wait"
        elif self.action_policy == "abort_episode":
            policy_action = "abort_episode"
            abort_episode = True

        if output is not None and output.in_window and policy_action != "log_only":
            self.detector.record_event(
                "policy_action_applied",
                {
                    "task_id": self.detector.current_task_id,
                    "episode_id": self.detector.current_episode_id,
                    "risk_score": output.risk_score,
                    "phase_id": output.phase_id,
                    "policy_action": policy_action,
                },
            )
        return {
            "action": action,
            "policy_action": policy_action,
            "abort_episode": abort_episode,
            "detector_output": output,
        }

    def flush(self) -> Dict[str, Any]:
        return self.detector.flush()
