"""Structured logging for risk_window detectors."""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict


FRAME_FIELDNAMES = [
    "timestamp",
    "task_id",
    "episode_id",
    "phase_id",
    "progress",
    "anchor_progress",
    "risk_score",
    "in_window",
    "status",
    "visibility_score",
]


class RiskWindowLogger:
    """Writes frame-level CSV and event-level JSONL logs."""

    def __init__(self, log_dir: str, session_name: str = "risk_window"):
        timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        root = os.path.abspath(os.path.expanduser(log_dir))
        os.makedirs(root, exist_ok=True)
        self.session_dir = os.path.join(root, f"{session_name}-{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.frame_log_path = os.path.join(self.session_dir, "frames.csv")
        self.event_log_path = os.path.join(self.session_dir, "events.jsonl")
        self._frame_file = open(self.frame_log_path, "w", newline="", encoding="utf-8")
        self._frame_writer = csv.DictWriter(self._frame_file, fieldnames=FRAME_FIELDNAMES)
        self._frame_writer.writeheader()
        self._event_file = open(self.event_log_path, "w", encoding="utf-8")
        self._event_count = 0
        self._frame_count = 0

    def log_frame(self, row: Dict[str, Any]) -> None:
        payload = {key: row.get(key) for key in FRAME_FIELDNAMES}
        self._frame_writer.writerow(payload)
        self._frame_file.flush()
        self._frame_count += 1

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {"event_type": str(event_type), **payload}
        self._event_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._event_file.flush()
        self._event_count += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "frame_log_path": self.frame_log_path,
            "event_log_path": self.event_log_path,
            "frame_count": self._frame_count,
            "event_count": self._event_count,
        }

    def close(self) -> None:
        self._frame_file.close()
        self._event_file.close()
