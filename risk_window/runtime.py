"""Runtime state helpers for the risk_window package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HysteresisState:
    in_window: bool = False
    enter_hits: int = 0
    exit_hits: int = 0


def update_hysteresis(
    state: HysteresisState,
    score: float,
    eligible: bool,
    enter_threshold: float,
    enter_consecutive: int,
    exit_threshold: float,
    exit_consecutive: int,
) -> Optional[str]:
    """Update enter/exit state and return an optional event name."""
    if not state.in_window:
        if eligible and float(score) >= float(enter_threshold):
            state.enter_hits += 1
        else:
            state.enter_hits = 0
        state.exit_hits = 0
        if state.enter_hits >= max(1, int(enter_consecutive)):
            state.in_window = True
            state.enter_hits = 0
            return "enter_window"
        return None

    if (not eligible) or float(score) <= float(exit_threshold):
        state.exit_hits += 1
    else:
        state.exit_hits = 0
    state.enter_hits = 0
    if state.exit_hits >= max(1, int(exit_consecutive)):
        state.in_window = False
        state.exit_hits = 0
        return "exit_window"
    return None
