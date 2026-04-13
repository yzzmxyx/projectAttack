"""Pure helpers for mapping rollout progress onto coarse GT phases."""

from __future__ import annotations

import math


def clamp_phase_boundary_ratios(contact_ratio: float, post_ratio: float) -> tuple[float, float]:
    """Clamp coarse phase boundaries into a valid non-decreasing pair."""
    contact = float(contact_ratio)
    post = float(post_ratio)
    if not math.isfinite(contact):
        contact = 0.3
    if not math.isfinite(post):
        post = 0.7
    contact = min(max(contact, 0.0), 1.0)
    post = min(max(post, contact), 1.0)
    return contact, post


def phase_name_for_progress(progress: float, contact_ratio: float, post_ratio: float) -> str:
    """Map absolute normalized progress onto a coarse phase name."""
    contact, post = clamp_phase_boundary_ratios(contact_ratio, post_ratio)
    value = min(max(float(progress), 0.0), 1.0)
    if value < contact:
        return "pre_contact"
    if value < post:
        return "contact_manipulate"
    return "post_contact"


def phase_start_ratio(phase_start_name: str, contact_ratio: float, post_ratio: float) -> float:
    """Return the absolute normalized progress implied by the rollout start state."""
    contact, post = clamp_phase_boundary_ratios(contact_ratio, post_ratio)
    if phase_start_name == "initial":
        return 0.0
    if phase_start_name == "contact_manipulate":
        return contact
    if phase_start_name == "post_contact":
        return post
    raise ValueError(f"Unsupported phase start name: {phase_start_name}")


def infer_gt_phase_for_step(
    step_idx: int,
    horizon: int,
    phase_start_name: str,
    contact_ratio: float,
    post_ratio: float,
) -> str:
    """Infer the GT phase for a rollout step under coarse phase-state starts."""
    start_ratio = phase_start_ratio(phase_start_name, contact_ratio, post_ratio)
    effective_horizon = max(1, int(horizon))
    if effective_horizon <= 1:
        relative_progress = 0.0
    else:
        relative_progress = float(step_idx) / float(max(1, effective_horizon - 1))
    relative_progress = min(max(relative_progress, 0.0), 1.0)
    absolute_progress = start_ratio + (relative_progress * max(0.0, 1.0 - start_ratio))
    return phase_name_for_progress(absolute_progress, contact_ratio, post_ratio)
