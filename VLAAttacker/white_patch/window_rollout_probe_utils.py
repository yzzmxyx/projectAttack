"""Pure helpers for phase-window rollout probe bookkeeping."""

from __future__ import annotations

from typing import Optional


VALID_WINDOW_PHASE_SCOPES = ("all", "initial", "contact_manipulate", "post_contact")
VALID_WINDOW_ROLLOUT_METRIC_MODES = ("delta_weighted", "adv_gt")


def normalize_window_rollout_phase_scope(phase_scope: str) -> str:
    value = str(phase_scope).lower().strip()
    aliases = {
        "": "all",
        "all": "all",
        "init": "initial",
        "initial": "initial",
        "contact": "contact_manipulate",
        "contact_manipulate": "contact_manipulate",
        "post": "post_contact",
        "post_contact": "post_contact",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported window rollout phase scope: `{phase_scope}`. "
            f"Expected one of {VALID_WINDOW_PHASE_SCOPES}."
        )
    return aliases[value]


def infer_phase_name_from_boundaries(
    absolute_step_idx: int,
    source_T: int,
    contact_step: Optional[int],
    post_step: Optional[int],
) -> str:
    del source_T
    step_idx = max(0, int(absolute_step_idx))
    if post_step is not None and step_idx >= int(post_step):
        return "post_contact"
    if contact_step is not None and step_idx >= int(contact_step):
        return "contact_manipulate"
    return "pre_contact"


def resolve_phase_window(
    source_T: int,
    contact_step: Optional[int],
    post_step: Optional[int],
    phase_name: str,
) -> Optional[dict]:
    total_steps = max(0, int(source_T))
    if total_steps <= 0:
        return None

    normalized_phase = normalize_window_rollout_phase_scope(phase_name)
    if normalized_phase == "all":
        return None

    contact_value = None if contact_step is None else int(contact_step)
    post_value = None if post_step is None else int(post_step)

    if normalized_phase == "initial":
        start_step = 0
        end_step = total_steps - 1 if contact_value is None else min(total_steps - 1, contact_value - 1)
    elif normalized_phase == "contact_manipulate":
        if contact_value is None:
            return None
        start_step = max(0, min(total_steps - 1, contact_value))
        end_step = total_steps - 1 if post_value is None else min(total_steps - 1, post_value - 1)
    else:
        if post_value is None:
            return None
        start_step = max(0, min(total_steps - 1, post_value))
        end_step = total_steps - 1

    if end_step < start_step:
        return None

    return {
        "phase_name": normalized_phase,
        "window_start_step": int(start_step),
        "window_end_step": int(end_step),
        "window_step_count": int(end_step - start_step + 1),
        "source_T": int(total_steps),
        "contact_step": contact_value,
        "post_step": post_value,
    }


def compute_window_rollout_weight(future_step_idx: int, exp_base: float) -> float:
    return float(exp_base) ** max(0, int(future_step_idx))


def compute_weighted_window_rollout_delta(step_deltas, exp_base: float) -> float:
    total = 0.0
    for future_step_idx, delta in enumerate(step_deltas):
        total += compute_window_rollout_weight(future_step_idx=future_step_idx, exp_base=exp_base) * float(delta)
    return float(total)


def normalize_window_rollout_metric_mode(metric_mode: str) -> str:
    value = str(metric_mode).lower().strip()
    aliases = {
        "": "delta_weighted",
        "delta": "delta_weighted",
        "delta_weighted": "delta_weighted",
        "adv": "adv_gt",
        "adv_gt": "adv_gt",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported window rollout metric mode: `{metric_mode}`. "
            f"Expected one of {VALID_WINDOW_ROLLOUT_METRIC_MODES}."
        )
    return aliases[value]


def select_window_rollout_metric_value(metric_mode: str, delta_weighted: float, adv_gt_action_gap: float) -> float:
    normalized_mode = normalize_window_rollout_metric_mode(metric_mode)
    if normalized_mode == "adv_gt":
        return float(adv_gt_action_gap)
    return float(delta_weighted)
