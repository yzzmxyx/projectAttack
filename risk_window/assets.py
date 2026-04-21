"""Asset loading and validation for risk_window."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .types import AssetBundle, PhaseWindowLabel, ReferencePrototype


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower().replace("\n", " ")
    return " ".join(text.split())


def phase_name_for_progress(progress: float, contact_ratio: float = 0.30, post_ratio: float = 0.70) -> str:
    value = min(max(float(progress), 0.0), 1.0)
    contact = min(max(float(contact_ratio), 0.0), 1.0)
    post = min(max(float(post_ratio), contact), 1.0)
    if value < contact:
        return "pre_contact"
    if value < post:
        return "contact_manipulate"
    return "post_contact"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON mapping at {path}, got {type(payload).__name__}")
    return payload


def discover_asset_files(asset_root: str) -> Dict[str, str]:
    root = Path(asset_root).expanduser().resolve()
    files = {
        "asset_root": str(root),
        "top_windows": str(root / "top_windows.json"),
        "window_annotations": str(root / "window_annotations.json"),
        "window_summary": str(root / "window_summary.json"),
        "run_config": str(root / "run_config.json"),
        "reference_prototypes": str(root / "reference_prototypes.json"),
        "roi_config": str(root / "roi_config.json"),
        "steps_parquet": str(root / "steps.parquet"),
        "phases_parquet": str(root / "phases.parquet"),
    }
    return files


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_prototypes(path: str) -> List[ReferencePrototype]:
    if not os.path.exists(path):
        return []
    payload = _load_json(path)
    rows = payload.get("prototypes", [])
    if not isinstance(rows, list):
        raise TypeError(f"Expected `prototypes` list in {path}")
    prototypes = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prototypes.append(
            ReferencePrototype(
                task_id=str(row.get("task_id", "")),
                init_state_idx=row.get("init_state_idx"),
                phase_id=str(row.get("phase_id", "unknown")),
                progress_points=[float(item) for item in row.get("progress_points", [])],
                feature_vectors=[[float(value) for value in vector] for vector in row.get("feature_vectors", [])],
                metadata=dict(row.get("metadata", {})),
            )
        )
    return prototypes


def _build_labels_from_generic_annotations(
    rows: Iterable[Dict[str, Any]],
    suite_name: str,
    task_id: Any,
    init_state_idx: Any,
    task_description: str,
    nominal_steps: Optional[int],
    window_size_default: int,
) -> List[PhaseWindowLabel]:
    labels: List[PhaseWindowLabel] = []
    denom = max(1, int(nominal_steps) - 1) if nominal_steps is not None else None
    for rank, row in enumerate(rows):
        start = _coerce_int(row.get("window_start_step", row.get("window_start", 0)), 0)
        end = _coerce_int(row.get("window_end_step", start + _coerce_int(row.get("window_size", window_size_default), window_size_default)), start + window_size_default)
        if end <= start:
            end = start + max(1, window_size_default)

        progress_start = row.get("phase_progress_start")
        progress_end = row.get("phase_progress_end")
        if progress_start is None or progress_end is None:
            local_denom = denom if denom is not None else max(1, end)
            progress_start = float(start) / float(local_denom)
            progress_end = float(end) / float(local_denom)
        progress_start = min(max(float(progress_start), 0.0), 1.0)
        progress_end = min(max(float(progress_end), progress_start), 1.0)
        phase_id = str(row.get("phase_id", phase_name_for_progress((progress_start + progress_end) / 2.0)))

        labels.append(
            PhaseWindowLabel(
                task_id=str(row.get("task_id", task_id)),
                init_state_idx=row.get("init_state_idx", init_state_idx),
                phase_id=phase_id,
                phase_progress_start=progress_start,
                phase_progress_end=progress_end,
                window_start_step=start,
                window_end_step=end,
                suite_name=str(row.get("suite_name", suite_name)),
                task_description=str(row.get("task_description", task_description)),
                rank=row.get("rank", rank),
                success_rate=row.get("success_rate"),
                mean_future20_action_l2=row.get("mean_future20_action_l2"),
                metadata=dict(row),
            )
        )
    return labels


def load_asset_bundle(asset_root: str) -> AssetBundle:
    files = discover_asset_files(asset_root)
    root = files["asset_root"]
    existing_top_windows = os.path.exists(files["top_windows"])
    existing_annotations = os.path.exists(files["window_annotations"])
    existing_window_summary = os.path.exists(files["window_summary"])
    if not (existing_top_windows or existing_annotations or existing_window_summary):
        raise FileNotFoundError(
            f"No supported window annotations found under {root}. "
            "Expected one of top_windows.json, window_annotations.json, or window_summary.json."
        )

    run_metadata = _load_json(files["run_config"]) if os.path.exists(files["run_config"]) else {}
    args_payload = run_metadata.get("args", {}) if isinstance(run_metadata.get("args"), dict) else {}

    suite_name = str(run_metadata.get("task_suite_name", args_payload.get("task_suite_name", "")))
    task_description = str(run_metadata.get("task_description", ""))
    task_id = run_metadata.get("task_id", args_payload.get("task_id", ""))
    init_state_idx = run_metadata.get("init_state_idx", args_payload.get("init_state_idx"))
    nominal_steps = run_metadata.get("gt_num_steps")
    if nominal_steps is None:
        nominal_steps = args_payload.get("gt_num_steps")
    if nominal_steps is not None:
        nominal_steps = _coerce_int(nominal_steps, 0)

    window_size_default = _coerce_int(args_payload.get("window_size", run_metadata.get("window_size", 4)), 4)

    if os.path.exists(files["top_windows"]):
        payload = _load_json(files["top_windows"])
        rows = payload.get("top_windows", [])
        source_kind = "top_windows"
    elif os.path.exists(files["window_annotations"]):
        payload = _load_json(files["window_annotations"])
        rows = payload.get("annotations", [])
        source_kind = "window_annotations"
    else:
        payload = _load_json(files["window_summary"])
        rows = payload.get("top_windows", payload.get("annotations", []))
        source_kind = "window_summary"

    if not isinstance(rows, list):
        raise TypeError("Window asset file must contain a list of window rows.")

    labels = _build_labels_from_generic_annotations(
        rows=rows,
        suite_name=suite_name,
        task_id=task_id,
        init_state_idx=init_state_idx,
        task_description=task_description,
        nominal_steps=nominal_steps,
        window_size_default=window_size_default,
    )

    steps_parquet_path = files["steps_parquet"]
    phases_parquet_path = files["phases_parquet"]
    if os.path.exists(files["run_config"]):
        steps_parquet_path = str(run_metadata.get("steps_parquet", steps_parquet_path) or steps_parquet_path)
        phases_parquet_path = str(run_metadata.get("phases_parquet", phases_parquet_path) or phases_parquet_path)

    roi_config = _load_json(files["roi_config"]) if os.path.exists(files["roi_config"]) else {}
    prototypes = _load_prototypes(files["reference_prototypes"])

    return AssetBundle(
        asset_root=root,
        suite_name=suite_name,
        task_description=task_description,
        nominal_steps=nominal_steps,
        labels=labels,
        prototypes=prototypes,
        roi_config=roi_config,
        run_metadata=run_metadata,
        source_kind=source_kind,
        top_windows_path=files["top_windows"] if os.path.exists(files["top_windows"]) else "",
        run_config_path=files["run_config"] if os.path.exists(files["run_config"]) else "",
        steps_parquet_path=steps_parquet_path if os.path.exists(steps_parquet_path) else "",
        phases_parquet_path=phases_parquet_path if os.path.exists(phases_parquet_path) else "",
    )


def validate_asset_root(asset_root: str) -> Dict[str, Any]:
    files = discover_asset_files(asset_root)
    result = {
        "asset_root": files["asset_root"],
        "exists": os.path.isdir(files["asset_root"]),
        "top_windows_exists": os.path.exists(files["top_windows"]),
        "window_annotations_exists": os.path.exists(files["window_annotations"]),
        "window_summary_exists": os.path.exists(files["window_summary"]),
        "run_config_exists": os.path.exists(files["run_config"]),
        "reference_prototypes_exists": os.path.exists(files["reference_prototypes"]),
        "roi_config_exists": os.path.exists(files["roi_config"]),
        "steps_parquet_exists": os.path.exists(files["steps_parquet"]),
        "phases_parquet_exists": os.path.exists(files["phases_parquet"]),
    }
    if result["top_windows_exists"] or result["window_annotations_exists"] or result["window_summary_exists"]:
        bundle = load_asset_bundle(asset_root)
        result.update(
            {
                "source_kind": bundle.source_kind,
                "label_count": len(bundle.labels),
                "prototype_count": len(bundle.prototypes),
                "suite_name": bundle.suite_name,
                "task_description": bundle.task_description,
                "nominal_steps": bundle.nominal_steps,
            }
        )
    return result


def task_matches(task_selector: Any, task_id: Any, task_description: str) -> bool:
    selector = _normalize_text(task_selector)
    if selector == "":
        return True
    return selector == _normalize_text(task_id) or selector == _normalize_text(task_description)
