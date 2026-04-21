import json
from pathlib import Path

from risk_window.assets import load_asset_bundle, validate_asset_root


def test_load_asset_bundle_from_existing_top_windows_layout(tmp_path: Path):
    asset_root = tmp_path / "window_run"
    asset_root.mkdir()
    (asset_root / "steps.parquet").write_bytes(b"")
    (asset_root / "phases.parquet").write_bytes(b"")

    run_config = {
        "task_suite_name": "libero_10",
        "task_id": 0,
        "init_state_idx": 2,
        "task_description": "put both the alphabet soup and the tomato sauce in the basket",
        "gt_num_steps": 300,
        "steps_parquet": str(asset_root / "steps.parquet"),
        "phases_parquet": str(asset_root / "phases.parquet"),
        "args": {"window_size": 4},
    }
    top_windows = {
        "top_windows": [
            {
                "rank": 0,
                "task_id": 0,
                "init_state_idx": 2,
                "window_start": 120,
                "window_size": 4,
                "success_rate": 0.0,
                "mean_future20_action_l2": 2.5,
            },
            {
                "rank": 1,
                "task_id": 0,
                "init_state_idx": 2,
                "window_start": 240,
                "window_size": 4,
                "success_rate": 0.2,
                "mean_future20_action_l2": 1.5,
            },
        ]
    }
    prototypes = {
        "prototypes": [
            {
                "task_id": "0",
                "init_state_idx": 2,
                "phase_id": "contact_manipulate",
                "progress_points": [0.0, 0.5, 1.0],
                "feature_vectors": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            }
        ]
    }
    roi_config = {"roi": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.7, "normalized": True}}

    (asset_root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (asset_root / "top_windows.json").write_text(json.dumps(top_windows), encoding="utf-8")
    (asset_root / "reference_prototypes.json").write_text(json.dumps(prototypes), encoding="utf-8")
    (asset_root / "roi_config.json").write_text(json.dumps(roi_config), encoding="utf-8")

    bundle = load_asset_bundle(str(asset_root))
    assert bundle.source_kind == "top_windows"
    assert bundle.suite_name == "libero_10"
    assert bundle.nominal_steps == 300
    assert bundle.steps_parquet_path.endswith("steps.parquet")
    assert bundle.phases_parquet_path.endswith("phases.parquet")
    assert len(bundle.labels) == 2
    assert bundle.labels[0].phase_id == "contact_manipulate"
    assert bundle.labels[1].phase_id == "post_contact"
    assert len(bundle.prototypes) == 1
    assert bundle.roi_config["roi"]["normalized"] is True


def test_validate_asset_root_reports_existing_files(tmp_path: Path):
    asset_root = tmp_path / "window_run"
    asset_root.mkdir()
    (asset_root / "top_windows.json").write_text(json.dumps({"top_windows": []}), encoding="utf-8")
    report = validate_asset_root(str(asset_root))
    assert report["exists"] is True
    assert report["top_windows_exists"] is True
    assert report["label_count"] == 0
