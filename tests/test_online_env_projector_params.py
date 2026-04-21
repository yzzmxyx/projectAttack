import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "VLAAttacker" / "white_patch" / "projector_photometric_params.py"
PATCH_COMPARE_PATH = REPO_ROOT / "evaluation_tool" / "eval_online_patch_compare.py"
WINDOW_SEARCH_PATH = REPO_ROOT / "evaluation_tool" / "eval_vulnerability_window_search.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learnable_projector_params_preserve_init_values_inside_bounds():
    module = _load_module(HELPER_PATH, "test_online_env_projector_photometric_params")
    params = module.LearnableProjectorPhotometricParams(
        projector_gain=1.35,
        projector_channel_gain=(1.08, 1.04, 1.00),
        learn_projector_gain=True,
        learn_projector_channel_gain=True,
    )
    gain, channel_gain = params.resolved_values()
    assert abs(float(gain.detach().cpu().item()) - 1.35) < 1e-5
    resolved_channel_gain = [float(v) for v in channel_gain.detach().cpu().tolist()]
    assert all(abs(a - b) < 1e-5 for a, b in zip(resolved_channel_gain, [1.08, 1.04, 1.00]))
    assert 0.5 <= float(gain.detach().cpu().item()) <= 2.0
    assert all(0.7 <= float(v) <= 1.3 for v in resolved_channel_gain)


def test_learnable_projector_params_disabled_mode_stays_fixed_and_untrainable():
    module = _load_module(HELPER_PATH, "test_online_env_projector_photometric_params_disabled")
    params = module.LearnableProjectorPhotometricParams(
        projector_gain=1.25,
        projector_channel_gain=(1.02, 0.98, 0.95),
        learn_projector_gain=False,
        learn_projector_channel_gain=False,
    )
    assert params.has_trainable_params() is False
    assert list(params.parameters()) == []
    gain, channel_gain = params.resolved_values()
    assert float(gain.detach().cpu().item()) == 1.25
    resolved_channel_gain = [float(v) for v in channel_gain.detach().cpu().tolist()]
    assert all(abs(a - b) < 1e-6 for a, b in zip(resolved_channel_gain, [1.02, 0.98, 0.95]))


def test_projector_params_sidecar_round_trip_and_missing_sidecar_fallback(tmp_path):
    module = _load_module(HELPER_PATH, "test_online_env_projector_photometric_params_io")

    checkpoint_dir = tmp_path / "run" / "last"
    checkpoint_dir.mkdir(parents=True)
    patch_path = checkpoint_dir / "patch.pt"
    patch_path.write_bytes(b"patch")

    sidecar_path = module.save_projector_params(
        output_dir=str(checkpoint_dir),
        projector_gain=1.6,
        projector_channel_gain=(1.1, 0.9, 1.2),
    )
    resolved = module.resolve_projector_params_for_patch(
        patch_path=str(patch_path),
        default_projector_gain=1.35,
        default_projector_channel_gain="1.08,1.04,1.00",
    )
    assert resolved["loaded_from_sidecar"] is True
    assert resolved["sidecar_path"] == sidecar_path
    assert resolved["schema_version"] == 1
    assert resolved["projector_gain"] == 1.6
    assert resolved["projector_channel_gain"] == (1.1, 0.9, 1.2)

    fallback_patch_path = tmp_path / "run" / "0" / "patch.pt"
    fallback_patch_path.parent.mkdir(parents=True)
    fallback_patch_path.write_bytes(b"patch")
    fallback = module.resolve_projector_params_for_patch(
        patch_path=str(fallback_patch_path),
        default_projector_gain=1.35,
        default_projector_channel_gain="1.08,1.04,1.00",
    )
    assert fallback["loaded_from_sidecar"] is False
    assert fallback["projector_gain"] == 1.35
    assert fallback["projector_channel_gain"] == (1.08, 1.04, 1.0)


def test_online_env_eval_entrypoints_resolve_projector_sidecars():
    patch_compare_contents = PATCH_COMPARE_PATH.read_text(encoding="utf-8")
    window_search_contents = WINDOW_SEARCH_PATH.read_text(encoding="utf-8")
    assert "resolve_projector_params_for_patch" in patch_compare_contents
    assert "resolve_projector_params_for_patch" in window_search_contents
