import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "VLAAttacker" / "white_patch" / "UADA_rollout_online_env.py"
WRAPPER_PATH = REPO_ROOT / "VLAAttacker" / "UADA_rollout_online_env_wrapper.py"
RUN_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_UADA_rollout_online_env.sh"


def _load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("test_online_env_resume_module", MODULE_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_resume_entrypoints_exist():
    wrapper_text = WRAPPER_PATH.read_text(encoding="utf-8")
    script_text = RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'parser.add_argument("--resume_run_dir", default="", type=str)' in wrapper_text
    assert 'RESUME_RUN_DIR_VALUE="${RESUME_RUN_DIR:-}"' in script_text
    assert 'resume_args+=(--resume_run_dir "${RESUME_RUN_DIR_VALUE}")' in script_text


def test_resume_helpers_round_trip_and_validate(tmp_path):
    module = _load_module()
    attacker = module.OpenVLAOnlineEnvAttacker.__new__(module.OpenVLAOnlineEnvAttacker)
    attacker.vla = SimpleNamespace(device=torch.device("cpu"))
    attacker.save_dir = str(tmp_path)
    attacker.best_rollout_score = 1.5
    attacker._wandb_run_id = "wandb-test-id"
    attacker._resume_capable = False
    attacker._latest_resume_checkpoint = ""
    attacker._last_completed_iter = -1
    attacker._init_projection_texture_path = ""
    attacker._initial_projection_texture_snapshot_path = ""
    attacker._initial_patch_snapshot_path = ""

    projection_texture = torch.rand((3, 4, 4), dtype=torch.float32)
    projection_texture.requires_grad_(True)
    photometric_params = module.LearnableProjectorPhotometricParams(
        projector_gain=1.2,
        projector_channel_gain=(1.0, 0.95, 0.9),
        learn_projector_gain=True,
        learn_projector_channel_gain=True,
        device="cpu",
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [projection_texture]},
            {"params": list(photometric_params.parameters())},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    config = {
        "num_iter": 10,
        "patch_size": [3, 4, 4],
        "projection_size": [3, 4, 4],
        "accumulate_steps": 1,
        "warmup": 2,
        "phase1_ratio": 0.4,
        "phase1_rollout": 8,
        "phase2_rollout": 24,
        "learn_projector_gain": True,
        "learn_projector_channel_gain": True,
    }

    last_dir = tmp_path / "last"
    checkpoint_path = attacker._save_resume_checkpoint(
        output_dir=str(last_dir),
        projection_texture=projection_texture,
        photometric_params=photometric_params,
        optimizer=optimizer,
        scheduler=scheduler,
        global_iter_completed=3,
        config=config,
        train_phase_start_counter=7,
        gpu_tuner_state={"level": 1},
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "wandb_run_id": "wandb-test-id",
                "init_projection_texture_path": "/tmp/init_projection_texture.pt",
                "initial_projection_texture_snapshot_path": "/tmp/initial_projection_texture.pt",
                "initial_patch_snapshot_path": "/tmp/initial_patch.pt",
            }
        ),
        encoding="utf-8",
    )

    assert Path(checkpoint_path).exists()

    loaded = attacker._load_resume_checkpoint(str(tmp_path))
    assert loaded["global_iter_completed"] == 3
    assert loaded["next_iter_idx"] == 4
    assert loaded["config"] == config
    assert attacker._wandb_run_id == "wandb-test-id"
    assert attacker._initial_patch_snapshot_path == "/tmp/initial_patch.pt"

    attacker._validate_resume_compatibility(config, loaded["config"])
    with pytest.raises(ValueError):
        attacker._validate_resume_compatibility(
            config,
            {
                **config,
                "phase2_rollout": 99,
            },
        )
