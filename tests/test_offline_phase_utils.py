from pathlib import Path

import pandas as pd
import torch
import random

from VLAAttacker.white_patch.offline_phase_utils import (
    InnerLoopBatchController,
    OfflinePhaseSelector,
    combine_rollout_objective,
)


def _build_phase_sidecar(path: Path) -> Path:
    rows = [
        {
            "instruction": "pick up the mug",
            "source_file_path": "/tmp/demo_a.hdf5",
            "t": 0,
            "T": 10,
            "phase": "pre_contact",
            "phase_start_t": 0,
        },
        {
            "instruction": "pick up the mug",
            "source_file_path": "/tmp/demo_a.hdf5",
            "t": 5,
            "T": 10,
            "phase": "contact_manipulate",
            "phase_start_t": 4,
        },
        {
            "instruction": "pick up the mug",
            "source_file_path": "/tmp/demo_a.hdf5",
            "t": 8,
            "T": 10,
            "phase": "post_contact",
            "phase_start_t": 8,
        },
    ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def test_phase_selector_exact_and_fallback_and_drop(tmp_path):
    sidecar = _build_phase_sidecar(tmp_path / "phases.parquet")
    selector = OfflinePhaseSelector.from_phase_parquet(
        phase_parquet_path=str(sidecar),
        target_phase="contact_manipulate",
        fallback_enabled=True,
    )
    mask, stats = selector.select_mask(
        instructions=["pick up the mug", "pick up the mug", "unknown task"],
        timesteps=[5, 4, 3],
        source_file_paths=["/tmp/demo_a.hdf5", "", ""],
        episode_lengths=[10, 10, 10],
    )
    assert mask == [True, True, False]
    assert stats["exact_hits"] >= 1
    assert stats["fallback_hits"] >= 1
    assert stats["unknown"] >= 1


def test_inner_loop_controller_reuses_batch_within_outer():
    controller = InnerLoopBatchController(inner_loop=4)
    batch_a = {"id": "a"}
    batch_b = {"id": "b"}

    repeated_a = list(controller.iter_batches_for_outer(batch_a))
    repeated_b = list(controller.iter_batches_for_outer(batch_b))

    assert len(repeated_a) == 4
    assert all(item is batch_a for item in repeated_a)
    assert len(repeated_b) == 4
    assert all(item is batch_b for item in repeated_b)
    assert repeated_a[0] is not repeated_b[0]


def test_combine_rollout_objective_keeps_gradient_chain():
    action_gap = torch.tensor(2.0, requires_grad=True)
    siglip_distance = torch.tensor(3.0, requires_grad=True)

    loss = combine_rollout_objective(
        action_gap_loss=action_gap,
        siglip_distance=siglip_distance,
        lambda_action_gap=1.0,
        lambda_siglip=0.15,
    )
    loss.backward()

    assert abs(loss.item() + 2.45) < 1e-6
    assert action_gap.grad is not None and abs(action_gap.grad.item() + 1.0) < 1e-6
    assert siglip_distance.grad is not None and abs(siglip_distance.grad.item() + 0.15) < 1e-6


def test_inner_loop_50_randomized_projection_params_change():
    class _MockProjectionSampler:
        def sample(self, randomization_enabled=True):
            if not randomization_enabled:
                return (0.8, 0.12, 0.95)
            return (
                round(0.8 + random.uniform(-0.05, 0.05), 6),
                round(0.12 + random.uniform(-0.03, 0.03), 6),
                round(random.uniform(0.8, 1.2), 6),
            )

    controller = InnerLoopBatchController(inner_loop=50)
    batch = {"id": "fixed"}
    sampler = _MockProjectionSampler()
    observed = set()

    for reused in controller.iter_batches_for_outer(batch):
        assert reused is batch
        observed.add(sampler.sample(randomization_enabled=True))

    assert len(observed) > 1
