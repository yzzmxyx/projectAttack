import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "VLAAttacker" / "white_patch" / "projector_attack_transform.py"


def _load_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("test_projector_attack_transform_module", MODULE_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_projection_aux_exposes_projected_tensor_with_texture_grad():
    module = _load_module()
    transform = module.ProjectorAttackTransform(device=torch.device("cpu"), resize_patch=False)
    texture = torch.full((3, 4, 4), 0.2, dtype=torch.float32, requires_grad=True)
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    mean = [torch.zeros(3), torch.zeros(3)]
    std = [torch.ones(3), torch.ones(3)]

    _output, aux = transform.apply_projection_batch(
        images=[image],
        projection_texture=texture,
        mean=mean,
        std=std,
        geometry=False,
        projection_alpha=0.5,
        projection_alpha_jitter=0.0,
        projection_soft_edge=0.0,
        projection_angle=0.0,
        projection_fixed_angle=True,
        projection_shear=0.0,
        projection_scale_min=1.0,
        projection_scale_max=1.0,
        projection_region="desk_center",
        projection_lower_start=0.0,
        projection_width_ratio=1.0,
        projection_height_ratio=1.0,
        projection_margin_x=0.0,
        projection_keystone=0.0,
        projection_keystone_jitter=0.0,
        projector_gamma=1.0,
        projector_gain=1.0,
        projector_channel_gain=(1.0, 1.0, 1.0),
        projector_ambient=0.0,
        projector_vignetting=0.0,
        projector_distance_falloff=0.0,
        projector_psf=False,
        projection_randomization_enabled=False,
        return_aux=True,
    )

    projected_tensor = aux["projected_input_tensors"][0]
    assert projected_tensor.requires_grad
    assert not aux["pre_projection_tensors"][0].requires_grad

    projected_tensor.sum().backward()
    assert texture.grad is not None
    assert texture.grad.abs().sum().item() > 0.0
