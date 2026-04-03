import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "VLAAttacker" / "white_patch" / "diffusion_lighting_augmentor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_diffusion_lighting_augmentor_module", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensure_torch_xpu_compat_creates_stub_when_missing(monkeypatch):
    module = _load_module()
    monkeypatch.delattr(torch, "xpu", raising=False)

    augmentor = module.DiffusionLightingAugmentor(enabled=False)
    augmentor._ensure_torch_xpu_compat()

    assert hasattr(torch, "xpu")
    assert hasattr(torch.xpu, "_is_in_bad_fork")
    assert torch.xpu._is_in_bad_fork() is False
    torch.manual_seed(1)


def test_ensure_torch_xpu_compat_patches_existing_xpu_namespace(monkeypatch):
    module = _load_module()

    class PartialXPU:
        @staticmethod
        def is_available():
            return False

    partial_xpu = PartialXPU()
    monkeypatch.setattr(torch, "xpu", partial_xpu, raising=False)

    augmentor = module.DiffusionLightingAugmentor(enabled=False)
    augmentor._ensure_torch_xpu_compat()

    assert torch.xpu is partial_xpu
    assert hasattr(torch.xpu, "manual_seed_all")
    assert hasattr(torch.xpu, "_is_in_bad_fork")
    assert torch.xpu._is_in_bad_fork() is False
    torch.manual_seed(2)
