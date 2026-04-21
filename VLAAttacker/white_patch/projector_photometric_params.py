import json
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


PROJECTOR_PARAMS_FILENAME = "projector_params.json"
PROJECTOR_PARAMS_SCHEMA_VERSION = 1
PROJECTOR_GAIN_BOUNDS = (0.5, 2.0)
PROJECTOR_CHANNEL_GAIN_BOUNDS = (0.7, 1.3)
_SIGMOID_EPS = 1e-6


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() <= 0:
            return float(default)
        return float(flat[0].item())
    return float(value)


def parse_projector_channel_gain(channel_gain: Any) -> Tuple[float, float, float]:
    if channel_gain is None:
        values = [1.0, 1.0, 1.0]
    elif torch.is_tensor(channel_gain):
        values = [float(v) for v in channel_gain.detach().cpu().reshape(-1).tolist()]
    elif isinstance(channel_gain, str):
        values = [float(v.strip()) for v in str(channel_gain).split(",") if v.strip() != ""]
    else:
        values = [float(v) for v in channel_gain]
    if len(values) != 3:
        values = (values + [1.0, 1.0, 1.0])[:3]
    return (float(values[0]), float(values[1]), float(values[2]))


def projector_gain_to_tensor(projector_gain: Any, device, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(projector_gain):
        tensor = projector_gain.to(device=device, dtype=dtype).reshape(-1)
        if tensor.numel() <= 0:
            tensor = torch.ones((1,), device=device, dtype=dtype)
        tensor = tensor[:1]
    else:
        tensor = torch.tensor([float(projector_gain)], device=device, dtype=dtype)
    return torch.clamp(tensor.reshape(()), min=0.0)


def projector_channel_gain_to_tensor(channel_gain: Any, device, dtype=torch.float32) -> torch.Tensor:
    if channel_gain is None:
        tensor = torch.ones((3,), device=device, dtype=dtype)
    elif torch.is_tensor(channel_gain):
        tensor = channel_gain.to(device=device, dtype=dtype).reshape(-1)
    elif isinstance(channel_gain, str):
        values = [float(v.strip()) for v in str(channel_gain).split(",") if v.strip() != ""]
        tensor = torch.tensor(values, device=device, dtype=dtype)
    else:
        tensor = torch.tensor([float(v) for v in channel_gain], device=device, dtype=dtype)
    if tensor.numel() < 3:
        pad = torch.ones((3 - tensor.numel(),), device=device, dtype=dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    tensor = tensor[:3].view(3, 1, 1)
    return torch.clamp(tensor, min=0.0)


def _inverse_bounded_sigmoid(value: float, bounds: Tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        raise ValueError(f"Invalid bounds: {bounds}")
    scaled = (float(value) - low) / (high - low)
    scaled = min(max(scaled, _SIGMOID_EPS), 1.0 - _SIGMOID_EPS)
    return float(math.log(scaled / (1.0 - scaled)))


def _bounded_sigmoid(raw: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        raise ValueError(f"Invalid bounds: {bounds}")
    return low + ((high - low) * torch.sigmoid(raw))


class LearnableProjectorPhotometricParams(torch.nn.Module):
    def __init__(
        self,
        projector_gain: Any,
        projector_channel_gain: Any,
        learn_projector_gain: bool = False,
        learn_projector_channel_gain: bool = False,
        device=None,
    ):
        super().__init__()
        device = torch.device(device) if device is not None else None

        init_gain = _to_float(projector_gain, default=1.0)
        init_channel_gain = parse_projector_channel_gain(projector_channel_gain)

        self.learn_projector_gain = bool(learn_projector_gain)
        self.learn_projector_channel_gain = bool(learn_projector_channel_gain)

        if self.learn_projector_gain:
            raw_value = _inverse_bounded_sigmoid(init_gain, PROJECTOR_GAIN_BOUNDS)
            self.projector_gain_raw = torch.nn.Parameter(
                torch.tensor(raw_value, device=device, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "projector_gain_fixed",
                torch.tensor(init_gain, device=device, dtype=torch.float32),
            )

        init_channel_tensor = torch.tensor(init_channel_gain, device=device, dtype=torch.float32)
        if self.learn_projector_channel_gain:
            raw_channel = [
                _inverse_bounded_sigmoid(float(value), PROJECTOR_CHANNEL_GAIN_BOUNDS)
                for value in init_channel_tensor.tolist()
            ]
            self.projector_channel_gain_raw = torch.nn.Parameter(
                torch.tensor(raw_channel, device=device, dtype=torch.float32)
            )
        else:
            self.register_buffer("projector_channel_gain_fixed", init_channel_tensor)

    def has_trainable_params(self) -> bool:
        return self.learn_projector_gain or self.learn_projector_channel_gain

    def resolved_projector_gain(self) -> torch.Tensor:
        if self.learn_projector_gain:
            return _bounded_sigmoid(self.projector_gain_raw, PROJECTOR_GAIN_BOUNDS).reshape(())
        return self.projector_gain_fixed.reshape(())

    def resolved_projector_channel_gain(self) -> torch.Tensor:
        if self.learn_projector_channel_gain:
            return _bounded_sigmoid(self.projector_channel_gain_raw, PROJECTOR_CHANNEL_GAIN_BOUNDS).reshape(3)
        return self.projector_channel_gain_fixed.reshape(3)

    def resolved_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.resolved_projector_gain(), self.resolved_projector_channel_gain()

    def resolved_payload(self) -> Dict[str, Any]:
        gain, channel_gain = self.resolved_values()
        return {
            "schema_version": int(PROJECTOR_PARAMS_SCHEMA_VERSION),
            "projector_gain": float(gain.detach().cpu().item()),
            "projector_channel_gain": [float(v) for v in channel_gain.detach().cpu().tolist()],
        }


def projector_params_sidecar_path(patch_path: str) -> str:
    patch_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(patch_path))))
    return os.path.join(patch_dir, PROJECTOR_PARAMS_FILENAME)


def save_projector_params(output_dir: str, projector_gain: Any, projector_channel_gain: Any) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PROJECTOR_PARAMS_FILENAME)
    payload = {
        "schema_version": int(PROJECTOR_PARAMS_SCHEMA_VERSION),
        "projector_gain": float(_to_float(projector_gain, default=1.0)),
        "projector_channel_gain": [float(v) for v in parse_projector_channel_gain(projector_channel_gain)],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def load_projector_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Projector params at `{path}` must be a JSON object.")
    gain = float(payload["projector_gain"])
    channel_gain = parse_projector_channel_gain(payload["projector_channel_gain"])
    schema_version = int(payload.get("schema_version", PROJECTOR_PARAMS_SCHEMA_VERSION))
    return {
        "schema_version": schema_version,
        "projector_gain": gain,
        "projector_channel_gain": channel_gain,
    }


def resolve_projector_params_for_patch(
    patch_path: str,
    default_projector_gain: Any,
    default_projector_channel_gain: Any,
) -> Dict[str, Any]:
    resolved_default_gain = float(_to_float(default_projector_gain, default=1.0))
    resolved_default_channel_gain = parse_projector_channel_gain(default_projector_channel_gain)
    sidecar_path = projector_params_sidecar_path(patch_path)
    if not os.path.exists(sidecar_path):
        return {
            "projector_gain": resolved_default_gain,
            "projector_channel_gain": resolved_default_channel_gain,
            "sidecar_path": sidecar_path,
            "loaded_from_sidecar": False,
        }
    payload = load_projector_params(sidecar_path)
    payload["sidecar_path"] = sidecar_path
    payload["loaded_from_sidecar"] = True
    return payload
