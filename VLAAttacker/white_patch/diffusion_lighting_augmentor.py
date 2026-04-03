import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter


class DiffusionLightingAugmentor:
    LIGHTING_PROMPTS = [
        "golden hour sunlight",
        "overcast daylight",
        "dramatic backlighting",
        "warm tungsten indoor lighting",
        "cool fluorescent office lighting",
        "soft studio spotlight",
        "sunset orange ambient lighting",
        "neon nightlife mixed lighting",
        "high contrast side lighting",
        "bright noon daylight",
    ]

    def __init__(
        self,
        enabled=False,
        model_id="stabilityai/sdxl-turbo",
        device="cuda",
        pool_size=8,
        refresh_interval=200,
        num_inference_steps=4,
        guidance_scale=0.0,
        blend_min=0.15,
        blend_max=0.5,
        apply_prob=1.0,
        seed=42,
    ):
        self.enabled = bool(enabled)
        self.model_id = str(model_id)
        self.device = self._resolve_device(device)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.pool_size = max(1, int(pool_size))
        self.refresh_interval = max(1, int(refresh_interval))
        self.num_inference_steps = max(1, int(num_inference_steps))
        self.guidance_scale = float(guidance_scale)
        self.blend_min = float(max(0.0, min(1.0, blend_min)))
        self.blend_max = float(max(self.blend_min, min(1.0, blend_max)))
        self.apply_prob = float(max(0.0, min(1.0, apply_prob)))
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.backend = "disabled" if not self.enabled else "uninitialized"
        self._pipeline = None
        self._load_attempted = False
        self._lighting_pool = []
        self._last_refresh_iter = -1_000_000_000
        self._warned = False
        self._fixed_lighting_cache = {}
        self._fixed_blend_cache = {}

    def _resolve_device(self, device):
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def _warn_once(self, msg):
        if self._warned:
            return
        print(f"[LightingAugmentor] {msg}")
        self._warned = True

    def force_procedural_fallback(self, reason=""):
        self._pipeline = None
        self.backend = "procedural_fallback"
        self._load_attempted = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reason_text = f" ({reason})" if str(reason).strip() else ""
        print(f"[LightingAugmentor] forcing procedural lighting fallback{reason_text}.")

    def _ensure_torch_xpu_compat(self):
        # diffusers>=0.37 may access torch.xpu at import time. torch==2.2 has no torch.xpu.
        def _xpu_is_available():
            return False

        def _xpu_empty_cache():
            return None

        def _xpu_device_count():
            return 0

        def _xpu_manual_seed(_seed):
            return None

        def _xpu_manual_seed_all(_seed):
            return None

        def _xpu_current_device():
            return 0

        def _xpu_synchronize():
            return None

        def _xpu_reset_peak_memory_stats(*_args, **_kwargs):
            return None

        def _xpu_max_memory_allocated(*_args, **_kwargs):
            return 0

        def _xpu_get_device_capability(*_args, **_kwargs):
            return {"architecture": "unknown"}

        def _xpu_is_in_bad_fork():
            return False

        required_attrs = {
            "is_available": _xpu_is_available,
            "empty_cache": _xpu_empty_cache,
            "device_count": _xpu_device_count,
            "manual_seed": _xpu_manual_seed,
            "manual_seed_all": _xpu_manual_seed_all,
            "current_device": _xpu_current_device,
            "synchronize": _xpu_synchronize,
            "reset_peak_memory_stats": _xpu_reset_peak_memory_stats,
            "max_memory_allocated": _xpu_max_memory_allocated,
            "get_device_capability": _xpu_get_device_capability,
            "_is_in_bad_fork": _xpu_is_in_bad_fork,
        }

        class _TorchXPUStub:
            @staticmethod
            def is_available():
                return _xpu_is_available()

            @staticmethod
            def empty_cache():
                return _xpu_empty_cache()

            @staticmethod
            def device_count():
                return _xpu_device_count()

            @staticmethod
            def manual_seed(_seed):
                return _xpu_manual_seed(_seed)

            @staticmethod
            def manual_seed_all(_seed):
                return _xpu_manual_seed_all(_seed)

            @staticmethod
            def current_device():
                return _xpu_current_device()

            @staticmethod
            def synchronize():
                return _xpu_synchronize()

            @staticmethod
            def reset_peak_memory_stats(*_args, **_kwargs):
                return _xpu_reset_peak_memory_stats(*_args, **_kwargs)

            @staticmethod
            def max_memory_allocated(*_args, **_kwargs):
                return _xpu_max_memory_allocated(*_args, **_kwargs)

            @staticmethod
            def get_device_capability(*_args, **_kwargs):
                return _xpu_get_device_capability(*_args, **_kwargs)

            @staticmethod
            def _is_in_bad_fork():
                return _xpu_is_in_bad_fork()

        xpu_obj = getattr(torch, "xpu", None)
        if xpu_obj is None:
            try:
                setattr(torch, "xpu", _TorchXPUStub())
            except Exception:
                # Keep original behavior if torch module disallows setattr.
                return
            xpu_obj = getattr(torch, "xpu", None)

        if xpu_obj is None:
            return

        # torch.manual_seed() in torch==2.2 checks torch.xpu._is_in_bad_fork() when torch.xpu exists.
        for attr_name, fallback in required_attrs.items():
            if hasattr(xpu_obj, attr_name):
                continue
            try:
                setattr(xpu_obj, attr_name, fallback)
            except Exception:
                continue

    def _maybe_load_pipeline(self):
        if (not self.enabled) or self._load_attempted:
            return
        self._load_attempted = True

        self._ensure_torch_xpu_compat()
        try:
            from diffusers import AutoPipelineForText2Image
        except Exception as err:
            self.backend = "procedural_fallback"
            self._warn_once(f"diffusers is unavailable ({err}); using procedural lighting fallback.")
            return

        try:
            load_kwargs = {"torch_dtype": self.dtype}
            if self.device.startswith("cuda"):
                load_kwargs["variant"] = "fp16"
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **load_kwargs)
            except Exception:
                load_kwargs.pop("variant", None)
                pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **load_kwargs)

            pipe = pipe.to(self.device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            self._pipeline = pipe
            self.backend = "diffusion"
        except Exception as err:
            self.backend = "procedural_fallback"
            self._warn_once(f"cannot load diffusion model '{self.model_id}' ({err}); using procedural lighting fallback.")

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if torch.is_tensor(image):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        elif isinstance(image, np.ndarray):
            array = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array).convert("RGB")

    def _build_prompt(self):
        base = self.rng.choice(self.LIGHTING_PROMPTS)
        return f"photorealistic lighting transfer only, keep object layout unchanged, {base}"

    def generate_lighting_map(self, image_size, prompt=None, seed=None):
        self._maybe_load_pipeline()
        if prompt is None:
            prompt = self._build_prompt()

        original_rng = self.rng
        if seed is not None:
            self.rng = random.Random(int(seed))
        try:
            if self._pipeline is None or self.backend != "diffusion":
                return self._procedural_lighting_map(image_size)

            width, height = image_size
            width = max(64, (int(width) // 8) * 8)
            height = max(64, (int(height) // 8) * 8)
            if self.device.startswith("cuda"):
                generator = torch.Generator(device=self.device)
            else:
                generator = torch.Generator()
            if seed is None:
                generator.manual_seed(self.rng.randint(0, 2**31 - 1))
            else:
                generator.manual_seed(int(seed))

            with torch.inference_mode():
                result = self._pipeline(
                    prompt=str(prompt),
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
            lighting = result.images[0].convert("RGB")
            if lighting.size != image_size:
                lighting = lighting.resize(image_size, resample=Image.BILINEAR)
            return lighting
        except Exception as err:
            self.backend = "procedural_fallback"
            self._warn_once(f"diffusion generation failed ({err}); switching to procedural lighting fallback.")
            return self._procedural_lighting_map(image_size)
        finally:
            self.rng = original_rng

    def _procedural_lighting_map(self, image_size):
        width, height = image_size
        np_rng = np.random.default_rng(self.rng.randint(0, 2**31 - 1))

        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        center_x = self.rng.uniform(-0.5, 0.5)
        center_y = self.rng.uniform(-0.5, 0.5)
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        radial = np.clip(1.0 - dist, 0.0, 1.0)

        theta = self.rng.uniform(-np.pi, np.pi)
        directional = np.clip(0.5 + 0.5 * np.cos((xx * np.cos(theta) + yy * np.sin(theta)) * np.pi), 0.0, 1.0)

        noise = np_rng.normal(loc=0.0, scale=0.07, size=(height, width)).astype(np.float32)
        lighting = np.clip(0.5 * radial + 0.4 * directional + noise + 0.1, 0.0, 1.0)

        warm = self.rng.random() > 0.5
        tint = np.array([1.12, 1.0, 0.9], dtype=np.float32) if warm else np.array([0.9, 1.0, 1.12], dtype=np.float32)
        lighting_rgb = np.clip(lighting[..., None] * tint[None, None, :], 0.0, 1.0)

        image = Image.fromarray((lighting_rgb * 255.0).astype(np.uint8))
        return image.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(4.0, 9.0)))

    def _diffusion_lighting_map(self, image_size):
        if self._pipeline is None:
            return self._procedural_lighting_map(image_size)

        width, height = image_size
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)

        return self.generate_lighting_map(image_size=image_size, prompt=self._build_prompt())

    def _refresh_pool_if_needed(self, iteration_idx, image_size):
        if not self.enabled:
            return
        if (len(self._lighting_pool) > 0) and ((iteration_idx - self._last_refresh_iter) < self.refresh_interval):
            return

        self._maybe_load_pipeline()
        use_diffusion = self.backend == "diffusion"
        new_pool = []
        for _ in range(self.pool_size):
            if use_diffusion:
                new_pool.append(self._diffusion_lighting_map(image_size))
            else:
                new_pool.append(self._procedural_lighting_map(image_size))
        self._lighting_pool = new_pool
        self._last_refresh_iter = int(iteration_idx)

    def _get_fixed_cache_key(self, fixed_map_idx, image_size):
        width, height = image_size
        return (int(fixed_map_idx), int(width), int(height))

    def _get_fixed_lighting_map(self, fixed_map_idx, image_size):
        cache_key = self._get_fixed_cache_key(fixed_map_idx=fixed_map_idx, image_size=image_size)
        if cache_key in self._fixed_lighting_cache:
            return self._fixed_lighting_cache[cache_key]

        self._maybe_load_pipeline()
        original_rng = self.rng
        self.rng = random.Random(self.seed + (9973 * int(fixed_map_idx)) + (31 * int(image_size[0])) + int(image_size[1]))
        try:
            if self.backend == "diffusion":
                lighting_map = self._diffusion_lighting_map(image_size)
            else:
                lighting_map = self._procedural_lighting_map(image_size)
        finally:
            self.rng = original_rng
        self._fixed_lighting_cache[cache_key] = lighting_map
        return lighting_map

    def _get_fixed_blend_params(self, fixed_map_idx):
        fixed_idx = int(fixed_map_idx)
        if fixed_idx not in self._fixed_blend_cache:
            rng = random.Random(self.seed + 104729 + (131 * fixed_idx))
            self._fixed_blend_cache[fixed_idx] = {
                "alpha": rng.uniform(self.blend_min, self.blend_max),
                "gamma": rng.uniform(0.85, 1.15),
                "exposure": rng.uniform(0.9, 1.1),
            }
        return dict(self._fixed_blend_cache[fixed_idx])

    def _blend_lighting(self, image, lighting_map, blend_params=None):
        image_np = np.asarray(image).astype(np.float32) / 255.0
        light_np = np.asarray(lighting_map).astype(np.float32) / 255.0

        if blend_params is None:
            alpha = self.rng.uniform(self.blend_min, self.blend_max)
            gamma = self.rng.uniform(0.85, 1.15)
            exposure = self.rng.uniform(0.9, 1.1)
        else:
            alpha = float(blend_params["alpha"])
            gamma = float(blend_params["gamma"])
            exposure = float(blend_params["exposure"])

        relit = np.clip((image_np**gamma) * (1.0 + alpha * (light_np - 0.5) * 2.0) * exposure, 0.0, 1.0)
        tint = np.mean(light_np, axis=(0, 1), keepdims=True)
        relit = np.clip(relit * (1.0 + 0.12 * (tint - 0.5)), 0.0, 1.0)
        return Image.fromarray((relit * 255.0).astype(np.uint8)).convert("RGB")

    def augment_batch(self, images: Sequence, iteration_idx=0, split="train", fixed_map_idx=None):
        del split
        if not self.enabled:
            return images
        if images is None or len(images) == 0:
            return images

        output_images = []
        use_fixed_map = fixed_map_idx is not None
        if use_fixed_map:
            fixed_idx = int(fixed_map_idx)
            fixed_blend_params = self._get_fixed_blend_params(fixed_idx)
        else:
            first_image = self._to_pil(images[0])
            self._refresh_pool_if_needed(iteration_idx=iteration_idx, image_size=first_image.size)
            if len(self._lighting_pool) == 0:
                return images
        for image in images:
            image_pil = self._to_pil(image)
            if (not use_fixed_map) and (self.rng.random() > self.apply_prob):
                output_images.append(image_pil)
                continue
            if use_fixed_map:
                lighting_map = self._get_fixed_lighting_map(fixed_map_idx=fixed_idx, image_size=image_pil.size)
                output_images.append(self._blend_lighting(image_pil, lighting_map, blend_params=fixed_blend_params))
            else:
                map_idx = self.rng.randint(0, len(self._lighting_pool) - 1)
                output_images.append(self._blend_lighting(image_pil, self._lighting_pool[map_idx]))
        return output_images

    @property
    def current_pool_size(self):
        return len(self._lighting_pool)
