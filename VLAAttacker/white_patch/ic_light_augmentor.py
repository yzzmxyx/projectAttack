import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from white_patch.diffusion_lighting_augmentor import DiffusionLightingAugmentor
except Exception:
    from diffusion_lighting_augmentor import DiffusionLightingAugmentor


class ICLightAugmentor:
    RECIPE_LIBRARY = [
        {"prompt": "golden hour sunlight", "bg_source": "neutral"},
        {"prompt": "overcast daylight", "bg_source": "neutral"},
        {"prompt": "dramatic backlighting", "bg_source": "soft_ambient"},
        {"prompt": "warm tungsten indoor lighting", "bg_source": "neutral"},
        {"prompt": "cool fluorescent office lighting", "bg_source": "neutral"},
        {"prompt": "soft studio spotlight", "bg_source": "top"},
        {"prompt": "sunset orange ambient lighting", "bg_source": "neutral"},
        {"prompt": "neon nightlife mixed lighting", "bg_source": "soft_ambient"},
        {"prompt": "high contrast side lighting", "bg_source": "left"},
        {"prompt": "bright noon daylight", "bg_source": "neutral"},
    ]

    def __init__(
        self,
        enabled=False,
        device="cuda",
        pool_size=8,
        refresh_interval=200,
        num_inference_steps=8,
        guidance_scale=0.0,
        seed=42,
        ic_light_repo="/home/yxx/IC-Light",
        ic_light_model_path="/home/yxx/IC-Light/models/iclight_sd15_fbc.safetensors",
        scope="full",
        bg_control="legacy_prompt",
        legacy_model_id="stabilityai/sdxl-turbo",
        legacy_blend_min=0.15,
        legacy_blend_max=0.5,
        legacy_apply_prob=1.0,
    ):
        self.enabled = bool(enabled)
        self.device = self._resolve_device(device)
        self.pool_size = max(1, int(pool_size))
        self.refresh_interval = max(1, int(refresh_interval))
        self.num_inference_steps = max(1, int(num_inference_steps))
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.scope = str(scope).lower().strip()
        self.bg_control = str(bg_control).lower().strip()
        self.ic_light_repo = str(ic_light_repo)
        self.ic_light_model_path = str(ic_light_model_path)

        self.backend = "disabled" if not self.enabled else "uninitialized"
        self._active_backend = "disabled" if not self.enabled else "ic_light"
        self._warned = False
        self._load_attempted = False
        self._recipe_pool: List[Dict[str, object]] = []
        self._last_refresh_iter = -1_000_000_000
        self._dataset_bg_paths: Optional[List[str]] = None
        self._legacy_prompt_bg_cache: Dict[Tuple[int, int, int, str], np.ndarray] = {}

        self._tokenizer = None
        self._text_encoder = None
        self._vae = None
        self._unet = None
        self._t2i_pipe = None
        self._i2i_pipe = None
        self._rmbg = None

        self._legacy_kwargs = {
            "enabled": self.enabled,
            "model_id": legacy_model_id,
            "device": self.device,
            "pool_size": self.pool_size,
            "refresh_interval": self.refresh_interval,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "blend_min": float(legacy_blend_min),
            "blend_max": float(legacy_blend_max),
            "apply_prob": float(legacy_apply_prob),
            "seed": self.seed,
        }
        self._legacy_augmentor: Optional[DiffusionLightingAugmentor] = None

        self._cfg_scale = float(self.guidance_scale if self.guidance_scale > 0 else 7.0)
        self._content_prompt = (
            "realistic robotic manipulation scene, a robot arm above a tabletop with household objects"
        )
        self._added_prompt = "best quality"
        self._negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        self._highres_scale = 1.5
        self._highres_denoise = 0.5
        self._min_generation_edge = 512
        self._sd15_name = "stablediffusionapi/realistic-vision-v51"

    def _resolve_device(self, device):
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def _warn_once(self, msg):
        if self._warned:
            return
        print(f"[ICLightAugmentor] {msg}")
        self._warned = True

    def _module_from_path(self, module_name: str, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Required IC-Light module not found: {path}")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module `{module_name}` from `{path}`.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_local_ic_light_modules(self):
        repo_path = Path(self.ic_light_repo)
        briarmbg_module = self._module_from_path("projectattack_ic_light_briarmbg", repo_path / "briarmbg.py")
        db_examples_module = self._module_from_path("projectattack_ic_light_db_examples", repo_path / "db_examples.py")
        return briarmbg_module, db_examples_module

    def _ensure_dataset_bg_paths(self):
        if self._dataset_bg_paths is not None:
            return self._dataset_bg_paths
        repo_path = Path(self.ic_light_repo)
        try:
            _briarmbg_module, db_examples_module = self._ensure_local_ic_light_modules()
            raw_paths = list(getattr(db_examples_module, "bg_samples", []))
        except Exception:
            raw_paths = []

        resolved = []
        for item in raw_paths:
            candidate = repo_path / str(item)
            if candidate.exists():
                resolved.append(str(candidate))

        if len(resolved) == 0:
            fallback_dir = repo_path / "imgs" / "bgs"
            if fallback_dir.exists():
                for candidate in sorted(fallback_dir.iterdir()):
                    if candidate.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        resolved.append(str(candidate))

        self._dataset_bg_paths = resolved
        return self._dataset_bg_paths

    def _activate_legacy_backend(self, reason=""):
        if self._legacy_augmentor is None:
            self._legacy_augmentor = DiffusionLightingAugmentor(**self._legacy_kwargs)
        self._active_backend = "legacy"
        self.backend = f"legacy:{self._legacy_augmentor.backend}"
        reason_text = f" ({reason})" if str(reason).strip() else ""
        print(f"[ICLightAugmentor] falling back to legacy lighting backend{reason_text}.")

    def force_procedural_fallback(self, reason=""):
        self._activate_legacy_backend(reason=reason)
        if self._legacy_augmentor is not None and hasattr(self._legacy_augmentor, "force_procedural_fallback"):
            self._legacy_augmentor.force_procedural_fallback(reason=reason)
            self.backend = f"legacy:{self._legacy_augmentor.backend}"

    def _maybe_load_pipeline(self):
        if (not self.enabled) or self._load_attempted or (self._active_backend != "ic_light"):
            return
        self._load_attempted = True

        try:
            import safetensors.torch as sf
            from diffusers import (
                AutoencoderKL,
                DPMSolverMultistepScheduler,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionPipeline,
                UNet2DConditionModel,
            )
            from diffusers.models.attention_processor import AttnProcessor2_0
            from transformers import CLIPTextModel, CLIPTokenizer

            briarmbg_module, _db_examples_module = self._ensure_local_ic_light_modules()
            BriaRMBG = briarmbg_module.BriaRMBG

            if not os.path.exists(self.ic_light_model_path):
                raise FileNotFoundError(f"IC-Light model not found: {self.ic_light_model_path}")

            tokenizer = CLIPTokenizer.from_pretrained(self._sd15_name, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(self._sd15_name, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(self._sd15_name, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(self._sd15_name, subfolder="unet")
            rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

            with torch.no_grad():
                new_conv_in = torch.nn.Conv2d(
                    12,
                    unet.conv_in.out_channels,
                    unet.conv_in.kernel_size,
                    unet.conv_in.stride,
                    unet.conv_in.padding,
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
                new_conv_in.bias = unet.conv_in.bias
                unet.conv_in = new_conv_in

            unet_original_forward = unet.forward

            def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
                c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
                c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
                new_sample = torch.cat([sample, c_concat], dim=1)
                kwargs["cross_attention_kwargs"] = {}
                return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

            unet.forward = hooked_unet_forward

            sd_offset = sf.load_file(self.ic_light_model_path)
            sd_origin = unet.state_dict()
            sd_merged = {key: sd_origin[key] + sd_offset[key] for key in sd_origin.keys()}
            unet.load_state_dict(sd_merged, strict=True)

            unet_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            vae_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            text_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

            text_encoder = text_encoder.to(device=self.device, dtype=text_dtype)
            vae = vae.to(device=self.device, dtype=vae_dtype)
            unet = unet.to(device=self.device, dtype=unet_dtype)
            rmbg = rmbg.to(device=self.device, dtype=torch.float32)

            if hasattr(unet, "set_attn_processor"):
                unet.set_attn_processor(AttnProcessor2_0())
            if hasattr(vae, "set_attn_processor"):
                vae.set_attn_processor(AttnProcessor2_0())

            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
                steps_offset=1,
            )

            t2i_pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
                image_encoder=None,
            )
            i2i_pipe = StableDiffusionImg2ImgPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
                image_encoder=None,
            )

            self._tokenizer = tokenizer
            self._text_encoder = text_encoder
            self._vae = vae
            self._unet = unet
            self._t2i_pipe = t2i_pipe
            self._i2i_pipe = i2i_pipe
            self._rmbg = rmbg
            self.backend = "ic_light"
        except Exception as err:
            self._warn_once(f"IC-Light load failed ({err}).")
            self._activate_legacy_backend(reason=f"ic_light_load_failed: {err}")

    @torch.inference_mode()
    def _encode_prompt_inner(self, txt: str):
        max_length = self._tokenizer.model_max_length
        chunk_length = self._tokenizer.model_max_length - 2
        id_start = self._tokenizer.bos_token_id
        id_end = self._tokenizer.eos_token_id
        id_pad = id_end

        def pad(values, pad_value, size):
            return values[:size] if len(values) >= size else values + [pad_value] * (size - len(values))

        tokens = self._tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i : i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(chunk, id_pad, max_length) for chunk in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self._text_encoder(token_ids).last_hidden_state
        return conds

    @torch.inference_mode()
    def _encode_prompt_pair(self, positive_prompt, negative_prompt):
        conds = self._encode_prompt_inner(positive_prompt)
        unconds = self._encode_prompt_inner(negative_prompt)

        cond_len = float(len(conds))
        uncond_len = float(len(unconds))
        max_count = max(cond_len, uncond_len)
        cond_repeat = int(math.ceil(max_count / cond_len))
        uncond_repeat = int(math.ceil(max_count / uncond_len))
        max_chunk = max(len(conds), len(unconds))

        conds = torch.cat([conds] * cond_repeat, dim=0)[:max_chunk]
        unconds = torch.cat([unconds] * uncond_repeat, dim=0)[:max_chunk]

        conds = torch.cat([item[None, ...] for item in conds], dim=1)
        unconds = torch.cat([item[None, ...] for item in unconds], dim=1)
        return conds, unconds

    @torch.inference_mode()
    def _pytorch2numpy(self, imgs, quant=True):
        outputs = []
        for tensor in imgs:
            array = tensor.movedim(0, -1)
            if quant:
                array = array * 127.5 + 127.5
                array = array.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                array = array * 0.5 + 0.5
                array = array.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
            outputs.append(array)
        return outputs

    @torch.inference_mode()
    def _numpy2pytorch(self, imgs):
        tensor = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        return tensor.movedim(-1, 1)

    def _resize_and_center_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        return np.array(resized_image.crop((left, top, right, bottom)))

    def _resize_without_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)

    def _compute_generation_size(self, width: int, height: int) -> Tuple[int, int]:
        width = max(64, int(width))
        height = max(64, int(height))
        min_edge = min(width, height)
        scale = max(1.0, float(self._min_generation_edge) / float(min_edge))
        gen_width = max(64, int(round((width * scale) / 64.0) * 64))
        gen_height = max(64, int(round((height * scale) / 64.0) * 64))
        return gen_width, gen_height

    def _build_positive_prompt(self, recipe: Dict[str, object]) -> str:
        recipe_prompt = str(recipe["prompt"]).strip()
        prompt_parts = [
            self._content_prompt,
            recipe_prompt,
            self._added_prompt,
        ]
        return ", ".join(part for part in prompt_parts if len(part) > 0)

    @torch.inference_mode()
    def _run_rmbg(self, image_np, sigma=0.0):
        height, width, channels = image_np.shape
        assert channels == 3
        scale = (256.0 / float(height * width)) ** 0.5
        feed = self._resize_without_crop(image_np, int(64 * round(width * scale)), int(64 * round(height * scale)))
        feed = self._numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha = self._rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(height, width), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (image_np.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

    def _recipe_from_fixed_index(self, fixed_map_idx: int) -> Dict[str, object]:
        rng = random.Random(int(self.seed) + int(fixed_map_idx))
        base = dict(self.RECIPE_LIBRARY[int(fixed_map_idx) % len(self.RECIPE_LIBRARY)])
        base["seed"] = int(rng.randint(0, (2**31) - 1))
        return base

    def _refresh_pool_if_needed(self, iteration_idx: int):
        if not self.enabled or (self._active_backend != "ic_light"):
            return
        if (len(self._recipe_pool) > 0) and ((int(iteration_idx) - self._last_refresh_iter) < self.refresh_interval):
            return
        new_pool = []
        for _ in range(self.pool_size):
            base = dict(random.choice(self.RECIPE_LIBRARY))
            base["seed"] = int(random.randint(0, (2**31) - 1))
            new_pool.append(base)
        self._recipe_pool = new_pool
        self._last_refresh_iter = int(iteration_idx)

    def _build_directional_background(self, width: int, height: int, bg_source: str):
        source = str(bg_source).lower().strip()
        if source == "left":
            gradient = np.linspace(224, 32, width, dtype=np.float32)
            image = np.tile(gradient, (height, 1))
        elif source == "right":
            gradient = np.linspace(32, 224, width, dtype=np.float32)
            image = np.tile(gradient, (height, 1))
        elif source == "top":
            gradient = np.linspace(224, 32, height, dtype=np.float32)[:, None]
            image = np.tile(gradient, (1, width))
        elif source == "bottom":
            gradient = np.linspace(32, 224, height, dtype=np.float32)[:, None]
            image = np.tile(gradient, (1, width))
        elif source == "soft_ambient":
            x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
            y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            radial = np.clip(1.0 - np.sqrt(xx**2 + yy**2), 0.0, 1.0)
            image = 96.0 + 96.0 * radial
        else:
            image = np.zeros((height, width), dtype=np.float32) + 64.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return np.stack((image,) * 3, axis=-1)

    def _build_dataset_background(self, width: int, height: int, recipe: Dict[str, object]):
        bg_paths = self._ensure_dataset_bg_paths()
        if len(bg_paths) == 0:
            return self._build_directional_background(width=width, height=height, bg_source=str(recipe.get("bg_source", "neutral")))

        bg_index = int(recipe.get("bg_dataset_idx", int(recipe["seed"]) % len(bg_paths))) % len(bg_paths)
        bg_path = bg_paths[bg_index]
        bg_image = Image.open(bg_path).convert("RGB")
        return self._resize_and_center_crop(np.asarray(bg_image, dtype=np.uint8), width, height)

    def _get_legacy_prompt_bg_cache_key(self, width: int, height: int, recipe: Dict[str, object]):
        return (
            int(width),
            int(height),
            int(recipe["seed"]),
            str(recipe["prompt"]).strip(),
        )

    def _build_legacy_prompt_background(self, width: int, height: int, recipe: Dict[str, object]):
        cache_key = self._get_legacy_prompt_bg_cache_key(width=width, height=height, recipe=recipe)
        if cache_key in self._legacy_prompt_bg_cache:
            return self._legacy_prompt_bg_cache[cache_key].copy()
        if self._legacy_augmentor is None:
            self._legacy_augmentor = DiffusionLightingAugmentor(**self._legacy_kwargs)
        prompt = (
            "photorealistic lighting transfer only, keep object layout unchanged, "
            f"{str(recipe['prompt']).strip()}"
        )
        lighting_map = self._legacy_augmentor.generate_lighting_map(
            image_size=(width, height),
            prompt=prompt,
            seed=int(recipe["seed"]),
        )
        lighting_np = np.asarray(lighting_map.convert("RGB"), dtype=np.uint8)
        self._legacy_prompt_bg_cache[cache_key] = lighting_np.copy()
        return lighting_np.copy()

    def _select_recipe(self, iteration_idx: int, fixed_map_idx=None):
        self._refresh_pool_if_needed(iteration_idx=iteration_idx)
        bg_paths = self._ensure_dataset_bg_paths() if self.bg_control == "ic_light_dataset" else []
        if fixed_map_idx is not None:
            recipe = self._recipe_from_fixed_index(int(fixed_map_idx))
            if len(bg_paths) > 0:
                recipe["bg_dataset_idx"] = int(fixed_map_idx) % len(bg_paths)
            return recipe
        if len(self._recipe_pool) == 0:
            self._refresh_pool_if_needed(iteration_idx=iteration_idx)
        if len(self._recipe_pool) == 0:
            recipe = dict(self.RECIPE_LIBRARY[0], seed=int(self.seed))
            if len(bg_paths) > 0:
                recipe["bg_dataset_idx"] = int(self.seed) % len(bg_paths)
            return recipe
        recipe = dict(self._recipe_pool[random.randint(0, len(self._recipe_pool) - 1)])
        if (len(bg_paths) > 0) and ("bg_dataset_idx" not in recipe):
            recipe["bg_dataset_idx"] = int(recipe["seed"]) % len(bg_paths)
        return recipe

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
            if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array).convert("RGB")

    def _build_background_condition(self, image_np, recipe):
        if self.bg_control == "current_image":
            return image_np.copy()
        if self.bg_control == "legacy_prompt":
            height, width = image_np.shape[:2]
            return self._build_legacy_prompt_background(width=width, height=height, recipe=recipe)
        if self.bg_control == "ic_light_dataset":
            height, width = image_np.shape[:2]
            return self._build_dataset_background(width=width, height=height, recipe=recipe)
        height, width = image_np.shape[:2]
        return self._build_directional_background(width=width, height=height, bg_source=str(recipe["bg_source"]))

    @torch.inference_mode()
    def _decode_latents_to_uint8_batch(self, latents):
        relit = self._vae.decode(latents.to(self._vae.dtype)).sample
        relit_np_list = self._pytorch2numpy(relit, quant=False)
        return [(item * 255.0).clip(0, 255).astype(np.uint8) for item in relit_np_list]

    @torch.inference_mode()
    def _relight_with_ic_light(self, image_pil: Image.Image, recipe: Dict[str, object], use_highres_refine: bool = False) -> Image.Image:
        self._maybe_load_pipeline()
        if self._active_backend != "ic_light":
            raise RuntimeError("IC-Light backend is not active.")

        image_np = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
        if self.scope == "foreground":
            foreground_np, _alpha = self._run_rmbg(image_np)
        else:
            foreground_np = image_np.copy()

        background_np = self._build_background_condition(image_np, recipe)
        width, height = image_pil.size
        proc_width, proc_height = self._compute_generation_size(width=width, height=height)
        positive_prompt = self._build_positive_prompt(recipe)
        conds, unconds = self._encode_prompt_pair(
            positive_prompt=positive_prompt,
            negative_prompt=self._negative_prompt,
        )

        generator = torch.Generator(device=self.device if self.device.startswith("cuda") else "cpu")
        generator.manual_seed(int(recipe["seed"]))

        fg = self._resize_and_center_crop(foreground_np, proc_width, proc_height)
        bg = self._resize_and_center_crop(background_np, proc_width, proc_height)
        concat_conds = self._numpy2pytorch([fg, bg]).to(device=self._vae.device, dtype=self._vae.dtype)
        concat_conds = self._vae.encode(concat_conds).latent_dist.mode() * self._vae.config.scaling_factor
        concat_conds = torch.cat([item[None, ...] for item in concat_conds], dim=1)

        latents = self._t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=proc_width,
            height=proc_height,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            output_type="latent",
            guidance_scale=self._cfg_scale,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self._vae.dtype) / self._vae.config.scaling_factor

        lowres_pixels = self._decode_latents_to_uint8_batch(latents)
        if not bool(use_highres_refine):
            relit_np = self._resize_without_crop(lowres_pixels[0], width, height)
            return Image.fromarray(relit_np).convert("RGB")

        target_width = int(round(width * self._highres_scale / 64.0) * 64)
        target_height = int(round(height * self._highres_scale / 64.0) * 64)
        pixels = [self._resize_without_crop(image=item, target_width=target_width, target_height=target_height) for item in lowres_pixels]

        pixels = self._numpy2pytorch(pixels).to(device=self._vae.device, dtype=self._vae.dtype)
        latents = self._vae.encode(pixels).latent_dist.mode() * self._vae.config.scaling_factor
        latents = latents.to(device=self._unet.device, dtype=self._unet.dtype)

        highres_height, highres_width = latents.shape[2] * 8, latents.shape[3] * 8
        fg_hr = self._resize_and_center_crop(foreground_np, highres_width, highres_height)
        bg_hr = self._resize_and_center_crop(background_np, highres_width, highres_height)
        concat_conds = self._numpy2pytorch([fg_hr, bg_hr]).to(device=self._vae.device, dtype=self._vae.dtype)
        concat_conds = self._vae.encode(concat_conds).latent_dist.mode() * self._vae.config.scaling_factor
        concat_conds = torch.cat([item[None, ...] for item in concat_conds], dim=1)

        latents = self._i2i_pipe(
            image=latents,
            strength=self._highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=highres_width,
            height=highres_height,
            num_inference_steps=max(1, int(round(self.num_inference_steps / self._highres_denoise))),
            num_images_per_prompt=1,
            generator=generator,
            output_type="latent",
            guidance_scale=self._cfg_scale,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self._vae.dtype) / self._vae.config.scaling_factor

        relit_np = self._decode_latents_to_uint8_batch(latents)[0]
        relit_np = self._resize_without_crop(relit_np, width, height)

        return Image.fromarray(relit_np).convert("RGB")

    def augment_batch(self, images: Sequence, iteration_idx=0, split="train", fixed_map_idx=None):
        if (not self.enabled) or images is None or len(images) == 0:
            return images

        if self._active_backend == "legacy":
            if self._legacy_augmentor is None:
                self._activate_legacy_backend(reason="legacy_backend_requested_without_instance")
            output_images = self._legacy_augmentor.augment_batch(
                images=images,
                iteration_idx=iteration_idx,
                split=split,
                fixed_map_idx=fixed_map_idx,
            )
            self.backend = f"legacy:{self._legacy_augmentor.backend}"
            return output_images

        output_images = []
        try:
            for image in images:
                image_pil = self._to_pil(image)
                recipe = self._select_recipe(iteration_idx=iteration_idx, fixed_map_idx=fixed_map_idx)
                output_images.append(self._relight_with_ic_light(image_pil, recipe))
            self.backend = "ic_light"
            return output_images
        except Exception as err:
            self._warn_once(f"IC-Light inference failed ({err}).")
            self._activate_legacy_backend(reason=f"ic_light_infer_failed: {err}")
            output_images = self._legacy_augmentor.augment_batch(
                images=images,
                iteration_idx=iteration_idx,
                split=split,
                fixed_map_idx=fixed_map_idx,
            )
            self.backend = f"legacy:{self._legacy_augmentor.backend}"
            return output_images

    @property
    def current_pool_size(self):
        if self._active_backend == "legacy" and self._legacy_augmentor is not None:
            return self._legacy_augmentor.current_pool_size
        return len(self._recipe_pool)
