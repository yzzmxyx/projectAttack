import random
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

try:
    from white_patch.appply_random_transform import RandomPatchTransform
except Exception:
    from appply_random_transform import RandomPatchTransform


class ProjectorAttackTransform:
    def __init__(self, device, resize_patch=False):
        self.device = device
        self.resize_patch = resize_patch
        self.patch_transform = RandomPatchTransform(device, resize_patch)
        self.last_alpha_mean = 0.0
        self.last_coverage_ratio = 0.0
        self.last_bottom_ratio = 0.0
        self.last_keystone = 0.0
        self.last_backend = "patch"

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self, images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def im_process(self, images, mean, std):
        return self.patch_transform.im_process(images, mean, std)

    def _to_tensor_image(self, image):
        if torch.is_tensor(image):
            tensor = image.detach().to(self.device).float()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor[:3, :, :]
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor[:3, :, :]
            raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")

        if isinstance(image, np.ndarray):
            array = image
        else:
            array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        # Ensure writable/contiguous memory before torch.from_numpy.
        array = np.array(array[:, :, :3], copy=True)
        tensor = torch.from_numpy(array).to(self.device).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.permute(2, 0, 1).contiguous()

    def _sample_position(
        self,
        img_h,
        img_w,
        tex_h,
        tex_w,
        region,
        projection_lower_start=0.5,
        projection_width_ratio=0.96,
        projection_height_ratio=1.0,
        projection_margin_x=0.02,
        randomization_enabled=True,
    ):
        if region == "lower_half_fixed":
            lower_start = float(np.clip(projection_lower_start, 0.0, 0.95))
            width_ratio = float(np.clip(projection_width_ratio, 0.1, 1.0))
            height_ratio = float(np.clip(projection_height_ratio, 0.1, 1.0))
            margin_ratio = float(np.clip(projection_margin_x, 0.0, 0.45))

            start_y = int(round(lower_start * img_h))
            start_y = min(max(0, start_y), max(0, img_h - 1))
            lower_h = max(1, img_h - start_y)

            margin_x = int(round(margin_ratio * img_w))
            margin_x = min(max(0, margin_x), max(0, img_w // 2))
            usable_w = max(1, img_w - (2 * margin_x))

            target_w = max(1, int(round(usable_w * width_ratio)))
            target_h = max(1, int(round(lower_h * height_ratio)))
            target_w = min(target_w, img_w)
            target_h = min(target_h, lower_h)

            x = margin_x + max(0, (usable_w - target_w) // 2)
            y = start_y + max(0, (lower_h - target_h) // 2)
            x = min(max(0, x), max(0, img_w - target_w))
            y = min(max(0, y), max(0, img_h - target_h))
            return x, y, target_h, target_w

        max_x = max(0, img_w - tex_w)
        max_y = max(0, img_h - tex_h)

        if region == "desk_bottom":
            min_y = int(0.45 * img_h)
            min_y = min(min_y, max_y)
            if not bool(randomization_enabled):
                x = max_x // 2
                y = (min_y + max_y) // 2 if max_y >= min_y else max_y
                return x, y, tex_h, tex_w
            y = random.randint(min_y, max_y) if max_y >= min_y else max_y
            x = random.randint(0, max_x) if max_x > 0 else 0
            return x, y, tex_h, tex_w

        if region == "desk_center":
            center_x = max_x // 2
            center_y = int(0.65 * img_h) - tex_h // 2
            if not bool(randomization_enabled):
                x = min(max(0, center_x), max_x)
                y = min(max(0, center_y), max_y)
                return x, y, tex_h, tex_w
            x = min(max(0, center_x + random.randint(-max(1, img_w // 10), max(1, img_w // 10))), max_x)
            y = min(max(0, center_y + random.randint(-max(1, img_h // 10), max(1, img_h // 10))), max_y)
            return x, y, tex_h, tex_w

        if not bool(randomization_enabled):
            return max_x // 2, max_y // 2, tex_h, tex_w

        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        return x, y, tex_h, tex_w

    def _soft_rect_mask(self, h, w, soft_edge):
        if soft_edge <= 0:
            return torch.ones((1, h, w), device=self.device, dtype=torch.float32)
        yy = torch.arange(h, device=self.device, dtype=torch.float32).view(h, 1)
        xx = torch.arange(w, device=self.device, dtype=torch.float32).view(1, w)
        dist_to_edge = torch.minimum(
            torch.minimum(xx, torch.tensor(float(w - 1), device=self.device) - xx),
            torch.minimum(yy, torch.tensor(float(h - 1), device=self.device) - yy),
        )
        mask = torch.clamp(dist_to_edge / float(soft_edge), 0.0, 1.0)
        return mask.unsqueeze(0)

    def _compose_affine_matrix(self, angle_deg, shear_x, shear_y, scale):
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        shear = np.array([[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        scl = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        mat = shear @ rot @ scl
        return torch.tensor(mat, dtype=torch.float32, device=self.device)

    def _apply_affine(self, image, matrix):
        affine_matrix = matrix[:2, :].unsqueeze(0)
        grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(
            image.unsqueeze(0),
            grid,
            align_corners=False,
            padding_mode="zeros",
        ).squeeze(0)

    def _apply_keystone(self, image, keystone):
        if abs(float(keystone)) < 1e-8:
            return image
        _, h, w = image.shape
        if h <= 1 or w <= 1:
            return image

        k = float(np.clip(keystone, -0.45, 0.45))
        delta = abs(k) * float(w - 1)

        startpoints = [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]]
        if k >= 0:
            endpoints = [
                [delta, 0.0],
                [float(w - 1) - delta, 0.0],
                [float(w - 1), float(h - 1)],
                [0.0, float(h - 1)],
            ]
        else:
            endpoints = [
                [0.0, 0.0],
                [float(w - 1), 0.0],
                [float(w - 1) - delta, float(h - 1)],
                [delta, float(h - 1)],
            ]

        try:
            return TVF.perspective(
                image,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        except Exception:
            return image

    def _parse_channel_gain(self, channel_gain):
        if channel_gain is None:
            values = [1.0, 1.0, 1.0]
        elif isinstance(channel_gain, str):
            values = [float(v.strip()) for v in channel_gain.split(",") if v.strip() != ""]
        else:
            values = [float(v) for v in channel_gain]
        if len(values) != 3:
            values = (values + [1.0, 1.0, 1.0])[:3]
        tensor = torch.tensor(values, device=self.device, dtype=torch.float32).view(3, 1, 1)
        return torch.clamp(tensor, min=0.0)

    def _build_physical_map(self, img_h, img_w, ambient, vignetting, distance_falloff):
        yy = torch.linspace(0.0, 1.0, steps=img_h, device=self.device, dtype=torch.float32).view(img_h, 1)
        xx = torch.linspace(0.0, 1.0, steps=img_w, device=self.device, dtype=torch.float32).view(1, img_w)

        # Approximate projector throw center on the lower half of table.
        cx = 0.5
        cy = 0.75
        radial = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radial = radial / (radial.max() + 1e-8)
        vignette_map = 1.0 - float(vignetting) * radial

        # Distance attenuation towards far end of table (top of lower-half band).
        yy_lower = torch.clamp((yy - 0.5) / 0.5, 0.0, 1.0)
        distance_map = 1.0 - float(distance_falloff) * yy_lower

        phys_map = torch.clamp(vignette_map * distance_map, 0.0, 1.0)
        ambient = float(np.clip(ambient, 0.0, 1.0))
        phys_map = ambient + (1.0 - ambient) * phys_map
        return phys_map.unsqueeze(0)

    def _projector_response(self, texture, gamma, gain, psf, channel_gain):
        out = torch.clamp(texture, 0.0, 1.0)
        out = torch.pow(out, gamma)
        out = torch.clamp(out * gain, 0.0, 1.0)
        out = torch.clamp(out * channel_gain.to(out.dtype), 0.0, 1.0)
        if psf:
            kernel = torch.ones((3, 1, 5, 5), device=self.device, dtype=out.dtype) / 25.0
            out = F.conv2d(out.unsqueeze(0), kernel, padding=2, groups=3).squeeze(0)
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def apply_projection_batch(
        self,
        images: Sequence,
        projection_texture: torch.Tensor,
        mean: Sequence[torch.Tensor],
        std: Sequence[torch.Tensor],
        geometry: bool,
        projection_alpha: float,
        projection_alpha_jitter: float,
        projection_soft_edge: float,
        projection_angle: float,
        projection_fixed_angle: bool,
        projection_shear: float,
        projection_scale_min: float,
        projection_scale_max: float,
        projection_region: str,
        projection_lower_start: float,
        projection_width_ratio: float,
        projection_height_ratio: float,
        projection_margin_x: float,
        projection_keystone: float,
        projection_keystone_jitter: float,
        projector_gamma: float,
        projector_gain: float,
        projector_channel_gain,
        projector_ambient: float,
        projector_vignetting: float,
        projector_distance_falloff: float,
        projector_psf: bool,
        projection_randomization_enabled: bool = True,
        return_aux: bool = False,
    ):
        modified_images = []
        projected_inputs: List[torch.Tensor] = []
        alpha_means: List[float] = []
        coverage_ratios: List[float] = []
        bottom_ratios: List[float] = []
        keystone_values: List[float] = []

        channel_gain_tensor = self._parse_channel_gain(projector_channel_gain)
        base_texture = self._projector_response(
            projection_texture,
            gamma=float(projector_gamma),
            gain=float(projector_gain),
            psf=bool(projector_psf),
            channel_gain=channel_gain_tensor,
        )

        for image in images:
            image_tensor = self._to_tensor_image(image)
            _, img_h, img_w = image_tensor.shape

            if bool(projection_randomization_enabled):
                scale = random.uniform(float(projection_scale_min), float(projection_scale_max))
            else:
                scale = 0.5 * (float(projection_scale_min) + float(projection_scale_max))
            tex_h = max(1, min(img_h, int(base_texture.shape[1] * scale)))
            tex_w = max(1, min(img_w, int(base_texture.shape[2] * scale)))
            texture_resized = F.interpolate(
                base_texture.unsqueeze(0),
                size=(tex_h, tex_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            x, y, tex_h, tex_w = self._sample_position(
                img_h,
                img_w,
                tex_h,
                tex_w,
                region=projection_region,
                projection_lower_start=projection_lower_start,
                projection_width_ratio=projection_width_ratio,
                projection_height_ratio=projection_height_ratio,
                projection_margin_x=projection_margin_x,
                randomization_enabled=projection_randomization_enabled,
            )
            texture_resized = F.interpolate(
                base_texture.unsqueeze(0),
                size=(tex_h, tex_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            if bool(projection_randomization_enabled):
                alpha_value = float(
                    np.clip(
                        projection_alpha + random.uniform(-projection_alpha_jitter, projection_alpha_jitter),
                        0.0,
                        1.0,
                    )
                )
            else:
                alpha_value = float(np.clip(projection_alpha, 0.0, 1.0))
            patch_rgb = texture_resized
            patch_alpha = self._soft_rect_mask(tex_h, tex_w, projection_soft_edge) * alpha_value

            if geometry:
                if bool(projection_randomization_enabled):
                    if projection_fixed_angle:
                        angle = float(projection_angle)
                        # Keep deterministic geometry when fixed-angle mode is enabled.
                        shx = float(projection_shear)
                        shy = 0.0
                    else:
                        angle = random.uniform(-float(projection_angle), float(projection_angle))
                        shx = random.uniform(-float(projection_shear), float(projection_shear))
                        shy = random.uniform(-float(projection_shear), float(projection_shear))
                    scale_geo = random.uniform(float(projection_scale_min), float(projection_scale_max))
                else:
                    angle = float(projection_angle) if projection_fixed_angle else 0.0
                    shx = float(projection_shear) if projection_fixed_angle else 0.0
                    shy = 0.0
                    scale_geo = 0.5 * (float(projection_scale_min) + float(projection_scale_max))
                matrix = self._compose_affine_matrix(angle, shx, shy, scale_geo)
                # Apply geometry on local patch to avoid rotating around full-image center.
                patch_rgb = self._apply_affine(patch_rgb, matrix)
                patch_alpha = self._apply_affine(patch_alpha, matrix)

                if bool(projection_randomization_enabled):
                    sampled_keystone = float(projection_keystone) + random.uniform(
                        -float(projection_keystone_jitter),
                        float(projection_keystone_jitter),
                    )
                else:
                    sampled_keystone = float(projection_keystone)
                # Temporarily keep keystone direction fixed to avoid trapezoid orientation flipping.
                patch_rgb = self._apply_keystone(patch_rgb, sampled_keystone)
                patch_alpha = self._apply_keystone(patch_alpha, sampled_keystone)
                keystone_values.append(abs(sampled_keystone))
            else:
                keystone_values.append(0.0)

            patch_alpha = torch.clamp(patch_alpha, 0.0, 1.0)
            proj_canvas = torch.zeros((3, img_h, img_w), device=self.device, dtype=torch.float32)
            alpha_canvas = torch.zeros((1, img_h, img_w), device=self.device, dtype=torch.float32)
            proj_canvas[:, y : y + tex_h, x : x + tex_w] = patch_rgb
            alpha_canvas[:, y : y + tex_h, x : x + tex_w] = patch_alpha

            projected_input = image_tensor
            physical_map = self._build_physical_map(
                img_h=img_h,
                img_w=img_w,
                ambient=projector_ambient,
                vignetting=projector_vignetting,
                distance_falloff=projector_distance_falloff,
            )
            adv_tensor = torch.clamp(projected_input + alpha_canvas * proj_canvas * physical_map, 0.0, 1.0)
            alpha_means.append(float(alpha_canvas.mean().detach().item()))
            changed_mask = alpha_canvas > 1e-4
            coverage_ratios.append(float(changed_mask.float().mean().detach().item()))
            changed_total = changed_mask.float().sum()
            if float(changed_total.item()) > 0.0:
                bottom_mask = torch.zeros_like(changed_mask, dtype=torch.float32)
                bottom_mask[:, int(0.5 * img_h) :, :] = 1.0
                bottom_hits = (changed_mask.float() * bottom_mask).sum()
                bottom_ratios.append(float((bottom_hits / changed_total).detach().item()))
            else:
                bottom_ratios.append(0.0)
            projected_inputs.append(adv_tensor.detach().cpu())

            im0 = self.normalize(adv_tensor, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(adv_tensor, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))

        self.last_alpha_mean = float(np.mean(alpha_means)) if alpha_means else 0.0
        self.last_coverage_ratio = float(np.mean(coverage_ratios)) if coverage_ratios else 0.0
        self.last_bottom_ratio = float(np.mean(bottom_ratios)) if bottom_ratios else 0.0
        self.last_keystone = float(np.mean(keystone_values)) if keystone_values else 0.0
        self.last_backend = "projection"

        output = torch.cat(modified_images, dim=0)
        if not return_aux:
            return output

        aux = {
            "projected_inputs": projected_inputs,
            "projection_alpha_mean": self.last_alpha_mean,
            "projection_coverage_ratio": self.last_coverage_ratio,
            "projection_bottom_ratio": self.last_bottom_ratio,
            "projection_keystone": self.last_keystone,
            "projection_backend": self.last_backend,
        }
        return output, aux

    def apply_attack_batch(
        self,
        images,
        attack_texture,
        mean,
        std,
        attack_mode="projection",
        geometry=True,
        projection_alpha=0.35,
        projection_alpha_jitter=0.10,
        projection_soft_edge=2.5,
        projection_angle=25.0,
        projection_fixed_angle=False,
        projection_shear=0.15,
        projection_scale_min=0.8,
        projection_scale_max=1.2,
        projection_region="desk_bottom",
        projection_lower_start=0.5,
        projection_width_ratio=0.96,
        projection_height_ratio=1.0,
        projection_margin_x=0.02,
        projection_keystone=0.12,
        projection_keystone_jitter=0.03,
        projector_gamma=2.2,
        projector_gain=1.0,
        projector_channel_gain=(1.0, 0.97, 0.94),
        projector_ambient=0.04,
        projector_vignetting=0.18,
        projector_distance_falloff=0.25,
        projector_psf=False,
        projection_randomization_enabled=True,
        return_aux=False,
    ):
        attack_mode = str(attack_mode).lower()
        if attack_mode in ("clean", "no_attack", "none"):
            output = self.im_process(images, mean, std)
            self.last_alpha_mean = 0.0
            self.last_coverage_ratio = 0.0
            self.last_bottom_ratio = 0.0
            self.last_keystone = 0.0
            self.last_backend = "clean"
            if not return_aux:
                return output
            return output, {
                "projected_inputs": [self._to_tensor_image(image).detach().cpu() for image in images],
                "projection_alpha_mean": self.last_alpha_mean,
                "projection_coverage_ratio": self.last_coverage_ratio,
                "projection_bottom_ratio": self.last_bottom_ratio,
                "projection_keystone": self.last_keystone,
                "projection_backend": self.last_backend,
            }
        if attack_mode == "patch":
            output = self.patch_transform.apply_random_patch_batch(
                images=images,
                patch=attack_texture,
                mean=mean,
                std=std,
                geometry=geometry,
            )
            self.last_alpha_mean = 1.0
            self.last_coverage_ratio = 0.0
            self.last_bottom_ratio = 0.0
            self.last_keystone = 0.0
            self.last_backend = "patch"
            if not return_aux:
                return output
            return output, {
                "projected_inputs": [self._to_tensor_image(image).detach().cpu() for image in images],
                "projection_alpha_mean": self.last_alpha_mean,
                "projection_coverage_ratio": self.last_coverage_ratio,
                "projection_bottom_ratio": self.last_bottom_ratio,
                "projection_keystone": self.last_keystone,
                "projection_backend": self.last_backend,
            }

        return self.apply_projection_batch(
            images=images,
            projection_texture=attack_texture,
            mean=mean,
            std=std,
            geometry=geometry,
            projection_alpha=projection_alpha,
            projection_alpha_jitter=projection_alpha_jitter,
            projection_soft_edge=projection_soft_edge,
            projection_angle=projection_angle,
            projection_fixed_angle=projection_fixed_angle,
            projection_shear=projection_shear,
            projection_scale_min=projection_scale_min,
            projection_scale_max=projection_scale_max,
            projection_region=projection_region,
            projection_lower_start=projection_lower_start,
            projection_width_ratio=projection_width_ratio,
            projection_height_ratio=projection_height_ratio,
            projection_margin_x=projection_margin_x,
            projection_keystone=projection_keystone,
            projection_keystone_jitter=projection_keystone_jitter,
            projector_gamma=projector_gamma,
            projector_gain=projector_gain,
            projector_channel_gain=projector_channel_gain,
            projector_ambient=projector_ambient,
            projector_vignetting=projector_vignetting,
            projector_distance_falloff=projector_distance_falloff,
            projector_psf=projector_psf,
            projection_randomization_enabled=projection_randomization_enabled,
            return_aux=return_aux,
        )
