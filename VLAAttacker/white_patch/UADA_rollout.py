import os
import pickle
import random
import csv
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
import transformers
from PIL import Image
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from white_patch.diffusion_lighting_augmentor import DiffusionLightingAugmentor
    from white_patch.ic_light_augmentor import ICLightAugmentor
    from white_patch.projector_attack_transform import ProjectorAttackTransform
except Exception:
    from diffusion_lighting_augmentor import DiffusionLightingAugmentor
    from ic_light_augmentor import ICLightAugmentor
    from projector_attack_transform import ProjectorAttackTransform

try:
    import wandb
except Exception:
    wandb = None


IGNORE_INDEX = -100


def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class OpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="pgd", resize_patch=False):
        self.vla = vla.eval()
        for param in self.vla.parameters():
            param.requires_grad_(False)

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        self.base_tokenizer = self.processor.tokenizer
        self.predict_stop_token = True
        self.pad_token_id = 32000
        self.model_max_length = 2048

        self.loss_buffer = []
        self.save_dir = save_dir
        self.optimizer = optimizer

        self.mean = [
            torch.tensor([0.484375, 0.455078125, 0.40625]),
            torch.tensor([0.5, 0.5, 0.5]),
        ]
        self.std = [
            torch.tensor([0.228515625, 0.2236328125, 0.224609375]),
            torch.tensor([0.5, 0.5, 0.5]),
        ]
        self.randomPatchTransform = ProjectorAttackTransform(self.vla.device, resize_patch)
        self.best_rollout_score = -1e10
        self.MSE_Distance_best = 10000
        self.action_bin_centers = torch.linspace(-1.0, 1.0, steps=256, device=self.vla.device, dtype=torch.float32)
        self.lighting_augmentor = None
        self.lighting_aug_train_only = False
        self.default_action_dim = 7

        self.input_sizes = [[3, 224, 224], [3, 224, 224]]
        self.tvf_resize_params = [
            {"antialias": True, "interpolation": 3, "max_size": None, "size": [224, 224]},
            {"antialias": True, "interpolation": 3, "max_size": None, "size": [224, 224]},
        ]
        self.tvf_crop_params = [{"output_size": [224, 224]}, {"output_size": [224, 224]}]
        self.tvf_normalize_params = [
            {"inplace": False, "mean": [0.484375, 0.455078125, 0.40625], "std": [0.228515625, 0.2236328125, 0.224609375]},
            {"inplace": False, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        ]

    def _reset_metric_buffers(self):
        # Legacy metrics
        self.train_CE_loss = []
        self.train_MSE_distance_loss = []
        self.train_UAD = []
        self.val_CE_loss = []
        self.val_MSE_Distance = []
        self.val_UAD = []

        # Rollout-v2 metrics
        self.train_rollout_action_gap = []
        self.train_rollout_action_gap_joints = []
        self.train_rollout_history_div = []
        self.train_rollout_score = []
        self.train_phase_id = []

        self.val_rollout_action_gap = []
        self.val_rollout_action_gap_joints = []
        self.val_rollout_history_div = []
        self.val_rollout_history_div_legacy = []
        self.val_rollout_score = []
        self.val_rollout_score_legacy = []

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)
        x_ticks = list(range(0, num_iters))
        plt.plot(x_ticks, self.loss_buffer, label="Rollout Score")
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.savefig("%s/loss_curve.png" % (self.save_dir))
        plt.clf()
        torch.save(self.loss_buffer, "%s/loss" % (self.save_dir))

    def patchattack_unconstrained(
        self,
        train_dataloader,
        val_dataloader,
        num_iter=5000,
        target_action=np.zeros(7),
        patch_size=[3, 50, 50],
        lr=1 / 255,
        accumulate_steps=1,
        maskidx=[],
        use_all_joints=True,
        gripper_weight=0.5,
        warmup=20,
        filterGripTrainTo1=False,
        geometry=False,
        innerLoop=1,
        args=None,
        phase1_ratio=0.4,
        phase1_rollout=8,
        phase2_rollout=24,
        lambda_action_gap=1.0,
        lambda_history=0.5,
        lambda_ce=0.1,
        eval_rollout=24,
        save_interval=100,
        eval_enabled=True,
        val_max_batches=1000,
        lighting_aug_enabled=False,
        lighting_model_id="stabilityai/sdxl-turbo",
        lighting_pool_size=8,
        lighting_refresh_interval=200,
        lighting_num_inference_steps=8,
        lighting_guidance_scale=0.0,
        lighting_blend_min=0.15,
        lighting_blend_max=0.5,
        lighting_apply_prob=1.0,
        lighting_seed=42,
        lighting_backend="ic_light",
        ic_light_repo="/home/yxx/IC-Light",
        ic_light_model_path="/home/yxx/IC-Light/models/iclight_sd15_fbc.safetensors",
        ic_light_scope="full",
        ic_light_bg_control="legacy_prompt",
        lighting_aug_train_only=False,
        attack_mode="projection",
        projection_size=None,
        projection_alpha=0.35,
        projection_alpha_jitter=0.10,
        projection_soft_edge=2.5,
        projection_angle=25.0,
        projection_fixed_angle=False,
        projection_shear=0.15,
        projection_scale_min=0.8,
        projection_scale_max=1.2,
        projection_region="lower_half_fixed",
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
        viz_enabled=True,
        viz_policy="milestone",
        viz_samples=4,
        viz_save_best=True,
        viz_save_last=True,
        val_deterministic=False,
        val_seed=42,
        val_disable_lighting=False,
    ):
        del target_action
        del innerLoop

        self._reset_metric_buffers()

        attack_mode = str(attack_mode).lower()
        if projection_size is None:
            projection_size = patch_size
        projection_size = [int(x) for x in projection_size]

        projection_texture = torch.rand(projection_size, device=self.vla.device)
        projection_texture.requires_grad_(True)
        projection_texture.retain_grad()

        if self.optimizer != "adamW":
            raise ValueError("UADA_rollout currently supports optimizer='adamW' only.")

        optimizer = transformers.AdamW([projection_texture], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=max(1, int(num_iter / max(1, accumulate_steps))),
            num_cycles=0.5,
            last_epoch=-1,
        )

        phase1_ratio = float(min(max(phase1_ratio, 0.0), 1.0))
        phase1_end_iter = int(num_iter * phase1_ratio)
        accumulate_steps = max(1, int(accumulate_steps))
        save_interval = max(1, int(save_interval))
        use_all_joints = bool(use_all_joints)
        gripper_weight = float(gripper_weight)
        phase1_rollout = max(1, int(phase1_rollout))
        phase2_rollout = max(1, int(phase2_rollout))
        eval_rollout = max(1, int(eval_rollout))
        val_max_batches = max(1, int(val_max_batches))
        eval_enabled = bool(eval_enabled)
        self.lighting_aug_train_only = bool(lighting_aug_train_only)
        viz_enabled = bool(viz_enabled)
        viz_policy = str(viz_policy).lower()
        viz_samples = max(1, int(viz_samples))
        viz_save_best = bool(viz_save_best)
        viz_save_last = bool(viz_save_last)
        val_deterministic = bool(val_deterministic)
        val_seed = int(val_seed)
        val_disable_lighting = bool(val_disable_lighting)
        milestone_iters = self._build_milestone_iters(num_iter=num_iter, phase1_end_iter=phase1_end_iter)

        if viz_enabled and not eval_enabled:
            print("[Viz] `viz_enabled=true` but `eval_enabled=false`; visualization will be skipped.")
        if self.lighting_aug_train_only and lighting_aug_enabled:
            print("[LightingAugmentor] train-only mode enabled; validation lighting augmentation disabled.")

        self._setup_lighting_augmentor(
            enabled=lighting_aug_enabled,
            backend=lighting_backend,
            model_id=lighting_model_id,
            pool_size=lighting_pool_size,
            refresh_interval=lighting_refresh_interval,
            num_inference_steps=lighting_num_inference_steps,
            guidance_scale=lighting_guidance_scale,
            blend_min=lighting_blend_min,
            blend_max=lighting_blend_max,
            apply_prob=lighting_apply_prob,
            seed=lighting_seed,
            ic_light_repo=ic_light_repo,
            ic_light_model_path=ic_light_model_path,
            ic_light_scope=ic_light_scope,
            ic_light_bg_control=ic_light_bg_control,
        )

        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        optimizer.zero_grad()

        for i in tqdm(range(num_iter)):
            phase_id = 1 if i < phase1_end_iter else 2
            rollout_steps = phase1_rollout if phase_id == 1 else phase2_rollout

            data, train_iterator = self._next_batch(train_iterator, train_dataloader)
            pixel_values, labels_full, attention_mask, input_ids = self._prepare_batch(
                data=data,
                maskidx=maskidx,
                filterGripTrainTo1=filterGripTrainTo1,
                use_all_joints=use_all_joints,
            )
            pixel_values = self._apply_lighting_augmentation(pixel_values, iteration_idx=i, split="train")

            labels_masked = self.mask_labels(labels_full.clone(), maskidx=maskidx, use_all_joints=use_all_joints)
            action_mask_full = self._build_action_mask(labels_full)

            if action_mask_full.sum().item() == 0:
                continue

            clean_images = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
            adv_images, attack_aux = self.randomPatchTransform.apply_attack_batch(
                images=pixel_values,
                attack_texture=projection_texture,
                mean=self.mean,
                std=self.std,
                attack_mode=attack_mode,
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
                return_aux=True,
            )

            clean_rollout_input_ids = input_ids.clone()
            adv_rollout_input_ids = input_ids.clone()
            total_action_gap = 0.0
            total_history_div = 0.0
            action_terms = 0
            history_terms = 0
            total_per_joint_gap = None
            final_output_adv = None
            weighted_train_loss_sum = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            train_weight_sum = 0.0
            need_history = lambda_history > 0

            for step_idx in range(rollout_steps):
                with torch.no_grad():
                    output_clean: CausalLMOutputWithPast = self.vla(
                        input_ids=clean_rollout_input_ids,
                        attention_mask=attention_mask,
                        pixel_values=clean_images.to(torch.bfloat16),
                        labels=labels_masked,
                        output_hidden_states=need_history,
                        use_cache=False,
                    )

                output_adv: CausalLMOutputWithPast = self.vla(
                    input_ids=adv_rollout_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=adv_images.to(torch.bfloat16),
                    labels=labels_masked,
                    output_hidden_states=need_history,
                    use_cache=False,
                )
                final_output_adv = output_adv

                step_action_gap_loss, step_action_gap_metric, step_per_joint_gap, adv_pred_tokens = self._compute_action_gap_losses(
                    adv_logits=output_adv.logits,
                    clean_logits=output_clean.logits,
                    labels_full=labels_full,
                    action_mask_full=action_mask_full,
                    maskidx=maskidx,
                    use_all_joints=use_all_joints,
                    gripper_weight=gripper_weight,
                )
                total_action_gap += step_action_gap_metric.item()
                action_terms += 1
                total_per_joint_gap = self._accumulate_joint_values(total_per_joint_gap, step_per_joint_gap)

                clean_pred_tokens = self._extract_pred_action_tokens_from_logits(output_clean.logits, labels_full)
                step_history_div = torch.zeros((), device=self.vla.device, dtype=torch.float32)
                if need_history:
                    step_history_div, _clean_history_state, _adv_history_state = self._compute_clean_adv_history_divergence(
                        clean_hidden_states=output_clean.hidden_states[-1],
                        adv_hidden_states=output_adv.hidden_states[-1],
                        labels_full=labels_full,
                        action_mask_full=action_mask_full,
                    )
                    total_history_div += step_history_div.detach().item()
                    history_terms += 1

                step_weight = float(self._get_rollout_step_weight(step_idx))
                train_weight_sum += step_weight

                step_loss = -(lambda_action_gap * step_action_gap_loss)
                if need_history:
                    step_loss = step_loss - (lambda_history * step_history_div)
                if step_idx == rollout_steps - 1:
                    step_loss = step_loss + lambda_ce * output_adv.loss

                weighted_train_loss_sum = weighted_train_loss_sum + (step_weight * step_loss)

                clean_rollout_input_ids = self._update_rollout_inputs(
                    rollout_input_ids=clean_rollout_input_ids,
                    pred_action_tokens=clean_pred_tokens,
                    action_mask_full=action_mask_full,
                )
                adv_rollout_input_ids = self._update_rollout_inputs(
                    rollout_input_ids=adv_rollout_input_ids,
                    pred_action_tokens=adv_pred_tokens,
                    action_mask_full=action_mask_full,
                )

            total_train_loss = weighted_train_loss_sum / float(max(train_weight_sum, 1e-8))
            (total_train_loss / float(accumulate_steps)).backward()

            log_patch_grad = 0.0
            if projection_texture.grad is not None:
                log_patch_grad = projection_texture.grad.detach().abs().mean().item()

            optimizer_step = ((i + 1) % accumulate_steps == 0) or ((i + 1) == num_iter)
            if optimizer_step:
                optimizer.step()
                projection_texture.data = projection_texture.data.clamp(0, 1)
                optimizer.zero_grad()
                scheduler.step()

            if final_output_adv is None:
                continue

            final_ce = final_output_adv.loss
            final_mse_distance, final_uad = self.weighted_loss(final_output_adv.logits, labels_masked)
            avg_action_gap = total_action_gap / float(max(1, action_terms))
            avg_history_div = total_history_div / float(max(1, history_terms))
            avg_per_joint_gap = self._normalize_joint_values(total_per_joint_gap, float(max(1, action_terms)))
            rollout_score = lambda_action_gap * avg_action_gap + lambda_history * avg_history_div

            self.train_CE_loss.append(final_ce.item())
            self.train_MSE_distance_loss.append(final_mse_distance.item())
            self.train_UAD.append(final_uad.item())
            self.train_rollout_action_gap.append(avg_action_gap)
            self.train_rollout_action_gap_joints.append(avg_per_joint_gap.detach().cpu().tolist())
            self.train_rollout_history_div.append(avg_history_div)
            self.train_rollout_score.append(rollout_score)
            self.train_phase_id.append(phase_id)
            self.loss_buffer.append(rollout_score)

            train_logdata = {
                "TRAIN_attack_loss(CE)": final_ce.item(),
                "TRAIN_attack_loss (MSE_Distance)": final_mse_distance.item(),
                "TRAIN_UAD": final_uad.item(),
                "TRAIN_patch_gradient": log_patch_grad,
                "TRAIN_LR": optimizer.param_groups[0]["lr"],
                "TRAIN_rollout_action_gap": avg_action_gap,
                "TRAIN_rollout_history_div": avg_history_div,
                "TRAIN_rollout_score": rollout_score,
                "phase_id": phase_id,
                "TRAIN_projection_alpha_mean": float(attack_aux["projection_alpha_mean"]),
                "TRAIN_projection_coverage_ratio": float(attack_aux["projection_coverage_ratio"]),
                "TRAIN_projection_bottom_ratio": float(attack_aux["projection_bottom_ratio"]),
                "TRAIN_projection_keystone": float(attack_aux["projection_keystone"]),
                "attack_mode": attack_mode,
            }
            for joint_idx, joint_gap_value in enumerate(avg_per_joint_gap.detach().cpu().tolist()):
                train_logdata[f"TRAIN_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
            if self.lighting_augmentor is not None:
                train_logdata["TRAIN_lighting_pool_size"] = self.lighting_augmentor.current_pool_size
                train_logdata["TRAIN_lighting_backend"] = self.lighting_augmentor.backend
                train_logdata["ic_light_scope"] = getattr(self.lighting_augmentor, "scope", "n/a")
                train_logdata["ic_light_bg_control"] = getattr(self.lighting_augmentor, "bg_control", "n/a")
            train_logdata["TRAIN_projection_backend"] = str(attack_aux["projection_backend"])
            if args is not None and args.wandb_project != "false" and wandb is not None:
                wandb.log(train_logdata, step=i)

            eval_due_to_interval = ((i + 1) % save_interval == 0) or (i == num_iter - 1)
            eval_due_to_milestone = viz_enabled and (viz_policy == "milestone") and (i in milestone_iters)
            if eval_enabled and (eval_due_to_interval or eval_due_to_milestone):
                if i % (save_interval * 2) == 0:
                    self.plot_loss()

                val_stats, val_iterator, visual_triplets, lighting_backend, projection_backend = self._evaluate_rollout(
                    projection_texture=projection_texture,
                    val_iterator=val_iterator,
                    val_dataloader=val_dataloader,
                    maskidx=maskidx,
                    eval_rollout=eval_rollout,
                    geometry=geometry,
                    filterGripTrainTo1=filterGripTrainTo1,
                    use_all_joints=use_all_joints,
                    gripper_weight=gripper_weight,
                    lambda_action_gap=lambda_action_gap,
                    lambda_history=lambda_history,
                    val_max_batches=val_max_batches,
                    global_iter=i,
                    viz_samples=viz_samples,
                    attack_mode=attack_mode,
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
                    val_deterministic=val_deterministic,
                    val_seed=val_seed,
                    val_disable_lighting=val_disable_lighting,
                )

                self.val_CE_loss.append(val_stats["VAL_CE_loss"])
                self.val_MSE_Distance.append(val_stats["VAL_MSE_Distance"])
                self.val_UAD.append(val_stats["VAL_UAD"])
                self.val_rollout_action_gap.append(val_stats["VAL_rollout_action_gap"])
                self.val_rollout_action_gap_joints.append(
                    self._extract_joint_metric_list(val_stats, prefix="VAL_rollout_action_gap_joint_")
                )
                self.val_rollout_history_div.append(val_stats["VAL_rollout_history_div"])
                self.val_rollout_history_div_legacy.append(val_stats["VAL_rollout_history_div_legacy"])
                self.val_rollout_score.append(val_stats["VAL_rollout_score"])
                self.val_rollout_score_legacy.append(val_stats["VAL_rollout_score_legacy"])

                if args is not None and args.wandb_project != "false" and wandb is not None:
                    wandb.log(val_stats, step=i)

                improved = val_stats["VAL_rollout_score"] > self.best_rollout_score
                if improved:
                    self.best_rollout_score = val_stats["VAL_rollout_score"]
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                    torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))

                temp_save_dir = os.path.join(self.save_dir, "last")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))

                if viz_enabled:
                    should_dump, vis_reason = self._should_dump_visualization(
                        iter_idx=i,
                        num_iter=num_iter,
                        milestone_iters=milestone_iters,
                        improved=improved,
                        viz_policy=viz_policy,
                        viz_save_best=viz_save_best,
                        viz_save_last=viz_save_last,
                    )
                    if should_dump:
                        self._dump_visual_triplets(
                            visual_triplets=visual_triplets,
                            iter_idx=i,
                            phase_id=phase_id,
                            is_best=improved,
                            val_rollout_score=val_stats["VAL_rollout_score"],
                            reason=vis_reason,
                            lighting_backend=lighting_backend,
                            projection_backend=projection_backend,
                            attack_mode=attack_mode,
                            args=args,
                        )

                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()
            elif (not eval_enabled) and (i == num_iter - 1):
                temp_save_dir = os.path.join(self.save_dir, "last")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()

    def _next_batch(self, iterator, dataloader):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            data = next(iterator)
        return data, iterator

    def _setup_lighting_augmentor(
        self,
        enabled,
        backend,
        model_id,
        pool_size,
        refresh_interval,
        num_inference_steps,
        guidance_scale,
        blend_min,
        blend_max,
        apply_prob,
        seed,
        ic_light_repo,
        ic_light_model_path,
        ic_light_scope,
        ic_light_bg_control,
    ):
        if not enabled:
            self.lighting_augmentor = None
            return

        backend_name = str(backend).lower().strip()
        if backend_name == "ic_light":
            self.lighting_augmentor = ICLightAugmentor(
                enabled=enabled,
                device=str(self.vla.device),
                pool_size=pool_size,
                refresh_interval=refresh_interval,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                ic_light_repo=ic_light_repo,
                ic_light_model_path=ic_light_model_path,
                scope=ic_light_scope,
                bg_control=ic_light_bg_control,
                legacy_model_id=model_id,
                legacy_blend_min=blend_min,
                legacy_blend_max=blend_max,
                legacy_apply_prob=apply_prob,
            )
        else:
            self.lighting_augmentor = DiffusionLightingAugmentor(
                enabled=enabled,
                model_id=model_id,
                device=str(self.vla.device),
                pool_size=pool_size,
                refresh_interval=refresh_interval,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                blend_min=blend_min,
                blend_max=blend_max,
                apply_prob=apply_prob,
                seed=seed,
            )
        print(
            "[LightingAugmentor] enabled "
            f"(backend={backend_name}, model_id={model_id}, pool_size={pool_size}, refresh_interval={refresh_interval})"
        )

    def _apply_lighting_augmentation(self, pixel_values, iteration_idx, split, fixed_map_idx=None):
        if not self._is_lighting_enabled_for_split(split):
            return pixel_values
        return self.lighting_augmentor.augment_batch(
            images=pixel_values,
            iteration_idx=iteration_idx,
            split=split,
            fixed_map_idx=fixed_map_idx,
        )

    def _is_lighting_enabled_for_split(self, split):
        if self.lighting_augmentor is None:
            return False
        split_name = str(split).lower()
        if split_name == "val" and self.lighting_aug_train_only:
            return False
        return True

    @contextmanager
    def _temporary_rng_seed(self, seed=None):
        if seed is None:
            yield
            return

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()

        seed_value = int(seed)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if (cuda_states is not None) and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)

    def _build_milestone_iters(self, num_iter, phase1_end_iter):
        milestone_iters = {0, max(0, num_iter - 1)}
        if 0 < phase1_end_iter < num_iter:
            milestone_iters.add(phase1_end_iter)
        return milestone_iters

    def _should_dump_visualization(
        self,
        iter_idx,
        num_iter,
        milestone_iters,
        improved,
        viz_policy,
        viz_save_best,
        viz_save_last,
    ):
        reasons = []
        if viz_policy == "every_eval":
            reasons.append("every_eval")
        elif (viz_policy == "milestone") and (iter_idx in milestone_iters):
            reasons.append("milestone")

        if viz_save_best and improved:
            reasons.append("best")
        if viz_save_last and (iter_idx == (num_iter - 1)):
            reasons.append("last")

        should_dump = len(reasons) > 0
        return should_dump, "|".join(reasons) if should_dump else ""

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
            raise TypeError(f"Unsupported image type for visualization: {type(image)}")

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array).convert("RGB")

    def _to_pil_batch(self, images, max_samples):
        pil_images = []
        for image in images:
            pil_images.append(self._to_pil(image))
            if len(pil_images) >= max_samples:
                break
        return pil_images

    def _build_visual_triplets(self, original_images, projected_input_images, adv_images, max_samples, relight_images=None):
        if adv_images is None:
            return []

        orig_pil = self._to_pil_batch(original_images, max_samples=max_samples)
        projected_input_pil = self._to_pil_batch(projected_input_images, max_samples=max_samples)
        relight_pil = self._to_pil_batch(relight_images, max_samples=max_samples) if relight_images is not None else []
        adv_images = self.randomPatchTransform.denormalize(
            adv_images[:, 0:3, :, :].detach().cpu(),
            mean=self.mean[0],
            std=self.std[0],
        ).clamp(0, 1)
        adv_pil = []
        for idx in range(min(int(adv_images.shape[0]), max_samples)):
            adv_pil.append(torchvision.transforms.ToPILImage()(adv_images[idx]))

        n = min(len(orig_pil), len(projected_input_pil), len(adv_pil))
        triplets = []
        for idx in range(n):
            item = {
                "sample_idx": idx,
                "orig": orig_pil[idx],
                "projected_input": projected_input_pil[idx],
                "adv": adv_pil[idx],
            }
            if idx < len(relight_pil):
                item["relight"] = relight_pil[idx]
            triplets.append(item)
        return triplets

    def _visualization_root(self):
        visual_root = os.path.join(self.save_dir, "visualization")
        os.makedirs(visual_root, exist_ok=True)
        return visual_root

    def _append_visual_manifest(self, row):
        visual_root = self._visualization_root()
        manifest_path = os.path.join(visual_root, "manifest.csv")
        file_exists = os.path.exists(manifest_path)
        fieldnames = [
            "iter_idx",
            "phase_id",
            "is_best",
            "val_rollout_score",
            "reason",
            "lighting_backend",
            "projection_backend",
            "attack_mode",
            "visual_dir",
        ]
        with open(manifest_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _dump_visual_triplets(
        self,
        visual_triplets,
        iter_idx,
        phase_id,
        is_best,
        val_rollout_score,
        reason,
        lighting_backend,
        projection_backend,
        attack_mode,
        args=None,
    ):
        if not visual_triplets:
            return

        visual_root = self._visualization_root()
        iter_dir = os.path.join(visual_root, f"iter_{int(iter_idx):06d}")
        os.makedirs(iter_dir, exist_ok=True)

        for triplet in visual_triplets:
            sample_idx = int(triplet["sample_idx"])
            triplet["orig"].save(os.path.join(iter_dir, f"sample_{sample_idx:03d}_orig.png"))
            triplet["projected_input"].save(os.path.join(iter_dir, f"sample_{sample_idx:03d}_projected_input.png"))
            if "relight" in triplet:
                triplet["relight"].save(os.path.join(iter_dir, f"sample_{sample_idx:03d}_relight.png"))
            triplet["adv"].save(os.path.join(iter_dir, f"sample_{sample_idx:03d}_adv.png"))

        self._append_visual_manifest(
            {
                "iter_idx": int(iter_idx),
                "phase_id": int(phase_id),
                "is_best": int(bool(is_best)),
                "val_rollout_score": float(val_rollout_score),
                "reason": reason,
                "lighting_backend": str(lighting_backend),
                "projection_backend": str(projection_backend),
                "attack_mode": str(attack_mode),
                "visual_dir": iter_dir,
            }
        )

        if args is not None and args.wandb_project != "false" and wandb is not None:
            wandb.log(
                {
                    "VIS/orig": [wandb.Image(item["orig"]) for item in visual_triplets],
                    "VIS/projected_input": [wandb.Image(item["projected_input"]) for item in visual_triplets],
                    "VIS/adv": [wandb.Image(item["adv"]) for item in visual_triplets],
                },
                step=int(iter_idx),
            )
            if "relight" in visual_triplets[0]:
                wandb.log(
                    {"VIS/relight": [wandb.Image(item["relight"]) for item in visual_triplets]},
                    step=int(iter_idx),
                )

    def _prepare_batch(self, data, maskidx, filterGripTrainTo1, use_all_joints):
        use_gripper_filter = (not use_all_joints) and (len(maskidx) == 1) and (maskidx[0] == 6) and filterGripTrainTo1
        if use_gripper_filter:
            labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            labels_full = labels
        else:
            pixel_values = data["pixel_values"]
            labels_full = data["labels"].to(self.vla.device)
            attention_mask = data["attention_mask"].to(self.vla.device)
            input_ids = data["input_ids"].to(self.vla.device)
        return pixel_values, labels_full, attention_mask, input_ids

    def _build_action_mask(self, labels_full):
        return labels_full[:, 1:] > self.action_tokenizer.action_token_begin_idx

    def _sanitize_mask_indices(self, maskidx, action_dim):
        unique_ids = sorted(set(int(x) for x in maskidx))
        valid_ids = [idx for idx in unique_ids if 0 <= idx < action_dim]
        if len(valid_ids) == 0:
            valid_ids = list(range(action_dim))
        return valid_ids

    def _select_joint_indices_and_weights(self, num_actions, maskidx, use_all_joints, gripper_weight):
        if num_actions <= 0:
            empty_idx = torch.zeros((0,), device=self.vla.device, dtype=torch.long)
            empty_weights = torch.zeros((0,), device=self.vla.device, dtype=torch.float32)
            return empty_idx, empty_weights

        if use_all_joints:
            selected_ids = list(range(num_actions))
        else:
            selected_ids = self._sanitize_mask_indices(maskidx, num_actions)

        idx_tensor = torch.tensor(selected_ids, device=self.vla.device, dtype=torch.long)
        joint_weights = torch.ones((len(selected_ids),), device=self.vla.device, dtype=torch.float32)
        if use_all_joints and len(selected_ids) > 0:
            gripper_idx = num_actions - 1
            for selected_pos, joint_id in enumerate(selected_ids):
                if int(joint_id) == gripper_idx:
                    joint_weights[selected_pos] = float(gripper_weight)
                    break
        return idx_tensor, joint_weights

    def _weighted_reduce(self, values, weights):
        if values.numel() == 0:
            return torch.zeros((), device=self.vla.device, dtype=torch.float32)
        denom = weights.sum().clamp(min=1e-6)
        return (values * weights).sum() / denom

    def _accumulate_joint_values(self, accumulator, joint_values):
        vector = joint_values.detach().to(torch.float32)
        if accumulator is None:
            return vector.clone()

        if vector.shape[0] > accumulator.shape[0]:
            expanded = torch.zeros((vector.shape[0],), device=accumulator.device, dtype=accumulator.dtype)
            expanded[: accumulator.shape[0]] = accumulator
            accumulator = expanded
        min_dim = min(accumulator.shape[0], vector.shape[0])
        accumulator[:min_dim] += vector[:min_dim]
        return accumulator

    def _normalize_joint_values(self, accumulated_joint_values, divisor):
        if accumulated_joint_values is None:
            return torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
        return accumulated_joint_values / float(max(1.0, divisor))

    def _extract_joint_metric_list(self, metric_dict, prefix):
        joint_values = []
        for key in sorted(metric_dict.keys()):
            if key.startswith(prefix):
                joint_values.append(float(metric_dict[key]))
        if len(joint_values) == 0:
            joint_values = [0.0 for _ in range(self.default_action_dim)]
        return joint_values

    def _extract_action_logits(self, logits, labels_full):
        temp_label = labels_full[:, 1:]
        temp_logits = logits[:, :, 31744:32000]
        action_logits = temp_logits[:, -temp_label.shape[-1] - 1 : -1, :]
        return action_logits

    def _extract_pred_action_tokens_from_logits(self, logits, labels_full):
        action_logits = self._extract_action_logits(logits, labels_full)
        if action_logits.numel() == 0:
            batch_size = int(logits.shape[0])
            return torch.zeros((batch_size, 0), device=logits.device, dtype=torch.long)
        return action_logits.argmax(dim=-1) + 31744

    def _compute_clean_adv_history_divergence(self, clean_hidden_states, adv_hidden_states, labels_full, action_mask_full):
        clean_history_state = self._extract_history_state(clean_hidden_states, labels_full, action_mask_full)
        adv_history_state = self._extract_history_state(adv_hidden_states, labels_full, action_mask_full)
        history_div = 1.0 - F.cosine_similarity(clean_history_state, adv_history_state, dim=-1).mean()
        return history_div, clean_history_state, adv_history_state

    def _compute_legacy_history_divergence(self, current_adv_history_state, prev_adv_history_state):
        if prev_adv_history_state is None:
            return torch.zeros((), device=self.vla.device, dtype=torch.float32)
        return 1.0 - F.cosine_similarity(current_adv_history_state, prev_adv_history_state, dim=-1).mean()

    def _get_rollout_step_weight(self, step_idx):
        step_idx = int(step_idx)
        if step_idx <= 9:
            return 1.5
        if step_idx >= 100:
            return 1.5
        return 1.0

    def _compute_action_gap_losses(
        self,
        adv_logits,
        clean_logits,
        labels_full,
        action_mask_full,
        maskidx,
        use_all_joints,
        gripper_weight,
    ):
        batch_size = labels_full.shape[0]
        adv_action_logits = self._extract_action_logits(adv_logits, labels_full)
        clean_action_logits = self._extract_action_logits(clean_logits, labels_full)

        seq_len = min(action_mask_full.shape[1], adv_action_logits.shape[1], clean_action_logits.shape[1])
        action_mask = action_mask_full[:, :seq_len]
        adv_action_logits = adv_action_logits[:, :seq_len, :]
        clean_action_logits = clean_action_logits[:, :seq_len, :]

        adv_pred_tokens = adv_action_logits.argmax(dim=-1) + 31744

        masked_adv_logits = adv_action_logits[action_mask]
        masked_clean_logits = clean_action_logits[action_mask]
        if masked_adv_logits.numel() == 0:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return zero, zero.detach(), per_joint_zero, adv_pred_tokens.detach()

        # Differentiable action-gap loss (for optimization)
        adv_probs = F.softmax(masked_adv_logits, dim=-1)
        clean_probs = F.softmax(masked_clean_logits, dim=-1)
        adv_soft_actions = (adv_probs * self.action_bin_centers).sum(dim=-1)
        clean_soft_actions = (clean_probs * self.action_bin_centers).sum(dim=-1)

        num_actions = int(masked_adv_logits.shape[0] // batch_size)
        if num_actions <= 0:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return zero, zero.detach(), per_joint_zero, adv_pred_tokens.detach()

        adv_soft_actions = adv_soft_actions.view(batch_size, num_actions)
        clean_soft_actions = clean_soft_actions.view(batch_size, num_actions)
        per_joint_mse = torch.square(adv_soft_actions - clean_soft_actions).mean(dim=0)
        idx_tensor, joint_weights = self._select_joint_indices_and_weights(
            num_actions=num_actions,
            maskidx=maskidx,
            use_all_joints=use_all_joints,
            gripper_weight=gripper_weight,
        )
        if idx_tensor.numel() == 0:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return zero, zero.detach(), per_joint_zero, adv_pred_tokens.detach()
        loss_action_gap = self._weighted_reduce(
            per_joint_mse.index_select(0, idx_tensor),
            joint_weights,
        )

        # Human-readable metric: decoded continuous action L1 gap
        adv_hard_tokens = masked_adv_logits.argmax(dim=-1) + 31744
        clean_hard_tokens = masked_clean_logits.argmax(dim=-1) + 31744
        adv_cont = self._decode_action_tokens(adv_hard_tokens, batch_size, num_actions)
        clean_cont = self._decode_action_tokens(clean_hard_tokens, batch_size, num_actions)
        per_joint_l1 = torch.abs(adv_cont - clean_cont).mean(dim=0)
        metric_action_gap = self._weighted_reduce(
            per_joint_l1.index_select(0, idx_tensor),
            joint_weights,
        )

        return loss_action_gap, metric_action_gap.detach(), per_joint_l1.detach(), adv_pred_tokens.detach()

    def _decode_action_tokens(self, token_ids_flat, batch_size, num_actions):
        decoded = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(token_ids_flat.detach().cpu().numpy()),
            device=self.vla.device,
            dtype=torch.float32,
        )
        if decoded.numel() != token_ids_flat.numel():
            normalized = ((token_ids_flat.float() - 31744.0) / 255.0) * 2.0 - 1.0
            decoded = normalized
        return decoded.view(batch_size, num_actions)

    def _extract_history_state(self, hidden_states, labels_full, action_mask_full):
        num_patches = self.vla.vision_backbone.featurizer.patch_embed.num_patches
        hidden_aligned = hidden_states[:, num_patches:-1, :]
        seq_len = min(hidden_aligned.shape[1], action_mask_full.shape[1])
        hidden_aligned = hidden_aligned[:, :seq_len, :]
        action_mask = action_mask_full[:, :seq_len]

        masked_hidden = hidden_aligned[action_mask]
        batch_size = hidden_aligned.shape[0]
        hidden_dim = hidden_aligned.shape[-1]
        if masked_hidden.numel() == 0:
            return torch.zeros((batch_size, hidden_dim), device=self.vla.device, dtype=hidden_aligned.dtype)

        num_actions = int(masked_hidden.shape[0] // batch_size)
        if num_actions <= 0:
            return torch.zeros((batch_size, hidden_dim), device=self.vla.device, dtype=hidden_aligned.dtype)

        masked_hidden = masked_hidden.view(batch_size, num_actions, hidden_dim)
        return masked_hidden.mean(dim=1)

    def _update_rollout_inputs(self, rollout_input_ids, pred_action_tokens, action_mask_full):
        updated_ids = rollout_input_ids.clone()
        seq_len = min(action_mask_full.shape[1], pred_action_tokens.shape[1], updated_ids.shape[1] - 1)
        if seq_len <= 0:
            return updated_ids

        target_slice = updated_ids[:, 1 : 1 + seq_len]
        action_mask = action_mask_full[:, :seq_len]
        token_slice = pred_action_tokens[:, :seq_len]
        target_slice[action_mask] = token_slice[action_mask]
        updated_ids[:, 1 : 1 + seq_len] = target_slice
        return updated_ids

    def _evaluate_rollout(
        self,
        projection_texture,
        val_iterator,
        val_dataloader,
        maskidx,
        eval_rollout,
        geometry,
        filterGripTrainTo1,
        use_all_joints,
        gripper_weight,
        lambda_action_gap,
        lambda_history,
        val_max_batches,
        global_iter,
        viz_samples,
        attack_mode,
        projection_alpha,
        projection_alpha_jitter,
        projection_soft_edge,
        projection_angle,
        projection_fixed_angle,
        projection_shear,
        projection_scale_min,
        projection_scale_max,
        projection_region,
        projection_lower_start,
        projection_width_ratio,
        projection_height_ratio,
        projection_margin_x,
        projection_keystone,
        projection_keystone_jitter,
        projector_gamma,
        projector_gain,
        projector_channel_gain,
        projector_ambient,
        projector_vignetting,
        projector_distance_falloff,
        projector_psf,
        val_deterministic,
        val_seed,
        val_disable_lighting,
    ):
        avg_ce = 0.0
        avg_mse_distance = 0.0
        avg_uad = 0.0
        avg_action_gap = 0.0
        avg_action_gap_joints = None
        avg_history_div = 0.0
        avg_history_div_legacy = 0.0
        avg_projection_alpha = 0.0
        avg_projection_coverage = 0.0
        avg_projection_bottom = 0.0
        avg_projection_keystone = 0.0

        val_batches = max(1, min(int(val_max_batches), len(val_dataloader)))
        processed_batches = 0
        visual_triplets = []
        val_lighting_enabled = (not val_disable_lighting) and self._is_lighting_enabled_for_split("val")
        if val_lighting_enabled:
            lighting_backend = self.lighting_augmentor.backend
        elif val_disable_lighting:
            lighting_backend = "disabled_by_flag"
        elif self.lighting_augmentor is None:
            lighting_backend = "disabled"
        else:
            lighting_backend = "disabled_train_only"
        projection_backend = "projection"

        with torch.no_grad():
            for val_batch_idx in tqdm(range(val_batches)):
                data, val_iterator = self._next_batch(val_iterator, val_dataloader)
                pixel_values, labels_full, attention_mask, input_ids = self._prepare_batch(
                    data=data,
                    maskidx=maskidx,
                    filterGripTrainTo1=filterGripTrainTo1,
                    use_all_joints=use_all_joints,
                )
                original_pixel_values = list(pixel_values)
                batch_seed = (int(val_seed) + int(val_batch_idx)) if val_deterministic else None
                val_lighting_enabled = (not val_disable_lighting) and self._is_lighting_enabled_for_split("val")
                if val_lighting_enabled:
                    with self._temporary_rng_seed(batch_seed):
                        pixel_values = self._apply_lighting_augmentation(
                            pixel_values,
                            iteration_idx=(global_iter * max(1, val_batches)) + val_batch_idx,
                            split="val",
                        )
                    lighting_backend = self.lighting_augmentor.backend
                elif val_disable_lighting:
                    lighting_backend = "disabled_by_flag"
                elif self.lighting_augmentor is None:
                    lighting_backend = "disabled"
                else:
                    lighting_backend = "disabled_train_only"

                labels_masked = self.mask_labels(labels_full.clone(), maskidx=maskidx, use_all_joints=use_all_joints)
                action_mask_full = self._build_action_mask(labels_full)
                if action_mask_full.sum().item() == 0:
                    continue

                clean_images = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
                with self._temporary_rng_seed(batch_seed):
                    modified_images, attack_aux = self.randomPatchTransform.apply_attack_batch(
                        images=pixel_values,
                        attack_texture=projection_texture,
                        mean=self.mean,
                        std=self.std,
                        attack_mode=attack_mode,
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
                        return_aux=True,
                    )
                projection_backend = str(attack_aux["projection_backend"])
                if len(visual_triplets) == 0:
                    visual_triplets = self._build_visual_triplets(
                        original_images=original_pixel_values,
                        projected_input_images=attack_aux["projected_inputs"],
                        adv_images=modified_images,
                        max_samples=viz_samples,
                        relight_images=pixel_values if val_lighting_enabled else None,
                    )

                clean_rollout_input_ids = input_ids.clone()
                adv_rollout_input_ids = input_ids.clone()
                sample_action_gap = 0.0
                sample_action_gap_joints = None
                sample_history_div = 0.0
                sample_history_div_legacy = 0.0
                action_terms = 0
                history_terms = 0
                history_terms_legacy = 0
                prev_adv_history_state = None
                final_output_adv = None
                need_history = True

                for _step_idx in range(eval_rollout):
                    output_clean: CausalLMOutputWithPast = self.vla(
                        input_ids=clean_rollout_input_ids,
                        attention_mask=attention_mask,
                        pixel_values=clean_images.to(torch.bfloat16),
                        labels=labels_masked,
                        output_hidden_states=need_history,
                        use_cache=False,
                    )
                    output_adv: CausalLMOutputWithPast = self.vla(
                        input_ids=adv_rollout_input_ids,
                        attention_mask=attention_mask,
                        pixel_values=modified_images.to(torch.bfloat16),
                        labels=labels_masked,
                        output_hidden_states=need_history,
                        use_cache=False,
                    )
                    final_output_adv = output_adv

                    _, metric_action_gap, step_per_joint_gap, adv_pred_tokens = self._compute_action_gap_losses(
                        adv_logits=output_adv.logits,
                        clean_logits=output_clean.logits,
                        labels_full=labels_full,
                        action_mask_full=action_mask_full,
                        maskidx=maskidx,
                        use_all_joints=use_all_joints,
                        gripper_weight=gripper_weight,
                    )
                    sample_action_gap += metric_action_gap.item()
                    action_terms += 1
                    sample_action_gap_joints = self._accumulate_joint_values(sample_action_gap_joints, step_per_joint_gap)

                    clean_pred_tokens = self._extract_pred_action_tokens_from_logits(output_clean.logits, labels_full)
                    history_div, _clean_history_state, adv_history_state = self._compute_clean_adv_history_divergence(
                        clean_hidden_states=output_clean.hidden_states[-1],
                        adv_hidden_states=output_adv.hidden_states[-1],
                        labels_full=labels_full,
                        action_mask_full=action_mask_full,
                    )
                    sample_history_div += history_div.item()
                    history_terms += 1
                    legacy_history_div = self._compute_legacy_history_divergence(adv_history_state, prev_adv_history_state)
                    if prev_adv_history_state is not None:
                        sample_history_div_legacy += legacy_history_div.item()
                        history_terms_legacy += 1
                    prev_adv_history_state = adv_history_state.detach()

                    clean_rollout_input_ids = self._update_rollout_inputs(
                        rollout_input_ids=clean_rollout_input_ids,
                        pred_action_tokens=clean_pred_tokens,
                        action_mask_full=action_mask_full,
                    )
                    adv_rollout_input_ids = self._update_rollout_inputs(
                        rollout_input_ids=adv_rollout_input_ids,
                        pred_action_tokens=adv_pred_tokens,
                        action_mask_full=action_mask_full,
                    )

                if final_output_adv is None:
                    continue

                val_mse_distance, val_uad = self.weighted_loss(final_output_adv.logits, labels_masked)

                avg_ce += final_output_adv.loss.item()
                avg_mse_distance += val_mse_distance.item()
                avg_uad += val_uad.item()
                avg_action_gap += sample_action_gap / float(max(1, action_terms))
                sample_avg_joint_gap = self._normalize_joint_values(
                    sample_action_gap_joints,
                    float(max(1, action_terms)),
                )
                avg_action_gap_joints = self._accumulate_joint_values(avg_action_gap_joints, sample_avg_joint_gap)
                avg_history_div += sample_history_div / float(max(1, history_terms))
                avg_history_div_legacy += sample_history_div_legacy / float(max(1, history_terms_legacy))
                avg_projection_alpha += float(attack_aux["projection_alpha_mean"])
                avg_projection_coverage += float(attack_aux["projection_coverage_ratio"])
                avg_projection_bottom += float(attack_aux["projection_bottom_ratio"])
                avg_projection_keystone += float(attack_aux["projection_keystone"])
                processed_batches += 1

        divisor = float(max(1, processed_batches))
        avg_ce /= divisor
        avg_mse_distance /= divisor
        avg_uad /= divisor
        avg_action_gap /= divisor
        avg_action_gap_joints = self._normalize_joint_values(avg_action_gap_joints, divisor)
        avg_history_div /= divisor
        avg_history_div_legacy /= divisor
        avg_projection_alpha /= divisor
        avg_projection_coverage /= divisor
        avg_projection_bottom /= divisor
        avg_projection_keystone /= divisor
        avg_rollout_score = lambda_action_gap * avg_action_gap + lambda_history * avg_history_div
        avg_rollout_score_legacy = lambda_action_gap * avg_action_gap + lambda_history * avg_history_div_legacy

        log_data = {
            "VAL_CE_loss": avg_ce,
            "VAL_MSE_Distance": avg_mse_distance,
            "VAL_UAD": avg_uad,
            "VAL_rollout_action_gap": avg_action_gap,
            "VAL_rollout_history_div": avg_history_div,
            "VAL_rollout_history_div_legacy": avg_history_div_legacy,
            "VAL_rollout_score": avg_rollout_score,
            "VAL_rollout_score_legacy": avg_rollout_score_legacy,
            "VAL_projection_alpha_mean": avg_projection_alpha,
            "VAL_projection_coverage_ratio": avg_projection_coverage,
            "VAL_projection_bottom_ratio": avg_projection_bottom,
            "VAL_projection_keystone": avg_projection_keystone,
            "attack_mode": str(attack_mode),
            "VAL_effective_lighting_enabled": int(bool(val_lighting_enabled)),
            "VAL_lighting_backend": str(lighting_backend),
            "ic_light_scope": getattr(self.lighting_augmentor, "scope", "n/a") if self.lighting_augmentor is not None else "n/a",
            "ic_light_bg_control": getattr(self.lighting_augmentor, "bg_control", "n/a")
            if self.lighting_augmentor is not None
            else "n/a",
            "projection_backend": projection_backend,
        }
        for joint_idx, joint_gap_value in enumerate(avg_action_gap_joints.detach().cpu().tolist()):
            log_data[f"VAL_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
        return log_data, val_iterator, visual_triplets, lighting_backend, projection_backend

    def _dump_val_images(self, modified_images, temp_save_dir, args=None, wandb_key="AdvImg"):
        if modified_images is None:
            return
        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
        os.makedirs(val_related_file_path, exist_ok=True)
        modified_images = self.randomPatchTransform.denormalize(
            modified_images[:, 0:3, :, :].detach().cpu(),
            mean=self.mean[0],
            std=self.std[0],
        )
        pil_imgs = []
        for o in range(modified_images.shape[0]):
            pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
            pil_imgs.append(pil_img)
        if args is not None and args.wandb_project != "false" and wandb is not None:
            wandb.log({wandb_key: [wandb.Image(pil_img) for pil_img in pil_imgs]})

    def modifiy_labels(
        self,
        labels,
        target_action={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8},
    ):
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            first_valid_index = (temp_label != -100).nonzero(as_tuple=True)[0].item()
            for key, value in target_action.items():
                if value != -100:
                    temp_label[int(first_valid_index + int(key))] = value
            newlabels.append(temp_label.unsqueeze(0))
        newlabels = torch.cat(newlabels, dim=0)
        return newlabels

    def filter_train(self, data):
        pixel_values = data["pixel_values"]
        labels = data["labels"].to(self.vla.device)
        attention_mask = data["attention_mask"].to(self.vla.device)
        input_ids = data["input_ids"].to(self.vla.device)

        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        one_index = []
        for idx in range(masked_labels.shape[0]):
            if masked_labels[idx, 6] == 31744:
                one_index.append(idx)
        if 1 < len(one_index) < 8:
            labels = labels[one_index, :]
            attention_mask = attention_mask[one_index, :]
            input_ids = input_ids[one_index, :]
            pixel_values = [pixel_values[i] for i in one_index]
        elif len(one_index) > 8:
            chosen = random.sample(one_index, k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        elif one_index is None:
            chosen = random.sample(range(labels.shape[0]), k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        return labels, attention_mask, input_ids, pixel_values

    def save_info(self, path):
        # Legacy metrics
        with open(os.path.join(path, "train_CE_loss.pkl"), "wb") as file:
            pickle.dump(self.train_CE_loss, file)
        with open(os.path.join(path, "train_MSE_distance_loss.pkl"), "wb") as file:
            pickle.dump(self.train_MSE_distance_loss, file)
        with open(os.path.join(path, "train_UAD.pkl"), "wb") as file:
            pickle.dump(self.train_UAD, file)
        with open(os.path.join(path, "val_CE_loss.pkl"), "wb") as file:
            pickle.dump(self.val_CE_loss, file)
        with open(os.path.join(path, "val_MSE_Distance.pkl"), "wb") as file:
            pickle.dump(self.val_MSE_Distance, file)
        with open(os.path.join(path, "val_UAD.pkl"), "wb") as file:
            pickle.dump(self.val_UAD, file)

        # Rollout-v2 metrics
        with open(os.path.join(path, "train_rollout_action_gap.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_action_gap, file)
        with open(os.path.join(path, "train_rollout_action_gap_joints.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_action_gap_joints, file)
        with open(os.path.join(path, "train_rollout_history_div.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_history_div, file)
        with open(os.path.join(path, "train_rollout_score.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_score, file)
        with open(os.path.join(path, "train_phase_id.pkl"), "wb") as file:
            pickle.dump(self.train_phase_id, file)
        with open(os.path.join(path, "val_rollout_action_gap.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_action_gap, file)
        with open(os.path.join(path, "val_rollout_action_gap_joints.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_action_gap_joints, file)
        with open(os.path.join(path, "val_rollout_history_div.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_history_div, file)
        with open(os.path.join(path, "val_rollout_history_div_legacy.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_history_div_legacy, file)
        with open(os.path.join(path, "val_rollout_score.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_score, file)
        with open(os.path.join(path, "val_rollout_score_legacy.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_score_legacy, file)

    def mask_labels(self, labels, maskidx, use_all_joints=False):
        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        if use_all_joints:
            maskidx = list(range(masked_labels.shape[1]))
        template_labels = torch.ones_like(masked_labels, device=masked_labels.device) * -100
        for idx in maskidx:
            template_labels[:, idx] = masked_labels[:, idx]
        labels[labels > 2] = template_labels.view(-1)
        return labels

    def weighted_loss(self, logits, labels):
        temp_label = labels[:, 1:].to(labels.device)
        action_mask = temp_label > 2
        temp_logits = logits[:, :, 31744:32000]
        action_logits = temp_logits[:, -temp_label.shape[-1] - 1 : -1, :]
        action_logits = action_logits[action_mask]
        if action_logits.numel() == 0:
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            return zero, zero

        reweigh = torch.arange(1, 257, device=logits.device, dtype=torch.float32) / 256.0
        temp_prob = F.softmax(action_logits, dim=-1)
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)

        hard_max_labels = temp_label[action_mask]
        hard_max_labels[hard_max_labels > 31872] = 31999
        hard_max_labels[hard_max_labels <= 31872] = 31744
        hard_max_labels[hard_max_labels == 31999] = 1 / 256
        hard_max_labels[hard_max_labels == 31744] = 1

        UAD = self.cal_UAD(action_logits.argmax(dim=-1) + 31744, temp_label[action_mask])
        distance_loss = F.mse_loss(5 * reweighted_prob.contiguous(), 5 * hard_max_labels.float().contiguous())
        return distance_loss, UAD

    def cal_UAD(self, pred, gt):
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(gt.clone().detach().cpu().numpy()),
            device=self.vla.device,
            dtype=torch.float32,
        )
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(pred.clone().detach().cpu().numpy()),
            device=self.vla.device,
            dtype=torch.float32,
        )
        max_distance = torch.where(
            continuous_actions_gt > 0,
            torch.abs(continuous_actions_gt - (-1)),
            torch.abs(continuous_actions_gt - 1),
        )
        distance = torch.abs(continuous_actions_pred - continuous_actions_gt)
        UAD = (distance / max_distance).mean()
        return UAD
