import csv
import json
import os
import pickle
import random
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import transformers
from PIL import Image
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from white_patch.UADA_rollout import IGNORE_INDEX, OpenVLAAttacker
except Exception:
    from UADA_rollout import IGNORE_INDEX, OpenVLAAttacker

try:
    import wandb
except Exception:
    wandb = None


class OpenVLAOnlineEnvAttacker(OpenVLAAttacker):
    def _reset_metric_buffers(self):
        super()._reset_metric_buffers()
        self.train_rollout_history_div_legacy = []
        self.train_rollout_objective_score = []
        self.val_rollout_objective_score = []
        self.train_online_done_rate = []
        self.train_online_episode_len = []
        self.val_online_done_rate = []
        self.val_online_episode_len = []
        self._recent_train_oom = False

    def _probe_metrics_path(self):
        return os.path.join(self.save_dir, "probe_metrics.csv")

    def _append_probe_metrics_row(self, row):
        metrics_path = self._probe_metrics_path()
        file_exists = os.path.exists(metrics_path)
        fieldnames = [
            "iter_idx",
            "phase_id",
            "split",
            "probe_variant",
            "rollout_steps_used",
            "horizon_type",
            "action_gap",
            "history1",
            "history2",
            "ce",
            "ce_objective",
            "rollout_score",
            "objective_score",
            "episode_len",
            "done_rate",
            "action_gap_joint_0",
            "action_gap_joint_1",
            "action_gap_joint_2",
            "projection_alpha_mean",
            "projection_coverage_ratio",
            "projection_bottom_ratio",
            "projection_keystone",
        ]
        with open(metrics_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_probe_final_val(self, payload):
        with open(os.path.join(self.save_dir, "probe_final_val.json"), "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def _online_video_root(self):
        video_root = os.path.join(self.save_dir, "videos")
        os.makedirs(video_root, exist_ok=True)
        return video_root

    def _append_online_video_manifest(self, row):
        video_root = self._online_video_root()
        manifest_path = os.path.join(video_root, "video_manifest.csv")
        file_exists = os.path.exists(manifest_path)
        fieldnames = [
            "iter_idx",
            "phase_id",
            "split",
            "frame_source",
            "lighting_backend",
            "task_id",
            "task_description",
            "episode_len",
            "done",
            "video_path",
        ]
        with open(manifest_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_mp4(self, video_path, frames, fps):
        if len(frames) == 0:
            return False
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        first_rgb = np.array(frames[0].convert("RGB"), dtype=np.uint8)
        height, width = first_rgb.shape[0], first_rgb.shape[1]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(max(1, fps)), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for `{video_path}`.")
        try:
            for frame in frames:
                rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
                if rgb.shape[0] != height or rgb.shape[1] != width:
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
        return True

    def _should_dump_online_episode_video(self, episode):
        if episode is None:
            return False
        return len(episode.get("video_frames", [])) > 0

    def _dump_online_episode_video(self, episode, iter_idx, phase_id, split, frame_source, fps):
        if not self._should_dump_online_episode_video(episode):
            return False
        split_name = str(split).lower().strip()
        video_root = self._online_video_root()
        split_dir = os.path.join(video_root, f"{split_name}_iter_{int(iter_idx):06d}")
        os.makedirs(split_dir, exist_ok=True)

        frame_source_name = str(frame_source).lower().strip()
        video_path = os.path.join(split_dir, f"{frame_source_name}.mp4")
        wrote_video = self._write_mp4(
            video_path=video_path,
            frames=episode.get("video_frames", []),
            fps=int(max(1, fps)),
        )
        if not wrote_video:
            return False

        self._append_online_video_manifest(
            {
                "iter_idx": int(iter_idx),
                "phase_id": int(phase_id),
                "split": split_name,
                "frame_source": frame_source_name,
                "lighting_backend": str(episode.get("lighting_backend", "unknown")),
                "task_id": int(episode.get("task_id", -1)),
                "task_description": str(episode.get("task_description", "")),
                "episode_len": int(episode.get("episode_len", 0)),
                "done": int(bool(episode.get("done", False))),
                "video_path": video_path,
            }
        )
        print(f"[OnlineVideo] saved {split_name} video to {video_path}")
        return True

    def online_attack_unconstrained(
        self,
        num_iter=5000,
        patch_size=[3, 50, 50],
        lr=1 / 255,
        accumulate_steps=1,
        maskidx=[],
        use_all_joints=True,
        gripper_weight=0.5,
        warmup=20,
        geometry=False,
        args=None,
        phase1_ratio=0.4,
        phase1_rollout=8,
        phase2_rollout=24,
        lambda_action_gap=1.0,
        lambda_history=0.5,
        lambda_history_legacy=0.0,
        lambda_ce=0.0,
        save_interval=100,
        eval_enabled=True,
        lighting_aug_enabled=False,
        lighting_model_id="stabilityai/sdxl-turbo",
        lighting_pool_size=8,
        lighting_refresh_interval=200,
        lighting_num_inference_steps=4,
        lighting_guidance_scale=0.0,
        lighting_blend_min=0.15,
        lighting_blend_max=0.5,
        lighting_apply_prob=1.0,
        lighting_seed=42,
        lighting_backend="ic_light",
        ic_light_repo="/home/yxx/IC-Light",
        ic_light_model_path="/home/yxx/IC-Light/models/iclight_sd15_fbc.safetensors",
        ic_light_scope="foreground",
        ic_light_bg_control="legacy_prompt",
        lighting_aug_train_only=False,
        phase1_disable_lighting=False,
        phase1_disable_projection_randomization=False,
        attack_mode="projection",
        projection_size=None,
        projection_alpha=0.55,
        projection_alpha_jitter=0.10,
        projection_soft_edge=1.2,
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
        projector_gamma=1.8,
        projector_gain=1.35,
        projector_channel_gain=(1.08, 1.04, 1.00),
        projector_ambient=0.08,
        projector_vignetting=0.08,
        projector_distance_falloff=0.10,
        projector_psf=False,
        viz_enabled=True,
        viz_policy="milestone",
        viz_samples=4,
        viz_save_best=True,
        viz_save_last=True,
        task_suite_name="auto",
        online_train_tasks_per_iter=1,
        online_train_episodes_per_task=1,
        online_val_episodes=8,
        num_steps_wait=10,
        max_env_steps="auto_by_suite",
        env_resolution=256,
        online_ce_mode="pseudo_clean",
        env_action_source="adv",
        env_seed=42,
        val_deterministic=False,
        val_seed=42,
        val_disable_lighting=False,
        probe_mode=False,
        probe_variant="",
        record_online_videos=False,
        record_online_videos_last_only=True,
        record_online_train_video=False,
        record_online_val_video=False,
        record_online_video_frame_source="projected_input",
        record_online_video_fps=10,
        auto_gpu_tune=False,
        gpu_tune_mode="stable",
        gpu_mem_low=0.82,
        gpu_mem_high=0.92,
        gpu_mem_hard_cap=0.95,
        gpu_util_low=70,
        gpu_tune_cooldown_iters=2,
        gpu_tune_min_rollout=8,
        gpu_tune_max_rollout=192,
        gpu_tune_min_tasks_per_iter=1,
        gpu_tune_max_tasks_per_iter=2,
    ):
        del env_action_source  # current implementation uses adv action for stepping by default
        self._reset_metric_buffers()
        self._set_online_env_seed(env_seed)

        attack_mode = str(attack_mode).lower()
        if projection_size is None:
            projection_size = patch_size
        projection_size = [int(x) for x in projection_size]

        projection_texture = torch.rand(projection_size, device=self.vla.device)
        projection_texture.requires_grad_(True)
        projection_texture.retain_grad()

        if self.optimizer != "adamW":
            raise ValueError("UADA_rollout_online_env currently supports optimizer='adamW' only.")

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
        phase1_rollout = max(1, int(phase1_rollout))
        phase2_rollout = max(1, int(phase2_rollout))
        use_all_joints = bool(use_all_joints)
        gripper_weight = float(gripper_weight)
        lambda_history_legacy = float(lambda_history_legacy)
        eval_enabled = bool(eval_enabled)
        self.lighting_aug_train_only = bool(lighting_aug_train_only)
        phase1_disable_lighting = bool(phase1_disable_lighting)
        phase1_disable_projection_randomization = bool(phase1_disable_projection_randomization)
        viz_enabled = bool(viz_enabled)
        viz_policy = str(viz_policy).lower()
        viz_samples = max(1, int(viz_samples))
        online_train_tasks_per_iter = max(1, int(online_train_tasks_per_iter))
        online_train_episodes_per_task = max(1, int(online_train_episodes_per_task))
        online_val_episodes = max(1, int(online_val_episodes))
        num_steps_wait = max(0, int(num_steps_wait))
        env_resolution = max(64, int(env_resolution))
        val_deterministic = bool(val_deterministic)
        val_seed = int(val_seed)
        val_disable_lighting = bool(val_disable_lighting)
        probe_mode = bool(probe_mode)
        probe_variant = str(probe_variant)
        record_online_videos = bool(record_online_videos)
        record_online_videos_last_only = bool(record_online_videos_last_only)
        record_online_train_video = bool(record_online_train_video)
        record_online_val_video = bool(record_online_val_video)
        record_online_video_frame_source = str(record_online_video_frame_source).lower().strip()
        record_online_video_fps = max(1, int(record_online_video_fps))
        auto_gpu_tune = bool(auto_gpu_tune)
        gpu_tune_mode = str(gpu_tune_mode).lower().strip()
        if gpu_tune_mode != "stable":
            raise ValueError("--gpu_tune_mode currently supports {'stable'} only.")
        gpu_mem_low = float(gpu_mem_low)
        gpu_mem_high = float(gpu_mem_high)
        gpu_mem_hard_cap = float(gpu_mem_hard_cap)
        gpu_util_low = float(gpu_util_low)
        gpu_tune_cooldown_iters = max(0, int(gpu_tune_cooldown_iters))
        gpu_tune_min_rollout = max(1, int(gpu_tune_min_rollout))
        gpu_tune_max_rollout = max(gpu_tune_min_rollout, int(gpu_tune_max_rollout))
        gpu_tune_min_tasks_per_iter = max(1, int(gpu_tune_min_tasks_per_iter))
        gpu_tune_max_tasks_per_iter = max(gpu_tune_min_tasks_per_iter, int(gpu_tune_max_tasks_per_iter))
        base_phase1_rollout = int(phase1_rollout)
        base_phase2_rollout = int(phase2_rollout)
        base_train_tasks_per_iter = int(online_train_tasks_per_iter)
        milestone_iters = self._build_milestone_iters(num_iter=num_iter, phase1_end_iter=phase1_end_iter)
        online_ce_mode = str(online_ce_mode).lower().strip()
        if online_ce_mode not in ("off", "pseudo_clean", "uada_inverse_pseudo_clean"):
            raise ValueError("--online_ce_mode must be one of {off, pseudo_clean, uada_inverse_pseudo_clean}")
        if online_ce_mode == "off":
            lambda_ce = 0.0

        if viz_enabled and not eval_enabled:
            print("[OnlineViz] `viz_enabled=true` but `eval_enabled=false`; visualization will be skipped.")

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

        benchmark_mod, get_libero_env, get_libero_image, get_libero_dummy_action = self._require_libero_modules()
        resolved_suite_name = self._resolve_task_suite_name(dataset=getattr(args, "dataset", "libero_spatial"), task_suite_name=task_suite_name)
        max_env_steps = self._resolve_max_env_steps(max_env_steps, resolved_suite_name)
        task_suite = benchmark_mod.get_benchmark_dict()[resolved_suite_name]()
        unnorm_key = self._resolve_unnorm_key(resolved_suite_name)
        action_stats = self.vla.get_action_stats(unnorm_key)
        action_dim = len(action_stats["q01"])
        print(
            "[OnlineEnv] initialized "
            f"(suite={resolved_suite_name}, action_dim={action_dim}, max_env_steps={max_env_steps}, ce_mode={online_ce_mode})"
        )
        gpu_tuner_state = self._init_gpu_tuner_state(
            enabled=auto_gpu_tune,
            mode=gpu_tune_mode,
            mem_low=gpu_mem_low,
            mem_high=gpu_mem_high,
            mem_hard_cap=gpu_mem_hard_cap,
            util_low=gpu_util_low,
            cooldown_iters=gpu_tune_cooldown_iters,
            min_rollout=gpu_tune_min_rollout,
            max_rollout=gpu_tune_max_rollout,
            min_tasks_per_iter=gpu_tune_min_tasks_per_iter,
            max_tasks_per_iter=gpu_tune_max_tasks_per_iter,
            base_phase1_rollout=base_phase1_rollout,
            base_phase2_rollout=base_phase2_rollout,
            base_train_tasks_per_iter=base_train_tasks_per_iter,
        )
        if auto_gpu_tune:
            print(
                "[OnlineEnv] auto_gpu_tune enabled "
                f"(device={gpu_tuner_state['device_index']}, mem_low={gpu_mem_low}, mem_high={gpu_mem_high}, mem_hard_cap={gpu_mem_hard_cap})"
            )

        optimizer.zero_grad()
        for i in tqdm(range(num_iter)):
            phase_id = 1 if i < phase1_end_iter else 2
            base_rollout_steps = base_phase1_rollout if phase_id == 1 else base_phase2_rollout
            tune_level_for_iter = int(gpu_tuner_state["level"]) if auto_gpu_tune else 2
            if auto_gpu_tune:
                eff_phase1_rollout, eff_phase2_rollout, eff_train_tasks_per_iter = self._resolve_effective_train_budget(
                    tune_level=tune_level_for_iter,
                    gpu_tuner_state=gpu_tuner_state,
                )
            else:
                eff_phase1_rollout = base_phase1_rollout
                eff_phase2_rollout = base_phase2_rollout
                eff_train_tasks_per_iter = base_train_tasks_per_iter

            rollout_steps = eff_phase1_rollout if phase_id == 1 else eff_phase2_rollout
            train_disable_lighting = bool((phase_id == 1) and phase1_disable_lighting)
            train_projection_randomization_enabled = not bool(
                (phase_id == 1) and phase1_disable_projection_randomization
            )
            train_effective_lighting_enabled = int(
                self._is_lighting_enabled_for_split("train") and (not train_disable_lighting)
            )
            train_effective_projection_randomization_enabled = int(
                (str(attack_mode).lower() == "projection") and train_projection_randomization_enabled
            )
            train_task_ids = self._sample_task_ids(task_suite=task_suite, num_tasks=eff_train_tasks_per_iter)
            should_record_train_video = bool(
                record_online_videos
                and record_online_train_video
                and ((not record_online_videos_last_only) or (i == num_iter - 1))
            )
            train_video_episode = None

            train_action_gap = 0.0
            train_history_div = 0.0
            train_history_div_legacy = 0.0
            train_ce = 0.0
            train_ce_objective = 0.0
            train_proj_alpha = 0.0
            train_proj_coverage = 0.0
            train_proj_bottom = 0.0
            train_proj_keystone = 0.0
            train_action_terms = 0
            train_history_terms = 0
            train_episodes_done = 0.0
            train_episode_len = 0.0
            train_per_joint_gap = None
            train_episode_count = 0

            for task_id in train_task_ids:
                task = task_suite.get_task(task_id)
                init_states = task_suite.get_task_init_states(task_id)
                env, task_description = get_libero_env(task, "openvla", resolution=env_resolution)
                try:
                    for episode_idx in range(online_train_episodes_per_task):
                        init_state = self._sample_init_state(init_states, i, episode_idx)
                        episode = self._run_online_episode(
                            env=env,
                            get_libero_image=get_libero_image,
                            get_libero_dummy_action=get_libero_dummy_action,
                            init_state=init_state,
                            task_description=task_description,
                            projection_texture=projection_texture,
                            rollout_steps=rollout_steps,
                            max_env_steps=max_env_steps,
                            num_steps_wait=num_steps_wait,
                            split="train",
                            global_iter=i,
                            geometry=geometry,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            action_stats=action_stats,
                            lambda_action_gap=lambda_action_gap,
                            lambda_history=lambda_history,
                            lambda_history_legacy=lambda_history_legacy,
                            lambda_ce=lambda_ce,
                            online_ce_mode=online_ce_mode,
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
                            projection_randomization_enabled=train_projection_randomization_enabled,
                            need_backward=True,
                            capture_visual=False,
                            deterministic_episode_seed=None,
                            disable_lighting=train_disable_lighting,
                            task_id=task_id,
                            record_video_frames=bool(should_record_train_video and (train_video_episode is None)),
                            video_frame_source=record_online_video_frame_source,
                        )
                        if episode is None:
                            continue

                        train_episode_count += 1
                        train_action_gap += episode["action_gap"]
                        train_history_div += episode["history_div"]
                        train_history_div_legacy += episode["history_div_legacy"]
                        train_ce += episode["ce_value"]
                        train_ce_objective += episode["ce_objective_value"]
                        train_proj_alpha += episode["projection_alpha"]
                        train_proj_coverage += episode["projection_coverage"]
                        train_proj_bottom += episode["projection_bottom"]
                        train_proj_keystone += episode["projection_keystone"]
                        train_action_terms += max(1, episode["action_terms"])
                        train_history_terms += max(0, episode["history_terms"])
                        train_episodes_done += float(episode["done"])
                        train_episode_len += float(episode["episode_len"])
                        train_per_joint_gap = self._accumulate_joint_values(train_per_joint_gap, episode["per_joint_gap"])
                        if should_record_train_video and (train_video_episode is None) and self._should_dump_online_episode_video(episode):
                            train_video_episode = episode
                finally:
                    if hasattr(env, "close"):
                        env.close()

            if train_episode_count <= 0:
                print(f"[OnlineEnv] iter {i}: no valid episodes processed, skipping optimizer step.")
                continue

            if projection_texture.grad is not None:
                grad_scale = float(max(1, train_episode_count * accumulate_steps))
                projection_texture.grad.div_(grad_scale)

            log_patch_grad = 0.0
            if projection_texture.grad is not None:
                log_patch_grad = projection_texture.grad.detach().abs().mean().item()

            optimizer_step = ((i + 1) % accumulate_steps == 0) or ((i + 1) == num_iter)
            if optimizer_step:
                optimizer.step()
                projection_texture.data = projection_texture.data.clamp(0, 1)
                optimizer.zero_grad()
                scheduler.step()

            avg_action_gap = train_action_gap / float(max(1, train_episode_count))
            avg_history_div = train_history_div / float(max(1, train_episode_count))
            avg_history_div_legacy = train_history_div_legacy / float(max(1, train_episode_count))
            avg_ce = train_ce / float(max(1, train_episode_count))
            avg_ce_objective = train_ce_objective / float(max(1, train_episode_count))
            avg_proj_alpha = train_proj_alpha / float(max(1, train_episode_count))
            avg_proj_coverage = train_proj_coverage / float(max(1, train_episode_count))
            avg_proj_bottom = train_proj_bottom / float(max(1, train_episode_count))
            avg_proj_keystone = train_proj_keystone / float(max(1, train_episode_count))
            avg_done_rate = train_episodes_done / float(max(1, train_episode_count))
            avg_ep_len = train_episode_len / float(max(1, train_episode_count))
            avg_per_joint_gap = self._normalize_joint_values(train_per_joint_gap, float(max(1, train_episode_count)))
            rollout_score = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
            )
            objective_score = rollout_score - (lambda_ce * avg_ce_objective)
            gpu_stats = self._collect_gpu_runtime_stats(device_index=gpu_tuner_state["device_index"])
            autotune_action = "hold(disabled)"
            if auto_gpu_tune:
                autotune_action = self._update_gpu_tuner_state(
                    gpu_tuner_state=gpu_tuner_state,
                    mem_ratio=gpu_stats["mem_ratio"],
                    gpu_util=gpu_stats["gpu_util"],
                    recent_oom=bool(self._recent_train_oom),
                )
            self._recent_train_oom = False

            self.train_rollout_action_gap.append(avg_action_gap)
            self.train_rollout_action_gap_joints.append(avg_per_joint_gap.detach().cpu().tolist())
            self.train_rollout_history_div.append(avg_history_div)
            self.train_rollout_history_div_legacy.append(avg_history_div_legacy)
            self.train_rollout_score.append(rollout_score)
            self.train_rollout_objective_score.append(objective_score)
            self.train_phase_id.append(phase_id)
            self.train_online_done_rate.append(avg_done_rate)
            self.train_online_episode_len.append(avg_ep_len)
            self.loss_buffer.append(objective_score if probe_mode else rollout_score)

            train_logdata = {
                "TRAIN_online_rollout_action_gap": avg_action_gap,
                "TRAIN_online_rollout_history_div": avg_history_div,
                "TRAIN_online_rollout_history_div_legacy": avg_history_div_legacy,
                "TRAIN_online_rollout_score": rollout_score,
                "TRAIN_online_objective_score": objective_score,
                "TRAIN_online_done_rate": avg_done_rate,
                "TRAIN_online_episode_len": avg_ep_len,
                "TRAIN_online_ce": avg_ce,
                "TRAIN_online_ce_objective": avg_ce_objective,
                "TRAIN_patch_gradient": log_patch_grad,
                "TRAIN_LR": optimizer.param_groups[0]["lr"],
                "TRAIN_projection_alpha_mean": avg_proj_alpha,
                "TRAIN_projection_coverage_ratio": avg_proj_coverage,
                "TRAIN_projection_bottom_ratio": avg_proj_bottom,
                "TRAIN_projection_keystone": avg_proj_keystone,
                "phase_id": phase_id,
                "attack_mode": attack_mode,
                "online_task_suite": resolved_suite_name,
                "TRAIN_effective_lighting_enabled": int(train_effective_lighting_enabled),
                "TRAIN_effective_projection_randomization_enabled": int(
                    train_effective_projection_randomization_enabled
                ),
                "TRAIN_gpu_util": float(gpu_stats["gpu_util"]) if gpu_stats["gpu_util"] is not None else -1.0,
                "TRAIN_gpu_mem_used_gb": float(gpu_stats["mem_used_gb"]) if gpu_stats["mem_used_gb"] is not None else -1.0,
                "TRAIN_gpu_mem_ratio": float(gpu_stats["mem_ratio"]) if gpu_stats["mem_ratio"] is not None else -1.0,
                "TRAIN_autotune_level": int(tune_level_for_iter),
                "TRAIN_eff_phase1_rollout": int(eff_phase1_rollout),
                "TRAIN_eff_phase2_rollout": int(eff_phase2_rollout),
                "TRAIN_eff_online_train_tasks_per_iter": int(eff_train_tasks_per_iter),
                "TRAIN_autotune_action": str(autotune_action),
            }
            for joint_idx, joint_gap_value in enumerate(avg_per_joint_gap.detach().cpu().tolist()):
                train_logdata[f"TRAIN_online_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
            if self.lighting_augmentor is not None:
                train_logdata["TRAIN_lighting_pool_size"] = self.lighting_augmentor.current_pool_size
                train_logdata["TRAIN_lighting_backend"] = self.lighting_augmentor.backend
                train_logdata["ic_light_scope"] = getattr(self.lighting_augmentor, "scope", "n/a")
                train_logdata["ic_light_bg_control"] = getattr(self.lighting_augmentor, "bg_control", "n/a")
            if probe_mode:
                self._append_probe_metrics_row(
                    {
                        "iter_idx": int(i),
                        "phase_id": int(phase_id),
                        "split": "train",
                        "probe_variant": probe_variant,
                        "rollout_steps_used": int(rollout_steps),
                        "horizon_type": "phase_rollout",
                        "action_gap": float(avg_action_gap),
                        "history1": float(avg_history_div),
                        "history2": float(avg_history_div_legacy),
                        "ce": float(avg_ce),
                        "ce_objective": float(avg_ce_objective),
                        "rollout_score": float(rollout_score),
                        "objective_score": float(objective_score),
                        "episode_len": float(avg_ep_len),
                        "action_gap_joint_0": float(avg_per_joint_gap[0].item()) if avg_per_joint_gap.numel() > 0 else "",
                        "action_gap_joint_1": float(avg_per_joint_gap[1].item()) if avg_per_joint_gap.numel() > 1 else "",
                        "action_gap_joint_2": float(avg_per_joint_gap[2].item()) if avg_per_joint_gap.numel() > 2 else "",
                        "projection_alpha_mean": float(avg_proj_alpha),
                        "projection_coverage_ratio": float(avg_proj_coverage),
                        "projection_bottom_ratio": float(avg_proj_bottom),
                        "projection_keystone": float(avg_proj_keystone),
                    }
                )

            if args is not None and args.wandb_project != "false" and wandb is not None:
                wandb.log(train_logdata, step=i)

            if should_record_train_video and (train_video_episode is not None):
                self._dump_online_episode_video(
                    episode=train_video_episode,
                    iter_idx=i,
                    phase_id=phase_id,
                    split="train",
                    frame_source=record_online_video_frame_source,
                    fps=record_online_video_fps,
                )

            eval_due_to_interval = ((i + 1) % save_interval == 0) or (i == num_iter - 1)
            eval_due_to_milestone = (not probe_mode) and viz_enabled and (viz_policy == "milestone") and (i in milestone_iters)
            if eval_enabled and (eval_due_to_interval or eval_due_to_milestone):
                if i % (save_interval * 2) == 0:
                    self.plot_loss()

                should_record_val_video = bool(
                    record_online_videos
                    and record_online_val_video
                    and ((not record_online_videos_last_only) or (i == num_iter - 1))
                )
                eval_rollout_steps = int(max_env_steps) if probe_mode else int(base_rollout_steps)
                val_stats, visual_frames, val_video_episode = self._evaluate_online_rollout(
                    task_suite=task_suite,
                    get_libero_env=get_libero_env,
                    get_libero_image=get_libero_image,
                    get_libero_dummy_action=get_libero_dummy_action,
                    projection_texture=projection_texture,
                    rollout_steps=eval_rollout_steps,
                    max_env_steps=max_env_steps,
                    num_steps_wait=num_steps_wait,
                    online_val_episodes=online_val_episodes,
                    global_iter=i,
                    geometry=geometry,
                    maskidx=maskidx,
                    use_all_joints=use_all_joints,
                    gripper_weight=gripper_weight,
                    action_stats=action_stats,
                    lambda_action_gap=lambda_action_gap,
                    lambda_history=lambda_history,
                    lambda_history_legacy=lambda_history_legacy,
                    lambda_ce=lambda_ce,
                    online_ce_mode=online_ce_mode,
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
                    env_resolution=env_resolution,
                    viz_samples=viz_samples,
                    val_deterministic=val_deterministic,
                    val_seed=val_seed,
                    val_disable_lighting=val_disable_lighting,
                    probe_mode=probe_mode,
                    record_video_frames=should_record_val_video,
                    video_frame_source=record_online_video_frame_source,
                )
                val_stats["VAL_effective_lighting_enabled"] = int(
                    self._is_lighting_enabled_for_split("val") and (not bool(val_disable_lighting))
                )
                val_stats["VAL_effective_projection_randomization_enabled"] = int(
                    str(attack_mode).lower() == "projection"
                )
                val_stats["VAL_autotune_enabled"] = int(auto_gpu_tune)
                val_stats["VAL_autotune_level_snapshot"] = int(tune_level_for_iter)

                self.val_rollout_action_gap.append(val_stats["VAL_online_rollout_action_gap"])
                self.val_rollout_action_gap_joints.append(
                    self._extract_joint_metric_list(val_stats, prefix="VAL_online_rollout_action_gap_joint_")
                )
                self.val_rollout_history_div.append(val_stats["VAL_online_rollout_history_div"])
                self.val_rollout_history_div_legacy.append(val_stats["VAL_online_rollout_history_div_legacy"])
                self.val_rollout_score.append(val_stats["VAL_online_rollout_score"])
                self.val_rollout_objective_score.append(val_stats["VAL_online_objective_score"])
                self.val_rollout_score_legacy.append(val_stats["VAL_online_rollout_score_legacy"])
                self.val_online_done_rate.append(val_stats["VAL_online_done_rate"])
                self.val_online_episode_len.append(val_stats["VAL_online_episode_len"])

                if args is not None and args.wandb_project != "false" and wandb is not None:
                    wandb.log(val_stats, step=i)

                if should_record_val_video and (val_video_episode is not None):
                    self._dump_online_episode_video(
                        episode=val_video_episode,
                        iter_idx=i,
                        phase_id=phase_id,
                        split="val",
                        frame_source=record_online_video_frame_source,
                        fps=record_online_video_fps,
                    )
                if probe_mode:
                    self._append_probe_metrics_row(
                        {
                            "iter_idx": int(i),
                            "phase_id": int(phase_id),
                            "split": "val",
                            "probe_variant": probe_variant,
                            "rollout_steps_used": int(eval_rollout_steps),
                            "horizon_type": "full_horizon",
                            "action_gap": float(val_stats["VAL_online_rollout_action_gap"]),
                            "history1": float(val_stats["VAL_online_rollout_history_div"]),
                            "history2": float(val_stats["VAL_online_rollout_history_div_legacy"]),
                            "ce": float(val_stats["VAL_online_ce"]),
                            "ce_objective": float(val_stats["VAL_online_ce_objective"]),
                            "rollout_score": float(val_stats["VAL_online_rollout_score"]),
                            "objective_score": float(val_stats["VAL_online_objective_score"]),
                            "episode_len": float(val_stats["VAL_online_episode_len"]),
                            "done_rate": float(val_stats["VAL_online_done_rate"]),
                            "action_gap_joint_0": float(val_stats.get("VAL_online_rollout_action_gap_joint_0", "")) if "VAL_online_rollout_action_gap_joint_0" in val_stats else "",
                            "action_gap_joint_1": float(val_stats.get("VAL_online_rollout_action_gap_joint_1", "")) if "VAL_online_rollout_action_gap_joint_1" in val_stats else "",
                            "action_gap_joint_2": float(val_stats.get("VAL_online_rollout_action_gap_joint_2", "")) if "VAL_online_rollout_action_gap_joint_2" in val_stats else "",
                            "projection_alpha_mean": float(val_stats["VAL_projection_alpha_mean"]),
                            "projection_coverage_ratio": float(val_stats["VAL_projection_coverage_ratio"]),
                            "projection_bottom_ratio": float(val_stats["VAL_projection_bottom_ratio"]),
                            "projection_keystone": float(val_stats["VAL_projection_keystone"]),
                        }
                    )
                    if i == (num_iter - 1):
                        self._write_probe_final_val(
                            {
                                "variant": probe_variant,
                                "run_dir": self.save_dir,
                                "iter": int(num_iter),
                                "iter_idx": int(i),
                                "phase_id": int(phase_id),
                                "lambda_action_gap": float(lambda_action_gap),
                                "lambda_history": float(lambda_history),
                                "lambda_history_legacy": float(lambda_history_legacy),
                                "lambda_ce": float(lambda_ce),
                                "online_ce_mode": str(online_ce_mode),
                                "final_val_done_rate": float(val_stats["VAL_online_done_rate"]),
                                "final_val_episode_len": float(val_stats["VAL_online_episode_len"]),
                                "final_val_action_gap": float(val_stats["VAL_online_rollout_action_gap"]),
                                "final_val_history1": float(val_stats["VAL_online_rollout_history_div"]),
                                "final_val_history2": float(val_stats["VAL_online_rollout_history_div_legacy"]),
                                "final_val_ce": float(val_stats["VAL_online_ce"]),
                                "final_val_ce_objective": float(val_stats["VAL_online_ce_objective"]),
                                "final_val_rollout_score": float(val_stats["VAL_online_rollout_score"]),
                                "final_val_objective_score": float(val_stats["VAL_online_objective_score"]),
                            }
                        )

                improved = val_stats["VAL_online_rollout_score"] > self.best_rollout_score
                if improved:
                    self.best_rollout_score = val_stats["VAL_online_rollout_score"]
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
                        self._dump_online_visual_frames(
                            visual_frames=visual_frames,
                            iter_idx=i,
                            phase_id=phase_id,
                            is_best=improved,
                            val_rollout_score=val_stats["VAL_online_rollout_score"],
                            reason=vis_reason,
                            attack_mode=attack_mode,
                            args=args,
                        )

                self.save_online_info(self.save_dir)
                torch.cuda.empty_cache()
            elif (not eval_enabled) and (i == num_iter - 1):
                temp_save_dir = os.path.join(self.save_dir, "last")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                self.save_online_info(self.save_dir)
                torch.cuda.empty_cache()

    def _evaluate_online_rollout(
        self,
        task_suite,
        get_libero_env,
        get_libero_image,
        get_libero_dummy_action,
        projection_texture,
        rollout_steps,
        max_env_steps,
        num_steps_wait,
        online_val_episodes,
        global_iter,
        geometry,
        maskidx,
        use_all_joints,
        gripper_weight,
        action_stats,
        lambda_action_gap,
        lambda_history,
        lambda_history_legacy,
        lambda_ce,
        online_ce_mode,
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
        env_resolution,
        viz_samples,
        val_deterministic,
        val_seed,
        val_disable_lighting,
        probe_mode,
        record_video_frames,
        video_frame_source,
    ):
        total_action_gap = 0.0
        total_history_div = 0.0
        total_history_div_legacy = 0.0
        total_ce = 0.0
        total_ce_objective = 0.0
        total_done = 0.0
        total_ep_len = 0.0
        total_proj_alpha = 0.0
        total_proj_cov = 0.0
        total_proj_bottom = 0.0
        total_proj_keystone = 0.0
        total_action_terms = 0
        total_history_terms = 0
        total_per_joint = None
        visual_frames = []
        video_episode = None

        n_tasks = task_suite.n_tasks
        with torch.no_grad():
            for ep_idx in range(int(online_val_episodes)):
                task_id = ep_idx % max(1, n_tasks)
                task = task_suite.get_task(task_id)
                init_states = task_suite.get_task_init_states(task_id)
                episode_seed = (int(val_seed) + int(ep_idx)) if val_deterministic else None
                init_state = self._sample_init_state(
                    init_states=init_states,
                    iter_idx=global_iter,
                    local_idx=ep_idx,
                    deterministic_seed=episode_seed,
                )
                env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
                try:
                    episode = self._run_online_episode(
                        env=env,
                        get_libero_image=get_libero_image,
                        get_libero_dummy_action=get_libero_dummy_action,
                        init_state=init_state,
                        task_description=task_description,
                        projection_texture=projection_texture,
                        rollout_steps=rollout_steps,
                        max_env_steps=max_env_steps,
                        num_steps_wait=num_steps_wait,
                        split="val",
                        global_iter=(global_iter * max(1, int(online_val_episodes))) + ep_idx,
                        geometry=geometry,
                        maskidx=maskidx,
                        use_all_joints=use_all_joints,
                        gripper_weight=gripper_weight,
                        action_stats=action_stats,
                        lambda_action_gap=lambda_action_gap,
                        lambda_history=lambda_history,
                        lambda_history_legacy=lambda_history_legacy,
                        lambda_ce=lambda_ce,
                        online_ce_mode=online_ce_mode,
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
                        projection_randomization_enabled=True,
                        need_backward=False,
                        capture_visual=(len(visual_frames) == 0),
                        deterministic_episode_seed=episode_seed,
                        disable_lighting=bool(val_disable_lighting),
                        task_id=task_id,
                        record_video_frames=bool(record_video_frames and (video_episode is None)),
                        video_frame_source=video_frame_source,
                    )
                    if episode is None:
                        continue

                    total_action_gap += episode["action_gap"]
                    total_history_div += episode["history_div"]
                    total_history_div_legacy += episode["history_div_legacy"]
                    total_ce += episode["ce_value"]
                    total_ce_objective += episode["ce_objective_value"]
                    total_done += float(episode["done"])
                    total_ep_len += float(episode["episode_len"])
                    total_proj_alpha += episode["projection_alpha"]
                    total_proj_cov += episode["projection_coverage"]
                    total_proj_bottom += episode["projection_bottom"]
                    total_proj_keystone += episode["projection_keystone"]
                    total_action_terms += max(1, episode["action_terms"])
                    total_history_terms += max(0, episode["history_terms"])
                    total_per_joint = self._accumulate_joint_values(total_per_joint, episode["per_joint_gap"])
                    if len(visual_frames) == 0 and len(episode["visual_frames"]) > 0:
                        visual_frames = episode["visual_frames"][: max(1, int(viz_samples))]
                    if bool(record_video_frames) and (video_episode is None) and self._should_dump_online_episode_video(episode):
                        video_episode = episode
                finally:
                    if hasattr(env, "close"):
                        env.close()

        divisor = float(max(1, int(online_val_episodes)))
        avg_action_gap = total_action_gap / divisor
        avg_history_div = total_history_div / divisor
        avg_history_div_legacy = total_history_div_legacy / divisor
        avg_ce = total_ce / divisor
        avg_ce_objective = total_ce_objective / divisor
        avg_done = total_done / divisor
        avg_ep_len = total_ep_len / divisor
        avg_proj_alpha = total_proj_alpha / divisor
        avg_proj_cov = total_proj_cov / divisor
        avg_proj_bottom = total_proj_bottom / divisor
        avg_proj_keystone = total_proj_keystone / divisor
        avg_per_joint = self._normalize_joint_values(total_per_joint, divisor)
        if probe_mode:
            avg_rollout_score = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
            )
            avg_rollout_score_legacy = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history_legacy * avg_history_div_legacy)
            )
        else:
            avg_rollout_score = (lambda_action_gap * avg_action_gap) + (lambda_history * avg_history_div)
            avg_rollout_score_legacy = (lambda_action_gap * avg_action_gap) + (lambda_history * avg_history_div_legacy)
        avg_objective_score = avg_rollout_score - (lambda_ce * avg_ce_objective)

        stats = {
            "VAL_online_rollout_action_gap": avg_action_gap,
            "VAL_online_rollout_history_div": avg_history_div,
            "VAL_online_rollout_history_div_legacy": avg_history_div_legacy,
            "VAL_online_rollout_score": avg_rollout_score,
            "VAL_online_rollout_score_legacy": avg_rollout_score_legacy,
            "VAL_online_objective_score": avg_objective_score,
            "VAL_online_done_rate": avg_done,
            "VAL_online_episode_len": avg_ep_len,
            "VAL_online_ce": avg_ce,
            "VAL_online_ce_objective": avg_ce_objective,
            "VAL_projection_alpha_mean": avg_proj_alpha,
            "VAL_projection_coverage_ratio": avg_proj_cov,
            "VAL_projection_bottom_ratio": avg_proj_bottom,
            "VAL_projection_keystone": avg_proj_keystone,
            "VAL_effective_lighting_enabled": int(self._is_lighting_enabled_for_split("val") and (not bool(val_disable_lighting))),
            "VAL_lighting_backend": str(getattr(self.lighting_augmentor, "backend", "disabled")),
            "ic_light_scope": getattr(self.lighting_augmentor, "scope", "n/a") if self.lighting_augmentor is not None else "n/a",
            "ic_light_bg_control": getattr(self.lighting_augmentor, "bg_control", "n/a")
            if self.lighting_augmentor is not None
            else "n/a",
        }
        for joint_idx, joint_gap_value in enumerate(avg_per_joint.detach().cpu().tolist()):
            stats[f"VAL_online_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
        return stats, visual_frames, video_episode

    def _run_online_episode(
        self,
        env,
        get_libero_image,
        get_libero_dummy_action,
        init_state,
        task_description,
        projection_texture,
        rollout_steps,
        max_env_steps,
        num_steps_wait,
        split,
        global_iter,
        geometry,
        maskidx,
        use_all_joints,
        gripper_weight,
        action_stats,
        lambda_action_gap,
        lambda_history,
        lambda_history_legacy,
        lambda_ce,
        online_ce_mode,
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
        projection_randomization_enabled,
        need_backward,
        capture_visual,
        deterministic_episode_seed=None,
        disable_lighting=False,
        task_id=None,
        record_video_frames=False,
        video_frame_source="projected_input",
    ):
        action_dim = len(action_stats["q01"])
        video_frame_source = str(video_frame_source).lower().strip()
        if video_frame_source not in ("next_obs", "orig", "projected_input", "adv"):
            video_frame_source = "projected_input"
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(max(0, int(num_steps_wait))):
            obs, _, done_wait, _ = env.step(get_libero_dummy_action("openvla"))
            if done_wait:
                break

        labels_full, attention_mask, input_ids = self._build_prompt_tensors(
            task_description=task_description,
            action_dim=action_dim,
        )
        labels_masked = self.mask_labels(labels_full.clone(), maskidx=maskidx, use_all_joints=use_all_joints)
        action_mask_full = self._build_action_mask(labels_full)
        if action_mask_full.sum().item() == 0:
            return None

        clean_rollout_input_ids = input_ids.clone()
        adv_rollout_input_ids = input_ids.clone()
        total_loss = torch.zeros((), device=self.vla.device, dtype=torch.float32)
        episode_weight_sum = 0.0
        grad_before_episode = None
        if need_backward and (projection_texture.grad is not None):
            grad_before_episode = projection_texture.grad.detach().clone()
        total_action_gap = 0.0
        total_history_div = 0.0
        total_history_div_legacy = 0.0
        total_ce = 0.0
        total_ce_objective = 0.0
        total_action_terms = 0
        total_history_terms = 0
        total_history_terms_legacy = 0
        total_per_joint = None
        total_proj_alpha = 0.0
        total_proj_cov = 0.0
        total_proj_bottom = 0.0
        total_proj_keystone = 0.0
        last_done = False
        visual_frames = []
        video_frames = []

        max_steps = int(min(int(rollout_steps), int(max_env_steps)))
        need_history = (max_steps > 0) and ((not need_backward) or (lambda_history > 0) or (lambda_history_legacy > 0))
        prev_adv_history_state = None
        episode_lighting_map_idx = None
        episode_lighting_backend = "disabled"
        episode_effective_lighting_enabled = 0
        if (not bool(disable_lighting)) and (self.lighting_augmentor is not None):
            if deterministic_episode_seed is not None:
                episode_lighting_map_idx = int(deterministic_episode_seed)
            else:
                episode_lighting_map_idx = random.randint(0, (2**31) - 1)
        for step_idx in range(max_steps):
            current_image = get_libero_image(obs, (224, 224))
            pixel_values = [current_image]
            step_seed = None
            if deterministic_episode_seed is not None:
                step_seed = int(deterministic_episode_seed) + (int(step_idx) * 10007)
            try:
                with self._temporary_rng_seed(step_seed):
                    if not bool(disable_lighting):
                        pixel_values = self._apply_lighting_augmentation(
                            pixel_values=pixel_values,
                            iteration_idx=(int(global_iter) * max(1, int(max_steps))) + int(step_idx),
                            split=split,
                            fixed_map_idx=episode_lighting_map_idx,
                        )
                        episode_effective_lighting_enabled = 1
                        episode_lighting_backend = str(getattr(self.lighting_augmentor, "backend", "disabled"))
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
                        projection_randomization_enabled=projection_randomization_enabled,
                        return_aux=True,
                    )
            except RuntimeError as err:
                recover_ok = self._recover_from_cuda_alloc_failure(err, split=split)
                if (not self._is_cuda_alloc_failure(err)) or (not recover_ok):
                    raise
                # Retry once with lighting disabled for this episode after forcing procedural fallback.
                disable_lighting = True
                episode_effective_lighting_enabled = 0
                episode_lighting_backend = str(getattr(self.lighting_augmentor, "backend", "disabled"))
                pixel_values = [current_image]
                with self._temporary_rng_seed(step_seed):
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
                        projection_randomization_enabled=projection_randomization_enabled,
                        return_aux=True,
                    )

            with torch.no_grad():
                output_clean: CausalLMOutputWithPast = self.vla(
                    input_ids=clean_rollout_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=clean_images.to(torch.bfloat16),
                    labels=None,
                    output_hidden_states=need_history,
                    use_cache=False,
                )
            adv_labels = labels_masked if online_ce_mode == "pseudo_clean" else None
            output_adv: CausalLMOutputWithPast = self.vla(
                input_ids=adv_rollout_input_ids,
                attention_mask=attention_mask,
                pixel_values=adv_images.to(torch.bfloat16),
                labels=adv_labels,
                output_hidden_states=need_history,
                use_cache=False,
            )

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
            total_action_terms += 1
            total_per_joint = self._accumulate_joint_values(total_per_joint, step_per_joint_gap)

            clean_pred_tokens = self._extract_pred_action_tokens_from_logits(output_clean.logits, labels_full)
            step_history_div = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            step_history_div_legacy = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            if need_history:
                step_history_div, _clean_history_state, adv_history_state = self._compute_clean_adv_history_divergence(
                    clean_hidden_states=output_clean.hidden_states[-1],
                    adv_hidden_states=output_adv.hidden_states[-1],
                    labels_full=labels_full,
                    action_mask_full=action_mask_full,
                )
                total_history_div += step_history_div.detach().item()
                total_history_terms += 1
                step_history_div_legacy = self._compute_legacy_history_divergence(adv_history_state, prev_adv_history_state)
                if prev_adv_history_state is not None:
                    total_history_div_legacy += step_history_div_legacy.detach().item()
                    total_history_terms_legacy += 1
                prev_adv_history_state = adv_history_state.detach()

            step_weight = float(self._get_rollout_step_weight(step_idx))
            episode_weight_sum += step_weight

            step_loss = -(lambda_action_gap * step_action_gap_loss)
            if need_history:
                step_loss = step_loss - (lambda_history * step_history_div)
                step_loss = step_loss - (lambda_history_legacy * step_history_div_legacy)
            step_ce = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            step_ce_objective = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            if online_ce_mode in ("pseudo_clean", "uada_inverse_pseudo_clean"):
                step_ce = self._compute_pseudo_clean_ce(
                    adv_logits=output_adv.logits,
                    clean_logits=output_clean.logits,
                    labels_full=labels_full,
                    action_mask_full=action_mask_full,
                )
                if online_ce_mode == "pseudo_clean":
                    step_ce_objective = step_ce
                else:
                    step_ce_objective = self._compute_uada_inverse_ce_objective(step_ce)
                step_loss = step_loss + (lambda_ce * step_ce_objective)
            weighted_step_loss = step_loss * step_weight
            total_ce += step_ce.detach().item()
            total_ce_objective += step_ce_objective.detach().item()
            if need_backward:
                if weighted_step_loss.requires_grad:
                    weighted_step_loss.backward()
                total_loss = total_loss + weighted_step_loss.detach()
            else:
                total_loss = total_loss + weighted_step_loss

            env_action = self._decode_env_action(adv_pred_tokens, action_mask_full, action_stats)
            obs, _, done, _ = env.step(env_action.tolist())
            last_done = bool(done)

            next_image = None
            if capture_visual or (record_video_frames and video_frame_source == "next_obs"):
                next_image = get_libero_image(obs, (224, 224))

            adv_frame_pil = None
            if capture_visual or (record_video_frames and video_frame_source == "adv"):
                adv_frame_pil = self._to_pil(
                    self.randomPatchTransform.denormalize(
                        adv_images[0, 0:3, :, :].detach().cpu().unsqueeze(0),
                        mean=self.mean[0],
                        std=self.std[0],
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                )

            if record_video_frames:
                if video_frame_source == "orig":
                    video_frame = self._to_pil(current_image)
                elif video_frame_source == "projected_input":
                    video_frame = self._to_pil(attack_aux["projected_inputs"][0])
                elif video_frame_source == "adv":
                    video_frame = adv_frame_pil
                else:
                    video_frame = self._to_pil(next_image if next_image is not None else current_image)
                video_frames.append(video_frame)

            if capture_visual and self._is_visual_step(step_idx, max_steps):
                if next_image is None:
                    next_image = get_libero_image(obs, (224, 224))
                visual_frames.append(
                    {
                        "step_idx": int(step_idx),
                        "orig": self._to_pil(current_image),
                        "projected_input": self._to_pil(attack_aux["projected_inputs"][0]),
                        "adv": adv_frame_pil
                        if adv_frame_pil is not None
                        else self._to_pil(
                            self.randomPatchTransform.denormalize(
                                adv_images[0, 0:3, :, :].detach().cpu().unsqueeze(0),
                                mean=self.mean[0],
                                std=self.std[0],
                            )
                            .squeeze(0)
                            .clamp(0, 1)
                        ),
                        "next_obs": self._to_pil(next_image),
                    }
                )

            total_proj_alpha += float(attack_aux["projection_alpha_mean"])
            total_proj_cov += float(attack_aux["projection_coverage_ratio"])
            total_proj_bottom += float(attack_aux["projection_bottom_ratio"])
            total_proj_keystone += float(attack_aux["projection_keystone"])

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
            if done:
                break

        weight_denom = float(max(episode_weight_sum, 1e-8))
        total_loss = total_loss / weight_denom
        if need_backward and (projection_texture.grad is not None):
            inv_weight_denom = 1.0 / weight_denom
            if grad_before_episode is None:
                projection_texture.grad.mul_(inv_weight_denom)
            else:
                episode_delta_grad = projection_texture.grad - grad_before_episode
                projection_texture.grad.copy_(grad_before_episode + (episode_delta_grad * inv_weight_denom))

        ep_steps = max(1, int(total_action_terms))
        return {
            "loss": total_loss if need_backward else total_loss.detach(),
            "action_gap": total_action_gap / float(ep_steps),
            "history_div": total_history_div / float(max(1, total_history_terms)),
            "history_div_legacy": total_history_div_legacy / float(max(1, total_history_terms_legacy)),
            "ce_value": total_ce / float(ep_steps),
            "ce_objective_value": total_ce_objective / float(ep_steps),
            "action_terms": total_action_terms,
            "history_terms": total_history_terms,
            "history_terms_legacy": total_history_terms_legacy,
            "done": last_done,
            "episode_len": total_action_terms,
            "per_joint_gap": self._normalize_joint_values(total_per_joint, float(ep_steps)),
            "projection_alpha": total_proj_alpha / float(ep_steps),
            "projection_coverage": total_proj_cov / float(ep_steps),
            "projection_bottom": total_proj_bottom / float(ep_steps),
            "projection_keystone": total_proj_keystone / float(ep_steps),
            "visual_frames": visual_frames,
            "video_frames": video_frames,
            "lighting_backend": str(episode_lighting_backend),
            "effective_lighting_enabled": int(episode_effective_lighting_enabled),
            "task_id": -1 if task_id is None else int(task_id),
            "task_description": str(task_description),
        }

    def _is_cuda_alloc_failure(self, err):
        message = str(err).lower()
        return (
            ("cuda out of memory" in message)
            or ("cublas_status_alloc_failed" in message)
            or ("cudnn_status_alloc_failed" in message)
        )

    def _recover_from_cuda_alloc_failure(self, err, split=None):
        if not self._is_cuda_alloc_failure(err):
            return False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if str(split).lower() == "train":
            self._recent_train_oom = True
        if self.lighting_augmentor is None:
            return False

        reason = str(err).strip().replace("\n", " ")
        if len(reason) > 240:
            reason = reason[:240] + "..."
        if hasattr(self.lighting_augmentor, "force_procedural_fallback"):
            self.lighting_augmentor.force_procedural_fallback(reason=reason)
        else:
            self.lighting_augmentor.backend = "procedural_fallback"
            self.lighting_augmentor._pipeline = None
            self.lighting_augmentor._load_attempted = True
            print(f"[LightingAugmentor] forcing procedural lighting fallback ({reason}).")
        return True

    def _build_prompt_tensors(self, task_description: str, action_dim: int):
        prompt_builder = PurePromptBuilder("openvla")
        dummy_action = np.zeros((action_dim,), dtype=np.float32)
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {str(task_description).lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(dummy_action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids_list = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        input_ids = torch.tensor(input_ids_list, device=self.vla.device, dtype=torch.long)
        labels = input_ids.clone()

        prefix_len = max(0, labels.shape[0] - (int(action_dim) + 1))
        if prefix_len > 0:
            labels[:prefix_len] = IGNORE_INDEX
        if (not self.predict_stop_token) and labels.numel() > 0:
            labels[-1] = IGNORE_INDEX

        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, device=self.vla.device, dtype=torch.long)
        return labels, attention_mask, input_ids

    def _compute_pseudo_clean_ce(self, adv_logits, clean_logits, labels_full, action_mask_full):
        adv_action_logits = self._extract_action_logits(adv_logits, labels_full)
        clean_action_logits = self._extract_action_logits(clean_logits, labels_full)

        seq_len = min(action_mask_full.shape[1], adv_action_logits.shape[1], clean_action_logits.shape[1])
        action_mask = action_mask_full[:, :seq_len]
        adv_selected = adv_action_logits[:, :seq_len, :][action_mask]
        clean_tokens = clean_action_logits[:, :seq_len, :][action_mask].argmax(dim=-1)
        if adv_selected.numel() == 0:
            return torch.zeros((), device=self.vla.device, dtype=torch.float32)
        return F.cross_entropy(adv_selected, clean_tokens.detach())

    def _compute_uada_inverse_ce_objective(self, ce_value, min_ce=1e-3):
        # Mimic original UADA's 1 / CE acceleration term, but use pseudo-clean action tokens
        # because online rollout has no expert action labels.
        safe_ce = torch.clamp(ce_value, min=float(min_ce))
        return torch.reciprocal(safe_ce)

    def _decode_env_action(self, adv_pred_tokens, action_mask_full, action_stats):
        seq_len = min(action_mask_full.shape[1], adv_pred_tokens.shape[1])
        action_mask = action_mask_full[:, :seq_len]
        selected_tokens = adv_pred_tokens[:, :seq_len][action_mask]
        if selected_tokens.numel() == 0:
            action_dim = len(action_stats["q01"])
            return np.zeros((action_dim,), dtype=np.float32)
        action_dim = len(action_stats["q01"])
        selected_tokens = selected_tokens.view(1, -1)[0]
        normalized_action = np.asarray(
            self.action_tokenizer.decode_token_ids_to_actions(selected_tokens.detach().cpu().numpy()),
            dtype=np.float32,
        )
        normalized_action = self._fit_action_dim(normalized_action, action_dim)
        unnormalized = self._unnormalize_action(normalized_action, action_stats)
        env_action = self._convert_to_env_action(unnormalized)
        return env_action

    def _unnormalize_action(self, normalized_action, action_stats):
        mask = np.array(action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool)), dtype=bool)
        high = np.array(action_stats["q99"], dtype=np.float32)
        low = np.array(action_stats["q01"], dtype=np.float32)
        return np.where(mask, 0.5 * (normalized_action + 1.0) * (high - low) + low, normalized_action).astype(np.float32)

    def _convert_to_env_action(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] > 0:
            action[-1] = 2.0 * (action[-1] - 0.0) / 1.0 - 1.0
            action[-1] = np.sign(action[-1])
            if action[-1] == 0.0:
                action[-1] = 1.0
            action[-1] = -1.0 * action[-1]
        return action

    def _fit_action_dim(self, action, action_dim):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == action_dim:
            return action
        if action.shape[0] > action_dim:
            return action[:action_dim]
        out = np.zeros((action_dim,), dtype=np.float32)
        out[: action.shape[0]] = action
        return out

    def _resolve_task_suite_name(self, dataset, task_suite_name):
        if str(task_suite_name).lower() not in ("", "auto", "none", "null"):
            return str(task_suite_name)
        dataset = str(dataset).lower()
        for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
            if suite_name in dataset:
                return suite_name
        return "libero_spatial"

    def _resolve_unnorm_key(self, task_suite_name):
        key = str(task_suite_name)
        if hasattr(self.vla, "norm_stats"):
            if key not in self.vla.norm_stats and f"{key}_no_noops" in self.vla.norm_stats:
                key = f"{key}_no_noops"
            if key not in self.vla.norm_stats:
                first_key = next(iter(self.vla.norm_stats.keys()))
                print(f"[OnlineEnv] Warning: unnorm key `{key}` not found, fallback to `{first_key}`.")
                key = first_key
        return key

    def _resolve_max_env_steps(self, max_env_steps, task_suite_name):
        if isinstance(max_env_steps, str) and max_env_steps.lower() == "auto_by_suite":
            suite = str(task_suite_name).lower()
            table = {
                "libero_spatial": 193,
                "libero_object": 254,
                "libero_goal": 270,
                "libero_10": 505,
                "libero_90": 373,
            }
            for key, value in table.items():
                if key in suite:
                    return int(value)
            return 270
        try:
            value = int(max_env_steps)
        except Exception:
            return 270
        return max(1, value)

    def _init_gpu_tuner_state(
        self,
        enabled,
        mode,
        mem_low,
        mem_high,
        mem_hard_cap,
        util_low,
        cooldown_iters,
        min_rollout,
        max_rollout,
        min_tasks_per_iter,
        max_tasks_per_iter,
        base_phase1_rollout,
        base_phase2_rollout,
        base_train_tasks_per_iter,
    ):
        return {
            "enabled": bool(enabled),
            "mode": str(mode),
            "level": 2,
            "cooldown": 0,
            "cooldown_iters": int(cooldown_iters),
            "low_counter": 0,
            "mem_low": float(mem_low),
            "mem_high": float(mem_high),
            "mem_hard_cap": float(mem_hard_cap),
            "util_low": float(util_low),
            "min_rollout": int(min_rollout),
            "max_rollout": int(max_rollout),
            "min_tasks_per_iter": int(min_tasks_per_iter),
            "max_tasks_per_iter": int(max_tasks_per_iter),
            "base_phase1_rollout": int(base_phase1_rollout),
            "base_phase2_rollout": int(base_phase2_rollout),
            "base_train_tasks_per_iter": int(base_train_tasks_per_iter),
            "device_index": self._resolve_cuda_device_index(),
        }

    def _resolve_cuda_device_index(self):
        if not torch.cuda.is_available():
            return None

        device = getattr(self.vla, "device", None)
        if isinstance(device, torch.device):
            if device.type != "cuda":
                return None
            return int(torch.cuda.current_device() if device.index is None else device.index)

        device_text = str(device) if device is not None else "cuda"
        if not device_text.startswith("cuda"):
            return None
        if ":" in device_text:
            try:
                return int(device_text.split(":")[-1])
            except Exception:
                return int(torch.cuda.current_device())
        return int(torch.cuda.current_device())

    def _resolve_effective_train_budget(self, tune_level, gpu_tuner_state):
        profile_table = {
            0: {"rollout_scale": 0.60, "tasks_per_iter": 1},
            1: {"rollout_scale": 0.80, "tasks_per_iter": 1},
            2: {"rollout_scale": 1.00, "tasks_per_iter": 1},
            3: {"rollout_scale": 1.15, "tasks_per_iter": 1},
            4: {"rollout_scale": 1.30, "tasks_per_iter": 2},
        }
        level = int(max(0, min(4, int(tune_level))))
        profile = profile_table[level]
        rollout_scale = float(profile["rollout_scale"])

        phase1_rollout = int(round(float(gpu_tuner_state["base_phase1_rollout"]) * rollout_scale))
        phase2_rollout = int(round(float(gpu_tuner_state["base_phase2_rollout"]) * rollout_scale))
        phase1_rollout = max(int(gpu_tuner_state["min_rollout"]), min(int(gpu_tuner_state["max_rollout"]), phase1_rollout))
        phase2_rollout = max(int(gpu_tuner_state["min_rollout"]), min(int(gpu_tuner_state["max_rollout"]), phase2_rollout))

        tasks_per_iter = int(profile["tasks_per_iter"])
        tasks_per_iter = max(
            int(gpu_tuner_state["min_tasks_per_iter"]),
            min(int(gpu_tuner_state["max_tasks_per_iter"]), tasks_per_iter),
        )
        return phase1_rollout, phase2_rollout, tasks_per_iter

    def _query_nvidia_smi_gpu_util(self, device_index):
        if device_index is None:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={int(device_index)}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if result.returncode != 0:
                return None
            line = str(result.stdout).strip().splitlines()
            if len(line) == 0:
                return None
            value_text = line[0].split(",")[0].strip()
            if value_text in ("", "N/A"):
                return None
            return float(value_text)
        except Exception:
            return None

    def _collect_gpu_runtime_stats(self, device_index):
        if (device_index is None) or (not torch.cuda.is_available()):
            return {"gpu_util": None, "mem_used_gb": None, "mem_ratio": None}

        mem_used_gb = None
        mem_ratio = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(int(device_index))
            used_bytes = max(0, int(total_bytes - free_bytes))
            mem_used_gb = float(used_bytes / float(1024**3))
            if total_bytes > 0:
                mem_ratio = float(used_bytes / float(total_bytes))
        except Exception:
            mem_used_gb = None
            mem_ratio = None

        gpu_util = self._query_nvidia_smi_gpu_util(device_index=device_index)
        return {"gpu_util": gpu_util, "mem_used_gb": mem_used_gb, "mem_ratio": mem_ratio}

    def _update_gpu_tuner_state(self, gpu_tuner_state, mem_ratio, gpu_util, recent_oom):
        current_level = int(gpu_tuner_state["level"])

        if bool(recent_oom):
            target_level = max(0, current_level - 2)
            gpu_tuner_state["low_counter"] = 0
            if target_level != current_level:
                gpu_tuner_state["level"] = target_level
                gpu_tuner_state["cooldown"] = int(gpu_tuner_state["cooldown_iters"])
                action = f"down(oom):L{current_level}->L{target_level}"
                print(f"[GpuAutoTune] {action}")
                return action
            return "hold(oom@floor)"

        cooldown = int(gpu_tuner_state["cooldown"])
        if cooldown > 0:
            gpu_tuner_state["cooldown"] = cooldown - 1
            gpu_tuner_state["low_counter"] = 0
            return f"hold(cooldown:{gpu_tuner_state['cooldown']})"

        if (mem_ratio is not None) and (float(mem_ratio) >= float(gpu_tuner_state["mem_hard_cap"])):
            target_level = max(0, current_level - 2)
            if target_level != current_level:
                gpu_tuner_state["level"] = target_level
                gpu_tuner_state["cooldown"] = int(gpu_tuner_state["cooldown_iters"])
                gpu_tuner_state["low_counter"] = 0
                action = f"down(mem_hard_cap):L{current_level}->L{target_level}"
                print(f"[GpuAutoTune] {action} (mem_ratio={float(mem_ratio):.4f})")
                return action
            return "hold(mem_hard_cap@floor)"

        if (mem_ratio is not None) and (float(mem_ratio) >= float(gpu_tuner_state["mem_high"])):
            target_level = max(0, current_level - 1)
            if target_level != current_level:
                gpu_tuner_state["level"] = target_level
                gpu_tuner_state["cooldown"] = int(gpu_tuner_state["cooldown_iters"])
                gpu_tuner_state["low_counter"] = 0
                action = f"down(mem_high):L{current_level}->L{target_level}"
                print(f"[GpuAutoTune] {action} (mem_ratio={float(mem_ratio):.4f})")
                return action
            return "hold(mem_high@floor)"

        low_condition = False
        if mem_ratio is not None:
            low_condition = float(mem_ratio) <= float(gpu_tuner_state["mem_low"])
            if gpu_util is not None:
                low_condition = low_condition and (float(gpu_util) <= float(gpu_tuner_state["util_low"]))

        if low_condition:
            gpu_tuner_state["low_counter"] = int(gpu_tuner_state["low_counter"]) + 1
        else:
            gpu_tuner_state["low_counter"] = 0

        if int(gpu_tuner_state["low_counter"]) >= 2:
            target_level = min(4, current_level + 1)
            gpu_tuner_state["low_counter"] = 0
            if target_level != current_level:
                gpu_tuner_state["level"] = target_level
                gpu_tuner_state["cooldown"] = int(gpu_tuner_state["cooldown_iters"])
                mem_text = f"{float(mem_ratio):.4f}" if mem_ratio is not None else "NA"
                util_text = f"{float(gpu_util):.1f}" if gpu_util is not None else "NA"
                action = f"up(low_util_mem):L{current_level}->L{target_level}"
                print(f"[GpuAutoTune] {action} (mem_ratio={mem_text}, gpu_util={util_text})")
                return action

        return "hold"

    def _sample_task_ids(self, task_suite, num_tasks):
        n_tasks = int(task_suite.n_tasks)
        if n_tasks <= 0:
            return [0]
        candidate_ids = list(range(n_tasks))
        if num_tasks >= n_tasks:
            return candidate_ids
        return random.sample(candidate_ids, k=int(num_tasks))

    def _sample_init_state(self, init_states, iter_idx, local_idx, deterministic_seed=None):
        if len(init_states) == 0:
            return None
        if deterministic_seed is not None:
            ridx = int(deterministic_seed) % len(init_states)
        else:
            ridx = (int(iter_idx) + int(local_idx) + random.randint(0, max(0, len(init_states) - 1))) % len(init_states)
        return init_states[ridx]

    def _set_online_env_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _is_visual_step(self, step_idx, max_steps):
        if max_steps <= 2:
            return True
        mid = max_steps // 2
        return step_idx in (0, mid, max_steps - 1)

    def _require_libero_modules(self):
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
            from experiments.robot.libero.libero_utils import (
                get_libero_dummy_action,
                get_libero_env,
                get_libero_image,
            )
        except Exception as exc:
            raise RuntimeError(
                "LIBERO online rollout dependencies are missing. "
                "Please install/import `libero` and ensure `OffScreenRenderEnv` is available."
            ) from exc
        return benchmark, get_libero_env, get_libero_image, get_libero_dummy_action

    def _online_visualization_root(self):
        visual_root = os.path.join(self.save_dir, "visualization")
        os.makedirs(visual_root, exist_ok=True)
        return visual_root

    def _append_online_visual_manifest(self, row):
        visual_root = self._online_visualization_root()
        manifest_path = os.path.join(visual_root, "online_manifest.csv")
        file_exists = os.path.exists(manifest_path)
        fieldnames = [
            "iter_idx",
            "phase_id",
            "is_best",
            "val_rollout_score",
            "reason",
            "attack_mode",
            "visual_dir",
            "steps",
        ]
        with open(manifest_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _dump_online_visual_frames(
        self,
        visual_frames,
        iter_idx,
        phase_id,
        is_best,
        val_rollout_score,
        reason,
        attack_mode,
        args=None,
    ):
        if not visual_frames:
            return

        visual_root = self._online_visualization_root()
        iter_dir = os.path.join(visual_root, f"iter_{int(iter_idx):06d}_online")
        os.makedirs(iter_dir, exist_ok=True)

        for frame in visual_frames:
            step_idx = int(frame["step_idx"])
            frame["orig"].save(os.path.join(iter_dir, f"step_{step_idx:03d}_orig.png"))
            frame["projected_input"].save(os.path.join(iter_dir, f"step_{step_idx:03d}_projected_input.png"))
            frame["adv"].save(os.path.join(iter_dir, f"step_{step_idx:03d}_adv.png"))
            frame["next_obs"].save(os.path.join(iter_dir, f"step_{step_idx:03d}_next_obs.png"))

        self._append_online_visual_manifest(
            {
                "iter_idx": int(iter_idx),
                "phase_id": int(phase_id),
                "is_best": int(bool(is_best)),
                "val_rollout_score": float(val_rollout_score),
                "reason": str(reason),
                "attack_mode": str(attack_mode),
                "visual_dir": iter_dir,
                "steps": ",".join(str(int(frame["step_idx"])) for frame in visual_frames),
            }
        )

        if args is not None and args.wandb_project != "false" and wandb is not None:
            wandb.log(
                {
                    "VIS_online/orig": [wandb.Image(item["orig"]) for item in visual_frames],
                    "VIS_online/projected_input": [wandb.Image(item["projected_input"]) for item in visual_frames],
                    "VIS_online/adv": [wandb.Image(item["adv"]) for item in visual_frames],
                    "VIS_online/next_obs": [wandb.Image(item["next_obs"]) for item in visual_frames],
                },
                step=int(iter_idx),
            )

    def save_online_info(self, path):
        with open(os.path.join(path, "train_online_rollout_action_gap.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_action_gap, file)
        with open(os.path.join(path, "train_online_rollout_action_gap_joints.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_action_gap_joints, file)
        with open(os.path.join(path, "train_online_rollout_history_div.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_history_div, file)
        with open(os.path.join(path, "train_online_rollout_history_div_legacy.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_history_div_legacy, file)
        with open(os.path.join(path, "train_online_rollout_score.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_score, file)
        with open(os.path.join(path, "train_online_objective_score.pkl"), "wb") as file:
            pickle.dump(self.train_rollout_objective_score, file)
        with open(os.path.join(path, "train_online_phase_id.pkl"), "wb") as file:
            pickle.dump(self.train_phase_id, file)
        with open(os.path.join(path, "train_online_done_rate.pkl"), "wb") as file:
            pickle.dump(self.train_online_done_rate, file)
        with open(os.path.join(path, "train_online_episode_len.pkl"), "wb") as file:
            pickle.dump(self.train_online_episode_len, file)
        with open(os.path.join(path, "val_online_rollout_action_gap.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_action_gap, file)
        with open(os.path.join(path, "val_online_rollout_action_gap_joints.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_action_gap_joints, file)
        with open(os.path.join(path, "val_online_rollout_history_div.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_history_div, file)
        with open(os.path.join(path, "val_online_rollout_history_div_legacy.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_history_div_legacy, file)
        with open(os.path.join(path, "val_online_rollout_score.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_score, file)
        with open(os.path.join(path, "val_online_objective_score.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_objective_score, file)
        with open(os.path.join(path, "val_online_rollout_score_legacy.pkl"), "wb") as file:
            pickle.dump(self.val_rollout_score_legacy, file)
        with open(os.path.join(path, "val_online_done_rate.pkl"), "wb") as file:
            pickle.dump(self.val_online_done_rate, file)
        with open(os.path.join(path, "val_online_episode_len.pkl"), "wb") as file:
            pickle.dump(self.val_online_episode_len, file)
