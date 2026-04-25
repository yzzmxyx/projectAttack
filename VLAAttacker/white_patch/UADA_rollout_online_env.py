import csv
import json
import math
import os
import pickle
import random
import subprocess
from pathlib import Path
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
    from white_patch.gt_phase_schedule import clamp_phase_boundary_ratios, infer_gt_phase_for_step
except Exception:
    from gt_phase_schedule import clamp_phase_boundary_ratios, infer_gt_phase_for_step

try:
    from white_patch.window_rollout_probe_utils import (
        compute_window_rollout_weight,
        infer_phase_name_from_boundaries,
        normalize_window_rollout_future_mode,
        normalize_window_rollout_metric_mode,
        normalize_window_rollout_phase_scope,
        resolve_phase_window,
        select_window_rollout_metric_value,
    )
except Exception:
    from window_rollout_probe_utils import (
        compute_window_rollout_weight,
        infer_phase_name_from_boundaries,
        normalize_window_rollout_future_mode,
        normalize_window_rollout_metric_mode,
        normalize_window_rollout_phase_scope,
        resolve_phase_window,
        select_window_rollout_metric_value,
    )

try:
    from white_patch.projector_photometric_params import (
        LearnableProjectorPhotometricParams,
        parse_projector_channel_gain,
        save_projector_params,
    )
except Exception:
    from projector_photometric_params import (
        LearnableProjectorPhotometricParams,
        parse_projector_channel_gain,
        save_projector_params,
    )

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"


class OpenVLAOnlineEnvAttacker(OpenVLAAttacker):
    RESUME_STATE_FILENAME = "resume_state.pt"

    def _reset_metric_buffers(self):
        super()._reset_metric_buffers()
        self.train_rollout_history_div_legacy = []
        self.train_rollout_siglip_distance = []
        self.train_rollout_objective_score = []
        self.train_gt_rollout_action_gap = []
        self.train_gt_rollout_action_gap_joints = []
        self.train_gt_rollout_score = []
        self.train_gt_rollout_objective_score = []
        self.train_continuous_clean_gt_action_gap = []
        self.train_continuous_adv_gt_action_gap = []
        self.train_continuous_rollout_delta = []
        self.train_impulse_rollout_area = []
        self.train_window_rollout_clean_gt_action_gap = []
        self.train_window_rollout_adv_gt_action_gap = []
        self.train_window_rollout_deattack_gt_action_gap = []
        self.train_window_rollout_selected_gt_action_gap = []
        self.train_window_rollout_delta_weighted = []
        self.train_window_rollout_delta_weighted_loss = []
        self.train_total_rollout_score = []
        self.train_total_objective_score = []
        self.train_active_rollout_action_gap = []
        self.train_active_rollout_score = []
        self.train_active_rollout_objective_score = []
        self.val_rollout_objective_score = []
        self.val_rollout_siglip_distance = []
        self.val_gt_rollout_action_gap = []
        self.val_gt_rollout_action_gap_joints = []
        self.val_gt_rollout_score = []
        self.val_gt_rollout_objective_score = []
        self.val_continuous_clean_gt_action_gap = []
        self.val_continuous_adv_gt_action_gap = []
        self.val_continuous_rollout_delta = []
        self.val_impulse_rollout_area = []
        self.val_window_rollout_clean_gt_action_gap = []
        self.val_window_rollout_adv_gt_action_gap = []
        self.val_window_rollout_deattack_gt_action_gap = []
        self.val_window_rollout_selected_gt_action_gap = []
        self.val_window_rollout_delta_weighted = []
        self.val_window_rollout_delta_weighted_loss = []
        self.val_total_rollout_score = []
        self.val_total_objective_score = []
        self.val_active_rollout_action_gap = []
        self.val_active_rollout_score = []
        self.val_active_rollout_objective_score = []
        self.train_online_done_rate = []
        self.train_online_episode_len = []
        self.val_online_done_rate = []
        self.val_online_episode_len = []
        self._recent_train_oom = False
        self._gt_expert_cache = {"train": {}, "val": {}}
        self._gt_expert_cache_stats = {"train": {}, "val": {}}
        self._action_gap_mode = "clean_adv"
        self._gt_dataset_root = ""
        self._gt_phase_action_bank = {}
        self._gt_phase_action_bank_stats = {}
        self._gt_phase_boundary_ratios = {}
        self._gt_action_bank_path = ""
        self._gt_softmin_tau = 0.05
        self._phase_state_cache = {}
        self._phase_state_cache_records = {}
        self._phase_state_cache_stats = {}
        self._phase_state_cache_path = ""
        self._init_projection_texture_path = ""
        self._initial_projection_texture_snapshot_path = ""
        self._initial_patch_snapshot_path = ""
        self._resume_capable = False
        self._latest_resume_checkpoint = ""
        self._last_completed_iter = -1
        self._wandb_run_id = ""

    def _history_metric_file_map(self):
        return [
            ("loss_buffer", "loss"),
            ("train_rollout_action_gap", "train_online_rollout_action_gap.pkl"),
            ("train_rollout_action_gap_joints", "train_online_rollout_action_gap_joints.pkl"),
            ("train_gt_rollout_action_gap", "train_online_gt_rollout_action_gap.pkl"),
            ("train_gt_rollout_action_gap_joints", "train_online_gt_rollout_action_gap_joints.pkl"),
            ("train_rollout_history_div", "train_online_rollout_history_div.pkl"),
            ("train_rollout_history_div_legacy", "train_online_rollout_history_div_legacy.pkl"),
            ("train_rollout_siglip_distance", "train_online_siglip_distance.pkl"),
            ("train_rollout_score", "train_online_rollout_score.pkl"),
            ("train_rollout_objective_score", "train_online_objective_score.pkl"),
            ("train_gt_rollout_score", "train_online_gt_rollout_score.pkl"),
            ("train_gt_rollout_objective_score", "train_online_gt_objective_score.pkl"),
            ("train_continuous_clean_gt_action_gap", "train_online_continuous_clean_gt_action_gap.pkl"),
            ("train_continuous_adv_gt_action_gap", "train_online_continuous_adv_gt_action_gap.pkl"),
            ("train_continuous_rollout_delta", "train_online_continuous_rollout_delta.pkl"),
            ("train_impulse_rollout_area", "train_online_impulse_rollout_area.pkl"),
            ("train_window_rollout_clean_gt_action_gap", "train_online_window_rollout_clean_gt_action_gap.pkl"),
            ("train_window_rollout_adv_gt_action_gap", "train_online_window_rollout_adv_gt_action_gap.pkl"),
            ("train_window_rollout_deattack_gt_action_gap", "train_online_window_rollout_deattack_gt_action_gap.pkl"),
            ("train_window_rollout_selected_gt_action_gap", "train_online_window_rollout_selected_gt_action_gap.pkl"),
            ("train_window_rollout_delta_weighted", "train_online_window_rollout_delta_weighted.pkl"),
            ("train_window_rollout_delta_weighted_loss", "train_online_window_rollout_delta_weighted_loss.pkl"),
            ("train_total_rollout_score", "train_online_total_rollout_score.pkl"),
            ("train_total_objective_score", "train_online_total_objective_score.pkl"),
            ("train_active_rollout_action_gap", "train_online_active_rollout_action_gap.pkl"),
            ("train_active_rollout_score", "train_online_active_rollout_score.pkl"),
            ("train_active_rollout_objective_score", "train_online_active_objective_score.pkl"),
            ("train_phase_id", "train_online_phase_id.pkl"),
            ("train_online_done_rate", "train_online_done_rate.pkl"),
            ("train_online_episode_len", "train_online_episode_len.pkl"),
            ("val_rollout_action_gap", "val_online_rollout_action_gap.pkl"),
            ("val_rollout_action_gap_joints", "val_online_rollout_action_gap_joints.pkl"),
            ("val_gt_rollout_action_gap", "val_online_gt_rollout_action_gap.pkl"),
            ("val_gt_rollout_action_gap_joints", "val_online_gt_rollout_action_gap_joints.pkl"),
            ("val_rollout_history_div", "val_online_rollout_history_div.pkl"),
            ("val_rollout_history_div_legacy", "val_online_rollout_history_div_legacy.pkl"),
            ("val_rollout_siglip_distance", "val_online_siglip_distance.pkl"),
            ("val_rollout_score", "val_online_rollout_score.pkl"),
            ("val_rollout_objective_score", "val_online_objective_score.pkl"),
            ("val_gt_rollout_score", "val_online_gt_rollout_score.pkl"),
            ("val_gt_rollout_objective_score", "val_online_gt_objective_score.pkl"),
            ("val_continuous_clean_gt_action_gap", "val_online_continuous_clean_gt_action_gap.pkl"),
            ("val_continuous_adv_gt_action_gap", "val_online_continuous_adv_gt_action_gap.pkl"),
            ("val_continuous_rollout_delta", "val_online_continuous_rollout_delta.pkl"),
            ("val_impulse_rollout_area", "val_online_impulse_rollout_area.pkl"),
            ("val_window_rollout_clean_gt_action_gap", "val_online_window_rollout_clean_gt_action_gap.pkl"),
            ("val_window_rollout_adv_gt_action_gap", "val_online_window_rollout_adv_gt_action_gap.pkl"),
            ("val_window_rollout_deattack_gt_action_gap", "val_online_window_rollout_deattack_gt_action_gap.pkl"),
            ("val_window_rollout_selected_gt_action_gap", "val_online_window_rollout_selected_gt_action_gap.pkl"),
            ("val_window_rollout_delta_weighted", "val_online_window_rollout_delta_weighted.pkl"),
            ("val_window_rollout_delta_weighted_loss", "val_online_window_rollout_delta_weighted_loss.pkl"),
            ("val_total_rollout_score", "val_online_total_rollout_score.pkl"),
            ("val_total_objective_score", "val_online_total_objective_score.pkl"),
            ("val_active_rollout_action_gap", "val_online_active_rollout_action_gap.pkl"),
            ("val_active_rollout_score", "val_online_active_rollout_score.pkl"),
            ("val_active_rollout_objective_score", "val_online_active_objective_score.pkl"),
            ("val_rollout_score_legacy", "val_online_rollout_score_legacy.pkl"),
            ("val_online_done_rate", "val_online_done_rate.pkl"),
            ("val_online_episode_len", "val_online_episode_len.pkl"),
        ]

    def _load_history_metrics(self, path):
        for attr_name, filename in self._history_metric_file_map():
            filepath = os.path.join(path, filename)
            if not os.path.exists(filepath):
                continue
            with open(filepath, "rb") as file:
                if filename == "loss":
                    loaded = torch.load(file, map_location="cpu")
                else:
                    loaded = pickle.load(file)
            setattr(self, attr_name, loaded)

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, rng_state):
        if not isinstance(rng_state, dict):
            return
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].cpu())
        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in torch_cuda_state])

    def _build_resume_state(
        self,
        projection_texture,
        photometric_params,
        optimizer,
        scheduler,
        global_iter_completed,
        config,
        train_phase_start_counter,
        gpu_tuner_state=None,
    ):
        return {
            "schema_version": 1,
            "projection_texture": projection_texture.detach().cpu(),
            "photometric_params_state_dict": photometric_params.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_iter_completed": int(global_iter_completed),
            "next_iter_idx": int(global_iter_completed) + 1,
            "config": dict(config),
            "rng_state": self._capture_rng_state(),
            "train_phase_start_counter": int(train_phase_start_counter),
            "best_rollout_score": float(self.best_rollout_score),
            "gpu_tuner_state": gpu_tuner_state,
            "wandb_run_id": str(self._wandb_run_id),
        }

    def _save_resume_checkpoint(
        self,
        output_dir,
        projection_texture,
        photometric_params,
        optimizer,
        scheduler,
        global_iter_completed,
        config,
        train_phase_start_counter,
        gpu_tuner_state=None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, self.RESUME_STATE_FILENAME)
        payload = self._build_resume_state(
            projection_texture=projection_texture,
            photometric_params=photometric_params,
            optimizer=optimizer,
            scheduler=scheduler,
            global_iter_completed=global_iter_completed,
            config=config,
            train_phase_start_counter=train_phase_start_counter,
            gpu_tuner_state=gpu_tuner_state,
        )
        torch.save(payload, checkpoint_path)
        self._resume_capable = True
        self._latest_resume_checkpoint = str(checkpoint_path)
        self._last_completed_iter = int(global_iter_completed)
        return checkpoint_path

    def _load_resume_checkpoint(self, resume_run_dir):
        checkpoint_path = os.path.join(resume_run_dir, "last", self.RESUME_STATE_FILENAME)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"resume checkpoint not found: `{checkpoint_path}`. "
                "This run does not appear to support full resume."
            )
        state = torch.load(checkpoint_path, map_location=self.vla.device)
        if not isinstance(state, dict):
            raise TypeError(f"Resume checkpoint `{checkpoint_path}` must contain a dict payload.")

        metadata_path = os.path.join(resume_run_dir, "run_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

        self._init_projection_texture_path = str(metadata.get("init_projection_texture_path", ""))
        self._initial_projection_texture_snapshot_path = str(
            metadata.get("initial_projection_texture_snapshot_path", "")
        )
        self._initial_patch_snapshot_path = str(metadata.get("initial_patch_snapshot_path", ""))
        self._wandb_run_id = str(state.get("wandb_run_id", metadata.get("wandb_run_id", self._wandb_run_id or "")))
        self._resume_capable = True
        self._latest_resume_checkpoint = str(checkpoint_path)
        self._last_completed_iter = int(state.get("global_iter_completed", -1))
        return state

    def _validate_resume_compatibility(self, current_config, resume_config):
        incompatible_keys = [
            "num_iter",
            "patch_size",
            "projection_size",
            "accumulate_steps",
            "warmup",
            "phase1_ratio",
            "phase1_rollout",
            "phase2_rollout",
            "learn_projector_gain",
            "learn_projector_channel_gain",
        ]
        mismatches = []
        for key in incompatible_keys:
            current_value = current_config.get(key)
            resume_value = resume_config.get(key)
            if current_value != resume_value:
                mismatches.append(f"{key}: current={current_value!r}, checkpoint={resume_value!r}")
        if mismatches:
            raise ValueError(
                "Resume configuration mismatch detected:\n" + "\n".join(mismatches)
            )


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
            "gt_action_gap",
            "history1",
            "history2",
            "ce",
            "ce_objective",
            "siglip_distance",
            "continuous_clean_gt_action_gap",
            "continuous_adv_gt_action_gap",
            "continuous_rollout_delta",
            "impulse_rollout_area",
            "window_rollout_clean_gt_action_gap",
            "window_rollout_adv_gt_action_gap",
            "window_rollout_deattack_gt_action_gap",
            "window_rollout_selected_gt_action_gap",
            "window_rollout_delta_weighted",
            "window_rollout_delta_weighted_loss",
            "window_rollout_future_steps",
            "window_phase_name",
            "window_start_step",
            "window_end_step",
            "lambda_window_rollout_loss",
            "window_rollout_metric_mode",
            "window_rollout_future_mode",
            "window_rollout_metric_value",
            "rollout_score",
            "gt_rollout_score",
            "objective_score",
            "gt_objective_score",
            "action_gap_mode_active",
            "active_action_gap",
            "active_rollout_score",
            "active_objective_score",
            "total_rollout_score",
            "total_objective_score",
            "episode_len",
            "done_rate",
            "action_gap_joint_0",
            "action_gap_joint_1",
            "action_gap_joint_2",
            "gt_action_gap_joint_0",
            "gt_action_gap_joint_1",
            "gt_action_gap_joint_2",
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

    def _sanitize_online_video_component(self, value):
        text = str(value).lower().strip()
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
        sanitized = sanitized.strip("_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized or "unknown"

    def _format_online_video_index(self, value, width):
        try:
            numeric_value = int(value)
        except Exception:
            return "na"
        if numeric_value < 0:
            return "na"
        return f"{numeric_value:0{int(width)}d}"

    def _append_online_video_manifest(self, row):
        video_root = self._online_video_root()
        manifest_path = os.path.join(video_root, "video_manifest.csv")
        file_exists = os.path.exists(manifest_path)
        fieldnames = [
            "iter_idx",
            "phase_id",
            "split",
            "episode_idx",
            "init_state_idx",
            "phase_start_name",
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
        episode_idx_text = self._format_online_video_index(episode.get("episode_idx", None), width=3)
        task_id_text = self._format_online_video_index(episode.get("task_id", None), width=2)
        init_state_idx_text = self._format_online_video_index(episode.get("init_state_idx", None), width=3)
        phase_start_name = self._sanitize_online_video_component(episode.get("phase_start_name", "unknown"))
        video_filename = (
            f"ep_{episode_idx_text}_task_{task_id_text}_init_{init_state_idx_text}"
            f"_phase_{phase_start_name}_{frame_source_name}.mp4"
        )
        video_path = os.path.join(split_dir, video_filename)
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
                "episode_idx": int(episode.get("episode_idx", -1)),
                "init_state_idx": int(episode.get("init_state_idx", -1)),
                "phase_start_name": str(episode.get("phase_start_name", "")),
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

    def _normalize_instruction_key(self, instruction):
        text = str(instruction).lower().replace("\n", " ").strip()
        text = " ".join(text.split())
        while text.endswith(".") or text.endswith("?"):
            text = text[:-1].rstrip()
        return text

    def _resolve_gt_dataset_name(self, dataset_name):
        dataset_key = str(dataset_name).lower().strip()
        mapping = {
            "bridge_orig": "bridge_orig",
            "libero_spatial": "libero_spatial_no_noops",
            "libero_object": "libero_object_no_noops",
            "libero_goal": "libero_goal_no_noops",
            "libero_10": "libero_10_no_noops",
        }
        if dataset_key not in mapping:
            raise ValueError(f"Unsupported GT dataset for online action-gap mode: `{dataset_name}`.")
        return mapping[dataset_key]

    def _resolve_default_sidecar_path(self, dataset_name, filename):
        sidecar_dataset = self._resolve_gt_dataset_name(dataset_name)
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "data" / "libero_sidecars" / sidecar_dataset / str(filename)

    def _resolve_optional_path(self, value, dataset_name, filename):
        path_value = "" if value is None else str(value).strip()
        if path_value.lower() in ("", "none", "null", "auto"):
            return self._resolve_default_sidecar_path(dataset_name, filename)
        return Path(os.path.abspath(os.path.expanduser(path_value)))

    def _normalize_phase_start_name(self, phase_name):
        name = str(phase_name).lower().strip()
        aliases = {
            "": "initial",
            "pre_contact": "initial",
            "default": "initial",
            "init": "initial",
            "initial": "initial",
            "contact": "contact_manipulate",
            "contact_manipulate": "contact_manipulate",
            "post": "post_contact",
            "post_contact": "post_contact",
        }
        if name not in aliases:
            raise ValueError(f"Unsupported phase start name: `{phase_name}`.")
        return aliases[name]

    def _phase_start_to_gt_phase(self, phase_start_name):
        name = self._normalize_phase_start_name(phase_start_name)
        if name == "initial":
            return "pre_contact"
        return name

    def _normalize_phase_state_mode(self, phase_state_mode):
        mode = str(phase_state_mode).lower().strip()
        aliases = {
            "initial_only": "initial_only",
            "initial": "initial_only",
            "contact_manipulate_only": "contact_manipulate_only",
            "contact_only": "contact_manipulate_only",
            "contact_manipulate": "contact_manipulate_only",
            "post_contact_only": "post_contact_only",
            "post_only": "post_contact_only",
            "post_contact": "post_contact_only",
            "phase_cycle": "phase_cycle",
            "cycle": "phase_cycle",
        }
        if mode not in aliases:
            raise ValueError(
                "--phase_state_mode must be one of "
                "{initial_only, contact_manipulate_only, post_contact_only, phase_cycle}"
            )
        return aliases[mode]

    def _phase_start_for_counter(self, phase_state_mode, counter):
        mode = self._normalize_phase_state_mode(phase_state_mode)
        if mode == "initial_only":
            return "initial"
        if mode == "contact_manipulate_only":
            return "contact_manipulate"
        if mode == "post_contact_only":
            return "post_contact"
        phases = ("initial", "contact_manipulate", "post_contact")
        return phases[int(counter) % len(phases)]

    def _load_gt_phase_boundary_ratios(self, phase_parquet_path, required_instruction_keys):
        ratios = {}
        phase_path = Path(phase_parquet_path)
        if not phase_path.exists():
            return ratios
        try:
            import pandas as pd
        except Exception:
            return ratios

        df = pd.read_parquet(
            phase_path,
            columns=["episode_key", "instruction", "T", "phase", "phase_start_t"],
        )
        required = set(required_instruction_keys)
        for instruction, inst_df in df.groupby("instruction", sort=False):
            instruction_key = self._normalize_instruction_key(instruction)
            if instruction_key not in required:
                continue
            contact_ratios = []
            post_ratios = []
            for _episode_key, episode_df in inst_df.groupby("episode_key", sort=False):
                T = int(episode_df["T"].iloc[0])
                if T <= 1:
                    continue
                contact_rows = episode_df[episode_df["phase"] == "contact_manipulate"]
                post_rows = episode_df[episode_df["phase"] == "post_contact"]
                if len(contact_rows) > 0:
                    contact_ratios.append(float(contact_rows["phase_start_t"].iloc[0]) / float(T))
                if len(post_rows) > 0:
                    post_ratios.append(float(post_rows["phase_start_t"].iloc[0]) / float(T))
            if contact_ratios and post_ratios:
                ratios[instruction_key] = {
                    "contact_ratio": float(np.median(np.asarray(contact_ratios, dtype=np.float32))),
                    "post_ratio": float(np.median(np.asarray(post_ratios, dtype=np.float32))),
                }
        return ratios

    def _build_gt_phase_action_bank(self, gt_action_bank_path, dataset_name, required_instruction_keys, action_dim):
        bank_path = self._resolve_optional_path(gt_action_bank_path, dataset_name, "action_bank.pt")
        if not bank_path.exists():
            raise FileNotFoundError(
                f"GT action bank not found: `{bank_path}`. Build it with build_libero_action_bank.py first."
            )

        raw_bank = torch.load(bank_path, map_location="cpu")
        required = set(required_instruction_keys)
        merged = {}
        source_group_counts = {}
        for key, value in raw_bank.items():
            if not isinstance(key, tuple) or len(key) < 3:
                continue
            instruction_key = self._normalize_instruction_key(key[0])
            phase_name = self._phase_start_to_gt_phase(key[1])
            if instruction_key not in required:
                continue
            actions = value["actions"] if isinstance(value, dict) else value
            actions = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, int(action_dim))
            merged.setdefault((instruction_key, phase_name), []).append(actions.cpu())
            source_group_counts[(instruction_key, phase_name)] = source_group_counts.get((instruction_key, phase_name), 0) + 1

        missing = [
            (instruction, phase)
            for instruction in sorted(required)
            for phase in ("pre_contact", "contact_manipulate", "post_contact")
            if (instruction, phase) not in merged
        ]
        if missing:
            preview = ", ".join(f"{instruction}/{phase}" for instruction, phase in missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            raise RuntimeError(f"Missing GT action-bank phase groups: {preview}")

        phase_bank = {}
        group_sizes = []
        for key, parts in merged.items():
            actions = torch.cat(parts, dim=0).contiguous().cpu()
            phase_bank[key] = actions
            group_sizes.append(int(actions.shape[0]))

        phase_parquet_path = bank_path.with_name("phases.parquet")
        self._gt_phase_action_bank = phase_bank
        self._gt_phase_boundary_ratios = self._load_gt_phase_boundary_ratios(
            phase_parquet_path=phase_parquet_path,
            required_instruction_keys=required,
        )
        self._gt_action_bank_path = str(bank_path)
        self._gt_phase_action_bank_stats = {
            "path": str(bank_path),
            "num_phase_groups": int(len(phase_bank)),
            "num_instructions": int(len(required)),
            "min_group_size": int(min(group_sizes)) if group_sizes else 0,
            "max_group_size": int(max(group_sizes)) if group_sizes else 0,
            "mean_group_size": float(sum(group_sizes) / float(max(1, len(group_sizes)))),
            "source_progress_bin_groups": int(sum(source_group_counts.values())),
            "phase_boundary_ratio_instructions": int(len(self._gt_phase_boundary_ratios)),
        }
        self._gt_expert_cache_stats = {
            "train": {
                "num_instructions": int(len(required)),
                "candidate_trajectories_mean": float(sum(group_sizes) / float(max(1, len(group_sizes)))),
                "candidate_trajectories_min": int(min(group_sizes)) if group_sizes else 0,
                "candidate_trajectories_max": int(max(group_sizes)) if group_sizes else 0,
            },
            "val": {
                "num_instructions": int(len(required)),
                "candidate_trajectories_mean": float(sum(group_sizes) / float(max(1, len(group_sizes)))),
                "candidate_trajectories_min": int(min(group_sizes)) if group_sizes else 0,
                "candidate_trajectories_max": int(max(group_sizes)) if group_sizes else 0,
            },
        }
        self._gt_dataset_root = str(bank_path.parent)

    def _load_phase_state_cache(self, phase_state_cache_path, dataset_name):
        cache_path = self._resolve_optional_path(phase_state_cache_path, dataset_name, "phase_state_cache.pt")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Phase-state cache not found: `{cache_path}`. Build it with build_libero_phase_state_cache.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        raw_states = payload.get("states", payload) if isinstance(payload, dict) else payload
        states = {}
        for key, value in raw_states.items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            task_id, init_state_idx, phase_name = key
            normalized_phase = self._normalize_phase_start_name(phase_name)
            states[(int(task_id), int(init_state_idx), normalized_phase)] = torch.as_tensor(value, dtype=torch.float32).cpu()
        if not states:
            raise RuntimeError(f"Phase-state cache `{cache_path}` did not contain any usable states.")
        self._phase_state_cache = states
        self._phase_state_cache_records = payload.get("records", {}) if isinstance(payload, dict) else {}
        self._phase_state_cache_path = str(cache_path)
        self._phase_state_cache_stats = {
            "path": str(cache_path),
            "num_states": int(len(states)),
            "num_records": int(len(self._phase_state_cache_records)),
            "schema_version": int(payload.get("schema_version", 0)) if isinstance(payload, dict) else 0,
        }

    def _resolve_phase_start_state(self, task_id, init_state_idx, phase_start_name, default_init_state):
        phase_name = self._normalize_phase_start_name(phase_start_name)
        if phase_name == "initial":
            return default_init_state
        key = (int(task_id), int(init_state_idx), phase_name)
        if key not in self._phase_state_cache:
            raise RuntimeError(
                f"Missing cached phase state for task_id={task_id}, init_state_idx={init_state_idx}, phase={phase_name}."
            )
        return self._phase_state_cache[key].detach().cpu().numpy()

    def _get_gt_phase_boundary_ratio_pair(self, task_description):
        instruction_key = self._normalize_instruction_key(task_description)
        ratios = self._gt_phase_boundary_ratios.get(instruction_key, {})
        return clamp_phase_boundary_ratios(
            ratios.get("contact_ratio", 0.3),
            ratios.get("post_ratio", 0.7),
        )

    def _get_phase_boundary_info(self, task_id, init_state_idx):
        phases = ("initial", "contact_manipulate", "post_contact")
        records = {}
        source_T = 0
        for phase_name in phases:
            key = (int(task_id), int(init_state_idx), phase_name)
            record = self._phase_state_cache_records.get(key)
            if not isinstance(record, dict):
                continue
            records[phase_name] = record
            source_T = max(source_T, int(record.get("source_T", 0) or 0))
        if source_T <= 0:
            return None
        contact_record = records.get("contact_manipulate", {})
        post_record = records.get("post_contact", {})
        contact_step = contact_record.get("source_boundary_t")
        post_step = post_record.get("source_boundary_t")
        return {
            "source_T": int(source_T),
            "contact_step": None if contact_step is None else int(contact_step),
            "post_step": None if post_step is None else int(post_step),
        }

    def _resolve_window_rollout_probe_case(
        self,
        task_id,
        init_state_idx,
        phase_scope,
        future_horizon,
        exp_base,
        max_env_steps,
    ):
        boundary_info = self._get_phase_boundary_info(task_id=task_id, init_state_idx=init_state_idx)
        if boundary_info is None:
            return None

        phase_name = normalize_window_rollout_phase_scope(phase_scope)
        window = resolve_phase_window(
            source_T=boundary_info["source_T"],
            contact_step=boundary_info["contact_step"],
            post_step=boundary_info["post_step"],
            phase_name=phase_name,
        )
        if window is None:
            return None

        available_steps_from_start = max(0, int(window["source_T"]) - int(window["window_start_step"]))
        max_total_steps = min(int(max_env_steps), available_steps_from_start)
        window_step_count = min(int(window["window_step_count"]), max_total_steps)
        if window_step_count <= 0:
            return None
        future_budget = max(0, max_total_steps - window_step_count)
        future_steps = min(max(0, int(future_horizon)), future_budget)
        total_rollout_steps = window_step_count + future_steps
        if total_rollout_steps <= 0:
            return None

        return {
            "phase_name": str(window["phase_name"]),
            "window_start_step": int(window["window_start_step"]),
            "window_end_step": int(window["window_end_step"]),
            "window_step_count": int(window_step_count),
            "future_steps": int(future_steps),
            "future_horizon": int(max(0, future_horizon)),
            "exp_base": float(exp_base),
            "source_T": int(window["source_T"]),
            "contact_step": window["contact_step"],
            "post_step": window["post_step"],
            "total_rollout_steps": int(total_rollout_steps),
        }

    def _resolve_window_rollout_phase_scopes(self, phase_scope):
        normalized = normalize_window_rollout_phase_scope(phase_scope)
        if normalized == "all":
            return ["initial", "contact_manipulate", "post_contact"]
        return [normalized]

    @staticmethod
    def _build_fixed_train_window_spec(
        phase_name,
        window_step_count,
        future_horizon,
        exp_base,
        max_env_steps,
    ):
        fixed_window_steps = max(1, int(window_step_count))
        max_total_steps = max(1, int(max_env_steps))
        fixed_window_steps = min(fixed_window_steps, max_total_steps)
        future_budget = max(0, max_total_steps - fixed_window_steps)
        future_steps = min(max(0, int(future_horizon)), future_budget)
        total_rollout_steps = fixed_window_steps + future_steps
        return {
            "phase_name": str(phase_name),
            "window_start_step": 0,
            "window_end_step": int(max(0, fixed_window_steps - 1)),
            "window_step_count": int(fixed_window_steps),
            "future_steps": int(future_steps),
            "future_horizon": int(max(0, int(future_horizon))),
            "exp_base": float(exp_base),
            "source_T": int(total_rollout_steps),
            "contact_step": None,
            "post_step": None,
            "total_rollout_steps": int(total_rollout_steps),
        }

    def _infer_gt_phase_for_step(self, split, task_description, step_idx, horizon, phase_start_name):
        phase_start = self._normalize_phase_start_name(phase_start_name)
        contact_ratio, post_ratio = self._get_gt_phase_boundary_ratio_pair(task_description)
        return infer_gt_phase_for_step(
            step_idx=step_idx,
            horizon=horizon,
            phase_start_name=phase_start,
            contact_ratio=contact_ratio,
            post_ratio=post_ratio,
        )

    def _infer_gt_phase_for_episode_step(
        self,
        split,
        task_description,
        step_idx,
        horizon,
        phase_start_name,
        phase_boundary_info=None,
        absolute_step_idx=None,
    ):
        if phase_boundary_info is not None and absolute_step_idx is not None:
            return infer_phase_name_from_boundaries(
                absolute_step_idx=int(absolute_step_idx),
                source_T=int(phase_boundary_info.get("source_T", 0) or 0),
                contact_step=phase_boundary_info.get("contact_step"),
                post_step=phase_boundary_info.get("post_step"),
            )
        return self._infer_gt_phase_for_step(
            split=split,
            task_description=task_description,
            step_idx=step_idx,
            horizon=horizon,
            phase_start_name=phase_start_name,
        )

    def _gt_cache_batch_transform(self, rlds_batch):
        instruction = rlds_batch["task"]["language_instruction"]
        if isinstance(instruction, bytes):
            instruction_text = instruction.decode()
        else:
            instruction_text = str(instruction)
        action = np.asarray(rlds_batch["action"], dtype=np.float32)
        if action.ndim > 1:
            action = action[0]
        return {
            "instruction_key": self._normalize_instruction_key(instruction_text),
            "normalized_action": np.asarray(action, dtype=np.float32),
        }

    def _fit_continuous_action_dim_tensor(self, action, action_dim):
        vector = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
        if vector.shape[0] == int(action_dim):
            return vector
        out = torch.zeros((int(action_dim),), dtype=torch.float32)
        copy_dim = min(int(action_dim), int(vector.shape[0]))
        if copy_dim > 0:
            out[:copy_dim] = vector[:copy_dim]
        return out

    def _collect_required_gt_instruction_keys(self, task_suite):
        required_keys = set()
        for task_id in range(int(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            instruction_key = self._normalize_instruction_key(getattr(task, "language", ""))
            if instruction_key == "":
                raise ValueError(f"Task {task_id} is missing a valid language instruction for GT cache alignment.")
            required_keys.add(instruction_key)
        if len(required_keys) == 0:
            raise ValueError("No task instructions found while preparing GT action-gap cache.")
        return required_keys

    def _build_gt_expert_cache(self, gt_dataset_root, dataset_name, required_instruction_keys, action_dim):
        try:
            from prismatic.vla.datasets import EpisodicRLDSDataset
        except Exception as exc:
            raise RuntimeError(
                "Failed to import EpisodicRLDSDataset for GT action-gap mode. "
                "Please ensure RLDS dataset dependencies are installed."
            ) from exc

        data_root_dir = Path(os.path.abspath(os.path.expanduser(str(gt_dataset_root))))
        if not data_root_dir.exists():
            raise FileNotFoundError(f"GT dataset root does not exist: `{data_root_dir}`.")

        gt_dataset_name = self._resolve_gt_dataset_name(dataset_name)
        required_keys_sorted = sorted(required_instruction_keys)
        expert_cache = {}
        expert_cache_stats = {}

        for split_name, train_flag in (("train", True), ("val", False)):
            split_cache = {key: [] for key in required_keys_sorted}
            dataset = EpisodicRLDSDataset(
                data_root_dir=data_root_dir,
                data_mix=gt_dataset_name,
                batch_transform=self._gt_cache_batch_transform,
                resize_resolution=(224, 224),
                shuffle_buffer_size=1,
                train=train_flag,
                image_aug=False,
            )
            for episode in dataset:
                if not episode:
                    continue
                instruction_key = str(episode[0].get("instruction_key", ""))
                if instruction_key not in required_instruction_keys:
                    continue
                trajectory_actions = [
                    self._fit_continuous_action_dim_tensor(step["normalized_action"], action_dim=action_dim) for step in episode
                ]
                if len(trajectory_actions) == 0:
                    continue
                split_cache[instruction_key].append(torch.stack(trajectory_actions, dim=0).cpu())

            missing_keys = [key for key in required_keys_sorted if len(split_cache.get(key, [])) == 0]
            if missing_keys:
                preview = ", ".join(missing_keys[:5])
                if len(missing_keys) > 5:
                    preview += ", ..."
                raise RuntimeError(
                    f"Missing GT expert trajectories for split `{split_name}` in `{data_root_dir}` "
                    f"dataset `{gt_dataset_name}`. Missing instructions: {preview}"
                )

            candidate_counts = [len(split_cache[key]) for key in required_keys_sorted]
            mean_count = float(sum(candidate_counts) / float(max(1, len(candidate_counts))))
            expert_cache[split_name] = split_cache
            expert_cache_stats[split_name] = {
                "num_instructions": int(len(required_keys_sorted)),
                "candidate_trajectories_mean": float(mean_count),
                "candidate_trajectories_min": int(min(candidate_counts)),
                "candidate_trajectories_max": int(max(candidate_counts)),
            }

        self._gt_expert_cache = expert_cache
        self._gt_expert_cache_stats = expert_cache_stats
        self._gt_dataset_root = str(data_root_dir)

    def _gt_cache_log_fields(self):
        train_stats = self._gt_expert_cache_stats.get("train", {})
        val_stats = self._gt_expert_cache_stats.get("val", {})
        fields = {
            "gt_dataset_root": str(self._gt_dataset_root),
            "gt_cache_train_num_instructions": int(train_stats.get("num_instructions", 0)),
            "gt_cache_val_num_instructions": int(val_stats.get("num_instructions", 0)),
            "gt_cache_train_candidate_trajectories_mean": float(train_stats.get("candidate_trajectories_mean", 0.0)),
            "gt_cache_train_candidate_trajectories_min": int(train_stats.get("candidate_trajectories_min", 0)),
            "gt_cache_train_candidate_trajectories_max": int(train_stats.get("candidate_trajectories_max", 0)),
            "gt_cache_val_candidate_trajectories_mean": float(val_stats.get("candidate_trajectories_mean", 0.0)),
            "gt_cache_val_candidate_trajectories_min": int(val_stats.get("candidate_trajectories_min", 0)),
            "gt_cache_val_candidate_trajectories_max": int(val_stats.get("candidate_trajectories_max", 0)),
        }
        phase_stats = self._gt_phase_action_bank_stats
        fields.update(
            {
                "gt_action_bank_path": str(self._gt_action_bank_path),
                "gt_action_bank_num_phase_groups": int(phase_stats.get("num_phase_groups", 0)),
                "gt_action_bank_min_group_size": int(phase_stats.get("min_group_size", 0)),
                "gt_action_bank_max_group_size": int(phase_stats.get("max_group_size", 0)),
                "gt_action_bank_mean_group_size": float(phase_stats.get("mean_group_size", 0.0)),
                "gt_action_bank_phase_boundary_ratio_instructions": int(
                    phase_stats.get("phase_boundary_ratio_instructions", 0)
                ),
                "gt_softmin_tau": float(self._gt_softmin_tau),
                "phase_state_cache_path": str(self._phase_state_cache_path),
                "phase_state_cache_num_states": int(self._phase_state_cache_stats.get("num_states", 0)),
                "init_projection_texture_path": str(self._init_projection_texture_path),
                "initial_projection_texture_snapshot_path": str(self._initial_projection_texture_snapshot_path),
                "initial_patch_snapshot_path": str(self._initial_patch_snapshot_path),
            }
        )
        return fields

    def _save_initial_projection_snapshot(self, projection_texture):
        initial_dir = os.path.join(self.save_dir, "initial")
        os.makedirs(initial_dir, exist_ok=True)
        projection_snapshot_path = os.path.join(initial_dir, "projection_texture.pt")
        patch_snapshot_path = os.path.join(initial_dir, "patch.pt")
        snapshot_tensor = projection_texture.detach().cpu()
        torch.save(snapshot_tensor, projection_snapshot_path)
        torch.save(snapshot_tensor, patch_snapshot_path)
        self._initial_projection_texture_snapshot_path = str(projection_snapshot_path)
        self._initial_patch_snapshot_path = str(patch_snapshot_path)

    def _write_run_metadata(self):
        metadata = {
            "run_dir": str(self.save_dir),
            "resume_capable": bool(self._resume_capable),
            "latest_resume_checkpoint": str(self._latest_resume_checkpoint),
            "last_completed_iter": int(self._last_completed_iter),
            "wandb_run_id": str(self._wandb_run_id),
        }
        metadata.update(self._gt_cache_log_fields())
        with open(os.path.join(self.save_dir, "run_metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

    def _get_gt_candidate_actions(self, split, task_description, step_idx, horizon, action_dim, phase_name=None):
        split_name = "val" if str(split).lower().strip() == "val" else "train"
        instruction_key = self._normalize_instruction_key(task_description)
        if self._gt_phase_action_bank:
            gt_phase = self._phase_start_to_gt_phase(phase_name or "initial")
            candidate_actions = self._gt_phase_action_bank.get((instruction_key, gt_phase))
            if candidate_actions is None or candidate_actions.numel() == 0:
                raise RuntimeError(
                    f"Missing GT action-bank candidates for instruction `{instruction_key}` phase `{gt_phase}` "
                    f"under `{self._gt_action_bank_path}`."
                )
            candidate_actions = candidate_actions.to(self.vla.device, dtype=torch.float32)
            if candidate_actions.shape[-1] != int(action_dim):
                candidate_actions = torch.stack(
                    [self._fit_continuous_action_dim_tensor(action, action_dim=int(action_dim)) for action in candidate_actions],
                    dim=0,
                ).to(self.vla.device, dtype=torch.float32)
            return candidate_actions, int(candidate_actions.shape[0]), instruction_key, gt_phase

        split_cache = self._gt_expert_cache.get(split_name, {})
        trajectories = split_cache.get(instruction_key, [])
        if len(trajectories) == 0:
            raise RuntimeError(
                f"Missing GT candidates for split `{split_name}` instruction `{instruction_key}` "
                f"under `{self._gt_dataset_root}`."
            )

        effective_horizon = max(1, int(horizon))
        progress = 0.0 if effective_horizon <= 1 else float(step_idx) / float(max(1, effective_horizon - 1))
        selected_actions = []
        for trajectory in trajectories:
            traj_len = int(trajectory.shape[0])
            candidate_idx = 0 if traj_len <= 1 else int(round(progress * float(traj_len - 1)))
            candidate_idx = max(0, min(traj_len - 1, candidate_idx))
            selected_actions.append(trajectory[candidate_idx])

        candidate_actions = torch.stack(selected_actions, dim=0).to(self.vla.device, dtype=torch.float32)
        if candidate_actions.shape[-1] != int(action_dim):
            candidate_actions = torch.stack(
            [self._fit_continuous_action_dim_tensor(action, action_dim=int(action_dim)) for action in candidate_actions],
                dim=0,
            ).to(self.vla.device, dtype=torch.float32)
        return candidate_actions, int(candidate_actions.shape[0]), instruction_key, self._phase_start_to_gt_phase(phase_name or "initial")

    def _compute_gt_action_gap_losses(
        self,
        adv_logits,
        labels_full,
        action_mask_full,
        gt_candidate_actions,
        maskidx,
        use_all_joints,
        gripper_weight,
        gt_softmin_tau=0.05,
    ):
        batch_size = labels_full.shape[0]
        adv_action_logits = self._extract_action_logits(adv_logits, labels_full)
        seq_len = min(action_mask_full.shape[1], adv_action_logits.shape[1])
        action_mask = action_mask_full[:, :seq_len]
        adv_action_logits = adv_action_logits[:, :seq_len, :]
        adv_pred_tokens = adv_action_logits.argmax(dim=-1) + 31744

        masked_adv_logits = adv_action_logits[action_mask]
        if masked_adv_logits.numel() == 0 or gt_candidate_actions is None or gt_candidate_actions.numel() == 0:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return zero, zero.detach(), per_joint_zero, adv_pred_tokens.detach()

        num_actions = int(masked_adv_logits.shape[0] // max(1, batch_size))
        if num_actions <= 0:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return zero, zero.detach(), per_joint_zero, adv_pred_tokens.detach()

        adv_probs = F.softmax(masked_adv_logits, dim=-1)
        adv_soft_actions = (adv_probs * self.action_bin_centers).sum(dim=-1).view(batch_size, num_actions)
        adv_hard_tokens = masked_adv_logits.argmax(dim=-1) + 31744
        adv_hard_actions = self._decode_action_tokens(adv_hard_tokens, batch_size, num_actions)

        gt_candidate_actions = gt_candidate_actions.to(self.vla.device, dtype=torch.float32)
        if gt_candidate_actions.ndim != 2:
            raise ValueError("GT candidate actions must have shape [num_candidates, action_dim].")
        if gt_candidate_actions.shape[1] != num_actions:
            gt_candidate_actions = torch.stack(
                [self._fit_continuous_action_dim_tensor(action, action_dim=num_actions) for action in gt_candidate_actions],
                dim=0,
            ).to(self.vla.device, dtype=torch.float32)

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

        loss_action_gap, metric_action_gap, per_joint_l1 = self._compute_weighted_action_set_objective(
            adv_soft_actions=adv_soft_actions,
            adv_hard_actions=adv_hard_actions,
            gt_candidate_actions=gt_candidate_actions,
            idx_tensor=idx_tensor,
            joint_weights=joint_weights,
            tau=gt_softmin_tau,
        )
        return loss_action_gap, metric_action_gap.detach(), per_joint_l1.detach(), adv_pred_tokens.detach()

    @staticmethod
    def _compute_weighted_action_set_objective(
        adv_soft_actions,
        adv_hard_actions,
        gt_candidate_actions,
        idx_tensor,
        joint_weights,
        tau,
    ):
        selected_adv_soft = adv_soft_actions.index_select(1, idx_tensor)
        selected_adv_hard = adv_hard_actions.index_select(1, idx_tensor)
        selected_gt_candidates = gt_candidate_actions.index_select(1, idx_tensor)
        weight_denom = joint_weights.sum().clamp(min=1e-6)

        soft_squared = torch.square(selected_adv_soft[:, None, :] - selected_gt_candidates[None, :, :])
        weighted_soft_squared = (soft_squared * joint_weights.view(1, 1, -1)).sum(dim=-1) / weight_denom
        tau = max(float(tau), 1e-6)
        num_candidates = max(1, int(selected_gt_candidates.shape[0]))
        soft_min_distance = -tau * (
            torch.logsumexp(-weighted_soft_squared / tau, dim=1) - math.log(float(num_candidates))
        )
        loss_action_gap = soft_min_distance.mean()

        hard_squared = torch.square(selected_adv_hard[:, None, :] - selected_gt_candidates[None, :, :])
        weighted_hard_squared = (hard_squared * joint_weights.view(1, 1, -1)).sum(dim=-1) / weight_denom
        nearest_idx = weighted_hard_squared.argmin(dim=1)
        nearest_actions = gt_candidate_actions.index_select(0, nearest_idx)
        per_joint_l1 = torch.abs(adv_hard_actions - nearest_actions).mean(dim=0)
        metric_action_gap = weighted_hard_squared.min(dim=1).values.mean()
        return loss_action_gap, metric_action_gap, per_joint_l1

    def _load_siglip_image_model(self, model_name):
        from transformers import SiglipModel

        model = SiglipModel.from_pretrained(str(model_name)).to(self.vla.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _as_siglip_image_batch(self, images):
        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                return torch.empty((0, 3, 1, 1), device=self.vla.device, dtype=torch.float32)
            batch = torch.stack([image.to(self.vla.device, dtype=torch.float32) for image in images], dim=0)
        else:
            batch = images.to(self.vla.device, dtype=torch.float32)
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
        if batch.ndim != 4:
            raise ValueError(f"Expected SigLIP image batch with 3 or 4 dims, got shape {tuple(batch.shape)}.")
        return torch.clamp(batch[:, :3, :, :], 0.0, 1.0)

    def _siglip_resize_center_crop(self, images, input_size):
        input_size = max(1, int(input_size))
        if images.numel() == 0:
            return images
        height = int(images.shape[-2])
        width = int(images.shape[-1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid SigLIP image shape: {tuple(images.shape)}.")
        scale = float(input_size) / float(min(height, width))
        resized_h = max(input_size, int(round(float(height) * scale)))
        resized_w = max(input_size, int(round(float(width) * scale)))
        resized = F.interpolate(images, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        top = max(0, (resized_h - input_size) // 2)
        left = max(0, (resized_w - input_size) // 2)
        return resized[:, :, top : top + input_size, left : left + input_size]

    def _compute_siglip_embedding_distance(self, siglip_model, reference_images, projected_images, input_size=384):
        if siglip_model is None:
            return torch.zeros((), device=self.vla.device, dtype=torch.float32)

        reference_batch = self._siglip_resize_center_crop(
            self._as_siglip_image_batch(reference_images).detach(),
            input_size=input_size,
        )
        projected_batch = self._siglip_resize_center_crop(
            self._as_siglip_image_batch(projected_images),
            input_size=input_size,
        )
        if projected_batch.numel() == 0:
            return torch.zeros((), device=self.vla.device, dtype=torch.float32)

        with torch.no_grad():
            reference_features = siglip_model.get_image_features(pixel_values=reference_batch)
            reference_features = F.normalize(reference_features.float(), dim=-1)
        projected_features = siglip_model.get_image_features(pixel_values=projected_batch)
        projected_features = F.normalize(projected_features.float(), dim=-1)
        cosine_similarity = F.cosine_similarity(reference_features, projected_features, dim=-1)
        return 1.0 - cosine_similarity.mean()

    def online_attack_unconstrained(
        self,
        num_iter=5000,
        patch_size=[3, 50, 50],
        init_projection_texture_path="",
        resume_run_dir="",
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
        lambda_history=0.0,
        lambda_history_legacy=0.0,
        lambda_ce=0.02,
        lambda_ce_phase2=0.0,
        lambda_continuous_rollout=0.0,
        lambda_window_rollout_loss=0.0,
        impulse_rollout_metric_enabled=False,
        window_rollout_probe_enabled=False,
        window_rollout_metric_mode="delta_weighted",
        window_rollout_future_mode="keep_adv",
        window_rollout_exp_base=0.9,
        window_rollout_future_horizon=8,
        window_rollout_phase_scope="all",
        lambda_siglip=0.15,
        siglip_model_name=DEFAULT_SIGLIP_MODEL_NAME,
        siglip_input_size=384,
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
        projection_randomization_enabled=True,
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
        learn_projector_gain=False,
        learn_projector_channel_gain=False,
        photometric_lr_ratio=0.1,
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
        online_train_episodes_per_task=10,
        online_val_episodes=8,
        num_steps_wait=10,
        max_env_steps="auto_by_suite",
        val_max_env_steps=120,
        env_resolution=256,
        online_ce_mode="pseudo_clean",
        env_action_source="adv",
        env_seed=42,
        dataset_name="libero_spatial",
        action_gap_mode="clean_adv",
        gt_dataset_root="/home/yxx/roboticAttack/openvla-main/dataset",
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
        phase_state_mode="initial_only",
        phase_state_cache_path="",
        gt_softmin_tau=0.05,
        gt_action_bank_path="",
    ):
        del env_action_source  # current implementation uses adv action for stepping by default
        self._reset_metric_buffers()
        self._set_online_env_seed(env_seed)
        self._wandb_run_id = str(getattr(args, "wandb_run_id", "") or "")

        attack_mode = str(attack_mode).lower()
        if projection_size is None:
            projection_size = patch_size
        projection_size = [int(x) for x in projection_size]
        resume_run_dir_value = "" if resume_run_dir is None else str(resume_run_dir).strip()
        if resume_run_dir_value.lower() in ("none", "null"):
            resume_run_dir_value = ""
        if resume_run_dir_value != "":
            resume_run_dir_value = os.path.abspath(os.path.expanduser(resume_run_dir_value))

        projection_texture = torch.rand(projection_size, device=self.vla.device)
        projection_texture.requires_grad_(True)
        projection_texture.retain_grad()
        init_projection_path_value = "" if init_projection_texture_path is None else str(init_projection_texture_path).strip()
        if init_projection_path_value.lower() in ("none", "null"):
            init_projection_path_value = ""
        if resume_run_dir_value != "" and init_projection_path_value != "":
            raise ValueError("--resume_run_dir and --init_projection_texture_path are mutually exclusive.")

        current_resume_config = {
            "num_iter": int(num_iter),
            "patch_size": [int(x) for x in patch_size],
            "projection_size": [int(x) for x in projection_size],
            "accumulate_steps": int(accumulate_steps),
            "warmup": int(warmup),
            "phase1_ratio": float(phase1_ratio),
            "phase1_rollout": int(phase1_rollout),
            "phase2_rollout": int(phase2_rollout),
            "learn_projector_gain": bool(learn_projector_gain),
            "learn_projector_channel_gain": bool(learn_projector_channel_gain),
        }
        resume_state = None
        start_iter = 0
        train_phase_start_counter = 0
        resumed_gpu_tuner_state = None
        if resume_run_dir_value != "":
            resume_state = self._load_resume_checkpoint(resume_run_dir_value)
            self._validate_resume_compatibility(current_resume_config, resume_state.get("config", {}))
            with torch.no_grad():
                projection_texture.copy_(
                    torch.as_tensor(
                        resume_state["projection_texture"],
                        device=self.vla.device,
                        dtype=projection_texture.dtype,
                    )
                )
            start_iter = int(resume_state.get("next_iter_idx", int(resume_state.get("global_iter_completed", -1)) + 1))
            train_phase_start_counter = int(resume_state.get("train_phase_start_counter", 0))
            resumed_gpu_tuner_state = resume_state.get("gpu_tuner_state")
            self.best_rollout_score = float(resume_state.get("best_rollout_score", self.best_rollout_score))
            self._load_history_metrics(self.save_dir)
        elif init_projection_path_value != "":
            init_projection_path = Path(os.path.abspath(os.path.expanduser(init_projection_path_value)))
            if not init_projection_path.exists():
                raise FileNotFoundError(f"init_projection_texture_path not found: `{init_projection_path}`.")
            loaded_projection = torch.load(init_projection_path, map_location=self.vla.device)
            loaded_projection = torch.as_tensor(loaded_projection, device=self.vla.device, dtype=projection_texture.dtype)
            if tuple(loaded_projection.shape) != tuple(projection_texture.shape):
                raise ValueError(
                    "Loaded projection texture shape mismatch: "
                    f"expected {tuple(projection_texture.shape)}, got {tuple(loaded_projection.shape)} "
                    f"from `{init_projection_path}`."
                )
            with torch.no_grad():
                projection_texture.copy_(loaded_projection)
            self._init_projection_texture_path = str(init_projection_path)
        else:
            self._init_projection_texture_path = ""
        if resume_run_dir_value == "":
            self._save_initial_projection_snapshot(projection_texture)

        phase1_ratio = float(min(max(phase1_ratio, 0.0), 1.0))
        phase1_end_iter = int(num_iter * phase1_ratio)
        accumulate_steps = max(1, int(accumulate_steps))
        save_interval = max(1, int(save_interval))
        phase1_rollout = max(1, int(phase1_rollout))
        phase2_rollout = max(1, int(phase2_rollout))
        use_all_joints = bool(use_all_joints)
        gripper_weight = float(gripper_weight)
        lambda_action_gap = float(lambda_action_gap)
        lambda_history = float(lambda_history)
        lambda_history_legacy = float(lambda_history_legacy)
        lambda_ce = float(lambda_ce)
        lambda_ce_phase2 = float(lambda_ce_phase2)
        lambda_continuous_rollout = float(lambda_continuous_rollout)
        lambda_window_rollout_loss = float(lambda_window_rollout_loss)
        impulse_rollout_metric_enabled = bool(impulse_rollout_metric_enabled)
        window_rollout_probe_enabled = bool(window_rollout_probe_enabled)
        window_rollout_metric_mode = normalize_window_rollout_metric_mode(window_rollout_metric_mode)
        window_rollout_future_mode = normalize_window_rollout_future_mode(window_rollout_future_mode)
        window_rollout_exp_base = float(window_rollout_exp_base)
        window_rollout_future_horizon = max(0, int(window_rollout_future_horizon))
        window_rollout_phase_scope = normalize_window_rollout_phase_scope(window_rollout_phase_scope)
        lambda_siglip = float(lambda_siglip)
        siglip_model_name = str(siglip_model_name)
        siglip_input_size = max(1, int(siglip_input_size))
        eval_enabled = bool(eval_enabled)
        learn_projector_gain = bool(learn_projector_gain)
        learn_projector_channel_gain = bool(learn_projector_channel_gain)
        photometric_lr_ratio = max(0.0, float(photometric_lr_ratio))
        self.lighting_aug_train_only = bool(lighting_aug_train_only)
        phase1_disable_lighting = bool(phase1_disable_lighting)
        projection_randomization_enabled = bool(projection_randomization_enabled)
        phase1_disable_projection_randomization = bool(phase1_disable_projection_randomization)
        viz_enabled = bool(viz_enabled)
        viz_policy = str(viz_policy).lower()
        viz_samples = max(1, int(viz_samples))
        online_train_tasks_per_iter = max(1, int(online_train_tasks_per_iter))
        online_train_episodes_per_task = max(1, int(online_train_episodes_per_task))
        online_val_episodes = max(1, int(online_val_episodes))
        num_steps_wait = max(0, int(num_steps_wait))
        env_resolution = max(64, int(env_resolution))
        val_max_env_steps = max(1, int(val_max_env_steps))
        dataset_name = str(dataset_name).strip()
        action_gap_mode = str(action_gap_mode).lower().strip()
        if action_gap_mode not in ("clean_adv", "gt_farthest"):
            raise ValueError("--action_gap_mode must be one of {clean_adv, gt_farthest}")
        gt_dataset_root = os.path.abspath(os.path.expanduser(str(gt_dataset_root)))
        gt_softmin_tau = max(float(gt_softmin_tau), 1e-6)
        self._gt_softmin_tau = gt_softmin_tau
        phase_state_mode = self._normalize_phase_state_mode(phase_state_mode)
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
            lambda_ce_phase2 = 0.0

        if self.optimizer != "adamW":
            raise ValueError("UADA_rollout_online_env currently supports optimizer='adamW' only.")

        photometric_params = LearnableProjectorPhotometricParams(
            projector_gain=projector_gain,
            projector_channel_gain=projector_channel_gain,
            learn_projector_gain=learn_projector_gain,
            learn_projector_channel_gain=learn_projector_channel_gain,
            device=self.vla.device,
        )
        optimizer_param_groups = [{"params": [projection_texture], "lr": lr}]
        if photometric_params.has_trainable_params():
            optimizer_param_groups.append(
                {
                    "params": list(photometric_params.parameters()),
                    "lr": float(lr) * float(photometric_lr_ratio),
                }
            )
        optimizer = transformers.AdamW(optimizer_param_groups, lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=max(1, int(num_iter / max(1, accumulate_steps))),
            num_cycles=0.5,
            last_epoch=-1,
        )
        if resume_state is not None:
            photometric_params.load_state_dict(resume_state["photometric_params_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            self._restore_rng_state(resume_state.get("rng_state"))

        siglip_model = None
        if lambda_siglip > 0.0:
            siglip_model = self._load_siglip_image_model(model_name=siglip_model_name)
            print(
                "[OnlineEnv] SigLIP objective enabled "
                f"(model={siglip_model_name}, input_size={siglip_input_size}, lambda_siglip={lambda_siglip})"
            )

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
        self._action_gap_mode = action_gap_mode
        self._gt_dataset_root = gt_dataset_root
        needs_gt_action_bank = (
            (action_gap_mode == "gt_farthest")
            or (lambda_continuous_rollout > 0.0)
            or (lambda_window_rollout_loss > 0.0)
            or bool(impulse_rollout_metric_enabled)
            or bool(window_rollout_probe_enabled)
        )
        if needs_gt_action_bank:
            required_instruction_keys = self._collect_required_gt_instruction_keys(task_suite)
            self._build_gt_phase_action_bank(
                gt_action_bank_path=gt_action_bank_path,
                dataset_name=dataset_name,
                required_instruction_keys=required_instruction_keys,
                action_dim=action_dim,
            )
        else:
            self._gt_expert_cache = {"train": {}, "val": {}}
            self._gt_expert_cache_stats = {"train": {}, "val": {}}
            self._gt_phase_action_bank = {}
            self._gt_phase_action_bank_stats = {}
            self._gt_phase_boundary_ratios = {}
            self._gt_action_bank_path = ""
        if phase_state_mode != "initial_only" or window_rollout_probe_enabled or (lambda_window_rollout_loss > 0.0):
            self._load_phase_state_cache(
                phase_state_cache_path=phase_state_cache_path,
                dataset_name=dataset_name,
            )
        else:
            self._phase_state_cache = {}
            self._phase_state_cache_records = {}
            self._phase_state_cache_stats = {}
            self._phase_state_cache_path = ""
        print(
            "[OnlineEnv] initialized "
            f"(suite={resolved_suite_name}, action_dim={action_dim}, max_env_steps={max_env_steps}, ce_mode={online_ce_mode}, "
            f"val_max_env_steps={val_max_env_steps}, "
            f"action_gap_mode={action_gap_mode}, lambda_ce={lambda_ce}, lambda_ce_phase2={lambda_ce_phase2}, "
            f"lambda_continuous_rollout={lambda_continuous_rollout}, "
            f"lambda_window_rollout_loss={lambda_window_rollout_loss}, "
            f"impulse_rollout_metric_enabled={int(impulse_rollout_metric_enabled)}, "
            f"init_projection_texture_path={init_projection_path_value}, "
            f"window_rollout_probe_enabled={int(window_rollout_probe_enabled)}, "
            f"window_rollout_metric_mode={window_rollout_metric_mode}, "
            f"window_rollout_future_mode={window_rollout_future_mode}, "
            f"window_rollout_exp_base={window_rollout_exp_base}, "
            f"window_rollout_future_horizon={window_rollout_future_horizon}, "
            f"window_rollout_phase_scope={window_rollout_phase_scope}, "
            f"lambda_siglip={lambda_siglip}, gt_dataset_root={gt_dataset_root}, "
            f"gt_action_bank_path={self._gt_action_bank_path}, phase_state_mode={phase_state_mode}, "
            f"phase_state_cache_path={self._phase_state_cache_path}, gt_softmin_tau={gt_softmin_tau}, "
            f"projection_randomization_enabled={int(projection_randomization_enabled)}, "
            f"initial_patch_snapshot_path={self._initial_patch_snapshot_path})"
        )
        self._write_run_metadata()
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
        if isinstance(resumed_gpu_tuner_state, dict):
            gpu_tuner_state.update(resumed_gpu_tuner_state)
        if auto_gpu_tune:
            print(
                "[OnlineEnv] auto_gpu_tune enabled "
                f"(device={gpu_tuner_state['device_index']}, mem_low={gpu_mem_low}, mem_high={gpu_mem_high}, mem_hard_cap={gpu_mem_hard_cap})"
            )

        optimizer.zero_grad()
        for i in tqdm(range(start_iter, num_iter)):
            phase_id = 1 if i < phase1_end_iter else 2
            effective_lambda_ce = lambda_ce if phase_id == 1 else lambda_ce_phase2
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
            train_projection_randomization_enabled = bool(
                projection_randomization_enabled
                and (not bool((phase_id == 1) and phase1_disable_projection_randomization))
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
            train_gt_action_gap = 0.0
            train_history_div = 0.0
            train_history_div_legacy = 0.0
            train_ce = 0.0
            train_ce_objective = 0.0
            train_siglip_distance = 0.0
            train_continuous_clean_gt_action_gap = 0.0
            train_continuous_adv_gt_action_gap = 0.0
            train_continuous_rollout_delta = 0.0
            train_impulse_rollout_area = 0.0
            train_window_rollout_clean_gt_action_gap = 0.0
            train_window_rollout_adv_gt_action_gap = 0.0
            train_window_rollout_deattack_gt_action_gap = 0.0
            train_window_rollout_selected_gt_action_gap = 0.0
            train_window_rollout_delta_weighted = 0.0
            train_window_rollout_delta_weighted_loss = 0.0
            train_proj_alpha = 0.0
            train_proj_coverage = 0.0
            train_proj_bottom = 0.0
            train_proj_keystone = 0.0
            train_action_terms = 0
            train_history_terms = 0
            train_episodes_done = 0.0
            train_episode_len = 0.0
            train_per_joint_gap = None
            train_gt_per_joint_gap = None
            train_episode_count = 0
            train_phase_start_counts = {"initial": 0, "contact_manipulate": 0, "post_contact": 0}

            for task_id in train_task_ids:
                task = task_suite.get_task(task_id)
                init_states = task_suite.get_task_init_states(task_id)
                env, task_description = get_libero_env(task, "openvla", resolution=env_resolution)
                try:
                    for episode_idx in range(online_train_episodes_per_task):
                        init_state_idx = self._sample_init_state_index(init_states, i, episode_idx)
                        if init_state_idx is None:
                            continue
                        init_state = init_states[init_state_idx]
                        phase_start_name = self._phase_start_for_counter(phase_state_mode, train_phase_start_counter)
                        train_phase_start_counter += 1
                        train_phase_start_counts[phase_start_name] = train_phase_start_counts.get(phase_start_name, 0) + 1
                        phase_init_state = self._resolve_phase_start_state(
                            task_id=task_id,
                            init_state_idx=init_state_idx,
                            phase_start_name=phase_start_name,
                            default_init_state=init_state,
                        )
                        train_window_rollout_spec = None
                        if window_rollout_probe_enabled or (lambda_window_rollout_loss > 0.0):
                            # Train-time window length follows current phase rollout budget:
                            # phase1 -> phase1_rollout, phase2 -> phase2_rollout, then append future k steps.
                            train_window_rollout_spec = self._build_fixed_train_window_spec(
                                phase_name=phase_start_name,
                                window_step_count=rollout_steps,
                                future_horizon=window_rollout_future_horizon,
                                exp_base=window_rollout_exp_base,
                                max_env_steps=max_env_steps,
                            )
                        episode_rollout_steps = int(rollout_steps)
                        if isinstance(train_window_rollout_spec, dict):
                            episode_rollout_steps = int(
                                train_window_rollout_spec.get("total_rollout_steps", episode_rollout_steps)
                            )
                        episode = self._run_online_episode(
                            env=env,
                            task=task,
                            get_libero_env=get_libero_env,
                            get_libero_image=get_libero_image,
                            get_libero_dummy_action=get_libero_dummy_action,
                            init_state=phase_init_state,
                            init_state_idx=init_state_idx,
                            task_description=task_description,
                            projection_texture=projection_texture,
                            rollout_steps=episode_rollout_steps,
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
                            lambda_ce=effective_lambda_ce,
                            lambda_continuous_rollout=lambda_continuous_rollout,
                            lambda_window_rollout_loss=lambda_window_rollout_loss,
                            impulse_rollout_metric_enabled=impulse_rollout_metric_enabled,
                            window_rollout_probe_enabled=bool(train_window_rollout_spec is not None),
                            window_rollout_metric_mode=window_rollout_metric_mode,
                            window_rollout_future_mode=window_rollout_future_mode,
                            window_rollout_spec=train_window_rollout_spec,
                            lambda_siglip=lambda_siglip,
                            siglip_model=siglip_model,
                            siglip_input_size=siglip_input_size,
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
                            learnable_projector_params=photometric_params,
                            projector_ambient=projector_ambient,
                            projector_vignetting=projector_vignetting,
                            projector_distance_falloff=projector_distance_falloff,
                            projector_psf=projector_psf,
                            env_resolution=env_resolution,
                            projection_randomization_enabled=train_projection_randomization_enabled,
                            need_backward=True,
                            capture_visual=False,
                            deterministic_episode_seed=None,
                            disable_lighting=train_disable_lighting,
                            task_id=task_id,
                            record_video_frames=bool(should_record_train_video and (train_video_episode is None)),
                            video_frame_source=record_online_video_frame_source,
                            action_gap_mode=action_gap_mode,
                            start_phase_name=phase_start_name,
                            gt_softmin_tau=gt_softmin_tau,
                        )
                        if episode is None:
                            continue

                        train_episode_count += 1
                        train_action_gap += episode["action_gap"]
                        train_gt_action_gap += episode["gt_action_gap"]
                        train_history_div += episode["history_div"]
                        train_history_div_legacy += episode["history_div_legacy"]
                        train_ce += episode["ce_value"]
                        train_ce_objective += episode["ce_objective_value"]
                        train_siglip_distance += episode["siglip_distance"]
                        train_continuous_clean_gt_action_gap += episode["continuous_clean_gt_action_gap"]
                        train_continuous_adv_gt_action_gap += episode["continuous_adv_gt_action_gap"]
                        train_continuous_rollout_delta += episode["continuous_rollout_delta"]
                        train_impulse_rollout_area += episode["impulse_rollout_area"]
                        train_window_rollout_clean_gt_action_gap += episode["window_rollout_clean_gt_action_gap"]
                        train_window_rollout_adv_gt_action_gap += episode["window_rollout_adv_gt_action_gap"]
                        train_window_rollout_deattack_gt_action_gap += episode["window_rollout_deattack_gt_action_gap"]
                        train_window_rollout_selected_gt_action_gap += episode["window_rollout_selected_gt_action_gap"]
                        train_window_rollout_delta_weighted += episode["window_rollout_delta_weighted"]
                        train_window_rollout_delta_weighted_loss += episode["window_rollout_delta_weighted_loss"]
                        train_proj_alpha += episode["projection_alpha"]
                        train_proj_coverage += episode["projection_coverage"]
                        train_proj_bottom += episode["projection_bottom"]
                        train_proj_keystone += episode["projection_keystone"]
                        train_action_terms += max(1, episode["action_terms"])
                        train_history_terms += max(0, episode["history_terms"])
                        train_episodes_done += float(episode["done"])
                        train_episode_len += float(episode["episode_len"])
                        train_per_joint_gap = self._accumulate_joint_values(train_per_joint_gap, episode["per_joint_gap"])
                        train_gt_per_joint_gap = self._accumulate_joint_values(train_gt_per_joint_gap, episode["gt_per_joint_gap"])
                        if should_record_train_video and (train_video_episode is None) and self._should_dump_online_episode_video(episode):
                            train_video_episode = episode
                finally:
                    if hasattr(env, "close"):
                        env.close()

            if train_episode_count <= 0:
                print(f"[OnlineEnv] iter {i}: no valid episodes processed, skipping optimizer step.")
                continue

            grad_scale = float(max(1, train_episode_count * accumulate_steps))
            if projection_texture.grad is not None:
                projection_texture.grad.div_(grad_scale)
            if photometric_params.has_trainable_params():
                for param in photometric_params.parameters():
                    if param.grad is not None:
                        param.grad.div_(grad_scale)

            log_patch_grad = 0.0
            if projection_texture.grad is not None:
                log_patch_grad = projection_texture.grad.detach().abs().mean().item()

            optimizer_step = ((i + 1) % accumulate_steps == 0) or ((i + 1) == num_iter)
            if optimizer_step:
                optimizer.step()
                projection_texture.data = projection_texture.data.clamp(0, 1)
                optimizer.zero_grad()
                scheduler.step()

            current_projector_gain, current_projector_channel_gain = photometric_params.resolved_values()
            current_projector_gain_value = float(current_projector_gain.detach().cpu().item())
            current_projector_channel_gain_values = [
                float(v) for v in current_projector_channel_gain.detach().cpu().tolist()
            ]

            avg_action_gap = train_action_gap / float(max(1, train_episode_count))
            avg_gt_action_gap = train_gt_action_gap / float(max(1, train_episode_count))
            avg_history_div = train_history_div / float(max(1, train_episode_count))
            avg_history_div_legacy = train_history_div_legacy / float(max(1, train_episode_count))
            avg_ce = train_ce / float(max(1, train_episode_count))
            avg_ce_objective = train_ce_objective / float(max(1, train_episode_count))
            avg_siglip_distance = train_siglip_distance / float(max(1, train_episode_count))
            avg_continuous_clean_gt_action_gap = train_continuous_clean_gt_action_gap / float(max(1, train_episode_count))
            avg_continuous_adv_gt_action_gap = train_continuous_adv_gt_action_gap / float(max(1, train_episode_count))
            avg_continuous_rollout_delta = train_continuous_rollout_delta / float(max(1, train_episode_count))
            avg_impulse_rollout_area = train_impulse_rollout_area / float(max(1, train_episode_count))
            avg_window_rollout_clean_gt_action_gap = train_window_rollout_clean_gt_action_gap / float(
                max(1, train_episode_count)
            )
            avg_window_rollout_adv_gt_action_gap = train_window_rollout_adv_gt_action_gap / float(
                max(1, train_episode_count)
            )
            avg_window_rollout_deattack_gt_action_gap = train_window_rollout_deattack_gt_action_gap / float(
                max(1, train_episode_count)
            )
            avg_window_rollout_selected_gt_action_gap = train_window_rollout_selected_gt_action_gap / float(
                max(1, train_episode_count)
            )
            avg_window_rollout_delta_weighted = train_window_rollout_delta_weighted / float(max(1, train_episode_count))
            avg_window_rollout_delta_weighted_loss = train_window_rollout_delta_weighted_loss / float(
                max(1, train_episode_count)
            )
            avg_window_rollout_metric_value = select_window_rollout_metric_value(
                metric_mode=window_rollout_metric_mode,
                delta_weighted=avg_window_rollout_delta_weighted,
                adv_gt_action_gap=avg_window_rollout_selected_gt_action_gap,
            )
            avg_proj_alpha = train_proj_alpha / float(max(1, train_episode_count))
            avg_proj_coverage = train_proj_coverage / float(max(1, train_episode_count))
            avg_proj_bottom = train_proj_bottom / float(max(1, train_episode_count))
            avg_proj_keystone = train_proj_keystone / float(max(1, train_episode_count))
            avg_done_rate = train_episodes_done / float(max(1, train_episode_count))
            avg_ep_len = train_episode_len / float(max(1, train_episode_count))
            avg_per_joint_gap = self._normalize_joint_values(train_per_joint_gap, float(max(1, train_episode_count)))
            avg_gt_per_joint_gap = self._normalize_joint_values(train_gt_per_joint_gap, float(max(1, train_episode_count)))
            rollout_score = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
                + (lambda_siglip * avg_siglip_distance)
            )
            gt_rollout_score = (
                (lambda_action_gap * avg_gt_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
                + (lambda_siglip * avg_siglip_distance)
            )
            objective_score = rollout_score - (effective_lambda_ce * avg_ce_objective)
            gt_objective_score = gt_rollout_score - (effective_lambda_ce * avg_ce_objective)
            active_action_gap = avg_gt_action_gap if action_gap_mode == "gt_farthest" else avg_action_gap
            active_rollout_score = gt_rollout_score if action_gap_mode == "gt_farthest" else rollout_score
            active_objective_score = gt_objective_score if action_gap_mode == "gt_farthest" else objective_score
            total_rollout_score = (
                active_rollout_score
                + (lambda_continuous_rollout * avg_continuous_rollout_delta)
                + (lambda_window_rollout_loss * avg_window_rollout_metric_value)
            )
            total_objective_score = total_rollout_score - (effective_lambda_ce * avg_ce_objective)
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
            self.train_rollout_siglip_distance.append(avg_siglip_distance)
            self.train_rollout_score.append(rollout_score)
            self.train_rollout_objective_score.append(objective_score)
            self.train_gt_rollout_action_gap.append(avg_gt_action_gap)
            self.train_gt_rollout_action_gap_joints.append(avg_gt_per_joint_gap.detach().cpu().tolist())
            self.train_gt_rollout_score.append(gt_rollout_score)
            self.train_gt_rollout_objective_score.append(gt_objective_score)
            self.train_continuous_clean_gt_action_gap.append(avg_continuous_clean_gt_action_gap)
            self.train_continuous_adv_gt_action_gap.append(avg_continuous_adv_gt_action_gap)
            self.train_continuous_rollout_delta.append(avg_continuous_rollout_delta)
            self.train_impulse_rollout_area.append(avg_impulse_rollout_area)
            self.train_window_rollout_clean_gt_action_gap.append(avg_window_rollout_clean_gt_action_gap)
            self.train_window_rollout_adv_gt_action_gap.append(avg_window_rollout_adv_gt_action_gap)
            self.train_window_rollout_deattack_gt_action_gap.append(avg_window_rollout_deattack_gt_action_gap)
            self.train_window_rollout_selected_gt_action_gap.append(avg_window_rollout_selected_gt_action_gap)
            self.train_window_rollout_delta_weighted.append(avg_window_rollout_delta_weighted)
            self.train_window_rollout_delta_weighted_loss.append(avg_window_rollout_delta_weighted_loss)
            self.train_total_rollout_score.append(total_rollout_score)
            self.train_total_objective_score.append(total_objective_score)
            self.train_active_rollout_action_gap.append(active_action_gap)
            self.train_active_rollout_score.append(active_rollout_score)
            self.train_active_rollout_objective_score.append(active_objective_score)
            self.train_phase_id.append(phase_id)
            self.train_online_done_rate.append(avg_done_rate)
            self.train_online_episode_len.append(avg_ep_len)
            self.loss_buffer.append(total_rollout_score)

            train_logdata = {
                "TRAIN_online_rollout_action_gap": avg_action_gap,
                "TRAIN_online_gt_action_gap": avg_gt_action_gap,
                "TRAIN_online_rollout_history_div": avg_history_div,
                "TRAIN_online_rollout_history_div_legacy": avg_history_div_legacy,
                "TRAIN_online_siglip_distance": avg_siglip_distance,
                "TRAIN_online_rollout_score": rollout_score,
                "TRAIN_online_gt_rollout_score": gt_rollout_score,
                "TRAIN_online_objective_score": objective_score,
                "TRAIN_online_gt_objective_score": gt_objective_score,
                "TRAIN_action_gap_mode_active": str(action_gap_mode),
                "TRAIN_online_active_action_gap": active_action_gap,
                "TRAIN_online_active_rollout_score": active_rollout_score,
                "TRAIN_online_active_objective_score": active_objective_score,
                "TRAIN_online_done_rate": avg_done_rate,
                "TRAIN_online_episode_len": avg_ep_len,
                "TRAIN_online_ce": avg_ce,
                "TRAIN_online_ce_objective": avg_ce_objective,
                "TRAIN_lambda_ce_effective": float(effective_lambda_ce),
                "TRAIN_lambda_continuous_rollout": float(lambda_continuous_rollout),
                "TRAIN_lambda_window_rollout_loss": float(lambda_window_rollout_loss),
                "TRAIN_impulse_rollout_metric_enabled": int(impulse_rollout_metric_enabled),
                "TRAIN_online_continuous_clean_gt_action_gap": avg_continuous_clean_gt_action_gap,
                "TRAIN_online_continuous_adv_gt_action_gap": avg_continuous_adv_gt_action_gap,
                "TRAIN_online_continuous_rollout_delta": avg_continuous_rollout_delta,
                "TRAIN_online_impulse_rollout_area": avg_impulse_rollout_area,
                "TRAIN_online_window_rollout_clean_gt_action_gap": avg_window_rollout_clean_gt_action_gap,
                "TRAIN_online_window_rollout_adv_gt_action_gap": avg_window_rollout_adv_gt_action_gap,
                "TRAIN_online_window_rollout_deattack_gt_action_gap": avg_window_rollout_deattack_gt_action_gap,
                "TRAIN_online_window_rollout_selected_gt_action_gap": avg_window_rollout_selected_gt_action_gap,
                "TRAIN_online_window_rollout_delta_weighted": avg_window_rollout_delta_weighted,
                "TRAIN_online_window_rollout_delta_weighted_loss": avg_window_rollout_delta_weighted_loss,
                "TRAIN_online_window_rollout_metric_mode": str(window_rollout_metric_mode),
                "TRAIN_online_window_rollout_future_mode": str(window_rollout_future_mode),
                "TRAIN_online_window_rollout_metric_value": avg_window_rollout_metric_value,
                "TRAIN_online_total_rollout_score": total_rollout_score,
                "TRAIN_online_total_objective_score": total_objective_score,
                "TRAIN_patch_gradient": log_patch_grad,
                "TRAIN_LR": optimizer.param_groups[0]["lr"],
                "TRAIN_projection_alpha_mean": avg_proj_alpha,
                "TRAIN_projection_coverage_ratio": avg_proj_coverage,
                "TRAIN_projection_bottom_ratio": avg_proj_bottom,
                "TRAIN_projection_keystone": avg_proj_keystone,
                "TRAIN_projector_gain": current_projector_gain_value,
                "TRAIN_projector_channel_gain_r": current_projector_channel_gain_values[0],
                "TRAIN_projector_channel_gain_g": current_projector_channel_gain_values[1],
                "TRAIN_projector_channel_gain_b": current_projector_channel_gain_values[2],
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
                "TRAIN_phase_state_mode": str(phase_state_mode),
                "TRAIN_phase_start_initial_count": int(train_phase_start_counts.get("initial", 0)),
                "TRAIN_phase_start_contact_manipulate_count": int(
                    train_phase_start_counts.get("contact_manipulate", 0)
                ),
                "TRAIN_phase_start_post_contact_count": int(train_phase_start_counts.get("post_contact", 0)),
            }
            for joint_idx, joint_gap_value in enumerate(avg_per_joint_gap.detach().cpu().tolist()):
                train_logdata[f"TRAIN_online_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
            for joint_idx, joint_gap_value in enumerate(avg_gt_per_joint_gap.detach().cpu().tolist()):
                train_logdata[f"TRAIN_online_gt_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
            train_logdata.update(self._gt_cache_log_fields())
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
                        "gt_action_gap": float(avg_gt_action_gap),
                        "history1": float(avg_history_div),
                        "history2": float(avg_history_div_legacy),
                        "ce": float(avg_ce),
                        "ce_objective": float(avg_ce_objective),
                        "siglip_distance": float(avg_siglip_distance),
                        "continuous_clean_gt_action_gap": float(avg_continuous_clean_gt_action_gap),
                        "continuous_adv_gt_action_gap": float(avg_continuous_adv_gt_action_gap),
                        "continuous_rollout_delta": float(avg_continuous_rollout_delta),
                        "impulse_rollout_area": float(avg_impulse_rollout_area),
                        "window_rollout_clean_gt_action_gap": float(avg_window_rollout_clean_gt_action_gap),
                        "window_rollout_adv_gt_action_gap": float(avg_window_rollout_adv_gt_action_gap),
                        "window_rollout_deattack_gt_action_gap": float(avg_window_rollout_deattack_gt_action_gap),
                        "window_rollout_selected_gt_action_gap": float(avg_window_rollout_selected_gt_action_gap),
                        "window_rollout_delta_weighted": float(avg_window_rollout_delta_weighted),
                        "window_rollout_delta_weighted_loss": float(avg_window_rollout_delta_weighted_loss),
                        "window_rollout_future_steps": "",
                        "window_phase_name": "",
                        "window_start_step": "",
                        "window_end_step": "",
                        "lambda_window_rollout_loss": float(lambda_window_rollout_loss),
                        "window_rollout_metric_mode": str(window_rollout_metric_mode),
                        "window_rollout_future_mode": str(window_rollout_future_mode),
                        "window_rollout_metric_value": float(avg_window_rollout_metric_value),
                        "rollout_score": float(rollout_score),
                        "gt_rollout_score": float(gt_rollout_score),
                        "objective_score": float(objective_score),
                        "gt_objective_score": float(gt_objective_score),
                        "action_gap_mode_active": str(action_gap_mode),
                        "active_action_gap": float(active_action_gap),
                        "active_rollout_score": float(active_rollout_score),
                        "active_objective_score": float(active_objective_score),
                        "total_rollout_score": float(total_rollout_score),
                        "total_objective_score": float(total_objective_score),
                        "episode_len": float(avg_ep_len),
                        "done_rate": float(avg_done_rate),
                        "action_gap_joint_0": float(avg_per_joint_gap[0].item()) if avg_per_joint_gap.numel() > 0 else "",
                        "action_gap_joint_1": float(avg_per_joint_gap[1].item()) if avg_per_joint_gap.numel() > 1 else "",
                        "action_gap_joint_2": float(avg_per_joint_gap[2].item()) if avg_per_joint_gap.numel() > 2 else "",
                        "gt_action_gap_joint_0": float(avg_gt_per_joint_gap[0].item()) if avg_gt_per_joint_gap.numel() > 0 else "",
                        "gt_action_gap_joint_1": float(avg_gt_per_joint_gap[1].item()) if avg_gt_per_joint_gap.numel() > 1 else "",
                        "gt_action_gap_joint_2": float(avg_gt_per_joint_gap[2].item()) if avg_gt_per_joint_gap.numel() > 2 else "",
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
                eval_projector_gain, eval_projector_channel_gain = photometric_params.resolved_values()
                eval_max_env_steps = int(min(int(max_env_steps), int(val_max_env_steps)))
                eval_rollout_steps = int(eval_max_env_steps)
                eval_projection_randomization_enabled = bool(
                    (str(attack_mode).lower() == "projection") and projection_randomization_enabled
                )
                val_stats, visual_frames, val_video_episodes = self._evaluate_online_rollout(
                    task_suite=task_suite,
                    get_libero_env=get_libero_env,
                    get_libero_image=get_libero_image,
                    get_libero_dummy_action=get_libero_dummy_action,
                    projection_texture=projection_texture,
                    rollout_steps=eval_rollout_steps,
                    max_env_steps=eval_max_env_steps,
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
                    lambda_ce=effective_lambda_ce,
                    lambda_continuous_rollout=lambda_continuous_rollout,
                    lambda_window_rollout_loss=lambda_window_rollout_loss,
                    impulse_rollout_metric_enabled=impulse_rollout_metric_enabled,
                    window_rollout_probe_enabled=window_rollout_probe_enabled,
                    window_rollout_metric_mode=window_rollout_metric_mode,
                    window_rollout_future_mode=window_rollout_future_mode,
                    window_rollout_exp_base=window_rollout_exp_base,
                    window_rollout_future_horizon=window_rollout_future_horizon,
                    window_rollout_phase_scope=window_rollout_phase_scope,
                    lambda_siglip=lambda_siglip,
                    siglip_model=siglip_model,
                    siglip_input_size=siglip_input_size,
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
                    projector_gain=eval_projector_gain,
                    projector_channel_gain=eval_projector_channel_gain,
                    projector_ambient=projector_ambient,
                    projector_vignetting=projector_vignetting,
                    projector_distance_falloff=projector_distance_falloff,
                    projector_psf=projector_psf,
                    env_resolution=env_resolution,
                    viz_samples=viz_samples,
                    val_deterministic=val_deterministic,
                    val_seed=val_seed,
                    val_disable_lighting=val_disable_lighting,
                    projection_randomization_enabled=eval_projection_randomization_enabled,
                    probe_mode=probe_mode,
                    record_video_frames=should_record_val_video,
                    video_frame_source=record_online_video_frame_source,
                    action_gap_mode=action_gap_mode,
                    gt_softmin_tau=gt_softmin_tau,
                )
                val_stats["VAL_effective_lighting_enabled"] = int(
                    self._is_lighting_enabled_for_split("val") and (not bool(val_disable_lighting))
                )
                val_stats["VAL_effective_projection_randomization_enabled"] = int(
                    eval_projection_randomization_enabled
                )
                val_stats["VAL_lambda_ce_effective"] = float(effective_lambda_ce)
                val_stats["VAL_autotune_enabled"] = int(auto_gpu_tune)
                val_stats["VAL_autotune_level_snapshot"] = int(tune_level_for_iter)

                self.val_rollout_action_gap.append(val_stats["VAL_online_rollout_action_gap"])
                self.val_rollout_action_gap_joints.append(
                    self._extract_joint_metric_list(val_stats, prefix="VAL_online_rollout_action_gap_joint_")
                )
                self.val_gt_rollout_action_gap.append(val_stats["VAL_online_gt_action_gap"])
                self.val_gt_rollout_action_gap_joints.append(
                    self._extract_joint_metric_list(val_stats, prefix="VAL_online_gt_action_gap_joint_")
                )
                self.val_rollout_history_div.append(val_stats["VAL_online_rollout_history_div"])
                self.val_rollout_history_div_legacy.append(val_stats["VAL_online_rollout_history_div_legacy"])
                self.val_rollout_siglip_distance.append(val_stats["VAL_online_siglip_distance"])
                self.val_rollout_score.append(val_stats["VAL_online_rollout_score"])
                self.val_rollout_objective_score.append(val_stats["VAL_online_objective_score"])
                self.val_gt_rollout_score.append(val_stats["VAL_online_gt_rollout_score"])
                self.val_gt_rollout_objective_score.append(val_stats["VAL_online_gt_objective_score"])
                self.val_continuous_clean_gt_action_gap.append(
                    val_stats["VAL_online_continuous_clean_gt_action_gap"]
                )
                self.val_continuous_adv_gt_action_gap.append(
                    val_stats["VAL_online_continuous_adv_gt_action_gap"]
                )
                self.val_continuous_rollout_delta.append(val_stats["VAL_online_continuous_rollout_delta"])
                self.val_impulse_rollout_area.append(val_stats["VAL_online_impulse_rollout_area"])
                self.val_window_rollout_clean_gt_action_gap.append(
                    val_stats["VAL_online_window_rollout_clean_gt_action_gap"]
                )
                self.val_window_rollout_adv_gt_action_gap.append(
                    val_stats["VAL_online_window_rollout_adv_gt_action_gap"]
                )
                self.val_window_rollout_deattack_gt_action_gap.append(
                    val_stats["VAL_online_window_rollout_deattack_gt_action_gap"]
                )
                self.val_window_rollout_selected_gt_action_gap.append(
                    val_stats["VAL_online_window_rollout_selected_gt_action_gap"]
                )
                self.val_window_rollout_delta_weighted.append(val_stats["VAL_online_window_rollout_delta_weighted"])
                self.val_window_rollout_delta_weighted_loss.append(
                    val_stats["VAL_online_window_rollout_delta_weighted_loss"]
                )
                self.val_total_rollout_score.append(val_stats["VAL_online_total_rollout_score"])
                self.val_total_objective_score.append(val_stats["VAL_online_total_objective_score"])
                self.val_active_rollout_action_gap.append(val_stats["VAL_online_active_action_gap"])
                self.val_active_rollout_score.append(val_stats["VAL_online_active_rollout_score"])
                self.val_active_rollout_objective_score.append(val_stats["VAL_online_active_objective_score"])
                self.val_rollout_score_legacy.append(val_stats["VAL_online_rollout_score_legacy"])
                self.val_online_done_rate.append(val_stats["VAL_online_done_rate"])
                self.val_online_episode_len.append(val_stats["VAL_online_episode_len"])

                if args is not None and args.wandb_project != "false" and wandb is not None:
                    wandb.log(val_stats, step=i)

                if should_record_val_video:
                    for val_video_episode in val_video_episodes:
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
                            "horizon_type": (
                                "full_horizon_with_phase_window_metric"
                                if window_rollout_probe_enabled
                                else "full_horizon"
                            ),
                            "action_gap": float(val_stats["VAL_online_rollout_action_gap"]),
                            "gt_action_gap": float(val_stats["VAL_online_gt_action_gap"]),
                            "history1": float(val_stats["VAL_online_rollout_history_div"]),
                            "history2": float(val_stats["VAL_online_rollout_history_div_legacy"]),
                            "ce": float(val_stats["VAL_online_ce"]),
                            "ce_objective": float(val_stats["VAL_online_ce_objective"]),
                            "siglip_distance": float(val_stats["VAL_online_siglip_distance"]),
                            "continuous_clean_gt_action_gap": float(
                                val_stats["VAL_online_continuous_clean_gt_action_gap"]
                            ),
                            "continuous_adv_gt_action_gap": float(
                                val_stats["VAL_online_continuous_adv_gt_action_gap"]
                            ),
                            "continuous_rollout_delta": float(val_stats["VAL_online_continuous_rollout_delta"]),
                            "impulse_rollout_area": float(val_stats["VAL_online_impulse_rollout_area"]),
                            "window_rollout_clean_gt_action_gap": float(
                                val_stats["VAL_online_window_rollout_clean_gt_action_gap"]
                            ),
                            "window_rollout_adv_gt_action_gap": float(
                                val_stats["VAL_online_window_rollout_adv_gt_action_gap"]
                            ),
                            "window_rollout_deattack_gt_action_gap": float(
                                val_stats["VAL_online_window_rollout_deattack_gt_action_gap"]
                            ),
                            "window_rollout_selected_gt_action_gap": float(
                                val_stats["VAL_online_window_rollout_selected_gt_action_gap"]
                            ),
                            "window_rollout_delta_weighted": float(
                                val_stats["VAL_online_window_rollout_delta_weighted"]
                            ),
                            "window_rollout_delta_weighted_loss": float(
                                val_stats["VAL_online_window_rollout_delta_weighted_loss"]
                            ),
                            "window_rollout_future_steps": float(
                                val_stats["VAL_online_window_rollout_future_steps"]
                            ),
                            "window_phase_name": str(val_stats["VAL_online_window_phase_name"]),
                            "window_start_step": float(val_stats["VAL_online_window_start_step"]),
                            "window_end_step": float(val_stats["VAL_online_window_end_step"]),
                            "lambda_window_rollout_loss": float(lambda_window_rollout_loss),
                            "window_rollout_metric_mode": str(window_rollout_metric_mode),
                            "window_rollout_future_mode": str(window_rollout_future_mode),
                            "window_rollout_metric_value": float(
                                val_stats["VAL_online_window_rollout_metric_value"]
                            ),
                            "rollout_score": float(val_stats["VAL_online_rollout_score"]),
                            "gt_rollout_score": float(val_stats["VAL_online_gt_rollout_score"]),
                            "objective_score": float(val_stats["VAL_online_objective_score"]),
                            "gt_objective_score": float(val_stats["VAL_online_gt_objective_score"]),
                            "action_gap_mode_active": str(action_gap_mode),
                            "active_action_gap": float(val_stats["VAL_online_active_action_gap"]),
                            "active_rollout_score": float(val_stats["VAL_online_active_rollout_score"]),
                            "active_objective_score": float(val_stats["VAL_online_active_objective_score"]),
                            "total_rollout_score": float(val_stats["VAL_online_total_rollout_score"]),
                            "total_objective_score": float(val_stats["VAL_online_total_objective_score"]),
                            "episode_len": float(val_stats["VAL_online_episode_len"]),
                            "done_rate": float(val_stats["VAL_online_done_rate"]),
                            "action_gap_joint_0": float(val_stats.get("VAL_online_rollout_action_gap_joint_0", "")) if "VAL_online_rollout_action_gap_joint_0" in val_stats else "",
                            "action_gap_joint_1": float(val_stats.get("VAL_online_rollout_action_gap_joint_1", "")) if "VAL_online_rollout_action_gap_joint_1" in val_stats else "",
                            "action_gap_joint_2": float(val_stats.get("VAL_online_rollout_action_gap_joint_2", "")) if "VAL_online_rollout_action_gap_joint_2" in val_stats else "",
                            "gt_action_gap_joint_0": float(val_stats.get("VAL_online_gt_action_gap_joint_0", "")) if "VAL_online_gt_action_gap_joint_0" in val_stats else "",
                            "gt_action_gap_joint_1": float(val_stats.get("VAL_online_gt_action_gap_joint_1", "")) if "VAL_online_gt_action_gap_joint_1" in val_stats else "",
                            "gt_action_gap_joint_2": float(val_stats.get("VAL_online_gt_action_gap_joint_2", "")) if "VAL_online_gt_action_gap_joint_2" in val_stats else "",
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
                                "lambda_ce": float(effective_lambda_ce),
                                "lambda_ce_configured": float(lambda_ce),
                                "lambda_ce_phase2": float(lambda_ce_phase2),
                                "lambda_continuous_rollout": float(lambda_continuous_rollout),
                                "lambda_window_rollout_loss": float(lambda_window_rollout_loss),
                                "impulse_rollout_metric_enabled": int(impulse_rollout_metric_enabled),
                                "window_rollout_probe_enabled": int(window_rollout_probe_enabled),
                                "window_rollout_metric_mode": str(window_rollout_metric_mode),
                                "window_rollout_future_mode": str(window_rollout_future_mode),
                                "window_rollout_exp_base": float(window_rollout_exp_base),
                                "window_rollout_future_horizon": int(window_rollout_future_horizon),
                                "window_rollout_phase_scope": str(window_rollout_phase_scope),
                                "lambda_siglip": float(lambda_siglip),
                                "online_ce_mode": str(online_ce_mode),
                                "action_gap_mode_active": str(action_gap_mode),
                                "final_val_done_rate": float(val_stats["VAL_online_done_rate"]),
                                "final_val_episode_len": float(val_stats["VAL_online_episode_len"]),
                                "final_val_action_gap": float(val_stats["VAL_online_rollout_action_gap"]),
                                "final_val_gt_action_gap": float(val_stats["VAL_online_gt_action_gap"]),
                                "final_val_history1": float(val_stats["VAL_online_rollout_history_div"]),
                                "final_val_history2": float(val_stats["VAL_online_rollout_history_div_legacy"]),
                                "final_val_ce": float(val_stats["VAL_online_ce"]),
                                "final_val_ce_objective": float(val_stats["VAL_online_ce_objective"]),
                                "final_val_siglip_distance": float(val_stats["VAL_online_siglip_distance"]),
                                "final_val_continuous_clean_gt_action_gap": float(
                                    val_stats["VAL_online_continuous_clean_gt_action_gap"]
                                ),
                                "final_val_continuous_adv_gt_action_gap": float(
                                    val_stats["VAL_online_continuous_adv_gt_action_gap"]
                                ),
                                "final_val_continuous_rollout_delta": float(
                                    val_stats["VAL_online_continuous_rollout_delta"]
                                ),
                                "final_val_impulse_rollout_area": float(
                                    val_stats["VAL_online_impulse_rollout_area"]
                                ),
                                "final_val_window_rollout_clean_gt_action_gap": float(
                                    val_stats["VAL_online_window_rollout_clean_gt_action_gap"]
                                ),
                                "final_val_window_rollout_adv_gt_action_gap": float(
                                    val_stats["VAL_online_window_rollout_adv_gt_action_gap"]
                                ),
                                "final_val_window_rollout_deattack_gt_action_gap": float(
                                    val_stats["VAL_online_window_rollout_deattack_gt_action_gap"]
                                ),
                                "final_val_window_rollout_selected_gt_action_gap": float(
                                    val_stats["VAL_online_window_rollout_selected_gt_action_gap"]
                                ),
                                "final_val_window_rollout_delta_weighted": float(
                                    val_stats["VAL_online_window_rollout_delta_weighted"]
                                ),
                                "final_val_window_rollout_delta_weighted_loss": float(
                                    val_stats["VAL_online_window_rollout_delta_weighted_loss"]
                                ),
                                "final_val_window_rollout_metric_value": float(
                                    val_stats["VAL_online_window_rollout_metric_value"]
                                ),
                                "final_val_window_rollout_future_steps": float(
                                    val_stats["VAL_online_window_rollout_future_steps"]
                                ),
                                "window_phase_name": str(val_stats["VAL_online_window_phase_name"]),
                                "window_start_step": float(val_stats["VAL_online_window_start_step"]),
                                "window_end_step": float(val_stats["VAL_online_window_end_step"]),
                                "final_val_rollout_score": float(val_stats["VAL_online_rollout_score"]),
                                "final_val_gt_rollout_score": float(val_stats["VAL_online_gt_rollout_score"]),
                                "final_val_objective_score": float(val_stats["VAL_online_objective_score"]),
                                "final_val_gt_objective_score": float(val_stats["VAL_online_gt_objective_score"]),
                                "final_val_active_action_gap": float(val_stats["VAL_online_active_action_gap"]),
                                "final_val_active_rollout_score": float(val_stats["VAL_online_active_rollout_score"]),
                                "final_val_active_objective_score": float(val_stats["VAL_online_active_objective_score"]),
                                "final_val_total_rollout_score": float(val_stats["VAL_online_total_rollout_score"]),
                                "final_val_total_objective_score": float(val_stats["VAL_online_total_objective_score"]),
                            }
                        )

                improved = val_stats["VAL_online_total_rollout_score"] > self.best_rollout_score
                if improved:
                    self.best_rollout_score = val_stats["VAL_online_total_rollout_score"]
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                    torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    save_projector_params(
                        output_dir=temp_save_dir,
                        projector_gain=current_projector_gain_value,
                        projector_channel_gain=current_projector_channel_gain_values,
                    )
                    self._save_resume_checkpoint(
                        output_dir=temp_save_dir,
                        projection_texture=projection_texture,
                        photometric_params=photometric_params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_iter_completed=i,
                        config=current_resume_config,
                        train_phase_start_counter=train_phase_start_counter,
                        gpu_tuner_state=gpu_tuner_state,
                    )

                temp_save_dir = os.path.join(self.save_dir, "last")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                save_projector_params(
                    output_dir=temp_save_dir,
                    projector_gain=current_projector_gain_value,
                    projector_channel_gain=current_projector_channel_gain_values,
                )
                self._save_resume_checkpoint(
                    output_dir=temp_save_dir,
                    projection_texture=projection_texture,
                    photometric_params=photometric_params,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_iter_completed=i,
                    config=current_resume_config,
                    train_phase_start_counter=train_phase_start_counter,
                    gpu_tuner_state=gpu_tuner_state,
                )

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
                            val_rollout_score=val_stats["VAL_online_total_rollout_score"],
                            reason=vis_reason,
                            attack_mode=attack_mode,
                            args=args,
                        )

                self.save_online_info(self.save_dir)
                torch.cuda.empty_cache()
            elif (not eval_enabled) and eval_due_to_interval:
                temp_save_dir = os.path.join(self.save_dir, "last")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "projection_texture.pt"))
                torch.save(projection_texture.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                final_projector_gain, final_projector_channel_gain = photometric_params.resolved_values()
                save_projector_params(
                    output_dir=temp_save_dir,
                    projector_gain=float(final_projector_gain.detach().cpu().item()),
                    projector_channel_gain=[float(v) for v in final_projector_channel_gain.detach().cpu().tolist()],
                )
                self._save_resume_checkpoint(
                    output_dir=temp_save_dir,
                    projection_texture=projection_texture,
                    photometric_params=photometric_params,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_iter_completed=i,
                    config=current_resume_config,
                    train_phase_start_counter=train_phase_start_counter,
                    gpu_tuner_state=gpu_tuner_state,
                )
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
        lambda_continuous_rollout,
        lambda_window_rollout_loss,
        impulse_rollout_metric_enabled,
        window_rollout_probe_enabled,
        window_rollout_metric_mode,
        window_rollout_future_mode,
        window_rollout_exp_base,
        window_rollout_future_horizon,
        window_rollout_phase_scope,
        lambda_siglip,
        siglip_model,
        siglip_input_size,
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
        projection_randomization_enabled,
        probe_mode,
        record_video_frames,
        video_frame_source,
        action_gap_mode,
        gt_softmin_tau=0.05,
    ):
        total_action_gap = 0.0
        total_gt_action_gap = 0.0
        total_history_div = 0.0
        total_history_div_legacy = 0.0
        total_ce = 0.0
        total_ce_objective = 0.0
        total_siglip_distance = 0.0
        total_continuous_clean_gt_action_gap = 0.0
        total_continuous_adv_gt_action_gap = 0.0
        total_continuous_rollout_delta = 0.0
        total_impulse_rollout_area = 0.0
        total_window_rollout_clean_gt_action_gap = 0.0
        total_window_rollout_adv_gt_action_gap = 0.0
        total_window_rollout_deattack_gt_action_gap = 0.0
        total_window_rollout_selected_gt_action_gap = 0.0
        total_window_rollout_delta_weighted = 0.0
        total_window_rollout_delta_weighted_loss = 0.0
        total_window_rollout_future_steps = 0.0
        total_window_start_step = 0.0
        total_window_end_step = 0.0
        total_done = 0.0
        total_ep_len = 0.0
        total_proj_alpha = 0.0
        total_proj_cov = 0.0
        total_proj_bottom = 0.0
        total_proj_keystone = 0.0
        total_action_terms = 0
        total_history_terms = 0
        total_per_joint = None
        total_gt_per_joint = None
        visual_frames = []
        video_episodes = []
        total_eval_cases = 0
        window_phase_name = ""

        n_tasks = task_suite.n_tasks
        with torch.no_grad():
            for ep_idx in range(int(online_val_episodes)):
                task_id = ep_idx % max(1, n_tasks)
                task = task_suite.get_task(task_id)
                init_states = task_suite.get_task_init_states(task_id)
                episode_seed = (int(val_seed) + int(ep_idx)) if val_deterministic else None
                init_state_idx = self._sample_init_state_index(
                    init_states=init_states,
                    iter_idx=global_iter,
                    local_idx=ep_idx,
                    deterministic_seed=episode_seed,
                )
                if init_state_idx is None:
                    continue
                init_state = init_states[init_state_idx]
                phase_scopes = ["initial"]
                if window_rollout_probe_enabled:
                    phase_scopes = self._resolve_window_rollout_phase_scopes(window_rollout_phase_scope)

                for phase_name in phase_scopes:
                    episode_window_rollout_spec = None
                    phase_init_state = init_state
                    episode_rollout_steps = rollout_steps
                    if window_rollout_probe_enabled:
                        episode_window_rollout_spec = self._resolve_window_rollout_probe_case(
                            task_id=task_id,
                            init_state_idx=init_state_idx,
                            phase_scope=phase_name,
                            future_horizon=window_rollout_future_horizon,
                            exp_base=window_rollout_exp_base,
                            max_env_steps=max_env_steps,
                        )
                        if episode_window_rollout_spec is None:
                            continue
                        phase_init_state = self._resolve_phase_start_state(
                            task_id=task_id,
                            init_state_idx=init_state_idx,
                            phase_start_name=phase_name,
                            default_init_state=init_state,
                        )
                    env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
                    try:
                        episode = self._run_online_episode(
                            env=env,
                            task=task,
                            get_libero_env=get_libero_env,
                            get_libero_image=get_libero_image,
                            get_libero_dummy_action=get_libero_dummy_action,
                            init_state=phase_init_state,
                            init_state_idx=init_state_idx,
                            task_description=task_description,
                            projection_texture=projection_texture,
                            rollout_steps=episode_rollout_steps,
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
                            lambda_continuous_rollout=lambda_continuous_rollout,
                            lambda_window_rollout_loss=lambda_window_rollout_loss,
                            impulse_rollout_metric_enabled=impulse_rollout_metric_enabled,
                            window_rollout_probe_enabled=bool(episode_window_rollout_spec is not None),
                            window_rollout_metric_mode=window_rollout_metric_mode,
                            window_rollout_future_mode=window_rollout_future_mode,
                            window_rollout_spec=episode_window_rollout_spec,
                            lambda_siglip=lambda_siglip,
                            siglip_model=siglip_model,
                            siglip_input_size=siglip_input_size,
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
                            projection_randomization_enabled=bool(projection_randomization_enabled),
                            need_backward=False,
                            capture_visual=(len(visual_frames) == 0),
                            deterministic_episode_seed=episode_seed,
                            disable_lighting=bool(val_disable_lighting),
                            task_id=task_id,
                            record_video_frames=bool(record_video_frames),
                            video_frame_source=video_frame_source,
                            action_gap_mode=action_gap_mode,
                            start_phase_name=phase_name if window_rollout_probe_enabled else "initial",
                            gt_softmin_tau=gt_softmin_tau,
                        )
                        if episode is None:
                            continue

                        episode["episode_idx"] = int(ep_idx)
                        episode["init_state_idx"] = int(init_state_idx)
                        total_eval_cases += 1
                        total_action_gap += episode["action_gap"]
                        total_gt_action_gap += episode["gt_action_gap"]
                        total_history_div += episode["history_div"]
                        total_history_div_legacy += episode["history_div_legacy"]
                        total_ce += episode["ce_value"]
                        total_ce_objective += episode["ce_objective_value"]
                        total_siglip_distance += episode["siglip_distance"]
                        total_continuous_clean_gt_action_gap += episode["continuous_clean_gt_action_gap"]
                        total_continuous_adv_gt_action_gap += episode["continuous_adv_gt_action_gap"]
                        total_continuous_rollout_delta += episode["continuous_rollout_delta"]
                        total_impulse_rollout_area += episode["impulse_rollout_area"]
                        total_window_rollout_clean_gt_action_gap += episode["window_rollout_clean_gt_action_gap"]
                        total_window_rollout_adv_gt_action_gap += episode["window_rollout_adv_gt_action_gap"]
                        total_window_rollout_deattack_gt_action_gap += episode["window_rollout_deattack_gt_action_gap"]
                        total_window_rollout_selected_gt_action_gap += episode["window_rollout_selected_gt_action_gap"]
                        total_window_rollout_delta_weighted += episode["window_rollout_delta_weighted"]
                        total_window_rollout_delta_weighted_loss += episode["window_rollout_delta_weighted_loss"]
                        total_window_rollout_future_steps += float(episode["window_rollout_future_steps"])
                        total_window_start_step += float(episode["window_start_step"])
                        total_window_end_step += float(episode["window_end_step"])
                        total_done += float(episode["done"])
                        total_ep_len += float(episode["episode_len"])
                        total_proj_alpha += episode["projection_alpha"]
                        total_proj_cov += episode["projection_coverage"]
                        total_proj_bottom += episode["projection_bottom"]
                        total_proj_keystone += episode["projection_keystone"]
                        total_action_terms += max(1, episode["action_terms"])
                        total_history_terms += max(0, episode["history_terms"])
                        total_per_joint = self._accumulate_joint_values(total_per_joint, episode["per_joint_gap"])
                        total_gt_per_joint = self._accumulate_joint_values(total_gt_per_joint, episode["gt_per_joint_gap"])
                        window_phase_name = str(episode.get("window_phase_name", window_phase_name or phase_name))
                        if len(visual_frames) == 0 and len(episode["visual_frames"]) > 0:
                            visual_frames = episode["visual_frames"][: max(1, int(viz_samples))]
                        if bool(record_video_frames) and self._should_dump_online_episode_video(episode):
                            video_episodes.append(episode)
                    finally:
                        if hasattr(env, "close"):
                            env.close()

        divisor = float(max(1, int(total_eval_cases if total_eval_cases > 0 else online_val_episodes)))
        avg_action_gap = total_action_gap / divisor
        avg_gt_action_gap = total_gt_action_gap / divisor
        avg_history_div = total_history_div / divisor
        avg_history_div_legacy = total_history_div_legacy / divisor
        avg_ce = total_ce / divisor
        avg_ce_objective = total_ce_objective / divisor
        avg_siglip_distance = total_siglip_distance / divisor
        avg_continuous_clean_gt_action_gap = total_continuous_clean_gt_action_gap / divisor
        avg_continuous_adv_gt_action_gap = total_continuous_adv_gt_action_gap / divisor
        avg_continuous_rollout_delta = total_continuous_rollout_delta / divisor
        avg_impulse_rollout_area = total_impulse_rollout_area / divisor
        avg_window_rollout_clean_gt_action_gap = total_window_rollout_clean_gt_action_gap / divisor
        avg_window_rollout_adv_gt_action_gap = total_window_rollout_adv_gt_action_gap / divisor
        avg_window_rollout_deattack_gt_action_gap = total_window_rollout_deattack_gt_action_gap / divisor
        avg_window_rollout_selected_gt_action_gap = total_window_rollout_selected_gt_action_gap / divisor
        avg_window_rollout_delta_weighted = total_window_rollout_delta_weighted / divisor
        avg_window_rollout_delta_weighted_loss = total_window_rollout_delta_weighted_loss / divisor
        avg_window_rollout_metric_value = select_window_rollout_metric_value(
            metric_mode=window_rollout_metric_mode,
            delta_weighted=avg_window_rollout_delta_weighted,
            adv_gt_action_gap=avg_window_rollout_selected_gt_action_gap,
        )
        avg_window_rollout_future_steps = total_window_rollout_future_steps / divisor
        avg_window_start_step = total_window_start_step / divisor
        avg_window_end_step = total_window_end_step / divisor
        avg_done = total_done / divisor
        avg_ep_len = total_ep_len / divisor
        avg_proj_alpha = total_proj_alpha / divisor
        avg_proj_cov = total_proj_cov / divisor
        avg_proj_bottom = total_proj_bottom / divisor
        avg_proj_keystone = total_proj_keystone / divisor
        avg_per_joint = self._normalize_joint_values(total_per_joint, divisor)
        avg_gt_per_joint = self._normalize_joint_values(total_gt_per_joint, divisor)
        if probe_mode:
            avg_rollout_score = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
                + (lambda_siglip * avg_siglip_distance)
            )
            avg_gt_rollout_score = (
                (lambda_action_gap * avg_gt_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_history_legacy * avg_history_div_legacy)
                + (lambda_siglip * avg_siglip_distance)
            )
            avg_rollout_score_legacy = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history_legacy * avg_history_div_legacy)
            )
        else:
            avg_rollout_score = (
                (lambda_action_gap * avg_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_siglip * avg_siglip_distance)
            )
            avg_gt_rollout_score = (
                (lambda_action_gap * avg_gt_action_gap)
                + (lambda_history * avg_history_div)
                + (lambda_siglip * avg_siglip_distance)
            )
            avg_rollout_score_legacy = (lambda_action_gap * avg_action_gap) + (lambda_history * avg_history_div_legacy)
        avg_objective_score = avg_rollout_score - (lambda_ce * avg_ce_objective)
        avg_gt_objective_score = avg_gt_rollout_score - (lambda_ce * avg_ce_objective)
        avg_active_action_gap = avg_gt_action_gap if action_gap_mode == "gt_farthest" else avg_action_gap
        avg_active_rollout_score = avg_gt_rollout_score if action_gap_mode == "gt_farthest" else avg_rollout_score
        avg_active_objective_score = avg_gt_objective_score if action_gap_mode == "gt_farthest" else avg_objective_score
        avg_total_rollout_score = (
            avg_active_rollout_score
            + (lambda_continuous_rollout * avg_continuous_rollout_delta)
            + (lambda_window_rollout_loss * avg_window_rollout_metric_value)
        )
        avg_total_objective_score = avg_total_rollout_score - (lambda_ce * avg_ce_objective)
        current_projector_gain = float(
            projector_gain.detach().cpu().item() if torch.is_tensor(projector_gain) else projector_gain
        )
        current_projector_channel_gain = parse_projector_channel_gain(projector_channel_gain)

        stats = {
            "VAL_online_rollout_action_gap": avg_action_gap,
            "VAL_online_gt_action_gap": avg_gt_action_gap,
            "VAL_online_rollout_history_div": avg_history_div,
            "VAL_online_rollout_history_div_legacy": avg_history_div_legacy,
            "VAL_online_siglip_distance": avg_siglip_distance,
            "VAL_online_rollout_score": avg_rollout_score,
            "VAL_online_gt_rollout_score": avg_gt_rollout_score,
            "VAL_online_rollout_score_legacy": avg_rollout_score_legacy,
            "VAL_online_objective_score": avg_objective_score,
            "VAL_online_gt_objective_score": avg_gt_objective_score,
            "VAL_action_gap_mode_active": str(action_gap_mode),
            "VAL_online_active_action_gap": avg_active_action_gap,
            "VAL_online_active_rollout_score": avg_active_rollout_score,
            "VAL_online_active_objective_score": avg_active_objective_score,
            "VAL_online_done_rate": avg_done,
            "VAL_online_episode_len": avg_ep_len,
            "VAL_online_ce": avg_ce,
            "VAL_online_ce_objective": avg_ce_objective,
            "VAL_lambda_continuous_rollout": float(lambda_continuous_rollout),
            "VAL_lambda_window_rollout_loss": float(lambda_window_rollout_loss),
            "VAL_impulse_rollout_metric_enabled": int(impulse_rollout_metric_enabled),
            "VAL_online_continuous_clean_gt_action_gap": avg_continuous_clean_gt_action_gap,
            "VAL_online_continuous_adv_gt_action_gap": avg_continuous_adv_gt_action_gap,
            "VAL_online_continuous_rollout_delta": avg_continuous_rollout_delta,
            "VAL_online_impulse_rollout_area": avg_impulse_rollout_area,
            "VAL_online_window_rollout_clean_gt_action_gap": avg_window_rollout_clean_gt_action_gap,
            "VAL_online_window_rollout_adv_gt_action_gap": avg_window_rollout_adv_gt_action_gap,
            "VAL_online_window_rollout_deattack_gt_action_gap": avg_window_rollout_deattack_gt_action_gap,
            "VAL_online_window_rollout_selected_gt_action_gap": avg_window_rollout_selected_gt_action_gap,
            "VAL_online_window_rollout_delta_weighted": avg_window_rollout_delta_weighted,
            "VAL_online_window_rollout_delta_weighted_loss": avg_window_rollout_delta_weighted_loss,
            "VAL_online_window_rollout_metric_mode": str(window_rollout_metric_mode),
            "VAL_online_window_rollout_future_mode": str(window_rollout_future_mode),
            "VAL_online_window_rollout_metric_value": avg_window_rollout_metric_value,
            "VAL_online_window_rollout_future_steps": avg_window_rollout_future_steps,
            "VAL_online_window_rollout_probe_enabled": int(window_rollout_probe_enabled),
            "VAL_online_window_rollout_exp_base": float(window_rollout_exp_base),
            "VAL_online_window_rollout_future_horizon": int(window_rollout_future_horizon),
            "VAL_online_window_phase_name": str(window_phase_name),
            "VAL_online_window_start_step": avg_window_start_step,
            "VAL_online_window_end_step": avg_window_end_step,
            "VAL_online_total_rollout_score": avg_total_rollout_score,
            "VAL_online_total_objective_score": avg_total_objective_score,
            "VAL_projection_alpha_mean": avg_proj_alpha,
            "VAL_projection_coverage_ratio": avg_proj_cov,
            "VAL_projection_bottom_ratio": avg_proj_bottom,
            "VAL_projection_keystone": avg_proj_keystone,
            "VAL_projector_gain": current_projector_gain,
            "VAL_projector_channel_gain_r": float(current_projector_channel_gain[0]),
            "VAL_projector_channel_gain_g": float(current_projector_channel_gain[1]),
            "VAL_projector_channel_gain_b": float(current_projector_channel_gain[2]),
            "VAL_effective_lighting_enabled": int(self._is_lighting_enabled_for_split("val") and (not bool(val_disable_lighting))),
            "VAL_lighting_backend": str(getattr(self.lighting_augmentor, "backend", "disabled")),
            "ic_light_scope": getattr(self.lighting_augmentor, "scope", "n/a") if self.lighting_augmentor is not None else "n/a",
            "ic_light_bg_control": getattr(self.lighting_augmentor, "bg_control", "n/a")
            if self.lighting_augmentor is not None
            else "n/a",
        }
        stats.update(self._gt_cache_log_fields())
        for joint_idx, joint_gap_value in enumerate(avg_per_joint.detach().cpu().tolist()):
            stats[f"VAL_online_rollout_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
        for joint_idx, joint_gap_value in enumerate(avg_gt_per_joint.detach().cpu().tolist()):
            stats[f"VAL_online_gt_action_gap_joint_{joint_idx}"] = float(joint_gap_value)
        return stats, visual_frames, video_episodes

    @staticmethod
    def _clone_init_state(init_state):
        return np.asarray(init_state, dtype=np.float32).copy()

    def _initialize_online_rollout_env(self, env, init_state, get_libero_dummy_action, effective_num_steps_wait):
        env.reset()
        obs = env.set_init_state(self._clone_init_state(init_state))
        for _ in range(max(0, int(effective_num_steps_wait))):
            obs, _, done_wait, _ = env.step(get_libero_dummy_action("openvla"))
            if done_wait:
                break
        return obs

    def _forward_clean_branch_gt_step(
        self,
        obs,
        rollout_input_ids,
        get_libero_image,
        attention_mask,
        labels_full,
        action_mask_full,
        gt_candidate_actions,
        maskidx,
        use_all_joints,
        gripper_weight,
        gt_softmin_tau,
        split,
        global_iter,
        max_steps,
        step_idx,
        step_seed,
        disable_lighting,
        episode_lighting_map_idx,
    ):
        branch_image = get_libero_image(obs, (224, 224))
        branch_pixels = [branch_image]

        with self._temporary_rng_seed(step_seed):
            if not bool(disable_lighting):
                branch_pixels = self._apply_lighting_augmentation(
                    pixel_values=branch_pixels,
                    iteration_idx=(int(global_iter) * max(1, int(max_steps))) + int(step_idx),
                    split=split,
                    fixed_map_idx=episode_lighting_map_idx,
                )
            branch_images = self.randomPatchTransform.im_process(branch_pixels, mean=self.mean, std=self.std)

        with torch.no_grad():
            output_branch: CausalLMOutputWithPast = self.vla(
                input_ids=rollout_input_ids,
                attention_mask=attention_mask,
                pixel_values=branch_images.to(torch.bfloat16),
                labels=None,
                output_hidden_states=False,
                use_cache=False,
            )
        branch_pred_tokens = self._extract_pred_action_tokens_from_logits(output_branch.logits, labels_full)
        if gt_candidate_actions is None:
            zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            per_joint_zero = torch.zeros((self.default_action_dim,), device=self.vla.device, dtype=torch.float32)
            return branch_pred_tokens.detach(), zero, zero.detach(), per_joint_zero

        branch_gt_loss, branch_gt_metric, branch_gt_per_joint, _ = self._compute_gt_action_gap_losses(
            adv_logits=output_branch.logits,
            labels_full=labels_full,
            action_mask_full=action_mask_full,
            gt_candidate_actions=gt_candidate_actions,
            maskidx=maskidx,
            use_all_joints=use_all_joints,
            gripper_weight=gripper_weight,
            gt_softmin_tau=gt_softmin_tau,
        )
        return (
            branch_pred_tokens.detach(),
            branch_gt_loss.detach(),
            branch_gt_metric.detach(),
            branch_gt_per_joint.detach(),
        )

    def _step_rollout_branch_env(self, env, obs, rollout_input_ids, pred_action_tokens, action_mask_full, action_stats):
        env_action = self._decode_env_action(pred_action_tokens, action_mask_full, action_stats)
        next_obs, _, done, _ = env.step(env_action.tolist())
        next_rollout_input_ids = self._update_rollout_inputs(
            rollout_input_ids=rollout_input_ids,
            pred_action_tokens=pred_action_tokens,
            action_mask_full=action_mask_full,
        )
        return next_obs, bool(done), next_rollout_input_ids

    def _run_online_episode(
        self,
        env,
        task,
        get_libero_env,
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
        lambda_continuous_rollout,
        lambda_window_rollout_loss,
        impulse_rollout_metric_enabled,
        lambda_siglip,
        siglip_model,
        siglip_input_size,
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
        projection_randomization_enabled,
        need_backward,
        capture_visual,
        deterministic_episode_seed=None,
        disable_lighting=False,
        task_id=None,
        record_video_frames=False,
        video_frame_source="projected_input",
        action_gap_mode="clean_adv",
        start_phase_name="initial",
        gt_softmin_tau=0.05,
        window_rollout_probe_enabled=False,
        window_rollout_metric_mode="delta_weighted",
        window_rollout_future_mode="keep_adv",
        window_rollout_spec=None,
        learnable_projector_params=None,
        init_state_idx=None,
    ):
        action_dim = len(action_stats["q01"])
        start_phase_name = self._normalize_phase_start_name(start_phase_name)
        video_frame_source = str(video_frame_source).lower().strip()
        if video_frame_source not in ("next_obs", "orig", "projected_input", "adv"):
            video_frame_source = "projected_input"

        continuous_enabled = float(lambda_continuous_rollout) > 0.0
        lambda_window_rollout_loss = float(lambda_window_rollout_loss)
        impulse_enabled = bool(impulse_rollout_metric_enabled)
        window_rollout_metric_mode = normalize_window_rollout_metric_mode(window_rollout_metric_mode)
        window_rollout_future_mode = normalize_window_rollout_future_mode(window_rollout_future_mode)
        window_rollout_enabled = bool(window_rollout_probe_enabled) and isinstance(window_rollout_spec, dict)
        deattack_future_enabled = window_rollout_enabled and (window_rollout_future_mode == "drop_attack_after_window")
        need_aux_clean_branch = continuous_enabled or impulse_enabled or window_rollout_enabled
        need_any_gt_rollout = (
            (action_gap_mode == "gt_farthest") or continuous_enabled or impulse_enabled or window_rollout_enabled
        )

        effective_num_steps_wait = 0 if start_phase_name != "initial" else max(0, int(num_steps_wait))
        obs = self._initialize_online_rollout_env(
            env=env,
            init_state=init_state,
            get_libero_dummy_action=get_libero_dummy_action,
            effective_num_steps_wait=effective_num_steps_wait,
        )

        clean_branch_env = None
        clean_branch_obs = None
        clean_branch_active = False
        clean_branch_rollout_input_ids = None

        impulse_branch_env = None
        impulse_branch_obs = None
        impulse_branch_active = False
        impulse_branch_rollout_input_ids = None

        deattack_branch_env = None
        deattack_branch_obs = None
        deattack_branch_active = False
        deattack_branch_rollout_input_ids = None

        try:
            if need_aux_clean_branch:
                clean_branch_env, _ = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
                clean_branch_obs = self._initialize_online_rollout_env(
                    env=clean_branch_env,
                    init_state=init_state,
                    get_libero_dummy_action=get_libero_dummy_action,
                    effective_num_steps_wait=effective_num_steps_wait,
                )
                clean_branch_active = True
            if impulse_enabled:
                impulse_branch_env, _ = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
                impulse_branch_obs = self._initialize_online_rollout_env(
                    env=impulse_branch_env,
                    init_state=init_state,
                    get_libero_dummy_action=get_libero_dummy_action,
                    effective_num_steps_wait=effective_num_steps_wait,
                )
                impulse_branch_active = True
            if deattack_future_enabled:
                deattack_branch_env, _ = get_libero_env(task, "openvla", resolution=max(64, int(env_resolution)))
                deattack_branch_obs = self._initialize_online_rollout_env(
                    env=deattack_branch_env,
                    init_state=init_state,
                    get_libero_dummy_action=get_libero_dummy_action,
                    effective_num_steps_wait=effective_num_steps_wait,
                )
                deattack_branch_active = True

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
            if need_aux_clean_branch:
                clean_branch_rollout_input_ids = input_ids.clone()
            if impulse_enabled:
                impulse_branch_rollout_input_ids = input_ids.clone()
            if deattack_future_enabled:
                deattack_branch_rollout_input_ids = input_ids.clone()

            total_loss = torch.zeros((), device=self.vla.device, dtype=torch.float32)
            episode_weight_sum = 0.0
            grad_before_episode = None
            if need_backward and (projection_texture.grad is not None):
                grad_before_episode = projection_texture.grad.detach().clone()
            photometric_grads_before_episode = None
            if need_backward and (learnable_projector_params is not None) and learnable_projector_params.has_trainable_params():
                photometric_grads_before_episode = []
                for param in learnable_projector_params.parameters():
                    if param.grad is None:
                        photometric_grads_before_episode.append(None)
                    else:
                        photometric_grads_before_episode.append(param.grad.detach().clone())
            total_action_gap = 0.0
            total_gt_action_gap = 0.0
            total_history_div = 0.0
            total_history_div_legacy = 0.0
            total_ce = 0.0
            total_ce_objective = 0.0
            total_siglip_distance = 0.0
            total_continuous_clean_gt_gap = 0.0
            total_continuous_adv_gt_gap = 0.0
            total_continuous_rollout_delta = 0.0
            total_impulse_rollout_area = 0.0
            total_action_terms = 0
            total_history_terms = 0
            total_history_terms_legacy = 0
            total_continuous_terms = 0
            total_impulse_terms = 0
            total_per_joint = None
            total_gt_per_joint = None
            total_proj_alpha = 0.0
            total_proj_cov = 0.0
            total_proj_bottom = 0.0
            total_proj_keystone = 0.0
            gt_candidate_count = 0
            gt_reference_instruction = self._normalize_instruction_key(task_description)
            gt_reference_phase = self._phase_start_to_gt_phase(start_phase_name)
            episode_phase_boundary_info = None
            if window_rollout_enabled and task_id is not None and init_state_idx is not None:
                episode_phase_boundary_info = self._get_phase_boundary_info(
                    task_id=task_id,
                    init_state_idx=init_state_idx,
                )
            last_done = False
            visual_frames = []
            video_frames = []
            window_step_count = int(window_rollout_spec.get("window_step_count", 0)) if window_rollout_enabled else 0
            window_future_steps = int(window_rollout_spec.get("future_steps", 0)) if window_rollout_enabled else 0
            window_exp_base = float(window_rollout_spec.get("exp_base", 0.9)) if window_rollout_enabled else 0.9
            total_window_rollout_clean_gt_gap = 0.0
            total_window_rollout_adv_gt_gap = 0.0
            total_window_rollout_deattack_gt_gap = 0.0
            total_window_rollout_selected_gt_gap = 0.0
            total_window_rollout_delta_weighted = 0.0
            total_window_rollout_delta_weighted_loss = 0.0
            total_window_rollout_terms = 0

            max_steps = int(min(int(rollout_steps), int(max_env_steps)))
            need_history = (max_steps > 0) and (
                (not need_backward) or (lambda_history > 0) or (lambda_history_legacy > 0)
            )
            prev_adv_history_state = None
            episode_lighting_map_idx = None
            episode_lighting_backend = "disabled"
            episode_effective_lighting_enabled = 0
            episode_projection_seed = None
            if (not bool(disable_lighting)) and (self.lighting_augmentor is not None):
                if deterministic_episode_seed is not None:
                    episode_lighting_map_idx = int(deterministic_episode_seed)
                else:
                    episode_lighting_map_idx = random.randint(0, (2**31) - 1)
            if bool(projection_randomization_enabled):
                if deterministic_episode_seed is not None:
                    episode_projection_seed = int(deterministic_episode_seed) + 17
                else:
                    episode_projection_seed = random.randint(0, (2**31) - 1)

            for step_idx in range(max_steps):
                current_image = get_libero_image(obs, (224, 224))
                pixel_values = [current_image]
                step_seed = None
                if deterministic_episode_seed is not None:
                    step_seed = int(deterministic_episode_seed) + (int(step_idx) * 10007)
                try:
                    current_projector_gain = projector_gain
                    current_projector_channel_gain = projector_channel_gain
                    if learnable_projector_params is not None:
                        current_projector_gain, current_projector_channel_gain = (
                            learnable_projector_params.resolved_values()
                        )
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
                    with self._temporary_rng_seed(episode_projection_seed):
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
                            projector_gain=current_projector_gain,
                            projector_channel_gain=current_projector_channel_gain,
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
                    disable_lighting = True
                    episode_effective_lighting_enabled = 0
                    episode_lighting_backend = str(getattr(self.lighting_augmentor, "backend", "disabled"))
                    pixel_values = [current_image]
                    current_projector_gain = projector_gain
                    current_projector_channel_gain = projector_channel_gain
                    if learnable_projector_params is not None:
                        current_projector_gain, current_projector_channel_gain = (
                            learnable_projector_params.resolved_values()
                        )
                    with self._temporary_rng_seed(step_seed):
                        clean_images = self.randomPatchTransform.im_process(pixel_values, mean=self.mean, std=self.std)
                    with self._temporary_rng_seed(episode_projection_seed):
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
                            projector_gain=current_projector_gain,
                            projector_channel_gain=current_projector_channel_gain,
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

                step_action_gap_loss, step_action_gap_metric, step_per_joint_gap, adv_pred_tokens = (
                    self._compute_action_gap_losses(
                        adv_logits=output_adv.logits,
                        clean_logits=output_clean.logits,
                        labels_full=labels_full,
                        action_mask_full=action_mask_full,
                        maskidx=maskidx,
                        use_all_joints=use_all_joints,
                        gripper_weight=gripper_weight,
                    )
                )
                total_action_gap += step_action_gap_metric.item()
                total_action_terms += 1
                total_per_joint = self._accumulate_joint_values(total_per_joint, step_per_joint_gap)

                gt_candidate_actions = None
                if need_any_gt_rollout:
                    absolute_step_idx = None
                    if window_rollout_enabled:
                        absolute_step_idx = int(window_rollout_spec.get("window_start_step", 0)) + int(step_idx)
                    gt_phase_name = self._infer_gt_phase_for_episode_step(
                        split=split,
                        task_description=task_description,
                        step_idx=step_idx,
                        horizon=max_steps,
                        phase_start_name=start_phase_name,
                        phase_boundary_info=episode_phase_boundary_info,
                        absolute_step_idx=absolute_step_idx,
                    )
                    gt_candidate_actions, gt_candidate_count, gt_reference_instruction, gt_reference_phase = (
                        self._get_gt_candidate_actions(
                            split=split,
                            task_description=task_description,
                            step_idx=step_idx,
                            horizon=max_steps,
                            action_dim=action_dim,
                            phase_name=gt_phase_name,
                        )
                    )

                zero = torch.zeros((), device=self.vla.device, dtype=torch.float32)
                step_gt_action_gap_loss = zero
                step_gt_action_gap_metric = zero.detach()
                step_gt_per_joint_gap = torch.zeros(
                    (self.default_action_dim,),
                    device=self.vla.device,
                    dtype=torch.float32,
                )
                if action_gap_mode == "gt_farthest":
                    step_gt_action_gap_loss, step_gt_action_gap_metric, step_gt_per_joint_gap, _ = (
                        self._compute_gt_action_gap_losses(
                            adv_logits=output_adv.logits,
                            labels_full=labels_full,
                            action_mask_full=action_mask_full,
                            gt_candidate_actions=gt_candidate_actions,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            gt_softmin_tau=gt_softmin_tau,
                        )
                    )
                elif continuous_enabled or impulse_enabled or window_rollout_enabled:
                    step_gt_action_gap_loss, step_gt_action_gap_metric, step_gt_per_joint_gap, _ = (
                        self._compute_gt_action_gap_losses(
                            adv_logits=output_adv.logits,
                            labels_full=labels_full,
                            action_mask_full=action_mask_full,
                            gt_candidate_actions=gt_candidate_actions,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            gt_softmin_tau=gt_softmin_tau,
                        )
                    )
                if action_gap_mode == "gt_farthest":
                    total_gt_action_gap += step_gt_action_gap_metric.item()
                    total_gt_per_joint = self._accumulate_joint_values(total_gt_per_joint, step_gt_per_joint_gap)

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
                    step_history_div_legacy = self._compute_legacy_history_divergence(
                        adv_history_state, prev_adv_history_state
                    )
                    if prev_adv_history_state is not None:
                        total_history_div_legacy += step_history_div_legacy.detach().item()
                        total_history_terms_legacy += 1
                    prev_adv_history_state = adv_history_state.detach()

                step_weight = float(self._get_rollout_step_weight(step_idx))
                episode_weight_sum += step_weight

                step_siglip_distance = torch.zeros((), device=self.vla.device, dtype=torch.float32)
                if (siglip_model is not None) and (lambda_siglip > 0.0):
                    step_siglip_distance = self._compute_siglip_embedding_distance(
                        siglip_model=siglip_model,
                        reference_images=attack_aux["pre_projection_tensors"],
                        projected_images=attack_aux["projected_input_tensors"],
                        input_size=siglip_input_size,
                    )

                clean_branch_pred_tokens = None
                clean_branch_gt_loss = zero.detach()
                clean_branch_gt_metric = zero.detach()
                if clean_branch_active:
                    if step_idx == 0:
                        clean_branch_pred_tokens = clean_pred_tokens.detach()
                        if gt_candidate_actions is not None:
                            clean_branch_gt_loss, clean_branch_gt_metric, _, _ = self._compute_gt_action_gap_losses(
                                adv_logits=output_clean.logits,
                                labels_full=labels_full,
                                action_mask_full=action_mask_full,
                                gt_candidate_actions=gt_candidate_actions,
                                maskidx=maskidx,
                                use_all_joints=use_all_joints,
                                gripper_weight=gripper_weight,
                                gt_softmin_tau=gt_softmin_tau,
                            )
                            clean_branch_gt_loss = clean_branch_gt_loss.detach()
                    else:
                        (
                            clean_branch_pred_tokens,
                            clean_branch_gt_loss,
                            clean_branch_gt_metric,
                            _,
                        ) = self._forward_clean_branch_gt_step(
                            obs=clean_branch_obs,
                            rollout_input_ids=clean_branch_rollout_input_ids,
                            get_libero_image=get_libero_image,
                            attention_mask=attention_mask,
                            labels_full=labels_full,
                            action_mask_full=action_mask_full,
                            gt_candidate_actions=gt_candidate_actions,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            gt_softmin_tau=gt_softmin_tau,
                            split=split,
                            global_iter=global_iter,
                            max_steps=max_steps,
                            step_idx=step_idx,
                            step_seed=step_seed,
                            disable_lighting=disable_lighting,
                            episode_lighting_map_idx=episode_lighting_map_idx,
                        )

                impulse_branch_pred_tokens = None
                impulse_branch_gt_metric = zero.detach()
                if impulse_branch_active:
                    if step_idx == 0:
                        impulse_branch_pred_tokens = adv_pred_tokens.detach()
                    else:
                        (
                            impulse_branch_pred_tokens,
                            _impulse_gt_loss_unused,
                            impulse_branch_gt_metric,
                            _,
                        ) = self._forward_clean_branch_gt_step(
                            obs=impulse_branch_obs,
                            rollout_input_ids=impulse_branch_rollout_input_ids,
                            get_libero_image=get_libero_image,
                            attention_mask=attention_mask,
                            labels_full=labels_full,
                            action_mask_full=action_mask_full,
                            gt_candidate_actions=gt_candidate_actions,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            gt_softmin_tau=gt_softmin_tau,
                            split=split,
                            global_iter=global_iter,
                            max_steps=max_steps,
                            step_idx=step_idx,
                            step_seed=step_seed,
                            disable_lighting=disable_lighting,
                            episode_lighting_map_idx=episode_lighting_map_idx,
                        )

                deattack_branch_pred_tokens = None
                deattack_branch_gt_loss = zero.detach()
                deattack_branch_gt_metric = zero.detach()
                if deattack_branch_active:
                    if step_idx < window_step_count:
                        deattack_branch_pred_tokens = adv_pred_tokens.detach()
                    elif step_idx == 0:
                        deattack_branch_pred_tokens = clean_pred_tokens.detach()
                        if gt_candidate_actions is not None:
                            deattack_branch_gt_loss, deattack_branch_gt_metric, _, _ = self._compute_gt_action_gap_losses(
                                adv_logits=output_clean.logits,
                                labels_full=labels_full,
                                action_mask_full=action_mask_full,
                                gt_candidate_actions=gt_candidate_actions,
                                maskidx=maskidx,
                                use_all_joints=use_all_joints,
                                gripper_weight=gripper_weight,
                                gt_softmin_tau=gt_softmin_tau,
                            )
                            deattack_branch_gt_loss = deattack_branch_gt_loss.detach()
                    else:
                        (
                            deattack_branch_pred_tokens,
                            deattack_branch_gt_loss,
                            deattack_branch_gt_metric,
                            _,
                        ) = self._forward_clean_branch_gt_step(
                            obs=deattack_branch_obs,
                            rollout_input_ids=deattack_branch_rollout_input_ids,
                            get_libero_image=get_libero_image,
                            attention_mask=attention_mask,
                            labels_full=labels_full,
                            action_mask_full=action_mask_full,
                            gt_candidate_actions=gt_candidate_actions,
                            maskidx=maskidx,
                            use_all_joints=use_all_joints,
                            gripper_weight=gripper_weight,
                            gt_softmin_tau=gt_softmin_tau,
                            split=split,
                            global_iter=global_iter,
                            max_steps=max_steps,
                            step_idx=step_idx,
                            step_seed=step_seed,
                            disable_lighting=disable_lighting,
                            episode_lighting_map_idx=episode_lighting_map_idx,
                        )

                active_step_action_gap_loss = (
                    step_gt_action_gap_loss if action_gap_mode == "gt_farthest" else step_action_gap_loss
                )
                step_loss = -(lambda_action_gap * active_step_action_gap_loss)
                if continuous_enabled and clean_branch_active:
                    continuous_step_delta_loss = step_gt_action_gap_loss - clean_branch_gt_loss.detach()
                    continuous_step_delta_metric = step_gt_action_gap_metric - clean_branch_gt_metric
                    step_loss = step_loss - (float(lambda_continuous_rollout) * continuous_step_delta_loss)
                    total_continuous_adv_gt_gap += float(step_gt_action_gap_metric.item())
                    total_continuous_clean_gt_gap += float(clean_branch_gt_metric.item())
                    total_continuous_rollout_delta += float(continuous_step_delta_metric.item())
                    total_continuous_terms += 1
                if window_rollout_enabled and clean_branch_active and (step_idx >= window_step_count):
                    future_step_idx = int(step_idx - window_step_count)
                    if future_step_idx < window_future_steps:
                        selected_gt_metric = step_gt_action_gap_metric
                        selected_gt_loss = step_gt_action_gap_loss
                        selected_branch_ready = True
                        if deattack_future_enabled:
                            selected_branch_ready = deattack_branch_pred_tokens is not None
                            selected_gt_metric = deattack_branch_gt_metric
                            selected_gt_loss = deattack_branch_gt_loss
                        if not selected_branch_ready:
                            selected_gt_metric = zero.detach()
                            selected_gt_loss = zero.detach()
                        step_window_delta = selected_gt_metric - clean_branch_gt_metric
                        step_window_delta_loss = selected_gt_loss - clean_branch_gt_loss.detach()
                        if window_rollout_metric_mode == "adv_gt":
                            step_window_metric_loss = selected_gt_loss
                        else:
                            step_window_metric_loss = step_window_delta_loss
                        step_window_weight = compute_window_rollout_weight(
                            future_step_idx=future_step_idx,
                            exp_base=window_exp_base,
                        )
                        if selected_branch_ready:
                            total_window_rollout_adv_gt_gap += float(step_gt_action_gap_metric.item())
                            total_window_rollout_clean_gt_gap += float(clean_branch_gt_metric.item())
                            total_window_rollout_deattack_gt_gap += float(deattack_branch_gt_metric.item())
                            total_window_rollout_selected_gt_gap += float(selected_gt_metric.item())
                            total_window_rollout_delta_weighted += float(step_window_delta.item()) * float(step_window_weight)
                            total_window_rollout_delta_weighted_loss += float(
                                step_window_delta_loss.detach().item()
                            ) * float(step_window_weight)
                            if lambda_window_rollout_loss != 0.0:
                                step_loss = step_loss - (
                                    float(lambda_window_rollout_loss) * float(step_window_weight) * step_window_metric_loss
                                )
                            total_window_rollout_terms += 1
                if need_history:
                    step_loss = step_loss - (lambda_history * step_history_div)
                    step_loss = step_loss - (lambda_history_legacy * step_history_div_legacy)
                step_loss = step_loss - (lambda_siglip * step_siglip_distance)

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
                total_siglip_distance += step_siglip_distance.detach().item()
                if need_backward:
                    if weighted_step_loss.requires_grad:
                        weighted_step_loss.backward()
                    total_loss = total_loss + weighted_step_loss.detach()
                else:
                    total_loss = total_loss + weighted_step_loss

                next_obs, done, adv_rollout_input_ids = self._step_rollout_branch_env(
                    env=env,
                    obs=obs,
                    rollout_input_ids=adv_rollout_input_ids,
                    pred_action_tokens=adv_pred_tokens.detach(),
                    action_mask_full=action_mask_full,
                    action_stats=action_stats,
                )
                last_done = bool(done)

                if clean_branch_active and (clean_branch_pred_tokens is not None):
                    clean_branch_obs, clean_branch_done, clean_branch_rollout_input_ids = self._step_rollout_branch_env(
                        env=clean_branch_env,
                        obs=clean_branch_obs,
                        rollout_input_ids=clean_branch_rollout_input_ids,
                        pred_action_tokens=clean_branch_pred_tokens,
                        action_mask_full=action_mask_full,
                        action_stats=action_stats,
                    )
                    clean_branch_active = not clean_branch_done
                if impulse_branch_active and (impulse_branch_pred_tokens is not None):
                    impulse_branch_obs, impulse_branch_done, impulse_branch_rollout_input_ids = self._step_rollout_branch_env(
                        env=impulse_branch_env,
                        obs=impulse_branch_obs,
                        rollout_input_ids=impulse_branch_rollout_input_ids,
                        pred_action_tokens=impulse_branch_pred_tokens,
                        action_mask_full=action_mask_full,
                        action_stats=action_stats,
                    )
                    impulse_branch_active = not impulse_branch_done
                if deattack_branch_active and (deattack_branch_pred_tokens is not None):
                    deattack_branch_obs, deattack_branch_done, deattack_branch_rollout_input_ids = self._step_rollout_branch_env(
                        env=deattack_branch_env,
                        obs=deattack_branch_obs,
                        rollout_input_ids=deattack_branch_rollout_input_ids,
                        pred_action_tokens=deattack_branch_pred_tokens,
                        action_mask_full=action_mask_full,
                        action_stats=action_stats,
                    )
                    deattack_branch_active = not deattack_branch_done

                if impulse_enabled and (step_idx >= 1) and clean_branch_pred_tokens is not None and (impulse_branch_pred_tokens is not None):
                    impulse_step_area = torch.clamp(impulse_branch_gt_metric - clean_branch_gt_metric, min=0.0)
                    total_impulse_rollout_area += float(impulse_step_area.item())
                    total_impulse_terms += 1

                obs = next_obs

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
            if (
                need_backward
                and (learnable_projector_params is not None)
                and learnable_projector_params.has_trainable_params()
                and (photometric_grads_before_episode is not None)
            ):
                inv_weight_denom = 1.0 / weight_denom
                for param, grad_before in zip(learnable_projector_params.parameters(), photometric_grads_before_episode):
                    if param.grad is None:
                        continue
                    if grad_before is None:
                        param.grad.mul_(inv_weight_denom)
                    else:
                        episode_delta_grad = param.grad - grad_before
                        param.grad.copy_(grad_before + (episode_delta_grad * inv_weight_denom))

            ep_steps = max(1, int(total_action_terms))
            avg_window_rollout_clean_gt_action_gap = (
                total_window_rollout_clean_gt_gap / float(max(1, total_window_rollout_terms))
                if total_window_rollout_terms > 0
                else 0.0
            )
            avg_window_rollout_adv_gt_action_gap = (
                total_window_rollout_adv_gt_gap / float(max(1, total_window_rollout_terms))
                if total_window_rollout_terms > 0
                else 0.0
            )
            avg_window_rollout_deattack_gt_action_gap = (
                total_window_rollout_deattack_gt_gap / float(max(1, total_window_rollout_terms))
                if total_window_rollout_terms > 0
                else 0.0
            )
            avg_window_rollout_selected_gt_action_gap = (
                total_window_rollout_selected_gt_gap / float(max(1, total_window_rollout_terms))
                if total_window_rollout_terms > 0
                else 0.0
            )
            window_rollout_metric_value = select_window_rollout_metric_value(
                metric_mode=window_rollout_metric_mode,
                delta_weighted=float(total_window_rollout_delta_weighted),
                adv_gt_action_gap=avg_window_rollout_selected_gt_action_gap,
            )
            return {
                "loss": total_loss if need_backward else total_loss.detach(),
                "action_gap": total_action_gap / float(ep_steps),
                "gt_action_gap": total_gt_action_gap / float(ep_steps),
                "history_div": total_history_div / float(max(1, total_history_terms)),
                "history_div_legacy": total_history_div_legacy / float(max(1, total_history_terms_legacy)),
                "ce_value": total_ce / float(ep_steps),
                "ce_objective_value": total_ce_objective / float(ep_steps),
                "siglip_distance": total_siglip_distance / float(ep_steps),
                "continuous_clean_gt_action_gap": (
                    total_continuous_clean_gt_gap / float(max(1, total_continuous_terms))
                    if total_continuous_terms > 0
                    else 0.0
                ),
                "continuous_adv_gt_action_gap": (
                    total_continuous_adv_gt_gap / float(max(1, total_continuous_terms))
                    if total_continuous_terms > 0
                    else 0.0
                ),
                "continuous_rollout_delta": (
                    total_continuous_rollout_delta / float(max(1, total_continuous_terms))
                    if total_continuous_terms > 0
                    else 0.0
                ),
                "impulse_rollout_area": total_impulse_rollout_area,
                "window_rollout_clean_gt_action_gap": (
                    avg_window_rollout_clean_gt_action_gap
                ),
                "window_rollout_adv_gt_action_gap": (
                    avg_window_rollout_adv_gt_action_gap
                ),
                "window_rollout_deattack_gt_action_gap": (
                    avg_window_rollout_deattack_gt_action_gap
                ),
                "window_rollout_selected_gt_action_gap": (
                    avg_window_rollout_selected_gt_action_gap
                ),
                "window_rollout_delta_weighted": float(total_window_rollout_delta_weighted),
                "window_rollout_delta_weighted_loss": float(total_window_rollout_delta_weighted_loss),
                "window_rollout_metric_mode": str(window_rollout_metric_mode),
                "window_rollout_future_mode": str(window_rollout_future_mode),
                "window_rollout_metric_value": float(window_rollout_metric_value),
                "window_rollout_future_steps": int(total_window_rollout_terms),
                "action_terms": total_action_terms,
                "history_terms": total_history_terms,
                "history_terms_legacy": total_history_terms_legacy,
                "continuous_terms": total_continuous_terms,
                "impulse_terms": total_impulse_terms,
                "done": last_done,
                "episode_len": total_action_terms,
                "per_joint_gap": self._normalize_joint_values(total_per_joint, float(ep_steps)),
                "gt_per_joint_gap": self._normalize_joint_values(total_gt_per_joint, float(ep_steps)),
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
                "phase_start_name": str(start_phase_name),
                "effective_num_steps_wait": int(effective_num_steps_wait),
                "gt_candidate_count": int(gt_candidate_count),
                "gt_reference_instruction": str(gt_reference_instruction),
                "gt_reference_phase": str(gt_reference_phase),
                "window_phase_name": str(window_rollout_spec.get("phase_name", "")) if window_rollout_enabled else "",
                "window_start_step": float(window_rollout_spec.get("window_start_step", 0.0))
                if window_rollout_enabled
                else 0.0,
                "window_end_step": float(window_rollout_spec.get("window_end_step", 0.0))
                if window_rollout_enabled
                else 0.0,
            }
        finally:
            for branch_env in (clean_branch_env, impulse_branch_env, deattack_branch_env):
                if (branch_env is not None) and hasattr(branch_env, "close"):
                    branch_env.close()

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

    def _sample_init_state_index(self, init_states, iter_idx, local_idx, deterministic_seed=None):
        if len(init_states) == 0:
            return None
        if deterministic_seed is not None:
            ridx = int(deterministic_seed) % len(init_states)
        else:
            ridx = (int(iter_idx) + int(local_idx) + random.randint(0, max(0, len(init_states) - 1))) % len(init_states)
        return int(ridx)

    def _sample_init_state(self, init_states, iter_idx, local_idx, deterministic_seed=None):
        ridx = self._sample_init_state_index(
            init_states=init_states,
            iter_idx=iter_idx,
            local_idx=local_idx,
            deterministic_seed=deterministic_seed,
        )
        if ridx is None:
            return None
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
        for attr_name, filename in self._history_metric_file_map():
            filepath = os.path.join(path, filename)
            value = getattr(self, attr_name)
            with open(filepath, "wb") as file:
                if filename == "loss":
                    torch.save(value, file)
                else:
                    pickle.dump(value, file)
        self._write_run_metadata()
