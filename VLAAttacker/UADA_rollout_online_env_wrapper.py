import argparse
import importlib
import json
import os
import random
import uuid

import numpy as np
import torch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from white_patch.UADA_rollout_online_env import OpenVLAOnlineEnvAttacker

try:
    import wandb
except Exception:
    wandb = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_online_env_dependencies():
    try:
        if importlib.util.find_spec("libero") is None:
            raise ModuleNotFoundError("libero")
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Online env rollout requires LIBERO dependencies, but they are not available. "
            "Please install `libero` and ensure `OffScreenRenderEnv` is importable."
        ) from exc


def resolve_vla_path(dataset: str) -> str:
    if "bridge_orig" in dataset:
        return "openvla/openvla-7b"
    if "libero_spatial" in dataset:
        return "openvla/openvla-7b-finetuned-libero-spatial"
    if "libero_object" in dataset:
        return "openvla/openvla-7b-finetuned-libero-object"
    if "libero_goal" in dataset:
        return "openvla/openvla-7b-finetuned-libero-goal"
    if "libero_10" in dataset:
        return "openvla/openvla-7b-finetuned-libero-10"
    raise ValueError("Invalid dataset")


def main(args):
    check_online_env_dependencies()

    pwd = os.getcwd()
    vla_path = resolve_vla_path(args.dataset)
    resume_run_dir = "" if args.resume_run_dir is None else str(args.resume_run_dir).strip()
    if resume_run_dir.lower() in ("none", "null"):
        resume_run_dir = ""
    if resume_run_dir != "":
        resume_run_dir = os.path.abspath(os.path.expanduser(resume_run_dir))
        if not os.path.isdir(resume_run_dir):
            raise FileNotFoundError(f"resume_run_dir not found: `{resume_run_dir}`.")
        exp_id = os.path.basename(resume_run_dir.rstrip(os.sep))
        path = resume_run_dir
    else:
        exp_id = str(uuid.uuid4())
        path = f"{pwd}/run/UADA_rollout_online_env/{exp_id}"
        os.makedirs(path, exist_ok=True)

    set_seed(42)
    target = "all" if args.use_all_joints else "".join(str(i) for i in args.maskidx)
    projection_size = args.projection_size if args.projection_size is not None else args.patch_size
    task_suite_name = (
        args.task_suite_name
        if str(args.task_suite_name).lower() not in ("auto", "", "none", "null")
        else f"auto({args.dataset})"
    )
    name = (
        f"{args.dataset}_UADA_rollout_onlineEnv_atk{args.attack_mode}_lr{format(args.lr, '.0e')}_iter{args.iter}_"
        f"phase1R{args.phase1_rollout}_phase2R{args.phase2_rollout}_"
        f"lamA{args.lambda_action_gap}_lamH{args.lambda_history}_lamHL{args.lambda_history_legacy}_"
        f"lamC{args.lambda_continuous_rollout}_"
        f"lamW{args.lambda_window_rollout_loss}_"
        f"winProbe{int(args.window_rollout_probe_enabled)}_"
        f"winMetric{args.window_rollout_metric_mode}_"
        f"winFuture{args.window_rollout_future_mode}_"
        f"winA{args.window_rollout_exp_base}_winK{args.window_rollout_future_horizon}_winPhase{args.window_rollout_phase_scope}_"
        f"lamS{args.lambda_siglip}_"
        f"impMetric{int(args.impulse_rollout_metric_enabled)}_"
        f"agMode{args.action_gap_mode}_p1Ag{args.phase1_action_gap_mode}_"
        f"texMode{args.texture_param_mode}_lat{args.latent_hw}_tv{args.lambda_tv}_"
        f"anchorH{args.train_anchor_horizon_iters}_anchorDet{int(args.deterministic_anchor_sampling)}_"
        f"phaseState{args.phase_state_mode}_tau{args.gt_softmin_tau}_"
        f"ceMode{args.online_ce_mode}_probe{int(args.probe_mode)}_{args.probe_variant}_"
        f"autoTune{int(args.auto_gpu_tune)}_{args.gpu_tune_mode}_"
        f"valDet{int(args.val_deterministic)}_valSeed{args.val_seed}_valNoLight{int(args.val_disable_lighting)}_"
        f"allJ{int(args.use_all_joints)}_gripW{args.gripper_weight}_"
        f"envAction{args.env_action_source}_taskBudget{args.online_train_tasks_per_iter}x{args.online_train_episodes_per_task}_"
        f"valEps{args.online_val_episodes}_suite{task_suite_name}_"
        f"lightAug{int(args.lighting_aug_enabled)}_{args.lighting_backend}_scope{args.ic_light_scope}_bg{args.ic_light_bg_control}_"
        f"trainOnly{int(args.lighting_aug_train_only)}_pool{args.lighting_pool_size}_"
        f"p1NoLight{int(args.phase1_disable_lighting)}_p1NoProjRand{int(args.phase1_disable_projection_randomization)}_"
        f"target{target}_proj{projection_size}_seed42-{exp_id}"
    )
    if resume_run_dir != "":
        name = f"{name}-resume"
    if args.use_all_joints:
        print("[Deprecation] --maskidx is ignored when --use_all_joints=true.")

    use_wandb = args.wandb_project != "false" and wandb is not None
    if args.wandb_project != "false" and wandb is None:
        print("Warning: wandb is not available, fallback to local logging only.")

    resume_wandb_id = ""
    if resume_run_dir != "":
        metadata_path = os.path.join(resume_run_dir, "run_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            resume_wandb_id = str(metadata.get("wandb_run_id", "") or "")

    if use_wandb:
        wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": name,
            "tags": args.tags,
        }
        if resume_wandb_id != "":
            wandb_init_kwargs["id"] = resume_wandb_id
            wandb_init_kwargs["resume"] = "must"
        wandb.init(**wandb_init_kwargs)
        args.wandb_run_id = str(getattr(wandb.run, "id", "") or resume_wandb_id)
        wandb.config = {
            "iteration": args.iter,
            "learning_rate": args.lr,
            "attack_target": "all" if args.use_all_joints else args.maskidx,
            "use_all_joints": args.use_all_joints,
            "gripper_weight": args.gripper_weight,
            "accumulate_steps": args.accumulate,
            "phase1_ratio": args.phase1_ratio,
            "phase1_rollout": args.phase1_rollout,
            "phase2_rollout": args.phase2_rollout,
            "lambda_action_gap": args.lambda_action_gap,
            "action_gap_mode": args.action_gap_mode,
            "phase1_action_gap_mode": args.phase1_action_gap_mode,
            "gt_action_bank_path": args.gt_action_bank_path,
            "gt_softmin_tau": args.gt_softmin_tau,
            "phase_state_mode": args.phase_state_mode,
            "phase_state_cache_path": args.phase_state_cache_path,
            "texture_param_mode": args.texture_param_mode,
            "latent_hw": args.latent_hw,
            "lambda_tv": args.lambda_tv,
            "train_anchor_horizon_iters": args.train_anchor_horizon_iters,
            "deterministic_anchor_sampling": args.deterministic_anchor_sampling,
            "lambda_history": args.lambda_history,
            "lambda_history_legacy": args.lambda_history_legacy,
            "lambda_ce": args.lambda_ce,
            "lambda_ce_phase2": args.lambda_ce_phase2,
            "lambda_continuous_rollout": args.lambda_continuous_rollout,
            "lambda_window_rollout_loss": args.lambda_window_rollout_loss,
            "impulse_rollout_metric_enabled": args.impulse_rollout_metric_enabled,
            "window_rollout_probe_enabled": args.window_rollout_probe_enabled,
            "window_rollout_metric_mode": args.window_rollout_metric_mode,
            "window_rollout_future_mode": args.window_rollout_future_mode,
            "window_rollout_exp_base": args.window_rollout_exp_base,
            "window_rollout_future_horizon": args.window_rollout_future_horizon,
            "window_rollout_phase_scope": args.window_rollout_phase_scope,
            "lambda_siglip": args.lambda_siglip,
            "siglip_model_name": args.siglip_model_name,
            "siglip_input_size": args.siglip_input_size,
            "probe_mode": args.probe_mode,
            "probe_variant": args.probe_variant,
            "save_interval": args.save_interval,
            "eval_enabled": args.eval_enabled,
            "val_deterministic": args.val_deterministic,
            "val_seed": args.val_seed,
            "val_disable_lighting": args.val_disable_lighting,
            "lighting_aug_enabled": args.lighting_aug_enabled,
            "lighting_aug_train_only": args.lighting_aug_train_only,
            "phase1_disable_lighting": args.phase1_disable_lighting,
            "projection_randomization_enabled": args.projection_randomization_enabled,
            "phase1_disable_projection_randomization": args.phase1_disable_projection_randomization,
            "lighting_backend": args.lighting_backend,
            "lighting_model_id": args.lighting_model_id,
            "lighting_pool_size": args.lighting_pool_size,
            "lighting_refresh_interval": args.lighting_refresh_interval,
            "lighting_num_inference_steps": args.lighting_num_inference_steps,
            "lighting_guidance_scale": args.lighting_guidance_scale,
            "lighting_blend_min": args.lighting_blend_min,
            "lighting_blend_max": args.lighting_blend_max,
            "lighting_apply_prob": args.lighting_apply_prob,
            "lighting_seed": args.lighting_seed,
            "ic_light_repo": args.ic_light_repo,
            "ic_light_model_path": args.ic_light_model_path,
            "ic_light_scope": args.ic_light_scope,
            "ic_light_bg_control": args.ic_light_bg_control,
            "attack_mode": args.attack_mode,
            "projection_size": projection_size,
            "init_projection_texture_path": args.init_projection_texture_path,
            "resume_run_dir": args.resume_run_dir,
            "projection_alpha": args.projection_alpha,
            "projection_alpha_jitter": args.projection_alpha_jitter,
            "projection_soft_edge": args.projection_soft_edge,
            "projection_angle": args.projection_angle,
            "projection_fixed_angle": args.projection_fixed_angle,
            "projection_shear": args.projection_shear,
            "projection_scale_min": args.projection_scale_min,
            "projection_scale_max": args.projection_scale_max,
            "projection_region": args.projection_region,
            "projection_lower_start": args.projection_lower_start,
            "projection_width_ratio": args.projection_width_ratio,
            "projection_height_ratio": args.projection_height_ratio,
            "projection_margin_x": args.projection_margin_x,
            "projection_keystone": args.projection_keystone,
            "projection_keystone_jitter": args.projection_keystone_jitter,
            "projector_gamma": args.projector_gamma,
            "projector_gain": args.projector_gain,
            "projector_channel_gain": args.projector_channel_gain,
            "learn_projector_gain": args.learn_projector_gain,
            "learn_projector_channel_gain": args.learn_projector_channel_gain,
            "photometric_lr_ratio": args.photometric_lr_ratio,
            "projector_ambient": args.projector_ambient,
            "projector_vignetting": args.projector_vignetting,
            "projector_distance_falloff": args.projector_distance_falloff,
            "projector_psf": args.projector_psf,
            "viz_enabled": args.viz_enabled,
            "viz_policy": args.viz_policy,
            "viz_samples": args.viz_samples,
            "viz_save_best": args.viz_save_best,
            "viz_save_last": args.viz_save_last,
            "record_online_videos": args.record_online_videos,
            "record_online_videos_last_only": args.record_online_videos_last_only,
            "record_online_train_video": args.record_online_train_video,
            "record_online_val_video": args.record_online_val_video,
            "record_online_video_frame_source": args.record_online_video_frame_source,
            "record_online_video_fps": args.record_online_video_fps,
            "task_suite_name": args.task_suite_name,
            "online_train_tasks_per_iter": args.online_train_tasks_per_iter,
            "online_train_episodes_per_task": args.online_train_episodes_per_task,
            "online_val_episodes": args.online_val_episodes,
            "num_steps_wait": args.num_steps_wait,
            "max_env_steps": args.max_env_steps,
            "val_max_env_steps": args.val_max_env_steps,
            "env_resolution": args.env_resolution,
            "online_ce_mode": args.online_ce_mode,
            "env_action_source": args.env_action_source,
            "env_seed": args.env_seed,
            "gt_dataset_root": args.gt_dataset_root,
            "auto_gpu_tune": args.auto_gpu_tune,
            "gpu_tune_mode": args.gpu_tune_mode,
            "gpu_mem_low": args.gpu_mem_low,
            "gpu_mem_high": args.gpu_mem_high,
            "gpu_mem_hard_cap": args.gpu_mem_hard_cap,
            "gpu_util_low": args.gpu_util_low,
            "gpu_tune_cooldown_iters": args.gpu_tune_cooldown_iters,
            "gpu_tune_min_rollout": args.gpu_tune_min_rollout,
            "gpu_tune_max_rollout": args.gpu_tune_max_rollout,
            "gpu_tune_min_tasks_per_iter": args.gpu_tune_min_tasks_per_iter,
            "gpu_tune_max_tasks_per_iter": args.gpu_tune_max_tasks_per_iter,
        }
    else:
        args.wandb_run_id = ""

    print(f"exp_id:{exp_id}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)

    attacker = OpenVLAOnlineEnvAttacker(vla, processor, path, optimizer="adamW", resize_patch=args.resize_patch)
    attacker.online_attack_unconstrained(
        num_iter=args.iter,
        patch_size=args.patch_size,
        projection_size=projection_size,
        init_projection_texture_path=args.init_projection_texture_path,
        resume_run_dir=args.resume_run_dir,
        lr=args.lr,
        accumulate_steps=args.accumulate,
        maskidx=args.maskidx,
        use_all_joints=args.use_all_joints,
        gripper_weight=args.gripper_weight,
        warmup=args.warmup,
        geometry=args.geometry,
        args=args,
        phase1_ratio=args.phase1_ratio,
        phase1_rollout=args.phase1_rollout,
        phase2_rollout=args.phase2_rollout,
        lambda_action_gap=args.lambda_action_gap,
        lambda_history=args.lambda_history,
        lambda_history_legacy=args.lambda_history_legacy,
        lambda_ce=args.lambda_ce,
        lambda_ce_phase2=args.lambda_ce_phase2,
        lambda_continuous_rollout=args.lambda_continuous_rollout,
        lambda_window_rollout_loss=args.lambda_window_rollout_loss,
        impulse_rollout_metric_enabled=args.impulse_rollout_metric_enabled,
        window_rollout_probe_enabled=args.window_rollout_probe_enabled,
        window_rollout_metric_mode=args.window_rollout_metric_mode,
        window_rollout_future_mode=args.window_rollout_future_mode,
        window_rollout_exp_base=args.window_rollout_exp_base,
        window_rollout_future_horizon=args.window_rollout_future_horizon,
        window_rollout_phase_scope=args.window_rollout_phase_scope,
        lambda_siglip=args.lambda_siglip,
        siglip_model_name=args.siglip_model_name,
        siglip_input_size=args.siglip_input_size,
        save_interval=args.save_interval,
        eval_enabled=args.eval_enabled,
        lighting_aug_enabled=args.lighting_aug_enabled,
        lighting_aug_train_only=args.lighting_aug_train_only,
        lighting_backend=args.lighting_backend,
        lighting_model_id=args.lighting_model_id,
        lighting_pool_size=args.lighting_pool_size,
        lighting_refresh_interval=args.lighting_refresh_interval,
        lighting_num_inference_steps=args.lighting_num_inference_steps,
        lighting_guidance_scale=args.lighting_guidance_scale,
        lighting_blend_min=args.lighting_blend_min,
        lighting_blend_max=args.lighting_blend_max,
        lighting_apply_prob=args.lighting_apply_prob,
        lighting_seed=args.lighting_seed,
        ic_light_repo=args.ic_light_repo,
        ic_light_model_path=args.ic_light_model_path,
        ic_light_scope=args.ic_light_scope,
        ic_light_bg_control=args.ic_light_bg_control,
        attack_mode=args.attack_mode,
        phase1_disable_lighting=args.phase1_disable_lighting,
        projection_randomization_enabled=args.projection_randomization_enabled,
        phase1_disable_projection_randomization=args.phase1_disable_projection_randomization,
        projection_alpha=args.projection_alpha,
        projection_alpha_jitter=args.projection_alpha_jitter,
        projection_soft_edge=args.projection_soft_edge,
        projection_angle=args.projection_angle,
        projection_fixed_angle=args.projection_fixed_angle,
        projection_shear=args.projection_shear,
        projection_scale_min=args.projection_scale_min,
        projection_scale_max=args.projection_scale_max,
        projection_region=args.projection_region,
        projection_lower_start=args.projection_lower_start,
        projection_width_ratio=args.projection_width_ratio,
        projection_height_ratio=args.projection_height_ratio,
        projection_margin_x=args.projection_margin_x,
        projection_keystone=args.projection_keystone,
        projection_keystone_jitter=args.projection_keystone_jitter,
        projector_gamma=args.projector_gamma,
        projector_gain=args.projector_gain,
        projector_channel_gain=args.projector_channel_gain,
        learn_projector_gain=args.learn_projector_gain,
        learn_projector_channel_gain=args.learn_projector_channel_gain,
        photometric_lr_ratio=args.photometric_lr_ratio,
        projector_ambient=args.projector_ambient,
        projector_vignetting=args.projector_vignetting,
        projector_distance_falloff=args.projector_distance_falloff,
        projector_psf=args.projector_psf,
        viz_enabled=args.viz_enabled,
        viz_policy=args.viz_policy,
        viz_samples=args.viz_samples,
        viz_save_best=args.viz_save_best,
        viz_save_last=args.viz_save_last,
        record_online_videos=args.record_online_videos,
        record_online_videos_last_only=args.record_online_videos_last_only,
        record_online_train_video=args.record_online_train_video,
        record_online_val_video=args.record_online_val_video,
        record_online_video_frame_source=args.record_online_video_frame_source,
        record_online_video_fps=args.record_online_video_fps,
        task_suite_name=args.task_suite_name,
        online_train_tasks_per_iter=args.online_train_tasks_per_iter,
        online_train_episodes_per_task=args.online_train_episodes_per_task,
        online_val_episodes=args.online_val_episodes,
        num_steps_wait=args.num_steps_wait,
        max_env_steps=args.max_env_steps,
        val_max_env_steps=args.val_max_env_steps,
        env_resolution=args.env_resolution,
        online_ce_mode=args.online_ce_mode,
        env_action_source=args.env_action_source,
        env_seed=args.env_seed,
        dataset_name=args.dataset,
        action_gap_mode=args.action_gap_mode,
        phase1_action_gap_mode=args.phase1_action_gap_mode,
        texture_param_mode=args.texture_param_mode,
        latent_hw=args.latent_hw,
        lambda_tv=args.lambda_tv,
        train_anchor_horizon_iters=args.train_anchor_horizon_iters,
        deterministic_anchor_sampling=args.deterministic_anchor_sampling,
        gt_dataset_root=args.gt_dataset_root,
        gt_action_bank_path=args.gt_action_bank_path,
        gt_softmin_tau=args.gt_softmin_tau,
        phase_state_mode=args.phase_state_mode,
        phase_state_cache_path=args.phase_state_cache_path,
        val_deterministic=args.val_deterministic,
        val_seed=args.val_seed,
        val_disable_lighting=args.val_disable_lighting,
        probe_mode=args.probe_mode,
        probe_variant=args.probe_variant,
        auto_gpu_tune=args.auto_gpu_tune,
        gpu_tune_mode=args.gpu_tune_mode,
        gpu_mem_low=args.gpu_mem_low,
        gpu_mem_high=args.gpu_mem_high,
        gpu_mem_hard_cap=args.gpu_mem_hard_cap,
        gpu_util_low=args.gpu_util_low,
        gpu_tune_cooldown_iters=args.gpu_tune_cooldown_iters,
        gpu_tune_min_rollout=args.gpu_tune_min_rollout,
        gpu_tune_max_rollout=args.gpu_tune_max_rollout,
        gpu_tune_min_tasks_per_iter=args.gpu_tune_min_tasks_per_iter,
        gpu_tune_max_tasks_per_iter=args.gpu_tune_max_tasks_per_iter,
    )
    print("Online env rollout attack done!")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maskidx", default="0,1,2", type=list_of_ints)
    parser.add_argument("--use_all_joints", type=str2bool, default=False)
    parser.add_argument("--gripper_weight", default=0.5, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--iter", default=2000, type=int)
    parser.add_argument("--accumulate", default=1, type=int)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--tags", nargs="+", default=[""])
    parser.add_argument("--geometry", type=str2bool, nargs="?", default=True)
    parser.add_argument("--patch_size", default="3,50,50", type=list_of_ints)
    parser.add_argument("--attack_mode", default="projection", type=str)
    parser.add_argument("--projection_size", default=None, type=list_of_ints_or_none)
    parser.add_argument("--init_projection_texture_path", default="", type=str)
    parser.add_argument("--resume_run_dir", default="", type=str)
    parser.add_argument("--projection_alpha", default=0.55, type=float)
    parser.add_argument("--projection_alpha_jitter", default=0.10, type=float)
    parser.add_argument("--projection_soft_edge", default=1.2, type=float)
    parser.add_argument("--projection_angle", default=25.0, type=float)
    parser.add_argument("--projection_fixed_angle", type=str2bool, default=False)
    parser.add_argument("--projection_shear", default=0.15, type=float)
    parser.add_argument("--projection_scale_min", default=0.8, type=float)
    parser.add_argument("--projection_scale_max", default=1.2, type=float)
    parser.add_argument("--projection_region", default="lower_half_fixed", type=str)
    parser.add_argument("--projection_lower_start", default=0.5, type=float)
    parser.add_argument("--projection_width_ratio", default=0.96, type=float)
    parser.add_argument("--projection_height_ratio", default=1.0, type=float)
    parser.add_argument("--projection_margin_x", default=0.02, type=float)
    parser.add_argument("--projection_keystone", default=0.12, type=float)
    parser.add_argument("--projection_keystone_jitter", default=0.03, type=float)
    parser.add_argument("--projector_gamma", default=1.8, type=float)
    parser.add_argument("--projector_gain", default=1.35, type=float)
    parser.add_argument("--projector_channel_gain", default="1.08,1.04,1.00", type=list_of_floats)
    parser.add_argument("--learn_projector_gain", type=str2bool, default=False)
    parser.add_argument("--learn_projector_channel_gain", type=str2bool, default=False)
    parser.add_argument("--photometric_lr_ratio", default=0.1, type=float)
    parser.add_argument("--projector_ambient", default=0.08, type=float)
    parser.add_argument("--projector_vignetting", default=0.08, type=float)
    parser.add_argument("--projector_distance_falloff", default=0.10, type=float)
    parser.add_argument("--projector_psf", type=str2bool, default=False)
    parser.add_argument("--wandb_project", default="false", type=str)
    parser.add_argument("--wandb_entity", default="xxx", type=str)
    parser.add_argument("--dataset", default="libero_10", type=str)
    parser.add_argument("--resize_patch", type=str2bool, default=False)
    parser.add_argument("--server", default="/home/yxx/projectAttack", type=str)

    parser.add_argument("--phase1_ratio", default=0.4, type=float)
    parser.add_argument("--phase1_rollout", default=8, type=int)
    parser.add_argument("--phase2_rollout", default=24, type=int)
    parser.add_argument("--lambda_action_gap", default=1.0, type=float)
    parser.add_argument("--action_gap_mode", default="gt_farthest", type=str)
    parser.add_argument("--phase1_action_gap_mode", default="inherit", type=str)
    parser.add_argument("--gt_action_bank_path", default="", type=str)
    parser.add_argument("--gt_softmin_tau", default=0.05, type=float)
    parser.add_argument("--phase_state_mode", default="phase_cycle", type=str)
    parser.add_argument("--phase_state_cache_path", default="", type=str)
    parser.add_argument("--texture_param_mode", default="direct", type=str)
    parser.add_argument("--latent_hw", default="12,12", type=list_of_ints)
    parser.add_argument("--lambda_tv", default=0.0, type=float)
    parser.add_argument("--train_anchor_horizon_iters", default=1, type=int)
    parser.add_argument("--deterministic_anchor_sampling", type=str2bool, default=False)
    parser.add_argument("--lambda_history", default=0.0, type=float)
    parser.add_argument("--lambda_history_legacy", default=0.0, type=float)
    parser.add_argument("--lambda_ce", default=0.02, type=float)
    parser.add_argument("--lambda_ce_phase2", default=0.0, type=float)
    parser.add_argument("--lambda_continuous_rollout", default=0.0, type=float)
    parser.add_argument("--lambda_window_rollout_loss", default=0.0, type=float)
    parser.add_argument("--impulse_rollout_metric_enabled", type=str2bool, default=False)
    parser.add_argument("--window_rollout_probe_enabled", type=str2bool, default=False)
    parser.add_argument("--window_rollout_metric_mode", default="delta_weighted", type=str)
    parser.add_argument("--window_rollout_future_mode", default="keep_adv", type=str)
    parser.add_argument("--window_rollout_exp_base", default=0.9, type=float)
    parser.add_argument("--window_rollout_future_horizon", default=8, type=int)
    parser.add_argument("--window_rollout_phase_scope", default="all", type=str)
    parser.add_argument("--lambda_siglip", default=0.15, type=float)
    parser.add_argument("--siglip_model_name", default="google/siglip-so400m-patch14-384", type=str)
    parser.add_argument("--siglip_input_size", default=384, type=int)
    parser.add_argument("--probe_mode", type=str2bool, default=False)
    parser.add_argument("--probe_variant", default="", type=str)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--eval_enabled", type=str2bool, default=True)
    parser.add_argument("--val_deterministic", type=str2bool, default=False)
    parser.add_argument("--val_seed", default=42, type=int)
    parser.add_argument("--val_disable_lighting", type=str2bool, default=False)
    parser.add_argument("--lighting_aug_enabled", type=str2bool, default=False)
    parser.add_argument("--lighting_aug_train_only", type=str2bool, default=False)
    parser.add_argument("--phase1_disable_lighting", type=str2bool, default=False)
    parser.add_argument("--projection_randomization_enabled", type=str2bool, default=True)
    parser.add_argument("--phase1_disable_projection_randomization", type=str2bool, default=False)
    parser.add_argument("--lighting_backend", default="ic_light", type=str)
    parser.add_argument("--lighting_model_id", default="stabilityai/sdxl-turbo", type=str)
    parser.add_argument("--lighting_pool_size", default=8, type=int)
    parser.add_argument("--lighting_refresh_interval", default=200, type=int)
    parser.add_argument("--lighting_num_inference_steps", default=8, type=int)
    parser.add_argument("--lighting_guidance_scale", default=7.0, type=float)
    parser.add_argument("--lighting_blend_min", default=0.15, type=float)
    parser.add_argument("--lighting_blend_max", default=0.5, type=float)
    parser.add_argument("--lighting_apply_prob", default=1.0, type=float)
    parser.add_argument("--lighting_seed", default=42, type=int)
    parser.add_argument("--ic_light_repo", default="/home/yxx/IC-Light", type=str)
    parser.add_argument(
        "--ic_light_model_path",
        default="/home/yxx/IC-Light/models/iclight_sd15_fbc.safetensors",
        type=str,
    )
    parser.add_argument("--ic_light_scope", default="full", type=str)
    parser.add_argument("--ic_light_bg_control", default="legacy_prompt", type=str)
    parser.add_argument("--viz_enabled", type=str2bool, default=True)
    parser.add_argument("--viz_policy", default="milestone", type=str)
    parser.add_argument("--viz_samples", default=4, type=int)
    parser.add_argument("--viz_save_best", type=str2bool, default=True)
    parser.add_argument("--viz_save_last", type=str2bool, default=True)
    parser.add_argument("--record_online_videos", type=str2bool, default=False)
    parser.add_argument("--record_online_videos_last_only", type=str2bool, default=True)
    parser.add_argument("--record_online_train_video", type=str2bool, default=False)
    parser.add_argument("--record_online_val_video", type=str2bool, default=False)
    parser.add_argument("--record_online_video_frame_source", default="projected_input", type=str)
    parser.add_argument("--record_online_video_fps", default=10, type=int)

    parser.add_argument("--task_suite_name", default="auto", type=str)
    parser.add_argument("--online_train_tasks_per_iter", default=1, type=int)
    parser.add_argument("--online_train_episodes_per_task", default=10, type=int)
    parser.add_argument("--online_val_episodes", default=8, type=int)
    parser.add_argument("--num_steps_wait", default=10, type=int)
    parser.add_argument("--max_env_steps", default="auto_by_suite", type=str)
    parser.add_argument("--val_max_env_steps", default=120, type=int)
    parser.add_argument("--env_resolution", default=256, type=int)
    parser.add_argument("--online_ce_mode", default="pseudo_clean", type=str)
    parser.add_argument("--env_action_source", default="adv", type=str)
    parser.add_argument("--env_seed", default=42, type=int)
    parser.add_argument("--gt_dataset_root", default="/home/yxx/roboticAttack/openvla-main/dataset", type=str)
    parser.add_argument("--auto_gpu_tune", type=str2bool, default=False)
    parser.add_argument("--gpu_tune_mode", default="stable", type=str)
    parser.add_argument("--gpu_mem_low", default=0.82, type=float)
    parser.add_argument("--gpu_mem_high", default=0.92, type=float)
    parser.add_argument("--gpu_mem_hard_cap", default=0.95, type=float)
    parser.add_argument("--gpu_util_low", default=70, type=int)
    parser.add_argument("--gpu_tune_cooldown_iters", default=2, type=int)
    parser.add_argument("--gpu_tune_min_rollout", default=8, type=int)
    parser.add_argument("--gpu_tune_max_rollout", default=192, type=int)
    parser.add_argument("--gpu_tune_min_tasks_per_iter", default=1, type=int)
    parser.add_argument("--gpu_tune_max_tasks_per_iter", default=2, type=int)
    return parser.parse_args()


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def list_of_ints_or_none(arg):
    if arg is None:
        return None
    value = str(arg).strip().lower()
    if value in ("none", "null", ""):
        return None
    return list(map(int, str(arg).split(",")))


def list_of_floats(arg):
    if isinstance(arg, (list, tuple)):
        return [float(x) for x in arg]
    return [float(x) for x in str(arg).split(",")]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = arg_parser()
    print(
        "Paramters:\n"
        f" dataset:{args.dataset}\n task_suite_name:{args.task_suite_name}\n"
        f" online_train_tasks_per_iter:{args.online_train_tasks_per_iter}\n"
        f" online_train_episodes_per_task:{args.online_train_episodes_per_task}\n"
        f" online_val_episodes:{args.online_val_episodes}\n"
        f" num_steps_wait:{args.num_steps_wait}\n max_env_steps:{args.max_env_steps}\n"
        f" val_max_env_steps:{args.val_max_env_steps}\n"
        f" env_resolution:{args.env_resolution}\n online_ce_mode:{args.online_ce_mode}\n"
        f" env_action_source:{args.env_action_source}\n env_seed:{args.env_seed}\n"
        f" auto_gpu_tune:{args.auto_gpu_tune}\n gpu_tune_mode:{args.gpu_tune_mode}\n"
        f" gpu_mem_low:{args.gpu_mem_low}\n gpu_mem_high:{args.gpu_mem_high}\n"
        f" gpu_mem_hard_cap:{args.gpu_mem_hard_cap}\n gpu_util_low:{args.gpu_util_low}\n"
        f" gpu_tune_cooldown_iters:{args.gpu_tune_cooldown_iters}\n"
        f" gpu_tune_min_rollout:{args.gpu_tune_min_rollout}\n gpu_tune_max_rollout:{args.gpu_tune_max_rollout}\n"
        f" gpu_tune_min_tasks_per_iter:{args.gpu_tune_min_tasks_per_iter}\n"
        f" gpu_tune_max_tasks_per_iter:{args.gpu_tune_max_tasks_per_iter}\n"
        f" maskidx:{args.maskidx}\n use_all_joints:{args.use_all_joints}\n gripper_weight:{args.gripper_weight}\n"
        f" lr:{args.lr}\n device:{args.device}\n tags:{args.tags}\n"
        f" attack_mode:{args.attack_mode}\n projection_size:{args.projection_size}\n"
        f" init_projection_texture_path:{args.init_projection_texture_path}\n"
        f" projector_gain:{args.projector_gain}\n projector_channel_gain:{args.projector_channel_gain}\n"
        f" learn_projector_gain:{args.learn_projector_gain}\n"
        f" learn_projector_channel_gain:{args.learn_projector_channel_gain}\n"
        f" photometric_lr_ratio:{args.photometric_lr_ratio}\n"
        f" phase1_ratio:{args.phase1_ratio}\n phase1_rollout:{args.phase1_rollout}\n"
        f" phase2_rollout:{args.phase2_rollout}\n lambda_action_gap:{args.lambda_action_gap}\n"
        f" action_gap_mode:{args.action_gap_mode}\n phase1_action_gap_mode:{args.phase1_action_gap_mode}\n"
        f" texture_param_mode:{args.texture_param_mode}\n latent_hw:{args.latent_hw}\n"
        f" lambda_tv:{args.lambda_tv}\n train_anchor_horizon_iters:{args.train_anchor_horizon_iters}\n"
        f" deterministic_anchor_sampling:{args.deterministic_anchor_sampling}\n"
        f" gt_dataset_root:{args.gt_dataset_root}\n"
        f" gt_action_bank_path:{args.gt_action_bank_path}\n gt_softmin_tau:{args.gt_softmin_tau}\n"
        f" phase_state_mode:{args.phase_state_mode}\n phase_state_cache_path:{args.phase_state_cache_path}\n"
        f" lambda_history:{args.lambda_history}\n lambda_history_legacy:{args.lambda_history_legacy}\n"
        f" lambda_ce:{args.lambda_ce}\n lambda_ce_phase2:{args.lambda_ce_phase2}\n"
        f" lambda_continuous_rollout:{args.lambda_continuous_rollout}\n"
        f" lambda_window_rollout_loss:{args.lambda_window_rollout_loss}\n"
        f" impulse_rollout_metric_enabled:{args.impulse_rollout_metric_enabled}\n"
        f" window_rollout_probe_enabled:{args.window_rollout_probe_enabled}\n"
        f" window_rollout_metric_mode:{args.window_rollout_metric_mode}\n"
        f" window_rollout_future_mode:{args.window_rollout_future_mode}\n"
        f" window_rollout_exp_base:{args.window_rollout_exp_base}\n"
        f" window_rollout_future_horizon:{args.window_rollout_future_horizon}\n"
        f" window_rollout_phase_scope:{args.window_rollout_phase_scope}\n"
        f" lambda_siglip:{args.lambda_siglip}\n"
        f" siglip_model_name:{args.siglip_model_name}\n siglip_input_size:{args.siglip_input_size}\n"
        f" probe_mode:{args.probe_mode}\n probe_variant:{args.probe_variant}\n"
        f" save_interval:{args.save_interval}\n eval_enabled:{args.eval_enabled}\n"
        f" val_deterministic:{args.val_deterministic}\n val_seed:{args.val_seed}\n"
        f" val_disable_lighting:{args.val_disable_lighting}\n"
        f" phase1_disable_lighting:{args.phase1_disable_lighting}\n"
        f" projection_randomization_enabled:{args.projection_randomization_enabled}\n"
        f" phase1_disable_projection_randomization:{args.phase1_disable_projection_randomization}\n"
        f" lighting_backend:{args.lighting_backend}\n lighting_model_id:{args.lighting_model_id}\n"
        f" lighting_pool_size:{args.lighting_pool_size}\n lighting_refresh_interval:{args.lighting_refresh_interval}\n"
        f" lighting_num_inference_steps:{args.lighting_num_inference_steps}\n lighting_guidance_scale:{args.lighting_guidance_scale}\n"
        f" lighting_blend_min:{args.lighting_blend_min}\n lighting_blend_max:{args.lighting_blend_max}\n"
        f" lighting_apply_prob:{args.lighting_apply_prob}\n lighting_seed:{args.lighting_seed}\n"
        f" ic_light_repo:{args.ic_light_repo}\n ic_light_model_path:{args.ic_light_model_path}\n"
        f" ic_light_scope:{args.ic_light_scope}\n ic_light_bg_control:{args.ic_light_bg_control}\n"
        f" record_online_videos:{args.record_online_videos}\n"
        f" record_online_videos_last_only:{args.record_online_videos_last_only}\n"
        f" record_online_train_video:{args.record_online_train_video}\n"
        f" record_online_val_video:{args.record_online_val_video}\n"
        f" record_online_video_frame_source:{args.record_online_video_frame_source}\n"
        f" record_online_video_fps:{args.record_online_video_fps}\n"
    )
    main(args)
