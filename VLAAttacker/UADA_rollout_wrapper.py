import argparse
import os
import random
import uuid

import numpy as np
import torch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from white_patch.openvla_dataloader import get_dataloader
from white_patch.UADA_rollout import OpenVLAAttacker

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
    pwd = os.getcwd()
    exp_id = str(uuid.uuid4())
    vla_path = resolve_vla_path(args.dataset)

    set_seed(42)
    target = "all" if args.use_all_joints else "".join(str(i) for i in args.maskidx)
    projection_size = args.projection_size if args.projection_size is not None else args.patch_size
    name = (
        f"{args.dataset}_UADA_rollout_v2_atk{args.attack_mode}_lr{format(args.lr, '.0e')}_iter{args.iter}_"
        f"phase1R{args.phase1_rollout}_phase2R{args.phase2_rollout}_"
        f"lamA{args.lambda_action_gap}_lamH{args.lambda_history}_lamCE{args.lambda_ce}_"
        f"valDet{int(args.val_deterministic)}_valSeed{args.val_seed}_valNoLight{int(args.val_disable_lighting)}_"
        f"allJ{int(args.use_all_joints)}_gripW{args.gripper_weight}_"
        f"lightAug{int(args.lighting_aug_enabled)}_{args.lighting_backend}_scope{args.ic_light_scope}_bg{args.ic_light_bg_control}_"
        f"trainOnly{int(args.lighting_aug_train_only)}_pool{args.lighting_pool_size}_"
        f"target{target}_proj{projection_size}_seed42-{exp_id}"
    )
    if args.use_all_joints:
        print("[Deprecation] --maskidx is ignored when --use_all_joints=true.")

    use_wandb = args.wandb_project != "false" and wandb is not None
    if args.wandb_project != "false" and wandb is None:
        print("Warning: wandb is not available, fallback to local logging only.")

    if use_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=name, tags=args.tags)
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
            "lambda_history": args.lambda_history,
            "lambda_ce": args.lambda_ce,
            "eval_rollout": args.eval_rollout,
            "save_interval": args.save_interval,
            "eval_enabled": args.eval_enabled,
            "eval_visual_only": args.eval_visual_only,
            "val_max_batches": args.val_max_batches,
            "val_deterministic": args.val_deterministic,
            "val_seed": args.val_seed,
            "val_disable_lighting": args.val_disable_lighting,
            "lighting_aug_enabled": args.lighting_aug_enabled,
            "lighting_aug_train_only": args.lighting_aug_train_only,
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
            "projection_randomization_enabled": args.projection_randomization_enabled,
            "projector_gamma": args.projector_gamma,
            "projector_gain": args.projector_gain,
            "projector_channel_gain": args.projector_channel_gain,
            "projector_ambient": args.projector_ambient,
            "projector_vignetting": args.projector_vignetting,
            "projector_distance_falloff": args.projector_distance_falloff,
            "projector_psf": args.projector_psf,
            "action_gap_mode": args.action_gap_mode,
            "lambda_siglip": args.lambda_siglip,
            "siglip_model_name": args.siglip_model_name,
            "siglip_device": args.siglip_device,
            "siglip_input_size": args.siglip_input_size,
            "gt_softmin_tau": args.gt_softmin_tau,
            "gt_action_bank_path": args.gt_action_bank_path,
            "offline_phase_scope": args.offline_phase_scope,
            "phase_state_cache_path": args.phase_state_cache_path,
            "offline_phase_fallback_enabled": args.offline_phase_fallback_enabled,
            "viz_enabled": args.viz_enabled,
            "viz_policy": args.viz_policy,
            "viz_samples": args.viz_samples,
            "viz_save_best": args.viz_save_best,
            "viz_save_last": args.viz_save_last,
        }

    print(f"exp_id:{exp_id}")
    path = f"{pwd}/run/UADA_rollout/{exp_id}"
    os.makedirs(path, exist_ok=True)

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

    train_dataloader, val_dataloader = get_dataloader(batch_size=args.bs, server=args.server, dataset=args.dataset)
    openvla_attacker = OpenVLAAttacker(vla, processor, path, optimizer="adamW", resize_patch=args.resize_patch)

    openvla_attacker.patchattack_unconstrained(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_iter=args.iter,
        target_action=np.zeros(7),
        patch_size=args.patch_size,
        projection_size=projection_size,
        lr=args.lr,
        accumulate_steps=args.accumulate,
        maskidx=args.maskidx,
        use_all_joints=args.use_all_joints,
        gripper_weight=args.gripper_weight,
        warmup=args.warmup,
        filterGripTrainTo1=args.filterGripTrainTo1,
        geometry=args.geometry,
        innerLoop=args.innerLoop,
        args=args,
        phase1_ratio=args.phase1_ratio,
        phase1_rollout=args.phase1_rollout,
        phase2_rollout=args.phase2_rollout,
        lambda_action_gap=args.lambda_action_gap,
        lambda_history=args.lambda_history,
        lambda_ce=args.lambda_ce,
        eval_rollout=args.eval_rollout,
        save_interval=args.save_interval,
        eval_enabled=args.eval_enabled,
        eval_visual_only=args.eval_visual_only,
        val_max_batches=args.val_max_batches,
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
        projector_ambient=args.projector_ambient,
        projector_vignetting=args.projector_vignetting,
        projector_distance_falloff=args.projector_distance_falloff,
        projector_psf=args.projector_psf,
        projection_randomization_enabled=args.projection_randomization_enabled,
        action_gap_mode=args.action_gap_mode,
        lambda_siglip=args.lambda_siglip,
        siglip_model_name=args.siglip_model_name,
        siglip_device=args.siglip_device,
        siglip_input_size=args.siglip_input_size,
        gt_softmin_tau=args.gt_softmin_tau,
        gt_action_bank_path=args.gt_action_bank_path,
        offline_phase_scope=args.offline_phase_scope,
        phase_state_cache_path=args.phase_state_cache_path,
        offline_phase_fallback_enabled=args.offline_phase_fallback_enabled,
        viz_enabled=args.viz_enabled,
        viz_policy=args.viz_policy,
        viz_samples=args.viz_samples,
        viz_save_best=args.viz_save_best,
        viz_save_last=args.viz_save_last,
        val_deterministic=args.val_deterministic,
        val_seed=args.val_seed,
        val_disable_lighting=args.val_disable_lighting,
        sanity_mode=args.sanity_mode,
        sanity_num_batches=args.sanity_num_batches,
        sanity_report_interval=args.sanity_report_interval,
        sanity_disable_randomization=args.sanity_disable_randomization,
    )
    print("Rollout-v2 attack done!")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maskidx", default="0,1,2", type=list_of_ints)
    parser.add_argument("--use_all_joints", type=str2bool, default=False)
    parser.add_argument("--gripper_weight", default=0.5, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--iter", default=2000, type=int)
    parser.add_argument("--accumulate", default=1, type=int)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--tags", nargs="+", default=[""])
    parser.add_argument(
        "--filterGripTrainTo1",
        type=str2bool,
        nargs="?",
        default=False,
        help="Remove gripper-0 training samples when attacking target grip dimension to 0.",
    )
    parser.add_argument("--geometry", type=str2bool, nargs="?", default=True, help="Apply geometry transforms to patch.")
    parser.add_argument("--patch_size", default="3,50,50", type=list_of_ints)
    parser.add_argument("--attack_mode", default="projection", type=str)
    parser.add_argument("--projection_size", default=None, type=list_of_ints_or_none)
    parser.add_argument("--projection_alpha", default=0.35, type=float)
    parser.add_argument("--projection_alpha_jitter", default=0.10, type=float)
    parser.add_argument("--projection_soft_edge", default=2.5, type=float)
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
    parser.add_argument("--projection_randomization_enabled", type=str2bool, default=True)
    parser.add_argument("--projector_gamma", default=2.2, type=float)
    parser.add_argument("--projector_gain", default=1.0, type=float)
    parser.add_argument("--projector_channel_gain", default="1.00,0.97,0.94", type=list_of_floats)
    parser.add_argument("--projector_ambient", default=0.04, type=float)
    parser.add_argument("--projector_vignetting", default=0.18, type=float)
    parser.add_argument("--projector_distance_falloff", default=0.25, type=float)
    parser.add_argument("--projector_psf", type=str2bool, default=False)
    parser.add_argument("--wandb_project", default="false", type=str)
    parser.add_argument("--wandb_entity", default="xxx", type=str)
    parser.add_argument("--innerLoop", default=50, type=int)
    parser.add_argument("--dataset", default="bridge_orig", type=str)
    parser.add_argument("--resize_patch", type=str2bool, default=False)
    parser.add_argument("--reverse_direction", type=str2bool, default=True)
    parser.add_argument("--server", default="/home/yxx/projectAttack", type=str)

    parser.add_argument("--phase1_ratio", default=0.4, type=float)
    parser.add_argument("--phase1_rollout", default=8, type=int)
    parser.add_argument("--phase2_rollout", default=24, type=int)
    parser.add_argument("--lambda_action_gap", default=1.0, type=float)
    parser.add_argument("--action_gap_mode", default="gt_farthest", type=str)
    parser.add_argument("--lambda_siglip", default=0.15, type=float)
    parser.add_argument("--siglip_model_name", default="google/siglip-so400m-patch14-384", type=str)
    parser.add_argument("--siglip_device", default="auto", type=str)
    parser.add_argument("--siglip_input_size", default=384, type=int)
    parser.add_argument("--gt_softmin_tau", default=0.05, type=float)
    parser.add_argument("--gt_action_bank_path", default="", type=str)
    parser.add_argument("--offline_phase_scope", default="contact_manipulate", type=str)
    parser.add_argument("--phase_state_cache_path", default="", type=str)
    parser.add_argument("--offline_phase_fallback_enabled", type=str2bool, default=True)
    parser.add_argument("--lambda_history", default=0.5, type=float)
    parser.add_argument("--lambda_ce", default=0.1, type=float)
    parser.add_argument("--eval_rollout", default=24, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--eval_enabled", type=str2bool, default=True)
    parser.add_argument("--eval_visual_only", type=str2bool, default=False)
    parser.add_argument("--val_max_batches", default=1000, type=int)
    parser.add_argument("--val_deterministic", type=str2bool, default=False)
    parser.add_argument("--val_seed", default=42, type=int)
    parser.add_argument("--val_disable_lighting", type=str2bool, default=False)
    parser.add_argument("--lighting_aug_enabled", type=str2bool, default=False)
    parser.add_argument("--lighting_aug_train_only", type=str2bool, default=False)
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
    parser.add_argument("--sanity_mode", type=str2bool, default=False)
    parser.add_argument("--sanity_num_batches", default=1, type=int)
    parser.add_argument("--sanity_report_interval", default=1, type=int)
    parser.add_argument("--sanity_disable_randomization", type=str2bool, default=True)
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
        f" maskidx:{args.maskidx}\n use_all_joints:{args.use_all_joints}\n gripper_weight:{args.gripper_weight}\n"
        f" lr:{args.lr}\n device:{args.device}\n tags:{args.tags}\n"
        f" attack_mode:{args.attack_mode}\n projection_size:{args.projection_size}\n"
        f" projection_alpha:{args.projection_alpha}\n projection_alpha_jitter:{args.projection_alpha_jitter}\n"
        f" projection_soft_edge:{args.projection_soft_edge}\n projection_angle:{args.projection_angle}\n"
        f" projection_fixed_angle:{args.projection_fixed_angle}\n"
        f" projection_shear:{args.projection_shear}\n projection_scale_min:{args.projection_scale_min}\n"
        f" projection_scale_max:{args.projection_scale_max}\n projection_region:{args.projection_region}\n"
        f" projection_lower_start:{args.projection_lower_start}\n projection_width_ratio:{args.projection_width_ratio}\n"
        f" projection_height_ratio:{args.projection_height_ratio}\n projection_margin_x:{args.projection_margin_x}\n"
        f" projection_keystone:{args.projection_keystone}\n projection_keystone_jitter:{args.projection_keystone_jitter}\n"
        f" projection_randomization_enabled:{args.projection_randomization_enabled}\n"
        f" projector_gamma:{args.projector_gamma}\n projector_gain:{args.projector_gain}\n"
        f" projector_channel_gain:{args.projector_channel_gain}\n projector_ambient:{args.projector_ambient}\n"
        f" projector_vignetting:{args.projector_vignetting}\n projector_distance_falloff:{args.projector_distance_falloff}\n"
        f" projector_psf:{args.projector_psf}\n"
        f" phase1_ratio:{args.phase1_ratio}\n phase1_rollout:{args.phase1_rollout}\n phase2_rollout:{args.phase2_rollout}\n"
        f" action_gap_mode:{args.action_gap_mode}\n lambda_action_gap:{args.lambda_action_gap}\n"
        f" lambda_siglip:{args.lambda_siglip}\n siglip_model_name:{args.siglip_model_name}\n"
        f" siglip_device:{args.siglip_device}\n"
        f" siglip_input_size:{args.siglip_input_size}\n gt_softmin_tau:{args.gt_softmin_tau}\n"
        f" gt_action_bank_path:{args.gt_action_bank_path}\n offline_phase_scope:{args.offline_phase_scope}\n"
        f" phase_state_cache_path:{args.phase_state_cache_path}\n"
        f" offline_phase_fallback_enabled:{args.offline_phase_fallback_enabled}\n"
        f" lambda_history:{args.lambda_history}\n lambda_ce:{args.lambda_ce}\n"
        f" eval_rollout:{args.eval_rollout}\n save_interval:{args.save_interval}\n"
        f" eval_enabled:{args.eval_enabled}\n eval_visual_only:{args.eval_visual_only}\n"
        f" val_max_batches:{args.val_max_batches}\n"
        f" val_deterministic:{args.val_deterministic}\n val_seed:{args.val_seed}\n"
        f" val_disable_lighting:{args.val_disable_lighting}\n"
        f" lighting_aug_enabled:{args.lighting_aug_enabled}\n"
        f" lighting_aug_train_only:{args.lighting_aug_train_only}\n"
        f" lighting_backend:{args.lighting_backend}\n"
        f" lighting_model_id:{args.lighting_model_id}\n"
        f" lighting_pool_size:{args.lighting_pool_size}\n lighting_refresh_interval:{args.lighting_refresh_interval}\n"
        f" lighting_num_inference_steps:{args.lighting_num_inference_steps}\n"
        f" lighting_guidance_scale:{args.lighting_guidance_scale}\n"
        f" lighting_blend_min:{args.lighting_blend_min}\n lighting_blend_max:{args.lighting_blend_max}\n"
        f" lighting_apply_prob:{args.lighting_apply_prob}\n lighting_seed:{args.lighting_seed}\n"
        f" ic_light_repo:{args.ic_light_repo}\n ic_light_model_path:{args.ic_light_model_path}\n"
        f" ic_light_scope:{args.ic_light_scope}\n ic_light_bg_control:{args.ic_light_bg_control}\n"
        f" viz_enabled:{args.viz_enabled}\n viz_policy:{args.viz_policy}\n"
        f" viz_samples:{args.viz_samples}\n viz_save_best:{args.viz_save_best}\n viz_save_last:{args.viz_save_last}"
    )
    main(args)
