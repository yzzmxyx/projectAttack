import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VLA_ATTACKER_ROOT = REPO_ROOT / "VLAAttacker"
if str(VLA_ATTACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(VLA_ATTACKER_ROOT))

from white_patch.UADA_rollout_online_env import OpenVLAOnlineEnvAttacker
from white_patch.projector_photometric_params import resolve_projector_params_for_patch


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "t", "y", "1"):
        return True
    if text in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def list_of_ints(arg):
    return list(map(int, str(arg).split(",")))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def infer_task_suite_name(exp_dir: str, task_suite_name: str) -> str:
    if str(task_suite_name).lower() not in ("auto", "", "none", "null"):
        return str(task_suite_name)
    path_text = str(exp_dir).lower()
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if suite_name in path_text:
            return suite_name
    print("[Eval] Warning: cannot infer task suite from exp_dir, fallback to `libero_spatial`.")
    return "libero_spatial"


def resolve_vla_path(task_suite_name: str) -> str:
    suite = str(task_suite_name).lower()
    if "libero_spatial" in suite:
        return "openvla/openvla-7b-finetuned-libero-spatial"
    if "libero_object" in suite:
        return "openvla/openvla-7b-finetuned-libero-object"
    if "libero_goal" in suite:
        return "openvla/openvla-7b-finetuned-libero-goal"
    if "libero_10" in suite:
        return "openvla/openvla-7b-finetuned-libero-10"
    if "libero_90" in suite:
        return "openvla/openvla-7b"
    return "openvla/openvla-7b-finetuned-libero-spatial"


def resolve_eval_rollout_steps(eval_rollout_steps, max_env_steps: int) -> int:
    if isinstance(eval_rollout_steps, str) and str(eval_rollout_steps).lower() in ("auto", "auto_by_suite"):
        return int(max_env_steps)
    try:
        value = int(eval_rollout_steps)
    except Exception:
        return int(max_env_steps)
    return max(1, value)


def load_patch_tensor(path: str, device: torch.device) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu")
    if not torch.is_tensor(loaded):
        raise TypeError(f"Patch at `{path}` is not a tensor.")
    patch = loaded.detach().to(device=device, dtype=torch.float32).clamp(0, 1)
    patch.requires_grad_(False)
    return patch


def _mean(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / float(len(values)))


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mp4(video_path: str, frames: List, fps: int) -> bool:
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Compare two online-env patch checkpoints under matched evaluation settings.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Run directory containing patch checkpoints.")
    parser.add_argument("--patch_a_subdir", type=str, default="0")
    parser.add_argument("--patch_b_subdir", type=str, default="last")
    parser.add_argument("--task_suite_name", type=str, default="auto")
    parser.add_argument("--trials_per_task", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=0, help="0 means all tasks in the suite.")
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--maskidx", default="0,1,2", type=list_of_ints)
    parser.add_argument("--use_all_joints", type=str2bool, default=False)
    parser.add_argument("--gripper_weight", default=0.5, type=float)
    parser.add_argument("--num_steps_wait", default=10, type=int)
    parser.add_argument("--max_env_steps", default="auto_by_suite", type=str)
    parser.add_argument("--eval_rollout_steps", default="auto_by_suite", type=str)
    parser.add_argument("--env_resolution", default=256, type=int)

    parser.add_argument("--lambda_action_gap", default=1.0, type=float)
    parser.add_argument("--lambda_history", default=0.0, type=float)
    parser.add_argument("--lambda_ce", default=0.1, type=float)
    parser.add_argument("--online_ce_mode", default="off", type=str)

    parser.add_argument("--geometry", type=str2bool, default=True)
    parser.add_argument("--attack_mode", default="projection", type=str)
    parser.add_argument("--projection_alpha", default=0.55, type=float)
    parser.add_argument("--projection_alpha_jitter", default=0.00, type=float)
    parser.add_argument("--projection_soft_edge", default=1.2, type=float)
    parser.add_argument("--projection_angle", default=25.0, type=float)
    parser.add_argument("--projection_fixed_angle", type=str2bool, default=False)
    parser.add_argument("--projection_shear", default=0.15, type=float)
    parser.add_argument("--projection_scale_min", default=0.8, type=float)
    parser.add_argument("--projection_scale_max", default=1.2, type=float)
    parser.add_argument("--projection_region", default="lower_half_fixed", type=str)
    parser.add_argument("--projection_lower_start", default=0.55, type=float)
    parser.add_argument("--projection_width_ratio", default=0.90, type=float)
    parser.add_argument("--projection_height_ratio", default=0.95, type=float)
    parser.add_argument("--projection_margin_x", default=0.04, type=float)
    parser.add_argument("--projection_keystone", default=0.22, type=float)
    parser.add_argument("--projection_keystone_jitter", default=0.03, type=float)
    parser.add_argument("--projector_gamma", default=1.8, type=float)
    parser.add_argument("--projector_gain", default=1.35, type=float)
    parser.add_argument("--projector_channel_gain", default="1.08,1.04,1.00", type=str)
    parser.add_argument("--projector_ambient", default=0.08, type=float)
    parser.add_argument("--projector_vignetting", default=0.08, type=float)
    parser.add_argument("--projector_distance_falloff", default=0.10, type=float)
    parser.add_argument("--projector_psf", type=str2bool, default=False)
    parser.add_argument("--projection_randomization_enabled", type=str2bool, default=True)

    parser.add_argument("--lighting_aug_enabled", type=str2bool, default=True)
    parser.add_argument("--lighting_backend", default="ic_light", type=str)
    parser.add_argument("--lighting_model_id", default="stabilityai/sdxl-turbo", type=str)
    parser.add_argument("--lighting_pool_size", default=2, type=int)
    parser.add_argument("--lighting_refresh_interval", default=100, type=int)
    parser.add_argument("--lighting_num_inference_steps", default=8, type=int)
    parser.add_argument("--lighting_guidance_scale", default=7.0, type=float)
    parser.add_argument("--lighting_blend_min", default=0.15, type=float)
    parser.add_argument("--lighting_blend_max", default=0.45, type=float)
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
    parser.add_argument("--val_disable_lighting", type=str2bool, default=False)

    parser.add_argument("--record_video", type=str2bool, default=True)
    parser.add_argument("--video_frame_source", type=str, default="projected_input")
    parser.add_argument("--video_fps", type=int, default=10)
    return parser


def summarize_patch_rows(rows: List[Dict]) -> Dict[str, float]:
    valid_rows = [row for row in rows if int(row.get("skipped", 0)) == 0]
    if len(valid_rows) == 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_episode_len": 0.0,
            "avg_rollout_score": 0.0,
            "avg_rollout_score_legacy": 0.0,
            "avg_action_gap": 0.0,
            "avg_history_div": 0.0,
            "avg_history_div_legacy": 0.0,
        }
    return {
        "episodes": len(valid_rows),
        "success_rate": _mean([float(row["success"]) for row in valid_rows]),
        "avg_episode_len": _mean([float(row["episode_len"]) for row in valid_rows]),
        "avg_rollout_score": _mean([float(row["rollout_score"]) for row in valid_rows]),
        "avg_rollout_score_legacy": _mean([float(row["rollout_score_legacy"]) for row in valid_rows]),
        "avg_action_gap": _mean([float(row["action_gap"]) for row in valid_rows]),
        "avg_history_div": _mean([float(row["history_div"]) for row in valid_rows]),
        "avg_history_div_legacy": _mean([float(row["history_div_legacy"]) for row in valid_rows]),
    }


def main():
    args = build_arg_parser().parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    patch_a_path = os.path.join(exp_dir, str(args.patch_a_subdir), "patch.pt")
    patch_b_path = os.path.join(exp_dir, str(args.patch_b_subdir), "patch.pt")
    if not os.path.exists(patch_a_path):
        raise FileNotFoundError(f"patch A not found: {patch_a_path}")
    if not os.path.exists(patch_b_path):
        raise FileNotFoundError(f"patch B not found: {patch_b_path}")

    set_seed(int(args.seed))
    task_suite_name = infer_task_suite_name(exp_dir=exp_dir, task_suite_name=args.task_suite_name)
    pretrained_checkpoint = resolve_vla_path(task_suite_name=task_suite_name)
    print(f"[Eval] task_suite={task_suite_name}, checkpoint={pretrained_checkpoint}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)

    output_root = os.path.join(exp_dir, f"eval_patch_compare_{args.patch_a_subdir}_vs_{args.patch_b_subdir}")
    os.makedirs(output_root, exist_ok=True)
    attacker = OpenVLAOnlineEnvAttacker(vla=vla, processor=processor, save_dir=output_root, optimizer="adamW", resize_patch=False)
    attacker.lighting_aug_train_only = False
    attacker._setup_lighting_augmentor(
        enabled=bool(args.lighting_aug_enabled),
        backend=str(args.lighting_backend),
        model_id=args.lighting_model_id,
        pool_size=int(args.lighting_pool_size),
        refresh_interval=int(args.lighting_refresh_interval),
        num_inference_steps=int(args.lighting_num_inference_steps),
        guidance_scale=float(args.lighting_guidance_scale),
        blend_min=float(args.lighting_blend_min),
        blend_max=float(args.lighting_blend_max),
        apply_prob=float(args.lighting_apply_prob),
        seed=int(args.lighting_seed),
        ic_light_repo=str(args.ic_light_repo),
        ic_light_model_path=str(args.ic_light_model_path),
        ic_light_scope=str(args.ic_light_scope),
        ic_light_bg_control=str(args.ic_light_bg_control),
    )

    benchmark_mod, get_libero_env, get_libero_image, get_libero_dummy_action = attacker._require_libero_modules()
    task_suite = benchmark_mod.get_benchmark_dict()[task_suite_name]()
    max_env_steps = attacker._resolve_max_env_steps(args.max_env_steps, task_suite_name=task_suite_name)
    eval_rollout_steps = resolve_eval_rollout_steps(args.eval_rollout_steps, max_env_steps=max_env_steps)
    unnorm_key = attacker._resolve_unnorm_key(task_suite_name)
    action_stats = attacker.vla.get_action_stats(unnorm_key)
    effective_lighting_enabled = bool(args.lighting_aug_enabled) and (not bool(args.val_disable_lighting))
    print(
        "[EvalConfig] "
        f"projection_randomization_enabled={bool(args.projection_randomization_enabled)} "
        f"val_disable_lighting={bool(args.val_disable_lighting)} "
        f"projection_region={str(args.projection_region)} "
        f"projection_lower_start={float(args.projection_lower_start):.2f} "
        f"projection_width_ratio={float(args.projection_width_ratio):.2f} "
        f"projection_height_ratio={float(args.projection_height_ratio):.2f} "
        f"projection_margin_x={float(args.projection_margin_x):.2f} "
        f"projection_angle={float(args.projection_angle):.2f} "
        f"projection_fixed_angle={bool(args.projection_fixed_angle)} "
        f"projection_keystone={float(args.projection_keystone):.2f} "
        f"lighting_backend={str(args.lighting_backend)} "
        f"lighting_enabled={effective_lighting_enabled} "
        f"lighting_seed={int(args.lighting_seed)} "
        f"ic_light_scope={str(args.ic_light_scope)} "
        f"ic_light_bg_control={str(args.ic_light_bg_control)}"
    )

    n_tasks = int(task_suite.n_tasks)
    task_ids = list(range(n_tasks))
    if int(args.max_tasks) > 0:
        task_ids = task_ids[: int(args.max_tasks)]

    patch_specs = [
        (
            "patch_a",
            str(args.patch_a_subdir),
            patch_a_path,
            resolve_projector_params_for_patch(
                patch_path=patch_a_path,
                default_projector_gain=args.projector_gain,
                default_projector_channel_gain=args.projector_channel_gain,
            ),
        ),
        (
            "patch_b",
            str(args.patch_b_subdir),
            patch_b_path,
            resolve_projector_params_for_patch(
                patch_path=patch_b_path,
                default_projector_gain=args.projector_gain,
                default_projector_channel_gain=args.projector_channel_gain,
            ),
        ),
    ]
    episode_rows: List[Dict] = []

    for patch_tag, patch_subdir, patch_path, projector_params in patch_specs:
        print(
            f"[Eval] running {patch_tag} ({patch_path}) "
            f"projector_params_source={'sidecar' if projector_params['loaded_from_sidecar'] else 'cli'}"
        )
        projection_texture = load_patch_tensor(path=patch_path, device=device)
        patch_projector_gain = float(projector_params["projector_gain"])
        patch_projector_channel_gain = tuple(float(x) for x in projector_params["projector_channel_gain"])
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(args.env_resolution)))
            try:
                for trial_idx in range(int(args.trials_per_task)):
                    episode_seed = int(args.val_seed) + (int(task_id) * int(args.trials_per_task)) + int(trial_idx)
                    init_state = attacker._sample_init_state(
                        init_states=init_states,
                        iter_idx=0,
                        local_idx=trial_idx,
                        deterministic_seed=episode_seed,
                    )
                    episode = attacker._run_online_episode(
                        env=env,
                        get_libero_image=get_libero_image,
                        get_libero_dummy_action=get_libero_dummy_action,
                        init_state=init_state,
                        task_description=task_description,
                        projection_texture=projection_texture,
                        rollout_steps=int(eval_rollout_steps),
                        max_env_steps=int(max_env_steps),
                        num_steps_wait=int(args.num_steps_wait),
                        split="val",
                        global_iter=trial_idx,
                        geometry=bool(args.geometry),
                        maskidx=args.maskidx,
                        use_all_joints=bool(args.use_all_joints),
                        gripper_weight=float(args.gripper_weight),
                        action_stats=action_stats,
                        lambda_action_gap=float(args.lambda_action_gap),
                        lambda_history=float(args.lambda_history),
                        lambda_ce=float(args.lambda_ce),
                        online_ce_mode=str(args.online_ce_mode),
                        attack_mode=str(args.attack_mode),
                        projection_alpha=float(args.projection_alpha),
                        projection_alpha_jitter=float(args.projection_alpha_jitter),
                        projection_soft_edge=float(args.projection_soft_edge),
                        projection_angle=float(args.projection_angle),
                        projection_fixed_angle=bool(args.projection_fixed_angle),
                        projection_shear=float(args.projection_shear),
                        projection_scale_min=float(args.projection_scale_min),
                        projection_scale_max=float(args.projection_scale_max),
                        projection_region=str(args.projection_region),
                        projection_lower_start=float(args.projection_lower_start),
                        projection_width_ratio=float(args.projection_width_ratio),
                        projection_height_ratio=float(args.projection_height_ratio),
                        projection_margin_x=float(args.projection_margin_x),
                        projection_keystone=float(args.projection_keystone),
                        projection_keystone_jitter=float(args.projection_keystone_jitter),
                        projector_gamma=float(args.projector_gamma),
                        projector_gain=patch_projector_gain,
                        projector_channel_gain=patch_projector_channel_gain,
                        projector_ambient=float(args.projector_ambient),
                        projector_vignetting=float(args.projector_vignetting),
                        projector_distance_falloff=float(args.projector_distance_falloff),
                        projector_psf=bool(args.projector_psf),
                        projection_randomization_enabled=bool(args.projection_randomization_enabled),
                        need_backward=False,
                        capture_visual=False,
                        deterministic_episode_seed=episode_seed,
                        disable_lighting=bool(args.val_disable_lighting),
                        record_video_frames=bool(args.record_video),
                        video_frame_source=str(args.video_frame_source),
                    )

                    row = {
                        "patch_tag": patch_tag,
                        "patch_subdir": patch_subdir,
                        "patch_path": patch_path,
                        "task_id": int(task_id),
                        "task_name": str(task_description),
                        "trial_idx": int(trial_idx),
                        "seed": int(episode_seed),
                        "skipped": 0,
                        "success": 0,
                        "episode_len": 0,
                        "rollout_score": 0.0,
                        "rollout_score_legacy": 0.0,
                        "action_gap": 0.0,
                        "history_div": 0.0,
                        "history_div_legacy": 0.0,
                        "ce_value": 0.0,
                        "ce_objective_value": 0.0,
                        "projection_alpha": 0.0,
                        "projection_coverage": 0.0,
                        "projection_bottom": 0.0,
                        "projection_keystone": 0.0,
                        "video_path": "",
                    }
                    if episode is None:
                        row["skipped"] = 1
                        episode_rows.append(row)
                        continue

                    rollout_score = float(args.lambda_action_gap) * float(episode["action_gap"]) + float(args.lambda_history) * float(
                        episode["history_div"]
                    )
                    rollout_score_legacy = float(args.lambda_action_gap) * float(episode["action_gap"]) + float(args.lambda_history) * float(
                        episode["history_div_legacy"]
                    )
                    row["success"] = int(bool(episode["done"]))
                    row["episode_len"] = int(episode["episode_len"])
                    row["rollout_score"] = float(rollout_score)
                    row["rollout_score_legacy"] = float(rollout_score_legacy)
                    row["action_gap"] = float(episode["action_gap"])
                    row["history_div"] = float(episode["history_div"])
                    row["history_div_legacy"] = float(episode["history_div_legacy"])
                    row["ce_value"] = float(episode["ce_value"])
                    row["ce_objective_value"] = float(episode["ce_objective_value"])
                    row["projection_alpha"] = float(episode["projection_alpha"])
                    row["projection_coverage"] = float(episode["projection_coverage"])
                    row["projection_bottom"] = float(episode["projection_bottom"])
                    row["projection_keystone"] = float(episode["projection_keystone"])

                    if bool(args.record_video):
                        video_path = os.path.join(
                            output_root,
                            "videos",
                            patch_tag,
                            f"task_{int(task_id):03d}",
                            f"trial_{int(trial_idx):03d}.mp4",
                        )
                        write_mp4(video_path=video_path, frames=episode.get("video_frames", []), fps=int(args.video_fps))
                        row["video_path"] = video_path

                    episode_rows.append(row)
            finally:
                if hasattr(env, "close"):
                    env.close()

    episode_results_path = os.path.join(output_root, "episode_results.csv")
    episode_fieldnames = [
        "patch_tag",
        "patch_subdir",
        "patch_path",
        "task_id",
        "task_name",
        "trial_idx",
        "seed",
        "skipped",
        "success",
        "episode_len",
        "rollout_score",
        "rollout_score_legacy",
        "action_gap",
        "history_div",
        "history_div_legacy",
        "ce_value",
        "ce_objective_value",
        "projection_alpha",
        "projection_coverage",
        "projection_bottom",
        "projection_keystone",
        "video_path",
    ]
    write_csv(episode_results_path, episode_rows, episode_fieldnames)

    grouped_rows = defaultdict(list)
    for row in episode_rows:
        grouped_rows[(int(row["task_id"]), str(row["patch_tag"]))].append(row)

    task_summary_rows = []
    for task_id in task_ids:
        rows_a = grouped_rows.get((int(task_id), "patch_a"), [])
        rows_b = grouped_rows.get((int(task_id), "patch_b"), [])
        task_name = ""
        if len(rows_a) > 0:
            task_name = str(rows_a[0]["task_name"])
        elif len(rows_b) > 0:
            task_name = str(rows_b[0]["task_name"])

        stat_a = summarize_patch_rows(rows_a)
        stat_b = summarize_patch_rows(rows_b)
        task_summary_rows.append(
            {
                "task_id": int(task_id),
                "task_name": task_name,
                "episodes_patch_a": int(stat_a["episodes"]),
                "success_rate_patch_a": float(stat_a["success_rate"]),
                "avg_episode_len_patch_a": float(stat_a["avg_episode_len"]),
                "avg_rollout_score_patch_a": float(stat_a["avg_rollout_score"]),
                "avg_rollout_score_legacy_patch_a": float(stat_a["avg_rollout_score_legacy"]),
                "avg_action_gap_patch_a": float(stat_a["avg_action_gap"]),
                "avg_history_div_patch_a": float(stat_a["avg_history_div"]),
                "avg_history_div_legacy_patch_a": float(stat_a["avg_history_div_legacy"]),
                "episodes_patch_b": int(stat_b["episodes"]),
                "success_rate_patch_b": float(stat_b["success_rate"]),
                "avg_episode_len_patch_b": float(stat_b["avg_episode_len"]),
                "avg_rollout_score_patch_b": float(stat_b["avg_rollout_score"]),
                "avg_rollout_score_legacy_patch_b": float(stat_b["avg_rollout_score_legacy"]),
                "avg_action_gap_patch_b": float(stat_b["avg_action_gap"]),
                "avg_history_div_patch_b": float(stat_b["avg_history_div"]),
                "avg_history_div_legacy_patch_b": float(stat_b["avg_history_div_legacy"]),
                "delta_success_rate_b_minus_a": float(stat_b["success_rate"] - stat_a["success_rate"]),
                "delta_avg_episode_len_b_minus_a": float(stat_b["avg_episode_len"] - stat_a["avg_episode_len"]),
                "delta_avg_rollout_score_b_minus_a": float(stat_b["avg_rollout_score"] - stat_a["avg_rollout_score"]),
                "delta_avg_rollout_score_legacy_b_minus_a": float(
                    stat_b["avg_rollout_score_legacy"] - stat_a["avg_rollout_score_legacy"]
                ),
                "delta_avg_history_div_b_minus_a": float(stat_b["avg_history_div"] - stat_a["avg_history_div"]),
                "delta_avg_history_div_legacy_b_minus_a": float(
                    stat_b["avg_history_div_legacy"] - stat_a["avg_history_div_legacy"]
                ),
            }
        )

    task_summary_path = os.path.join(output_root, "task_summary.csv")
    write_csv(task_summary_path, task_summary_rows, list(task_summary_rows[0].keys()) if task_summary_rows else ["task_id", "task_name"])

    rows_patch_a = [row for row in episode_rows if row["patch_tag"] == "patch_a"]
    rows_patch_b = [row for row in episode_rows if row["patch_tag"] == "patch_b"]
    overall_a = summarize_patch_rows(rows_patch_a)
    overall_b = summarize_patch_rows(rows_patch_b)
    overall_rows = [
        {
            "patch_tag": "patch_a",
            "episodes": int(overall_a["episodes"]),
            "success_rate": float(overall_a["success_rate"]),
            "avg_episode_len": float(overall_a["avg_episode_len"]),
            "avg_rollout_score": float(overall_a["avg_rollout_score"]),
            "avg_rollout_score_legacy": float(overall_a["avg_rollout_score_legacy"]),
            "avg_action_gap": float(overall_a["avg_action_gap"]),
            "avg_history_div": float(overall_a["avg_history_div"]),
            "avg_history_div_legacy": float(overall_a["avg_history_div_legacy"]),
        },
        {
            "patch_tag": "patch_b",
            "episodes": int(overall_b["episodes"]),
            "success_rate": float(overall_b["success_rate"]),
            "avg_episode_len": float(overall_b["avg_episode_len"]),
            "avg_rollout_score": float(overall_b["avg_rollout_score"]),
            "avg_rollout_score_legacy": float(overall_b["avg_rollout_score_legacy"]),
            "avg_action_gap": float(overall_b["avg_action_gap"]),
            "avg_history_div": float(overall_b["avg_history_div"]),
            "avg_history_div_legacy": float(overall_b["avg_history_div_legacy"]),
        },
        {
            "patch_tag": "delta_b_minus_a",
            "episodes": int(overall_b["episodes"] - overall_a["episodes"]),
            "success_rate": float(overall_b["success_rate"] - overall_a["success_rate"]),
            "avg_episode_len": float(overall_b["avg_episode_len"] - overall_a["avg_episode_len"]),
            "avg_rollout_score": float(overall_b["avg_rollout_score"] - overall_a["avg_rollout_score"]),
            "avg_rollout_score_legacy": float(overall_b["avg_rollout_score_legacy"] - overall_a["avg_rollout_score_legacy"]),
            "avg_action_gap": float(overall_b["avg_action_gap"] - overall_a["avg_action_gap"]),
            "avg_history_div": float(overall_b["avg_history_div"] - overall_a["avg_history_div"]),
            "avg_history_div_legacy": float(overall_b["avg_history_div_legacy"] - overall_a["avg_history_div_legacy"]),
        },
    ]
    overall_summary_path = os.path.join(output_root, "overall_summary.csv")
    write_csv(overall_summary_path, overall_rows, list(overall_rows[0].keys()))

    print("\n=== Overall Comparison ===")
    for row in overall_rows:
        print(
            f"{row['patch_tag']}: episodes={row['episodes']}, success_rate={row['success_rate']:.4f}, "
            f"avg_len={row['avg_episode_len']:.2f}, avg_score={row['avg_rollout_score']:.6f}, "
            f"avg_score_legacy={row['avg_rollout_score_legacy']:.6f}, "
            f"avg_action_gap={row['avg_action_gap']:.6f}, avg_history_div={row['avg_history_div']:.6f}, "
            f"avg_history_div_legacy={row['avg_history_div_legacy']:.6f}"
        )

    print("\n=== Per-Task Comparison (patch_b - patch_a) ===")
    for row in task_summary_rows:
        print(
            f"task_id={row['task_id']:03d}, task_name={row['task_name']}, "
            f"success_a={row['success_rate_patch_a']:.4f}, success_b={row['success_rate_patch_b']:.4f}, "
            f"delta_success={row['delta_success_rate_b_minus_a']:+.4f}, "
            f"delta_score={row['delta_avg_rollout_score_b_minus_a']:+.6f}, "
            f"delta_score_legacy={row['delta_avg_rollout_score_legacy_b_minus_a']:+.6f}"
        )

    print("\n=== Output Files ===")
    print(f"episode_results: {episode_results_path}")
    print(f"task_summary:    {task_summary_path}")
    print(f"overall_summary: {overall_summary_path}")
    if bool(args.record_video):
        print(f"videos_root:     {os.path.join(output_root, 'videos')}")


if __name__ == "__main__":
    main()
