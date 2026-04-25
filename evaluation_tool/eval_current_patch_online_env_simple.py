import argparse
import csv
from datetime import datetime
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLATTACKER_ROOT = PROJECT_ROOT / "VLAAttacker"
if str(VLATTACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(VLATTACKER_ROOT))

from white_patch.UADA_rollout_online_env import OpenVLAOnlineEnvAttacker  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).lower().strip()
    if text in ("yes", "true", "t", "y", "1"):
        return True
    if text in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def list_of_ints_or_auto(value):
    text = str(value).strip().lower()
    if text in ("", "auto", "none", "null"):
        return None
    return [int(item) for item in str(value).split(",")]


def list_of_floats(value):
    return [float(item) for item in str(value).split(",")]


def resolve_vla_path(dataset):
    dataset = str(dataset)
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
    raise ValueError(f"Invalid dataset: {dataset}")


def sanitize_filename(text, max_len=90):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "task"
    return text[:max_len]


def write_mp4(video_path, frames, fps):
    if not frames:
        return False
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    first = np.array(frames[0].convert("RGB"), dtype=np.uint8)
    height, width = first.shape[:2]
    writer = imageio.get_writer(str(video_path), fps=int(max(1, fps)))
    try:
        for frame in frames:
            rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
            if rgb.shape[0] != height or rgb.shape[1] != width:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.append_data(rgb)
    finally:
        writer.close()
    return True


def append_manifest(manifest_path, row):
    fieldnames = [
        "run_idx",
        "task_id",
        "task_description",
        "init_state_idx",
        "success",
        "episode_len",
        "runtime_sec",
        "video_path",
    ]
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_one_episode(
    *,
    attacker,
    env,
    task,
    get_libero_image,
    get_libero_dummy_action,
    init_state,
    init_state_idx,
    task_description,
    action_stats,
    projection_texture,
    args,
    run_idx,
):
    action_dim = len(action_stats["q01"])
    obs = attacker._initialize_online_rollout_env(
        env=env,
        init_state=init_state,
        get_libero_dummy_action=get_libero_dummy_action,
        effective_num_steps_wait=args.num_steps_wait,
    )
    labels_full, attention_mask, rollout_input_ids = attacker._build_prompt_tensors(
        task_description=task_description,
        action_dim=action_dim,
    )
    action_mask_full = attacker._build_action_mask(labels_full)
    if action_mask_full.sum().item() == 0:
        raise RuntimeError(f"No action tokens found for task: {task_description}")

    frames = []
    done = False
    episode_len = 0
    start_time = time.time()
    for step_idx in range(int(args.max_env_steps)):
        current_image = get_libero_image(obs, (224, 224))
        step_seed = int(args.seed) + (int(run_idx) * 1000003) + (int(step_idx) * 10007)

        with attacker._temporary_rng_seed(step_seed):
            adv_images, attack_aux = attacker.randomPatchTransform.apply_attack_batch(
                images=[current_image],
                attack_texture=projection_texture,
                mean=attacker.mean,
                std=attacker.std,
                attack_mode="projection",
                geometry=True,
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
                return_aux=True,
            )

        with torch.no_grad():
            output = attacker.vla(
                input_ids=rollout_input_ids,
                attention_mask=attention_mask,
                pixel_values=adv_images.to(torch.bfloat16),
                labels=None,
                output_hidden_states=False,
                use_cache=False,
            )
        pred_action_tokens = attacker._extract_pred_action_tokens_from_logits(output.logits, labels_full)
        obs, done, rollout_input_ids = attacker._step_rollout_branch_env(
            env=env,
            obs=obs,
            rollout_input_ids=rollout_input_ids,
            pred_action_tokens=pred_action_tokens.detach(),
            action_mask_full=action_mask_full,
            action_stats=action_stats,
        )
        episode_len += 1

        frame_source = str(args.video_frame_source).lower().strip()
        if frame_source == "orig":
            frame = attacker._to_pil(current_image)
        elif frame_source == "next_obs":
            frame = attacker._to_pil(get_libero_image(obs, (224, 224)))
        elif frame_source == "adv":
            frame = attacker._to_pil(
                attacker.randomPatchTransform.denormalize(
                    adv_images[0, 0:3, :, :].detach().cpu().unsqueeze(0),
                    mean=attacker.mean[0],
                    std=attacker.std[0],
                )
                .squeeze(0)
                .clamp(0, 1)
            )
        else:
            frame = attacker._to_pil(attack_aux["projected_inputs"][0])
        frames.append(frame)

        if done:
            break

    runtime_sec = time.time() - start_time
    return {
        "done": bool(done),
        "episode_len": int(episode_len),
        "runtime_sec": float(runtime_sec),
        "frames": frames,
        "init_state_idx": int(init_state_idx),
    }


def main(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    date_text = datetime.now().strftime("%Y_%m_%d")
    date_time_text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    video_dir = output_dir / "rollouts" / date_text
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "video_manifest.csv"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla_path = resolve_vla_path(args.dataset)
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

    attacker = OpenVLAOnlineEnvAttacker(vla, processor, str(output_dir), optimizer="adamW", resize_patch=False)
    benchmark_mod, get_libero_env, get_libero_image, get_libero_dummy_action = attacker._require_libero_modules()
    task_suite = benchmark_mod.get_benchmark_dict()[args.task_suite_name]()
    unnorm_key = attacker._resolve_unnorm_key(args.task_suite_name)
    action_stats = vla.get_action_stats(unnorm_key)

    patch = torch.load(args.patch_path, map_location=device)
    projection_texture = torch.as_tensor(patch, device=device, dtype=torch.float32)
    expected_shape = args.projection_size if args.projection_size is not None else list(projection_texture.shape)
    if tuple(projection_texture.shape) != tuple(expected_shape):
        raise ValueError(
            f"Loaded patch shape mismatch: expected {tuple(expected_shape)}, "
            f"got {tuple(projection_texture.shape)} from {args.patch_path}"
        )

    print(f"output_dir:{output_dir}")
    print(f"patch_path:{args.patch_path}")
    print(f"task_suite_name:{args.task_suite_name}")
    print(f"eval_runs:{args.eval_runs}")
    print(f"max_env_steps:{args.max_env_steps}")
    print(f"video_manifest:{manifest_path}")

    success_count = 0
    for run_idx in range(int(args.eval_runs)):
        task_id = int(run_idx % max(1, int(task_suite.n_tasks)))
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        init_state_idx = attacker._sample_init_state_index(
            init_states=init_states,
            iter_idx=0,
            local_idx=run_idx,
            deterministic_seed=(int(args.val_seed) + int(run_idx)) if args.val_deterministic else None,
        )
        if init_state_idx is None:
            print(f"[EvalSimple] run={run_idx:03d} task_id={task_id} skipped: no init states")
            continue
        init_state = init_states[init_state_idx]
        env, task_description = get_libero_env(task, "openvla", resolution=max(64, int(args.env_resolution)))
        try:
            result = run_one_episode(
                attacker=attacker,
                env=env,
                task=task,
                get_libero_image=get_libero_image,
                get_libero_dummy_action=get_libero_dummy_action,
                init_state=init_state,
                init_state_idx=init_state_idx,
                task_description=task_description,
                action_stats=action_stats,
                projection_texture=projection_texture,
                args=args,
                run_idx=run_idx,
            )
        finally:
            if hasattr(env, "close"):
                env.close()

        status = "success" if result["done"] else "failure"
        success_count += int(result["done"])
        task_slug = sanitize_filename(task_description)
        video_name = (
            f"{date_time_text}--episode={run_idx}"
            f"--success={status}"
            f"--episode_len={int(result['episode_len'])}"
            f"--task={task_slug}.mp4"
        )
        video_path = video_dir / video_name
        write_mp4(video_path, result["frames"], fps=args.video_fps)

        row = {
            "run_idx": int(run_idx),
            "task_id": int(task_id),
            "task_description": str(task_description),
            "init_state_idx": int(result["init_state_idx"]),
            "success": int(result["done"]),
            "episode_len": int(result["episode_len"]),
            "runtime_sec": f"{result['runtime_sec']:.3f}",
            "video_path": str(video_path),
        }
        append_manifest(manifest_path, row)
        print(
            f"[EvalSimple] run={run_idx:03d}/{int(args.eval_runs)} task_id={task_id} "
            f"status={status} episode_len={result['episode_len']} video={video_path}"
        )

    done_rate = success_count / float(max(1, int(args.eval_runs)))
    print(f"[EvalSimple] completed runs={int(args.eval_runs)} success={success_count} done_rate={done_rate:.4f}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--dataset", default="libero_spatial", type=str)
    parser.add_argument("--task_suite_name", default="libero_spatial", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--eval_runs", default=25, type=int)
    parser.add_argument("--max_env_steps", default=180, type=int)
    parser.add_argument("--num_steps_wait", default=10, type=int)
    parser.add_argument("--env_resolution", default=256, type=int)
    parser.add_argument("--val_deterministic", type=str2bool, default=True)
    parser.add_argument("--val_seed", default=42, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--projection_size", default=None, type=list_of_ints_or_auto)
    parser.add_argument("--projection_alpha", default=0.55, type=float)
    parser.add_argument("--projection_alpha_jitter", default=0.0, type=float)
    parser.add_argument("--projection_soft_edge", default=1.2, type=float)
    parser.add_argument("--projection_angle", default=25.0, type=float)
    parser.add_argument("--projection_fixed_angle", type=str2bool, default=False)
    parser.add_argument("--projection_shear", default=0.15, type=float)
    parser.add_argument("--projection_scale_min", default=0.8, type=float)
    parser.add_argument("--projection_scale_max", default=1.2, type=float)
    parser.add_argument("--projection_region", default="lower_half_fixed", type=str)
    parser.add_argument("--projection_lower_start", default=0.55, type=float)
    parser.add_argument("--projection_width_ratio", default=0.35, type=float)
    parser.add_argument("--projection_height_ratio", default=0.35, type=float)
    parser.add_argument("--projection_margin_x", default=0.04, type=float)
    parser.add_argument("--projection_keystone", default=0.22, type=float)
    parser.add_argument("--projection_keystone_jitter", default=0.03, type=float)
    parser.add_argument("--projection_randomization_enabled", type=str2bool, default=False)
    parser.add_argument("--projector_gamma", default=1.8, type=float)
    parser.add_argument("--projector_gain", default=1.35, type=float)
    parser.add_argument("--projector_channel_gain", default="1.08,1.04,1.00", type=list_of_floats)
    parser.add_argument("--projector_ambient", default=0.08, type=float)
    parser.add_argument("--projector_vignetting", default=0.08, type=float)
    parser.add_argument("--projector_distance_falloff", default=0.10, type=float)
    parser.add_argument("--projector_psf", type=str2bool, default=False)
    parser.add_argument("--video_frame_source", default="projected_input", type=str)
    parser.add_argument("--video_fps", default=10, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
