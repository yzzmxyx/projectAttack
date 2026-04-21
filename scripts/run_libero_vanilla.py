import os
import argparse
import json
import time
import numpy as np
import imageio
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# 导入 LIBERO 官方库
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print("错误: 未找到 LIBERO 库")
    exit(1)

# 导入 OpenVLA 官方实验工具函数
try:
    from experiments.robot.libero.run_libero_eval import GenerateConfig, run_task
    from experiments.robot.openvla_utils_oft import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
except ImportError:
    print("错误: 无法导入 OpenVLA 实验工具")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="运行 LIBERO 环境 (基于官方 OFT 实验逻辑)")
    parser.add_argument("--task_suite", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=25, help="每个任务测试的 episode 数量")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, default="/home/yxx/projectAttack/openvla-7b-oft-finetuned-libero-10")
    args = parser.parse_args()

    # ==========================================
    # 0. 创建结果保存的独立文件夹
    # ==========================================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.task_suite}_task{args.task_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "eval_log.txt")

    # ==========================================
    # 1. 按照示例代码配置模型参数
    # ==========================================
    unnorm_key = f"{args.task_suite}_no_noops"
    
    cfg = GenerateConfig(
        pretrained_checkpoint = args.model_path,
        use_l1_regression = True,
        use_diffusion = False,
        use_film = False,
        num_images_in_input = 2,
        use_proprio = True,
        load_in_8bit = False,
        load_in_4bit = False,
        center_crop = True, 
        num_open_loop_steps = NUM_ACTIONS_CHUNK,
        unnorm_key = unnorm_key, 
        model_family = "openvla",
        env_img_res = 256,
        num_trials_per_task = args.num_trials,
        initial_states_path = "DEFAULT",
        use_wandb = False
    )

    print(f"[*] 加载 OpenVLA-OFT 策略...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    # ==========================================
    # 2. 初始化 LIBERO 环境
    # ==========================================
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise ValueError(f"Task suite '{args.task_suite}' 不存在. 可选项: {list(benchmark_dict.keys())}")
    
    task_suite = benchmark_dict[args.task_suite]()
    
    # ==========================================
    # 3. 直接调用官方 run_task 跑满该任务的所有评测
    # ==========================================
    print(f"\n[*] 准备开始官方评测流程 (共 {args.num_trials} 个 trials)...")
    with open(log_file_path, "w", encoding='utf-8') as log_file:
        total_episodes, total_successes = run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=args.task_id, # 比如测试第 0 个任务
            model=vla,
            resize_size=256, # 或者 224 等（取决于模型预期）
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=None,
            total_episodes=0,
            total_successes=0,
            log_file=log_file,
            task_save_dir = run_dir
        )
        
    print("\n" + "="*50)
    print(f"[*] 评测任务 {args.task_id} 完成！")
    print(f"[*] 总测试次数: {total_episodes}")
    print(f"[*] 总成功次数: {total_successes}")
    if total_episodes > 0:
        print(f"[*] 最终成功率: {(total_successes / total_episodes) * 100:.2f}%")
    print(f"[*] 详细日志与录像保存在: {run_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
