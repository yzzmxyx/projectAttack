"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 60                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    risk_window_enable: bool = False                            # Enable independent risk-window detector
    risk_window_config: str = ""                                # Optional JSON/YAML config path
    risk_window_asset_root: str = ""                            # Existing vulnerability-window asset root
    risk_window_log_dir: str = ""                               # Optional detector log root
    risk_window_action: str = "log_only"                        # log_only | hold_last_action | dummy_wait | abort_episode
    risk_window_overlay: bool = False                           # Reserved for future visualization overlay support

    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "bridge_orig"

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize the WidowX environment
    env = get_widowx_env(cfg, model)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    risk_window_adapter = None
    if getattr(cfg, "risk_window_enable", False):
        if str(getattr(cfg, "risk_window_asset_root", "")).strip() == "":
            raise ValueError("risk_window_enable=True requires a non-empty `risk_window_asset_root`.")
        from risk_window.adapters.bridge import BridgeRiskWindowAdapter

        risk_window_adapter = BridgeRiskWindowAdapter.from_runtime_args(
            config_path=str(getattr(cfg, "risk_window_config", "")),
            asset_root=str(cfg.risk_window_asset_root),
            log_dir=str(getattr(cfg, "risk_window_log_dir", "")),
            action_policy=str(getattr(cfg, "risk_window_action", "log_only")),
            overlay=bool(getattr(cfg, "risk_window_overlay", False)),
        )

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()
        if risk_window_adapter is not None:
            risk_window_adapter.reset(task_id=task_label, episode_id=episode_idx)

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    if risk_window_adapter is not None:
                        risk_window_adapter.inspect(frame=obs["full_image"], timestamp=curr_tstamp)

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        task_label,
                        processor=processor,
                    )

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    if risk_window_adapter is not None:
                        policy_result = risk_window_adapter.apply_action_policy(action)
                        if policy_result["abort_episode"]:
                            print("[risk_window] abort_episode triggered; ending current episode early.")
                            break
                        action = policy_result["action"]

                    # Execute action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        if risk_window_adapter is not None:
            summary = risk_window_adapter.flush()
            print(f"[risk_window] episode summary: {summary}")

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
