"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import cv2        # 新增：用于 CSRT 追踪和可视化
import imageio    # 新增：用于保存追踪的可视化视频
import torch
import torch.nn.functional as F
from PIL import Image

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils_oft import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils_oft import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils_oft import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
# =======================================================================
# 自包含的动态水波纹光效投影逻辑 (无 SAPIEN 依赖，纯 PyTorch + OpenCV)
# =======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_enhanced_water_wave_offset(h, w, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, t=0.0, device=DEVICE):
    x = torch.linspace(-1, 1, w, device=device).repeat(h, 1)
    y = torch.linspace(-1, 1, h, device=device).repeat(w, 1).transpose(0, 1)
    x = x - (wave_center[0] - 0.5) * 2
    y = y - (wave_center[1] - 0.5) * 2

    r1 = torch.sqrt(x**2 + y**2)
    wave1 = torch.sin(r1 * wave_freq - t * flow_speed) * torch.exp(-r1**2 / (2 * 0.6**2))
    r2 = torch.sqrt(x**2 + y**2)
    wave2 = torch.sin(r2 * second_wave_freq - t * flow_speed * 0.7) * torch.exp(-r2**2 / (2 * 1.0**2))
    
    wave = (wave1 * 0.7 + wave2 * 0.3) * 1.5
    theta = torch.atan2(y, x)

    dx = wave * wave_amp * torch.cos(theta) / w * 2
    dy = wave * wave_amp * torch.sin(theta) / h * 2
    return dx, dy, wave

def generate_yellow_water_wave_light(h, w, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, color_variance, light_radius, light_brightness, color, t=0.0, device=DEVICE):
    x = torch.linspace(0, 1, w, device=device).repeat(h, 1)
    y = torch.linspace(0, 1, h, device=device).repeat(w, 1).transpose(0, 1)
    r = torch.sqrt((x - wave_center[0])**2 + (y - wave_center[1])**2)
    base_light = torch.exp(-r**2 / (2 * light_radius**2)) * light_brightness

    dx, dy, wave = generate_enhanced_water_wave_offset(h, w, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, t, device)
    color_factor = wave * color_variance

    r_channel = (color[0] + color_factor) * base_light
    g_channel = (color[1] + color_factor * 0.8) * base_light
    b_channel = (color[2] - color_factor * 0.5) * base_light
    r_channel = torch.clamp(r_channel, 0, light_brightness)
    g_channel = torch.clamp(g_channel, 0, light_brightness)
    b_channel = torch.clamp(b_channel, 0, light_brightness * 0.3)

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = grid_x.float() + dx * w
    grid_y = grid_y.float() + dy * h
    grid_x = (grid_x / (w - 1) * 2) - 1
    grid_y = (grid_y / (h - 1) * 2) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def distort_channel(channel):
        channel = channel.unsqueeze(0).unsqueeze(0)
        return F.grid_sample(channel, grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)

    r_distorted = distort_channel(r_channel)
    g_distorted = distort_channel(g_channel)
    b_distorted = distort_channel(b_channel)

    light_layer = torch.cat([r_distorted, g_distorted, b_distorted], dim=0).unsqueeze(0)
    blur_kernel = max(3, int(0.08 * 30) // 2 * 2 + 1)
    blur_padding = (blur_kernel - 1) // 2
    light_layer = F.avg_pool2d(light_layer, kernel_size=blur_kernel, padding=blur_padding, stride=1)
    return light_layer

def apply_yellow_water_wave_effect(img_tensor, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, color_variance, light_radius, light_brightness, color, t=0.0):
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    B, C, H, W = img_tensor.shape
    light_layer = generate_yellow_water_wave_light(H, W, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, color_variance, light_radius, light_brightness, color, t, DEVICE)
    return torch.clamp(light_layer, 0, 1)

def project_3d_to_2d_custom(corners_3d, camera_wrapper):
    """通用的3D到2D投影逻辑，只依赖内外参矩阵"""
    extrinsic = camera_wrapper.get_extrinsic_matrix()
    intrinsic = camera_wrapper.get_intrinsic_matrix()
    corners_homo = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis=1)
    corners_cam_homo = (extrinsic @ corners_homo.T).T
    corners_cam = corners_cam_homo[:, :3] / corners_cam_homo[:, 3:4]
    img_homo = (intrinsic @ corners_cam.T).T
    img_coords = img_homo[:, :2] / img_homo[:, 2:3]
    return np.array(img_coords, dtype=np.float32)

def project_wave_to_table(
    img, cam_wrapper, table_corners_3d, wave_center=(0.5, 0.5),
    wave_freq=22, second_wave_freq=15, flow_speed=4, wave_amp=45,
    color_variance=0.45, light_radius=0.6, light_brightness=1.5,
    color=(0.95, 0.92, 0.25), opacity=0.7, t=0.0
):
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(DEVICE) / 255.0
    wave_tensor = apply_yellow_water_wave_effect(img_tensor, wave_center, wave_freq, second_wave_freq, flow_speed, wave_amp, color_variance, light_radius, light_brightness, color, t)
    wave_img = wave_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
    wave_np = wave_img.astype(np.uint8)

    table_corners_2d = project_3d_to_2d_custom(table_corners_3d, cam_wrapper)

    h, w = wave_np.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = table_corners_2d
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    wave_projected = cv2.warpPerspective(
        wave_np, M, (img.shape[1], img.shape[0]),
        borderMode=cv2.BORDER_TRANSPARENT
    )

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_pts)], 255)
    mask = mask / 255.0

    img_np = np.array(img).astype(np.float32)
    wave_projected = wave_projected.astype(np.float32)
    img_np[mask > 0] += wave_projected[mask > 0] * opacity

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


# =======================================================================
# ENV / EVAL 配置与主逻辑
# =======================================================================


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # 新增: Tracking (CSRT) parameters
    #################################################################################################################
    track_csrt: bool = True                          # 是否开启追踪
    track_point1_x: float = 143.0                    # 追踪点1初始X坐标
    track_point1_y: float = 34.0                     # 追踪点1初始Y坐标
    track_render_size: int = 256                     # 用于追踪的画面分辨率大小
    
    # 交点计算及平滑参数
    table_z: float = 0.41                            # Robosuite环境默认桌面基准高度大致为0.8米，视环境而定
    forward_offset: float = 0.03                     # 垂足光影向前移动的偏移量 (米) 0.07
    right_offset: float = 0.03                       # 垂足光影向机器臂右侧(主视角左侧)平移的偏移量 (米)
    max_move_per_frame: float = 0.04                 # 限制单帧光斑最大移动距离 (米) 0.08
    ema_alpha: float = 0.55                          # 指数移动平均 (EMA) 学习率
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    combined_img = np.concatenate([img, wrist_img], axis=1) #新增：拼接视频图像

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, combined_img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


# =======================================================================
# 新增：MuJoCo 物理引擎专用的相机坐标计算辅助函数
# =======================================================================
class MujocoCameraWrapper:
    """封装 MuJoCo 相机参数，以便原生匹配透视投影 API"""
    def __init__(self, sim, cam_name, width, height):
        self.sim = sim
        self.cam_name = cam_name
        self.width = width
        self.height = height
        
    def get_intrinsic_matrix(self):
        cam_id = self.sim.model.camera_name2id(self.cam_name)
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1]])
        
    def get_extrinsic_matrix(self):
        cam_pos = self.sim.data.get_camera_xpos(self.cam_name)
        cam_mat = self.sim.data.get_camera_xmat(self.cam_name)
        convert_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_cv = convert_mat @ cam_mat.T
        t_cv = -R_cv @ cam_pos
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_cv
        extrinsic[:3, 3] = t_cv
        return extrinsic

def check_corners_in_front(cam_wrapper, corners_3d):
    """安全检查：确保所有桌角都在相机前方，防止腕部相机透视变换崩溃"""
    ext = cam_wrapper.get_extrinsic_matrix()
    corners_homo = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis=1)
    corners_cam = (ext @ corners_homo.T).T
    return np.all(corners_cam[:, 2] > 0.01)


def get_mujoco_cam_matrices(sim, cam_name, width, height):
    cam_id = sim.model.camera_name2id(cam_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    
    cam_pos = sim.data.get_camera_xpos(cam_name)
    cam_mat = sim.data.get_camera_xmat(cam_name) # 3x3 rotation matrix
    return K, cam_pos, cam_mat

def pixel_to_world_mujoco(u, v, d_map, sim, K, cam_pos, cam_mat):
    # Convert MuJoCo non-linear depth buffer [0, 1] to real depth in meters
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    d = near * far / (far - d_map * (far - near))
    
    # MuJoCo OpenGL camera frame: Image Y goes down, Camera Y goes up
    x_cam = -(u - K[0, 2]) * d / K[0, 0]
    y_cam = -(v - K[1, 2]) * d / K[1, 1] 
    z_cam = -d
    
    pt_cam = np.array([x_cam, y_cam, z_cam])
    pt_world = cam_pos + cam_mat @ pt_cam
    return pt_world

def world_to_pixel_mujoco(pt_world, sim, K, cam_pos, cam_mat):
    pt_cam = np.linalg.inv(cam_mat) @ (pt_world - cam_pos)
    x_cam, y_cam, z_cam = pt_cam
    if z_cam >= 0:
        return None # Behind camera

    d = -z_cam
    u = -(x_cam * K[0, 0] / d) + K[0, 2]
    v = -(y_cam * K[1, 1] / d) + K[1, 2]
    return int(u), int(v)


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    episode_idx=0, # 新增: 用于视频和图片命名
    ep_save_dir=None, # 新增: 当前 episode 保存文件的独立文件夹路径
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = 1000 # 修改：最大步数上调

    # ==================== 新增：追踪与可视化配置初始化 ====================
    track_w, track_h = cfg.track_render_size, cfg.track_render_size
    tracking_frames = []
    
    # 提取底层 MuJoCo sim 对象
    _env = env
    while not hasattr(_env, "sim") and hasattr(_env, "env"):
        _env = _env.env
    sim = _env.sim if hasattr(_env, "sim") else None
    
    # 追踪器及平滑状态初始化
    if cfg.track_csrt and sim is not None:
        K_fixed, cam_pos_fixed, cam_mat_fixed = get_mujoco_cam_matrices(sim, "agentview", track_w, track_h)
        
        smoothed_intersection = None
        initial_pts = [[cfg.track_point1_x, cfg.track_point1_y]]
        trackers = []
        box_size = 15
        tracking_initialized = False
        last_valid_pixels = []
    # ======================================================================

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            # ==================== 1. CSRT 追踪物理点(完全在原图上进行) ====================
            if cfg.track_csrt and sim is not None:
                track_rgb_raw, track_depth_map = sim.render(camera_name="agentview", width=track_w, height=track_h, depth=True)
                track_rgb_raw = track_rgb_raw[::-1, ::-1, :]
                track_depth_map = track_depth_map[::-1, ::-1]

                track_bgr_raw = track_rgb_raw[:, :, ::-1].copy()
                
                current_pixels = []
                tracking_flags = []
                
                if not tracking_initialized:
                    for pt in initial_pts:
                        tracker = cv2.TrackerCSRT_create()
                        bbox = (int(pt[0]) - box_size // 2, int(pt[1]) - box_size // 2, box_size, box_size)
                        tracker.init(track_bgr_raw, bbox)
                        trackers.append(tracker)
                    last_valid_pixels = initial_pts.copy()
                    current_pixels = initial_pts.copy()
                    tracking_flags = [True] * len(initial_pts)
                    tracking_initialized = True
                else:
                    for i, tracker in enumerate(trackers):
                        trk_success, bbox = tracker.update(track_bgr_raw)
                        if trk_success:
                            cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
                            current_pixels.append([cx, cy])
                            last_valid_pixels[i] = [cx, cy]
                            tracking_flags.append(True)
                        else:
                            current_pixels.append(last_valid_pixels[i])
                            tracking_flags.append(False)

                current_frame_3d_points = []
                for idx, (u, v) in enumerate(current_pixels):
                    if tracking_flags[idx]:
                        d_val = track_depth_map[int(np.clip(v, 0, track_h-1)), int(np.clip(u, 0, track_w-1))]
                        pt_3d = pixel_to_world_mujoco(u, v, d_val, sim, K_fixed, cam_pos_fixed, cam_mat_fixed)
                        current_frame_3d_points.append(pt_3d)
                    else:
                        current_frame_3d_points.append(np.array([np.nan, np.nan, np.nan]))

                intersection_3d = np.array([np.nan, np.nan, np.nan])
                p_forward = np.array([np.nan, np.nan, np.nan])

                valid_points = [pt for pt in current_frame_3d_points if not np.isnan(pt).any()]
                if len(valid_points) > 0:
                    p_forward = valid_points[0]
                    target_x = p_forward[0] + cfg.forward_offset 
                    target_y = p_forward[1] + cfg.right_offset
                    raw_intersection_3d = np.array([target_x, target_y, cfg.table_z])

                    if smoothed_intersection is None or np.isnan(smoothed_intersection).any():
                        smoothed_intersection = raw_intersection_3d.copy()
                    else:
                        diff = raw_intersection_3d - smoothed_intersection
                        dist = np.linalg.norm(diff)
                        if dist > cfg.max_move_per_frame:
                            raw_intersection_3d = smoothed_intersection + (diff / dist) * cfg.max_move_per_frame
                        smoothed_intersection = cfg.ema_alpha * raw_intersection_3d + (1.0 - cfg.ema_alpha) * smoothed_intersection

                    intersection_3d = smoothed_intersection.copy()
                else:
                    smoothed_intersection = None
                    
            # ==================== 2. 全局参数配置与目标投影生成 ====================
            img = get_libero_image(obs)             # Inference Base Image 
            wrist_img = get_libero_wrist_image(obs) # Inference Wrist Image

            if sim is not None:
                params = [0.38, 21.21, 14.45, 3.57, 43.54, 0.59, 0.1, 1.23, 0.86, 0.93, 0.19]
                center_def, wave_freq, second_wave_freq, flow_speed, wave_amp, color_variance, light_radius, light_brightness, yellow_r, yellow_g, yellow_b = params
                
                desk_center_x, desk_center_y = (0.0, 0.0) 
                desk_size_x, desk_size_y = (0.8, 0.8) 
                desk_height = cfg.table_z
                
                min_x = desk_center_x - desk_size_x / 2
                max_x = desk_center_x + desk_size_x / 2
                min_y = desk_center_y - desk_size_y / 2
                max_y = desk_center_y + desk_size_y / 2
                
                wave_time = (t - cfg.num_steps_wait) * 0.05
                wave_center = (center_def, center_def)

                if cfg.track_csrt and smoothed_intersection is not None and not np.isnan(smoothed_intersection).any():
                    ray_pt = smoothed_intersection
                    u_wave = (ray_pt[0] - min_x) / desk_size_x
                    v_wave = (max_y - ray_pt[1]) / desk_size_y
                    u_wave = np.clip(u_wave, 0.0, 1.0)
                    v_wave = np.clip(v_wave, 0.0, 1.0)
                    wave_center = (u_wave, v_wave)
                    
                table_corners = np.array([
                    [min_x, max_y, desk_height],
                    [max_x, max_y, desk_height],
                    [max_x, min_y, desk_height],
                    [min_x, min_y, desk_height]
                ])

                cam_main = MujocoCameraWrapper(sim, "agentview", img.shape[1], img.shape[0])
                cam_wrist = MujocoCameraWrapper(sim, "robot0_eye_in_hand", wrist_img.shape[1], wrist_img.shape[0])
                
                if check_corners_in_front(cam_main, table_corners):
                    img = project_wave_to_table(
                        img, cam_main, table_corners_3d=table_corners, wave_center=wave_center, 
                        wave_freq=wave_freq, second_wave_freq=second_wave_freq, flow_speed=flow_speed, 
                        wave_amp=wave_amp, color_variance=color_variance, light_radius=light_radius, 
                        light_brightness=light_brightness, color=(yellow_r, yellow_g, yellow_b), opacity=0.6, t=wave_time
                    )
                
                if check_corners_in_front(cam_wrist, table_corners):
                    wrist_img = project_wave_to_table(
                        wrist_img, cam_wrist, table_corners_3d=table_corners, wave_center=wave_center, 
                        wave_freq=wave_freq, second_wave_freq=second_wave_freq, flow_speed=flow_speed, 
                        wave_amp=wave_amp, color_variance=color_variance, light_radius=light_radius, 
                        light_brightness=light_brightness, color=(yellow_r, yellow_g, yellow_b), opacity=0.6, t=wave_time
                    )
                
                # ==================== 3. 追踪可视化加入光效 ====================
                if cfg.track_csrt:
                    cam_track = MujocoCameraWrapper(sim, "agentview", track_w, track_h)
                    track_rgb_proj = track_rgb_raw.copy()
                    
                    if check_corners_in_front(cam_track, table_corners):
                        track_rgb_proj = project_wave_to_table(
                            track_rgb_proj, cam_track, table_corners_3d=table_corners, wave_center=wave_center, 
                            wave_freq=wave_freq, second_wave_freq=second_wave_freq, flow_speed=flow_speed, 
                            wave_amp=wave_amp, color_variance=color_variance, light_radius=light_radius, 
                            light_brightness=light_brightness, color=(yellow_r, yellow_g, yellow_b), opacity=0.6, t=wave_time
                        )
                        
                    track_bgr_draw = track_rgb_proj[:, :, ::-1].copy()
                    text_y_offset = 20
                    for idx in range(len(current_pixels)):
                        if tracking_flags[idx]:
                            u, v = int(current_pixels[idx][0]), int(current_pixels[idx][1])
                            cv2.rectangle(track_bgr_draw, (u - 15, v - 15), (u + 15, v + 15), (0, 255, 0), 2)
                            cv2.putText(track_bgr_draw, f"P{idx+1}", (u - 15, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            pt_3d = current_frame_3d_points[idx]
                            if not np.isnan(pt_3d).any():
                                cv2.putText(track_bgr_draw, f"P{idx+1}: ({pt_3d[0]:.2f}, {pt_3d[1]:.2f}, {pt_3d[2]:.2f})",
                                            (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                                text_y_offset += 20
                        else:
                            cv2.putText(track_bgr_draw, f"P{idx+1}: Lost", (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                            text_y_offset += 20

                    if not np.isnan(intersection_3d).any() and not np.isnan(p_forward).any():
                        p_f_2d = world_to_pixel_mujoco(p_forward, sim, K_fixed, cam_pos_fixed, cam_mat_fixed)
                        inter_2d = world_to_pixel_mujoco(intersection_3d, sim, K_fixed, cam_pos_fixed, cam_mat_fixed)

                        if p_f_2d and inter_2d:
                            cv2.line(track_bgr_draw, p_f_2d, inter_2d, (255, 0, 255), 2)
                            cv2.circle(track_bgr_draw, inter_2d, 5, (0, 0, 255), -1)

                        cv2.putText(track_bgr_draw, f"Center: ({intersection_3d[0]:.2f}, {intersection_3d[1]:.2f}, {intersection_3d[2]:.2f})",
                                    (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                        text_y_offset += 20

                    tracking_frames.append(cv2.cvtColor(track_bgr_draw, cv2.COLOR_BGR2RGB))

            observation, combined_img = prepare_observation(obs, img, wrist_img, resize_size)
            replay_images.append(combined_img)
            
            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # ==================== 新增：保存追踪视频 ====================
    if cfg.track_csrt and len(tracking_frames) > 0 and ep_save_dir is not None:
        video_path = os.path.join(ep_save_dir, "tracking_vis.mp4")
        # 同步将追踪可视化视频调为 30 fps
        imageio.mimsave(video_path, tracking_frames, fps=30)
    # ============================================================

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    task_save_dir=None
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    
    os.makedirs(task_save_dir, exist_ok=True)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # 新建当前 episode 的独立文件夹
        ep_save_dir = os.path.join(task_save_dir, f"episode_{episode_idx}")
        os.makedirs(ep_save_dir, exist_ok=True)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            episode_idx=episode_idx,
            ep_save_dir=ep_save_dir,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # 将原任务输出视频保存到 episode 独立文件夹中，使用原仓库一致的保存逻辑
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        rollout_video_path = os.path.join(
            ep_save_dir, 
            f"{DATE_TIME}--openvla_oft--episode={episode_idx}--success={success}--task={processed_task_description}.mp4"
        )
        
        video_writer = imageio.get_writer(rollout_video_path, fps=30)
        for img in replay_images:
            video_writer.append_data(img)
        video_writer.close()
        log_message(f"Saved rollout MP4 at path {rollout_video_path}", log_file)

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
