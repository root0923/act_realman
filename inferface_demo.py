#!/usr/bin/env python3
import torch
import numpy as np
import time
import pickle
from einops import rearrange

from policy import ACTPolicy
from constants import FPS, PUPPET_GRIPPER_JOINT_OPEN
from utils import set_seed
from detr.models.latent_model import Latent_Model_Transformer

# 机械臂控制接口
from aloha_scripts.robot_utils import move_grippers
from aloha_scripts.real_env import make_real_env

# ---------- 配置 ----------
CKPT_DIR = "ckpts/real_single_arm"
CKPT_NAME = "policy_best.ckpt"
STATS_PATH = f"{CKPT_DIR}/dataset_stats.pkl"
NUM_STEPS = 500
CAMERA_NAMES = ["cam01"]
SEED = 0

# ---------- 初始化 ----------
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载模型 ----------
with open(STATS_PATH, "rb") as f:
    stats = pickle.load(f)

policy_config = {
    'lr': 1e-5,
    'num_queries': 8,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': CAMERA_NAMES,
    'vq': False,
    'state_dim': 8,
    'action_dim': 8,
    'no_encoder': False,
}
policy = ACTPolicy(policy_config)
policy.deserialize(torch.load(f"{CKPT_DIR}/{CKPT_NAME}"))
policy.to(device)
policy.eval()

# 如果使用 VQ，需要加载 latent model（这里假设你没用 vq）
use_vq = policy_config['vq']
if use_vq:
    vq_dim = policy_config['vq_dim']
    vq_class = policy_config['vq_class']
    latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
    latent_model_ckpt_path = f"{CKPT_DIR}/latent_model_last.ckpt"
    latent_model.deserialize(torch.load(latent_model_ckpt_path))
    latent_model.eval()
    latent_model.to(device)

# ---------- 环境 ----------
env = make_real_env(init_node=True, setup_robots=True, setup_base=True)

# ---------- 辅助函数 ----------
def get_image(ts, camera_names):
    imgs = []
    for cam in camera_names:
        img = rearrange(ts.observation['images'][cam], 'h w c -> c h w')
        imgs.append(img)
    img = np.stack(imgs, axis=0)
    img = torch.from_numpy(img / 255.0).float().to(device).unsqueeze(0)
    return img

def pre_process_qpos(qpos):
    return torch.from_numpy((qpos - stats['qpos_mean']) / stats['qpos_std']).float().to(device).unsqueeze(0)

def post_process_action(raw_action):
    return raw_action * stats['action_std'] + stats['action_mean']

# ---------- 推理循环 ----------
DT = 1.0 / FPS
for step in range(NUM_STEPS):
    ts = env.get_observation()  # 获取机械臂当前状态
    qpos = np.array(ts['qpos'])
    image = get_image(ts, CAMERA_NAMES)
    qpos_input = pre_process_qpos(qpos)

    with torch.no_grad():
        if use_vq:
            vq_sample = latent_model.generate(1, temperature=1, x=None)
            raw_action = policy(qpos_input, image, vq_sample=vq_sample)[:, 0]
        else:
            raw_action = policy(qpos_input, image)[:, 0]

    raw_action = raw_action.cpu().numpy().squeeze()
    action = post_process_action(raw_action)
    target_qpos = action[:-2]
    gripper_action = action[-2:]

    # 执行动作
    ts = env.step(target_qpos, gripper_action)

    # 可视化或延时
    time.sleep(DT)

# 打开夹爪
move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
