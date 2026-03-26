#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import time
import h5py
from policy import ACTPolicy
from Robotic_Arm.rm_robot_interface import *

# ========== 配置 ==========
CKPT_DIR = "ckpts/real_single_arm"
H5_PATH = "data/real_episodes_prepared/single_arm_data_1758190252.hdf5"
CKPT_PATH = "ckpts/real_single_arm/policy_best.ckpt"
CONFIG_PATH = "ckpts/real_single_arm/config.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 8
DT = 0.05  # 每步延时，保证动作平滑

# ======== 初始化机械臂 ========
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm(ip="169.254.128.18", port=8080)
# arm.rm_movej([0,0,0,0,0,0,0], 3, 0, 0, 1)
print("机械臂初始化完成")

# ======== 加载策略模型 ========
args_override = {
    # ----------------------------------------
    # 基本训练/优化参数（即使不训练也要）
    # ----------------------------------------
    'lr': 1e-4,
    'lr_backbone': 1e-5,
    'batch_size': 2,
    'weight_decay': 1e-4,
    'epochs': 300,
    'lr_drop': 200,
    'clip_max_norm': 0.1,

    # ----------------------------------------
    # Backbone 参数
    # ----------------------------------------
    'backbone': 'resnet18',
    'dilation': False,
    'position_embedding': 'sine',  # 'sine' 或 'learned'
    'camera_names': ['cam01'],     # 单摄像头
    'masks': False,

    # ----------------------------------------
    # Transformer / DETR 参数
    # ----------------------------------------
    'enc_layers': 4,
    'dec_layers': 6,
    'dim_feedforward': 3200,
    'hidden_dim': 512,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 400,
    'pre_norm': False,

    # ----------------------------------------
    # ACT / Diffusion 参数
    # ----------------------------------------
    'action_dim': 8,               # 机械臂自由度 + 夹爪
    'state_dim': 7,                # qpos维度
    'kl_weight': 10,
    'vq': False,
    'vq_class': 512,               # 仅在使用 VQ 时有效
    'vq_dim': 512,                 # 仅在使用 VQ 时有效
    'observation_horizon': 1,
    'action_horizon': 1,
    'prediction_horizon': 1,
    'num_inference_timesteps': 50,
    'ema_power': 0.995,
    'no_encoder': False,

    # ----------------------------------------
    # 占位参数（DETR / ACT 中必须）
    # ----------------------------------------
    'eval': False,
    'onscreen_render': False,
    'ckpt_dir': 'ckpts/real_single_arm/ckpt1',          # 这里填你的模型路径
    'policy_class': 'ACT',          # 'ACT', 'CNNMLP' 或 'Diffusion'
    'task_name': 'real_single_arm',
    'seed': 0,
    'num_steps': 500,
    'chunk_size': 8,
    'temporal_agg': False,
    'load_pretrain': False,
    'eval_every': 500,
    'validate_every': 500,
    'save_every': 500,
    'resume_ckpt_path': None,
    'skip_mirrored_data': False,
    'actuator_network_dir': None,
}
policy = ACTPolicy(args_override).to(DEVICE)
policy.eval()
print("策略模型加载完成")

# ======== 读取 HDF5 数据 ========
with h5py.File(H5_PATH, "r") as f:
    images_h5 = np.array(f["observations/images/cam01"])  # [N,H,W,3]
    qpos_h5 = np.array(f["observations/qpos"])            # [N,7]
    actions_h5 = np.array(f["action"])                    # [N,8]

print(f"Loaded {len(actions_h5)} frames from HDF5.")

# ======== 转 Tensor (用于模型推理) ========
images_tensor = torch.from_numpy(images_h5).float().permute(0,3,1,2).to(DEVICE) / 255.0
images_tensor = images_tensor.unsqueeze(0)
qpos_tensor = torch.from_numpy(np.deg2rad(qpos_h5)).float().to(DEVICE)  # 弧度
qpos_tensor = qpos_tensor.unsqueeze(0) 

# ======== 执行抓取函数 ========
def execute_grasp(mode="model"):
    """
    mode: "h5" 使用 HDF5 原始动作复现
          "model" 使用模型预测动作
    """
    if mode == "model":
        with torch.no_grad():
            out = policy.model(qpos_tensor, images_tensor, env_state=None)
            action_seq = out[0].cpu().numpy()  # [num_queries, 8]
            print(f"预测动作序列长度: {len(action_seq)}")
    else:
        action_seq = actions_h5
        print(f"HDF5 动作序列长度: {len(action_seq)}")

    for i, step_action in enumerate(action_seq):
        # ====== 关节角度 ======
        joint_angles_deg = np.rad2deg(step_action[:7]).tolist() if mode=="model" else step_action[:7].tolist()
        # ====== 夹爪映射 ======
        gripper_cmd = step_action[7]
        gripper_target = ((gripper_cmd + 1)/2 * 100) if mode=="model" else gripper_cmd

        # ====== 执行机械臂动作 ======
        # arm.rm_movej(joint_angles_deg, 5, 0, 0, 1)
        # arm.rm_set_gripper_position(int(gripper_target), True, 0)
        print("arm:", joint_angles_deg),
        print("gripper:", gripper_target)

        # ====== 可视化 HDF5 图像 ======
        # frame = images_h5[i]
        # cv2.imshow("Camera", frame.astype(np.uint8))
        # cv2.waitKey(int(DT*1000))

        print(f"Step {i+1}/{len(action_seq)}")
    arm.rm_delete_robot_arm()
    cv2.destroyAllWindows()
    print("抓取复现完成")

# ======== 主程序 ========
if __name__ == "__main__":
    execute_grasp(mode="model")   # "model" 或 "h5"
