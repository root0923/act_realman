#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import time
from policy import ACTPolicy
from Robotic_Arm.rm_robot_interface import *

# ========== 配置 ==========
CKPT_DIR = "ckpts/real_single_arm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 8
DT = 0.05  # 每步延时，保证动作平滑

# args_override 模板
# =====================================
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

# ======== 初始化机械臂 + 摄像头 ========
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm(ip="169.254.128.18", port=8080)
cap = cv2.VideoCapture("http://169.254.128.20:5000/video_feed")
print("机械臂和相机初始化完成")

# ======== 加载策略模型 ========
policy = ACTPolicy(args_override).to(DEVICE)
policy.eval()
print("策略模型加载完成")

# ======== 获取观测函数 ========
def get_obs():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("获取相机画面失败")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float().to(DEVICE) / 255.0
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(1)  # [B, num_cam=1, C, H, W]
    
    qpos = torch.from_numpy(np.array(arm.rm_get_joint_degree()[1])).unsqueeze(0).float().to(DEVICE)  # [B, 7]
    
    return image_tensor, qpos, frame  # frame 用于可视化

# ======== 执行抓取函数 ========
def execute_grasp():
    with torch.no_grad():
        images, qpos, frame = get_obs()
        
        # 模型前向
        out = policy.model(qpos, images, env_state=None)
        action_all = out[0].squeeze(0).cpu().numpy()  # [T, 8]
        
        print(f"预测动作序列长度: {len(action_all)}")
        
        # 逐步执行动作
        for i, step_action in enumerate(action_all):
            # print("step_action: ", step_action)
            # target_qpos = step_action[:7].tolist()
            # gripper_cmd = step_action[7]
            qpos_target_deg = np.array(arm.rm_get_joint_degree()[1]) + np.degrees(step_action[:7])
            gripper_target = int((step_action[7] + 1)/2 * (999 - 0) + 0)
            # # 执行机械臂动作
            # arm.rm_movej(qpos_target_deg, 3, 0, 0, 1)
            # arm.rm_set_gripper_position(gripper_target, True, 0)
            
            # # 可视化相机
            cv2.imshow("Camera", frame)
            cv2.waitKey(int(DT*1000))  # 延时
            print("arm:", qpos_target_deg)
            print("gripper:", gripper_target)
            print(f"Step {i+1}/{len(action_all)} 执行完成")

    cv2.destroyAllWindows()

# ======== 主程序 ========
if __name__ == "__main__":
    execute_grasp()
    print("抓取复现完成")
