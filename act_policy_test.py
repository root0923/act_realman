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
print("机械臂初始化完成")

# ======== 加载策略模型 ========
args_override = {
    # 基本训练/优化参数
    'lr': 5e-5,
    'lr_backbone': 1e-5,
    'batch_size': 16,
    'weight_decay': 1e-4,
    'epochs': 300,
    'lr_drop': 200,
    'clip_max_norm': 0.1,

    # Backbone 参数
    'backbone': 'resnet18',
    'dilation': False,
    'position_embedding': 'sine',
    'camera_names': ['cam01'],
    'masks': False,

    # Transformer / DETR 参数
    'enc_layers': 4,
    'dec_layers': 6,
    'dim_feedforward': 1024,
    'hidden_dim': 512,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 400,
    'pre_norm': False,

    # ACT / Diffusion 参数
    'action_dim': 8,
    'state_dim': 8,
    'kl_weight': 10,
    'vq': False,
    'vq_class': 512,
    'vq_dim': 512,
    'observation_horizon': 1,
    'action_horizon': 1,
    'prediction_horizon': 1,
    'num_inference_timesteps': 50,
    'ema_power': 0.995,
    'no_encoder': False,

    # 占位参数
    'eval': False,
    'onscreen_render': False,
    'ckpt_dir': 'ckpts/real_single_arm',
    'policy_class': 'ACT',
    'task_name': 'real_single_arm',
    'seed': 42,
    'num_steps': 10000,
    'chunk_size': 16,
    'temporal_agg': False,
    'load_pretrain': False,
    'eval_every': 500,
    'validate_every': 500,
    'save_every': 500,
    'resume_ckpt_path': None,
    'skip_mirrored_data': False,
    'actuator_network_dir': None,
}
# state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
policy = ACTPolicy(args_override).to(DEVICE)
# policy.load_state_dict(state_dict['model'])
policy.eval()
print("策略模型加载完成")

# ======== 读取 HDF5 数据 ========
with h5py.File(H5_PATH, "r") as f:
    images_h5 = np.array(f["observations/images/cam01"])  # [N,H,W,3]
    qpos_h5 = np.array(f["observations/qpos"])            # [N,8]
    actions_h5 = np.array(f["action"])                    # [N,8]

print(f"Loaded {len(actions_h5)} frames from HDF5.")
print(f"qpos_h5 shape: {qpos_h5.shape}")
print(f"images_h5 shape: {images_h5.shape}")

# ======== 修复图像输入格式 ========
# 预处理第一帧图像
image_np = images_h5[0]  # [480, 640, 3]

# 转换为tensor并调整维度
image_tensor = torch.from_numpy(image_np).float()  # [480, 640, 3]
image_tensor = image_tensor.permute(2, 0, 1)      # [3, 480, 640]
image_tensor = image_tensor / 255.0               # 归一化到[0, 1]

# 扩展为多摄像头格式 [1, 1, 3, 480, 640]
image_tensor = image_tensor.unsqueeze(0)          # [1, 3, 480, 640] -> 添加batch维度
image_tensor = image_tensor.unsqueeze(1)          # [1, 1, 3, 480, 640] -> 添加摄像头维度
image_tensor = image_tensor.to(DEVICE)

# 预处理qpos
qpos_tensor = torch.from_numpy(np.deg2rad(qpos_h5[0:1])).float().to(DEVICE)  # [1, 8]

print(f"修复后的 images_tensor shape: {image_tensor.shape}")
print(f"修复后的 qpos_tensor shape: {qpos_tensor.shape}")

# ======== 执行抓取函数 ========
def execute_grasp(mode="model"):
    """
    mode: "h5" 使用 HDF5 原始动作复现
          "model" 使用模型预测动作
    """
    if mode == "model":
        print(f"\n=== 开始模型推理 ===")
        print(f"输入 qpos_tensor shape: {qpos_tensor.shape}")
        print(f"输入 images_tensor shape: {image_tensor.shape}")
        
        with torch.no_grad():
            try:
                model = policy.model
                out = model.forward(qpos=qpos_tensor, image=image_tensor, env_state=None)
                
                # 处理输出 - 关键修复！
                if isinstance(out, tuple) or isinstance(out, list):
                    action_pred = out[0].cpu().numpy()  # [1, 400, 8]
                else:
                    action_pred = out.cpu().numpy()
                    
                print(f"模型输出形状: {action_pred.shape}")
                
                # 关键：从400个查询中选择一个动作序列
                # 方法1：选择第一个查询的动作序列
                action_seq = action_pred[0]  # [400, 8]
                
                # 方法2：或者对所有查询取平均
                # action_seq = np.mean(action_pred[0], axis=0, keepdims=True)  # [1, 8]
                
                print(f"处理后的动作序列形状: {action_seq.shape}")
                
            except Exception as e:
                print(f"模型推理失败: {e}")
                import traceback
                traceback.print_exc()
                return
    else:
        action_seq = actions_h5
        print(f"HDF5 动作序列长度: {len(action_seq)}")

    # 确保动作序列是2维的
    if len(action_seq.shape) == 1:
        action_seq = action_seq.reshape(1, -1)
    
    print(f"\n=== 开始执行动作序列 ===")
    print(f"动作序列形状: {action_seq.shape}")
    
    # 限制执行步数，避免太长
    max_steps = len(action_seq)  # 最多执行50步
    
    for i in range(max_steps):
        step_action = action_seq[i]
        
        # 调试信息
        print(f"步骤 {i+1}/{max_steps}, 动作形状: {step_action.shape}")
        print(f"原始动作值: {step_action}")
        
        # ====== 关节角度 ======
        if len(step_action) >= 7:
            if mode == "model":
                # 模型输出的是弧度，需要转换为角度
                joint_angles_deg = np.rad2deg(step_action[:7]).tolist()
            else:
                # HDF5数据已经是角度
                joint_angles_deg = step_action[:7].tolist()
        else:
            print(f"警告: 步骤 {i} 的动作维度不足: {len(step_action)}")
            continue
            
        # ====== 夹爪映射 ======
        if len(step_action) >= 8:
            gripper_cmd = step_action[7]
            if mode == "model":
                # 模型输出范围可能是[-1,1]或其他，需要映射到[0,100]
                gripper_target = ((gripper_cmd + 1) / 2 * 100)  # 假设模型输出在[-1,1]
                gripper_target = np.clip(gripper_target, 0, 100)  # 限制在0-100之间
            else:
                # HDF5数据已经是0-100的范围
                gripper_target = gripper_cmd
        else:
            gripper_target = 50  # 默认值

        print(f"Step {i+1}/{max_steps}")
        print(f"  关节角度(度): {[f'{x:.2f}' for x in joint_angles_deg]}")
        print(f"  夹爪位置: {gripper_target}")

        # ====== 执行机械臂动作 ======
        # arm.rm_movej(joint_angles_deg, 5, 0, 0, 1)
        # arm.rm_set_gripper_position(int(gripper_target), True, 0)

        # ====== 可视化 HDF5 图像 ======
        if i < len(images_h5):
            frame = images_h5[i]
            cv2.imshow("Camera", frame.astype(np.uint8))
            if cv2.waitKey(int(DT*1000)) & 0xFF == ord('q'):
                break

    arm.rm_delete_robot_arm()
    cv2.destroyAllWindows()
    print("抓取复现完成")

# ======== 主程序 ========
if __name__ == "__main__":
    try:
        execute_grasp(mode="model")   # "model" 或 "h5"
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()