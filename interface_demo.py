#!/usr/bin/env python3
import torch
import pickle
import h5py
import numpy as np
from policy import ACTPolicy
from Robotic_Arm.rm_robot_interface import *

# ----------------------------
# 1️⃣ 配置路径
# ----------------------------
H5_PATH = "data/real_episodes_prepared/single_arm_data_1758190252.hdf5"
CKPT_PATH = "ckpts/real_single_arm/policy_best.ckpt"
CONFIG_PATH = "ckpts/real_single_arm/config.pkl"
ROBOT_IP = "169.254.128.18"
ROBOT_PORT = 8080

# ----------------------------
# 2️⃣ 设备
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ----------------------------
# 3️⃣ 初始化机械臂
# ----------------------------
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm(ip=ROBOT_IP, port=ROBOT_PORT)

# ----------------------------
# 4️⃣ 加载训练配置
# ----------------------------
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)

# ----------------------------
# 5️⃣ 构建 policy 所需参数
# ----------------------------
args_override = {
    "state_dim": 8,
    "action_dim": 8,
    "position_embedding": "sine",
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "masks": False,
    "dilation": False,
    "temporal_agg": False,
    "use_vq": False,
    "vq_class": None,
    "vq_dim": None,
    "vq": False,
    "backbone": "resnet18",       # 确保 build_backbone 支持
    "pretrain_backbone": False,
    "return_interm_layers": True,
    "device": DEVICE,
    "camera_names": ["cam01"],    # ✅ 关键，非空
    "num_queries": 8,            # 根据你模型训练设置
    "vq": False,
    "vq_class": None,
    "vq_dim": None,
    "no_encoder": False,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "kl_weight": 10,
}

# ----------------------------
# 6️⃣ 初始化策略
# ----------------------------
# policy = ACTPolicy(args_override)

# # 加载权重
# state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
# policy.model.to(DEVICE)
# policy.model.eval()
policy = ACTPolicy(args_override).to(DEVICE)
policy.eval()
print("策略模型加载完成")

# ----------------------------
# 7️⃣ 读取 HDF5 数据
# ----------------------------
with h5py.File(H5_PATH, "r") as f:
    images = np.array(f["observations/images/cam01"])  # [N,H,W,3]
    qpos = np.array(f["observations/qpos"])            # [N,7]

# 转 tensor，维度 [N,3,H,W]
images_tensor = torch.tensor(images, dtype=torch.float32).permute(0,3,1,2).unsqueeze(1).to(DEVICE)  # [N,1,3,H,W]
qpos_tensor_full = torch.tensor(qpos, dtype=torch.float32).to(DEVICE)  # 直接用 qpos
print(f"Loaded {len(images_tensor)} frames from HDF5.")

# ----------------------------
# 8️⃣ 推理动作
# ----------------------------
with torch.no_grad():
    out = policy.model(qpos_tensor_full, images_tensor, env_state=None)
    action_pred = out[0].cpu().numpy()  # 形状: [num_queries, 8]
print(f"预测动作序列长度: {len(action_pred)}")

# ----------------------------
# 9️⃣ 控制机械臂执行动作
# ----------------------------
for step_action in action_pred:  # 确保不超出范围
    # joint_angles = action[:-1].tolist()  # 7 个关节
    # gripper_cmd = action[-1]    # 夹爪开合
    
    joint_angles = step_action[:7].tolist()
    gripper_cmd = step_action[7]
    # arm.rm_movej(joint_angles, 10, 0, 0, 1)
    # arm.rm_set_gripper_position(gripper_cmd, True, 0)
    print("arm:",joint_angles)
    print("gripper:", gripper_cmd)

print("All actions executed.")

# ----------------------------
# 🔟 清理
# ----------------------------
arm.rm_delete_robot_arm()
print("回放完成！")
