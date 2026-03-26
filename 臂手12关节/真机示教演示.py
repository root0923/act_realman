#!/usr/bin/env python3
# filepath: /home/shuxiangzhang/act-yang/部署真机_teach.py

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
import cv2
import pyrealsense2 as rs
from torchvision import transforms
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import sys

# 在导入可能会修改参数解析器的模块前保存命令行参数
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # 只保留程序名称，移除所有参数

# 导入策略模型
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

# 恢复命令行参数
sys.argv = original_argv


def make_policy(policy_class, policy_config):
    """创建策略模型"""
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError(f"不支持的策略类型: {policy_class}")
    return policy


def image_preprocess(image1, image2, rand_crop_resize=False):
    """处理两个摄像头图像为模型输入格式"""
    output = []
    for image in [image1, image2]:
        img = rearrange(image, 'h w c -> c h w')
        output.append(img)
    output = np.stack(output, axis=0)
    output = torch.from_numpy(output / 255.0).float().cuda().unsqueeze(0)
    if rand_crop_resize:
        original_size = output.shape[-2:]
        ratio = 0.95
        output = output[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        output = output.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        output = resize_transform(output)
        output = output.unsqueeze(0)
    return output


def init_cameras(resolution_width=640, resolution_height=480, fps=30):
    """初始化两个RealSense相机"""
    print("初始化相机...")
    
    # 创建上下文以查找相机
    ctx = rs.context()
    devices = list(ctx.query_devices())
    
    if len(devices) < 2:
        raise RuntimeError(f"需要两个RealSense相机，但只找到{len(devices)}个")
    
    pipelines = []
    
    for i, device in enumerate(devices[:2]):
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 获取设备序列号
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"连接到相机 {i+1}: {device.get_info(rs.camera_info.name)} (S/N: {serial})")
        
        # 启用特定设备的颜色流
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, fps)
        
        # 启动管道
        pipeline.start(config)
        pipelines.append(pipeline)
    
    # 等待相机启动
    time.sleep(1)
    print(f"成功初始化 {len(pipelines)} 个相机")
    return pipelines


def get_camera_frames(pipelines):
    """从相机获取图像帧"""
    images = []
    
    for pipeline in pipelines:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # 将图像转换为numpy数组
        image = np.asanyarray(color_frame.get_data())
        images.append(image)
    
    return images


def init_robot_arm(ip_address="192.168.110.118", port=8080, init_pose=None):
    """初始化机械臂"""
    print(f"连接机械臂 {ip_address}:{port}...")
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_handle = robot.rm_create_robot_arm(ip_address, port)
    
    if init_pose is not None:
        # 移动到初始位置
        result = robot.rm_movej(init_pose, 30, 0, 0, 1)
        if result != 0:
            print(f"移动到初始位置失败，错误码: {result}")
        else:
            print(f"机械臂已移动到初始位置: {init_pose}")
    
    return robot, arm_handle


def get_current_joint_angles(robot):
    """获取当前机械臂关节角度"""
    arm_status = robot.rm_get_current_arm_state()
    if arm_status[0] == 0:  # 状态正常
        return arm_status[1]['joint'].copy()
    else:
        print(f"获取关节角度失败，状态码: {arm_status[0]}")
        return None


def get_current_hand_state(robot):
    """获取当前灵巧手关节状态"""
    try:
        hand_status = robot.rm_get_hand_state()
        if isinstance(hand_status, dict) and 'hand_pos' in hand_status:
            return hand_status['hand_pos']
        else:
            print(f"获取灵巧手状态失败，返回: {hand_status}")
            return [0, 0, 0, 0, 0, 0]  # 默认值
    except Exception as e:
        print(f"获取灵巧手状态出错: {e}")
        return [0, 0, 0, 0, 0, 0]  # 默认值


def control_robot_hand(robot, hand_action):
    """控制灵巧手6个关节"""
    try:
        # 确保手部动作是整数列表
        hand_pos = [int(x) for x in hand_action]
        # 发送控制命令到灵巧手
        result = robot.rm_set_hand_follow_pos(hand_pos, 0)
        if result != 0:
            print(f"控制灵巧手失败，错误码: {result}")
        return result
    except Exception as e:
        print(f"控制灵巧手出错: {e}")
        return -1


def pad_qpos(qpos, hand_state, target_dim=12):
    """将机械臂关节状态和灵巧手状态合并为一个完整状态"""
    # 确保输入是列表或numpy数组
    qpos = np.array(qpos, dtype=np.float32)
    hand_state = np.array(hand_state, dtype=np.float32)
    
    # 合并状态
    full_state = np.concatenate([qpos, hand_state])
    
    # 如果维度不足，补零
    if len(full_state) < target_dim:
        pad_width = target_dim - len(full_state)
        full_state = np.pad(full_state, (0, pad_width), 'constant')
    
    # 如果维度过多，截断
    if len(full_state) > target_dim:
        full_state = full_state[:target_dim]
    
    return full_state


def main(args):
    # 设置随机种子以确保可重复性
    if 'seed' in args:
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
    
    # 从配置中获取必要参数
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    temporal_agg = args['temporal_agg']
    use_vq = args.get('use_vq', False)
    vq_class = args.get('vq_class', None)
    vq_dim = args.get('vq_dim', None)
    no_encoder = args.get('no_encoder', False)
    
    # 初始位姿 - 可以根据实际情况调整
    init_pose = [36.43, -3.88, 82.67, 5.19, 77.72, -3.09]  
    
    # 初始化机械臂
    master_arm, arm_handle = init_robot_arm(
        args.get('robot_ip', "192.168.110.118"), 
        args.get('robot_port', 8080), 
        init_pose
    )
    
    # 初始化相机
    pipelines = init_cameras(
        args.get('width', 640), 
        args.get('height', 480), 
        args.get('fps', 30)
    )
    
    # 设置12维的状态和动作空间
    state_dim = 12
    action_dim = 12
    camera_names = ['cam01', 'cam02']
    
    # 配置策略
    policy_config = {
        'lr': 1e-5,
        'num_queries': args.get('chunk_size', 100),
        'kl_weight': args.get('kl_weight', 10),
        'hidden_dim': args.get('hidden_dim', 512),
        'dim_feedforward': args.get('dim_feedforward', 3200),
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'vq': use_vq,
        'vq_class': vq_class,
        'vq_dim': vq_dim,
        'state_dim': state_dim,    # 使用12维状态
        'action_dim': action_dim,  # 使用12维动作
        'no_encoder': no_encoder,
    }
    
    # 加载模型和统计数据
    print(f"从 {ckpt_dir} 加载模型...")
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    
    # 加载统计数据（用于归一化）
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # 定义预处理和后处理函数，处理12维状态和动作
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean'][:state_dim]) / stats['qpos_std'][:state_dim]
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'][:action_dim] - stats['action_min'][:action_dim]) + stats['action_min'][:action_dim]
    else:
        post_process = lambda a: a * stats['action_std'][:action_dim] + stats['action_mean'][:action_dim]
    
    # 创建显示窗口
    cv2.namedWindow("相机 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("相机 2", cv2.WINDOW_NORMAL)
    
    # 初始化时间聚合
    if temporal_agg:
        print('启用时序聚合')
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([1, 1000 + num_queries, action_dim]).cuda()  # 假设最多1000帧
        frame_count = 0
    
    # 初始化帧计数和时间记录
    frame_count = 0
    last_time = time.time()
    joint_errors = []
    
    print("开始实时控制循环。按'q'退出。")
    try:
        with torch.inference_mode():
            while True:
                start_time = time.time()
                
                # 1. 获取相机图像
                images = get_camera_frames(pipelines)
                if images is None or len(images) < 2:
                    print("获取相机图像失败，重试中...")
                    continue
                
                # 2. 显示相机图像
                cv2.imshow("相机 1", images[0])
                cv2.imshow("相机 2", images[1])
                
                # 检查按键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 3. 获取机械臂当前关节状态和灵巧手状态
                qpos_numpy = get_current_joint_angles(master_arm)
                hand_state = get_current_hand_state(master_arm)
                
                if qpos_numpy is None:
                    print("获取关节角度失败，重试中...")
                    continue
                
                # 合并机械臂和灵巧手状态为12维状态
                full_qpos = pad_qpos(qpos_numpy, hand_state, state_dim)
                
                # 4. 预处理数据
                qpos = pre_process(full_qpos)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                image_tensor = image_preprocess(images[0], images[1], rand_crop_resize=(policy_class == 'Diffusion'))
                
                # 5. 模型推理
                inference_start = time.time()
                all_action = policy(qpos, image_tensor)
                inference_time = time.time() - inference_start
                
                # 6. 时序聚合（如果启用）
                if temporal_agg:
                    # 更新时序动作缓冲区
                    all_time_actions[[0], frame_count:frame_count+num_queries] = all_action
                    actions_for_curr_step = all_time_actions[:, frame_count]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    # 计算加权平均
                    if len(actions_for_curr_step) > 0:
                        k = 0.01  # 指数衰减系数
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_action[:, 0]
                else:
                    # 不使用时序聚合，直接取第一个预测动作
                    raw_action = all_action[:, 0]
                
                # 7. 后处理获得实际控制命令
                raw_action = raw_action.squeeze(0).detach().cpu().numpy()
                action = post_process(raw_action)
                
                # 8. 将12维动作拆分为机械臂动作（前6维）和灵巧手动作（后6维）
                arm_action = action[:6]
                hand_action = action[6:12]
                
                # 9. 发送命令到机械臂（如果启用）
                if args.get('execute_actions', False):
                    # 控制机械臂
                    arm_result = master_arm.rm_movej_canfd(arm_action.tolist(), False, 0, 1, 50)
                    if arm_result != 0:
                        print(f"执行机械臂动作失败，错误码: {arm_result}")
                    
                    # 控制灵巧手
                    hand_result = control_robot_hand(master_arm, hand_action)
                    if hand_result != 0:
                        print(f"执行灵巧手动作失败，错误码: {hand_result}")
                
                # 10. 打印推理结果
                loop_time = time.time() - start_time
                if frame_count % 10 == 0:
                    print(f"\n=== 帧 {frame_count} ===")
                    print(f"当前机械臂状态: {[round(x, 2) for x in qpos_numpy]}")
                    print(f"当前灵巧手状态: {[round(x, 2) for x in hand_state]}")
                    print(f"预测机械臂动作: {[round(x, 2) for x in arm_action]}")
                    print(f"预测灵巧手动作: {[round(x, 2) for x in hand_action]}")
                    print(f"推理时间: {inference_time*1000:.2f}毫秒")
                    print(f"总循环时间: {loop_time*1000:.2f}毫秒")
                    current_fps = 10 / (time.time() - last_time)
                    print(f"FPS: {current_fps:.1f}")
                    last_time = time.time()
                
                # 11. 帧计数增加
                frame_count += 1
                
                # 12. 控制循环频率
                elapsed = time.time() - start_time
                target_loop_time = 1.0 / args.get('control_fps', 10)
                if elapsed < target_loop_time:
                    time.sleep(target_loop_time - elapsed)
    
    except KeyboardInterrupt:
        print("用户中断控制")
    
    finally:
        # 清理资源
        print("清理资源...")
        
        # 停止相机
        for pipeline in pipelines:
            pipeline.stop()
        
        # 关闭显示窗口
        cv2.destroyAllWindows()
        
        # 将机械臂移回初始位置
        print("将机械臂移回初始位置...")
        master_arm.rm_movej(init_pose, 30, 0, 0, 1)
        
        # 将灵巧手回到初始状态
        print("将灵巧手回到初始状态...")
        control_robot_hand(master_arm, [0, 0, 0, 0, 0, 0])
        
        # 释放机械臂资源
        master_arm.rm_delete_robot_arm()
        
        print("程序结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RealMan机械臂与灵巧手实时控制")
    
    # DETR需要的参数
    parser.add_argument('--task_name', default='real_test01', type=str, help='任务名称')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--num_steps', default=20000, type=int, help='训练步数')
    
    # 模型和ckpt相关参数
    parser.add_argument('--ckpt_dir', default='/home/shuxiangzhang/act-yang/ckpt/real_test01', type=str, help='模型检查点目录')
    parser.add_argument('--policy_class', default='ACT', type=str, help='策略类型 (ACT, CNNMLP, Diffusion)')
    parser.add_argument('--chunk_size', default=100, type=int, help='块大小')
    parser.add_argument('--kl_weight', default=10, type=int, help='KL权重')
    parser.add_argument('--hidden_dim', default=512, type=int, help='隐藏层维度')
    parser.add_argument('--dim_feedforward', default=3200, type=int, help='前馈网络维度')
    parser.add_argument('--temporal_agg', action='store_true', default=False, help='使用时序聚合')
    parser.add_argument('--use_vq', action='store_true', help='使用VQ')
    parser.add_argument('--vq_class', type=int, default=None, help='VQ类别数')
    parser.add_argument('--vq_dim', type=int, default=None, help='VQ维度')
    parser.add_argument('--no_encoder', action='store_true', help='不使用编码器')
    
    # 机械臂和相机参数
    parser.add_argument('--robot_ip', default='192.168.110.118', type=str, help='机械臂IP地址')
    parser.add_argument('--robot_port', default=8080, type=int, help='机械臂端口')
    parser.add_argument('--execute_actions', action='store_true', help='是否执行动作控制机械臂')
    parser.add_argument('--width', default=640, type=int, help='相机分辨率宽度')
    parser.add_argument('--height', default=480, type=int, help='相机分辨率高度')
    parser.add_argument('--fps', default=30, type=int, help='相机帧率')
    parser.add_argument('--control_fps', default=10, type=int, help='控制循环频率')
    
    # 使用parse_known_args避免DETR冲突
    args, unknown = parser.parse_known_args()
    main(vars(args))