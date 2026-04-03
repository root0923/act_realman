#!/usr/bin/env python3
"""
单臂（7关节）+ 夹爪（1维）实时推理部署脚本

硬件要求:
  - Realman 7轴机械臂
  - 夹爪
  - Intel RealSense相机（支持多个）

数据维度:
  - qpos/action: 8维 [7关节角度 + 1夹爪宽度]
"""

import torch
import numpy as np
import os
import pickle
import argparse
import time
import cv2
import pyrealsense2 as rs
from torchvision import transforms
from einops import rearrange
import sys
from pynput import keyboard

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 保存原始命令行参数
original_argv = sys.argv
sys.argv = [sys.argv[0]]

# 导入策略模型
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

# 恢复命令行参数
sys.argv = original_argv

# 导入Realman机械臂接口
from Robotic_Arm.rm_robot_interface import *


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


def init_cameras(camera_names, resolution_width=640, resolution_height=480, fps=30):
    """
    初始化RealSense相机

    Args:
        camera_names: 相机名称列表（用于识别）
        resolution_width: 图像宽度
        resolution_height: 图像高度
        fps: 帧率

    Returns:
        pipelines: 相机pipeline列表
    """
    print(f"初始化 {len(camera_names)} 个相机...")

    ctx = rs.context()
    devices = list(ctx.query_devices())

    num_cameras = len(camera_names)
    if len(devices) < num_cameras:
        raise RuntimeError(f"需要 {num_cameras} 个RealSense相机，但只找到 {len(devices)} 个")

    pipelines = []

    for i in range(num_cameras):
        device = devices[i]
        pipeline = rs.pipeline()
        config = rs.config()

        serial = device.get_info(rs.camera_info.serial_number)
        print(f"  连接相机 {i+1} ({camera_names[i]}): S/N {serial}")

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height,
                           rs.format.bgr8, fps)

        pipeline.start(config)
        pipelines.append(pipeline)

    # 等待相机稳定
    time.sleep(1)

    # 丢弃前几帧
    for _ in range(30):
        for pipeline in pipelines:
            pipeline.wait_for_frames()

    print(f"✓ 成功初始化 {len(pipelines)} 个相机")
    return pipelines


def get_camera_frames(pipelines):
    """从相机获取图像帧"""
    images = []

    for pipeline in pipelines:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        image = np.asanyarray(color_frame.get_data())
        images.append(image)

    return images


def image_preprocess(images, rand_crop_resize=False):
    """
    预处理多相机图像

    Args:
        images: 图像列表 [(H, W, 3), ...]
        rand_crop_resize: 是否使用随机裁剪（Diffusion Policy）

    Returns:
        Tensor (1, num_cams, 3, H, W)
    """
    processed = []
    for image in images:
        # HWC -> CHW
        img = rearrange(image, 'h w c -> c h w')
        processed.append(img)

    # Stack: (num_cams, 3, H, W)
    output = np.stack(processed, axis=0)

    # 归一化到[0, 1]并转换为tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.from_numpy(output / 255.0).float().to(device).unsqueeze(0)

    if rand_crop_resize:
        original_size = output.shape[-2:]
        ratio = 0.95
        output = output[...,
                       int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                       int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        output = output.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        output = resize_transform(output)
        output = output.unsqueeze(0)

    return output


def init_robot_arm(ip_address, port=8080):
    """初始化Realman机械臂"""
    print(f"连接机械臂: {ip_address}:{port}")

    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm(ip_address, port)

    # 验证连接是否成功
    result = robot.rm_get_current_arm_state()
    if result[0] != 0:
        raise RuntimeError(f"机械臂连接失败，无法获取状态，错误码: {result[0]}")

    print("✓ 机械臂连接成功")
    return robot


def get_current_state(robot):
    """
    获取机械臂当前状态

    Returns:
        qpos: numpy array (8,) - [7关节角度 + 1夹爪宽度]
    """
    # 获取关节角度
    result = robot.rm_get_current_arm_state()
    if result[0] != 0:
        print(f"获取关节状态失败，错误码: {result[0]}")
        return None

    joint_angles = np.array(result[1]['joint'], dtype=np.float32)  # (7,)

    # 获取夹爪宽度
    gripper_result = robot.rm_get_gripper_state()
    if gripper_result[0] != 0:
        print(f"获取夹爪状态失败，错误码: {gripper_result[0]}")
        gripper_width = 0.0
    else:
        gripper_width = float(gripper_result[1].get('actpos', 0))

    # 组合: [7关节 + 1夹爪]
    qpos = np.concatenate([joint_angles, [gripper_width]])

    return qpos


def execute_action(robot, action, use_canfd=True):
    """
    执行动作命令

    Args:
        robot: 机械臂对象
        action: numpy array (8,) - [7关节角度 + 1夹爪宽度] (已反归一化)
        use_canfd: 是否使用CAN FD通信（更快）
    """
    # 分离关节和夹爪指令
    joint_action = action[:7].tolist()
    gripper_action = float(action[7])

    # 控制机械臂关节
    if use_canfd:
        result = robot.rm_movej_canfd(joint_action, False, 0, 1, 60)
    else:
        result = robot.rm_movej(joint_action, 50, 0, 0, 1)

    if result != 0:
        print(f"  警告: 机械臂执行失败，错误码: {result}")

    # 控制夹爪 - 强制关闭状态
    gripper_result = robot.rm_set_gripper_position(0, False, 1)
    if gripper_result != 0:
        print(f"  警告: 夹爪执行失败，错误码: {gripper_result}")


def main(args):
    """主控制循环"""

    # 键盘状态
    key_state = {'right_arrow': False, 'space': False, 'quit': False}

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                key_state['right_arrow'] = True
            elif key == keyboard.Key.space:
                key_state['space'] = True
            elif key == keyboard.Key.esc:
                key_state['quit'] = True
        except:
            pass

    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 设置随机种子
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])

    # 参数
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    temporal_agg = args['temporal_agg']
    state_dim = 8  # 7关节 + 1夹爪
    action_dim = 8
    camera_names = args['camera_names']

    # 初始化机械臂
    robot = init_robot_arm(args['robot_ip'], args['robot_port'])

    # 移动到初始位置（如果指定）
    if args.get('init_joints'):
        print(f"移动到初始位置: {args['init_joints']}")
        robot.rm_movej(args['init_joints'], 20, 0, 0, 1)
        time.sleep(2)

    # 初始化相机
    pipelines = init_cameras(
        camera_names,
        args['cam_width'],
        args['cam_height'],
        args['cam_fps']
    )

    # 等待用户确认
    print("\n" + "=" * 80)
    print("准备完成！")
    print("按 'g' + Enter 开始")
    print("=" * 80)

    while input().lower() != 'g':
        print("请输入 'g' + Enter 开始...")

    # 加载模型
    print(f"\n加载模型: {ckpt_dir}")

    policy_config = {
        'lr': 1e-5,
        'num_queries': args['chunk_size'],
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'vq': args.get('use_vq', False),
        'vq_class': args.get('vq_class'),
        'vq_dim': args.get('vq_dim'),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'no_encoder': args.get('no_encoder', False),
    }

    policy = make_policy(policy_class, policy_config)

    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')

    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(f"模型加载状态: {loading_status}")

    policy.to(device)
    policy.eval()

    # 加载归一化统计数据
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 归一化函数
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 创建显示窗口
    for i, cam_name in enumerate(camera_names):
        cv2.namedWindow(f"Camera {i+1}: {cam_name}", cv2.WINDOW_NORMAL)

    # 时序聚合buffer
    if temporal_agg:
        print("启用时序聚合")
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([1, 10000 + num_queries, action_dim]).to(device)

    frame_count = 0
    last_time = time.time()
    is_inferring = False
    num_actions_to_execute = 100  # 每次推理执行的动作数量

    print("\n" + "=" * 80)
    print("控制说明:")
    print("  右方向键 → 开始/停止推理")
    print("  空格键   → 移动到预设位置")
    print("  ESC键    → 退出程序")
    print(f"  每次推理执行前 {num_actions_to_execute} 个动作")
    print("=" * 80)
    print("\n等待环境重置，按右方向键开始推理...\n")

    try:
        with torch.inference_mode():
            while True:
                # 检查退出
                if key_state['quit']:
                    print("\n用户按ESC退出")
                    break

                # 检查空格键 - 移动到预设位置
                if key_state['space']:
                    key_state['space'] = False
                    if args.get('init_joints'):
                        print("移动到预设位置...")
                        robot.rm_movej(args['init_joints'], 20, 0, 0, 1)
                        time.sleep(1)
                        print("已到达预设位置")
                    else:
                        print("未设置预设位置 (--init_joints)")

                # 检查右方向键 - 切换推理状态
                if key_state['right_arrow']:
                    key_state['right_arrow'] = False
                    is_inferring = not is_inferring
                    if is_inferring:
                        print("\n>>> 开始推理 <<<\n")
                        frame_count = 0
                        if temporal_agg:
                            all_time_actions.zero_()
                    else:
                        print("\n>>> 停止推理，等待环境重置... <<<\n")

                loop_start = time.time()

                # 1. 获取相机图像
                images = get_camera_frames(pipelines)

                if images is None:
                    print("获取图像失败，跳过")
                    continue

                # 2. 显示图像
                for i, (img, cam_name) in enumerate(zip(images, camera_names)):
                    cv2.imshow(f"Camera {i+1}: {cam_name}", img)
                cv2.waitKey(1)

                # 如果未在推理状态，跳过推理和执行
                if not is_inferring:
                    time.sleep(0.02)
                    continue

                # 3. 获取当前状态
                qpos_numpy = get_current_state(robot)
                if qpos_numpy is None:
                    continue

                # 4. 归一化
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

                # 5. 图像预处理
                image_tensor = image_preprocess(images,
                                               rand_crop_resize=(policy_class == 'Diffusion'))

                # 6. 模型推理
                t_inference = time.time()
                all_actions = policy(qpos, image_tensor)  # (1, chunk_size, 8)
                inference_time = time.time() - t_inference

                # 7. 执行前N个动作
                num_to_exec = min(num_actions_to_execute, all_actions.shape[1])

                for action_idx in range(num_to_exec):
                    raw_action = all_actions[:, action_idx]

                    # 反归一化
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)

                    # 执行动作
                    if args['execute_actions']:
                        t_action = time.time()
                        execute_action(robot, action, use_canfd=args['use_canfd'])
                        action_time = time.time() - t_action
                        print(f"  Action {action_idx+1}/{num_to_exec}: {action_time*1000:.1f}ms")

                    frame_count += 1

                # 打印信息（每次推理后打印一次）
                loop_time = time.time() - loop_start
                print(f"=== Inference {frame_count//num_to_exec} ===")
                print(f"  Inference: {inference_time*1000:.1f}ms, Total loop: {loop_time*1000:.1f}ms")

                # # 控制循环频率
                # elapsed = time.time() - loop_start
                # target_time = 1.0 / args['control_fps']
                # if elapsed < target_time:
                #     time.sleep(target_time - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")

    finally:
        print("\n清理资源...")

        # 停止相机
        for pipeline in pipelines:
            pipeline.stop()

        cv2.destroyAllWindows()

        # 复位选项
        if args.get('reset_on_exit') and args.get('init_joints'):
            print("复位到初始位置...")
            robot.rm_movej(args['init_joints'], 15, 0, 0, 1)
            time.sleep(2)

        # 释放机械臂
        robot.rm_delete_robot_arm()

        # 停止键盘监听
        listener.stop()

        print("程序结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="单臂+夹爪实时推理部署")

    # 模型参数
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='模型checkpoint目录')
    parser.add_argument('--policy_class', type=str, default='ACT',
                       choices=['ACT', 'CNNMLP', 'Diffusion'],
                       help='策略类型')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Action chunk size')
    parser.add_argument('--kl_weight', type=int, default=10,
                       help='KL weight for ACT')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                       help='Feedforward dimension')
    parser.add_argument('--temporal_agg', action='store_true',
                       help='启用时序聚合')
    parser.add_argument('--use_vq', action='store_true',
                       help='使用VQ')
    parser.add_argument('--vq_class', type=int, default=None)
    parser.add_argument('--vq_dim', type=int, default=None)
    parser.add_argument('--no_encoder', action='store_true')

    # 硬件参数
    parser.add_argument('--robot_ip', type=str, default='192.168.1.18',
                       help='机械臂IP地址')
    parser.add_argument('--robot_port', type=int, default=8080,
                       help='机械臂端口')
    parser.add_argument('--init_joints', type=float, nargs=7,
                       default=[15.022, 17.552, -8.356, -112.476, 2.656, 2.709, -11.533],
                       help='初始关节角度 (7个值)')
    parser.add_argument('--camera_names', type=str, nargs='+',
                       default=['camera_head_image', 'camera_left_image'],
                       help='相机名称列表')
    parser.add_argument('--cam_width', type=int, default=640,
                       help='相机宽度')
    parser.add_argument('--cam_height', type=int, default=480,
                       help='相机高度')
    parser.add_argument('--cam_fps', type=int, default=30,
                       help='相机帧率')

    # 控制参数
    parser.add_argument('--execute_actions', action='store_true', default=True,
                       help='是否执行动作')
    parser.add_argument('--use_canfd', action='store_true', default=False,
                       help='使用CAN FD通信（更快）')
    parser.add_argument('--control_fps', type=int, default=30,
                       help='控制循环频率')
    parser.add_argument('--reset_on_exit', action='store_true',
                       help='退出时复位')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')

    args, unknown = parser.parse_known_args()
    main(vars(args))
