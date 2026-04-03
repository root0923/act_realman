#!/usr/bin/env python3
"""
单臂（7关节）+ 夹爪（1维）数据重播脚本
从HDF5数据文件中读取action序列，在真机上重复采集的轨迹

数据维度:
  - qpos/action: 8维 [7关节角度 + 1夹爪宽度]
  - 角度单位: 度 (degrees)
  - 夹爪范围: 1-1000
"""

import h5py
import numpy as np
import argparse
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入Realman机械臂接口
from Robotic_Arm.rm_robot_interface import *


def init_robot_arm(ip_address, port=8080):
    """初始化Realman机械臂"""
    print(f"连接机械臂: {ip_address}:{port}")

    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm(ip_address, port)

    result = robot.rm_get_current_arm_state()
    if result[0] != 0:
        raise RuntimeError(f"机械臂连接失败，错误码: {result[0]}")

    print("✓ 机械臂连接成功")
    return robot


def get_current_state(robot):
    """获取机械臂当前状态"""
    result = robot.rm_get_current_arm_state()
    if result[0] != 0:
        print(f"获取关节状态失败，错误码: {result[0]}")
        return None

    joint_angles = np.array(result[1]['joint'], dtype=np.float32)

    gripper_result = robot.rm_get_gripper_state()
    if gripper_result[0] != 0:
        gripper_width = 0.0
    else:
        gripper_width = float(gripper_result[1].get('actpos', 0))

    qpos = np.concatenate([joint_angles, [gripper_width]])
    return qpos


def execute_action(robot, action, use_canfd=True):
    """执行动作命令"""
    joint_action = action[:7].tolist()
    # 夹爪值需要在API允许的范围[1, 1000]内
    gripper_action = int(np.clip(action[7], 1, 1000))

    if use_canfd:
        result = robot.rm_movej_canfd(joint_action, False, 0, 1, 60)
    else:
        result = robot.rm_movej(joint_action, 50, 0, 0, 1)

    if result != 0:
        print(f"  警告: 机械臂执行失败，错误码: {result}")

    gripper_result = robot.rm_set_gripper_position(gripper_action, False, 1)
    if gripper_result != 0:
        print(f"  警告: 夹爪执行失败，错误码: {gripper_result}")


def load_episode_data(dataset_path, episode_idx=None):
    """
    加载episode数据 (直接从 .h5 原始文件读取)

    Args:
        dataset_path: 数据集目录或单个文件路径
        episode_idx: episode索引，如果为None则dataset_path是单个文件路径

    Returns:
        actions: (T, 8) 动作序列 [7关节角度 + 1夹爪]
        qpos: (T, 8) 关节位置序列 [7关节角度 + 1夹爪]
    """
    # 判断是目录还是文件
    if os.path.isdir(dataset_path):
        if episode_idx is None:
            raise ValueError("当dataset_path是目录时，必须指定episode_idx")

        # 构建文件名: 0000.h5, 0001.h5, ...
        filename = f'{episode_idx:04d}.h5'
        full_path = os.path.join(dataset_path, filename)
    else:
        full_path = dataset_path

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f'数据文件不存在: {full_path}')

    print(f"加载数据: {full_path}")

    with h5py.File(full_path, 'r') as f:
        # 读取关节角度 (T, 7)
        joint_status = f['observations/left_arm_joint_status'][()]
        # 读取夹爪宽度 (T, 1) -> squeeze 到 (T,)
        gripper_width = f['observations/left_arm_gripper_width'][()]
        if gripper_width.ndim == 2:
            gripper_width = gripper_width.squeeze(axis=-1)

        # 组合 qpos: [7关节角度 + 1夹爪] = 8维
        qpos = np.concatenate([joint_status, gripper_width[..., None]], axis=-1).astype(np.float32)

        # Action = 下一时刻的qpos (teacher forcing)
        actions = np.zeros_like(qpos)
        actions[:-1] = qpos[1:]
        actions[-1] = qpos[-1]

        # 打印数据信息
        episode_len = actions.shape[0]
        print(f"  序列长度: {episode_len}")
        print(f"  qpos/action 维度: {actions.shape}")

        # 打印数据范围
        print(f"\n  关节角度范围 (度):")
        for i in range(7):
            print(f"    关节{i+1}: [{qpos[:, i].min():.2f}, {qpos[:, i].max():.2f}]")

        print(f"  夹爪范围: [{qpos[:, 7].min():.1f}, {qpos[:, 7].max():.1f}]")

    return actions, qpos


def replay_episode(robot, actions, qpos=None, speed_factor=1.0, 
                   execute=True, show_progress=True):
    """
    重播单个episode
    
    Args:
        robot: 机械臂对象
        actions: (T, 8) 动作序列
        qpos: (T, 8) 关节位置序列 (可选，用于显示误差)
        speed_factor: 速度因子 (>1更快, <1更慢)
        execute: 是否真正执行动作 (False则只打印不执行)
        show_progress: 是否显示进度
    """
    episode_len = actions.shape[0]
    print(f"\n开始重播 {episode_len} 个动作")
    print(f"速度因子: {speed_factor}x")
    print(f"执行模式: {'真实执行' if execute else '仅模拟'}")
    print("=" * 80)
    
    # 计算时间间隔 (原始数据采集频率为30Hz)
    base_dt = 1.0 / 30.0
    dt = base_dt / speed_factor
    
    start_time = time.time()
    errors = []
    
    for t in range(episode_len):
        step_start = time.time()
        
        action = actions[t]
        
        # 如果execute=False，只打印不执行
        if execute:
            execute_action(robot, action, use_canfd=True)
        
        # 获取当前实际状态并计算误差
        if qpos is not None:
            current_state = get_current_state(robot)
            if current_state is not None:
                error = np.abs(current_state - qpos[t])
                errors.append(error)
                
                if show_progress and (t % 10 == 0 or t == episode_len - 1):
                    joint_error = error[:7]
                    gripper_error = error[7]
                    print(f"  Step {t+1}/{episode_len} | "
                          f"最大关节误差: {joint_error.max():.2f}° | "
                          f"夹爪误差: {gripper_error:.1f}")
        
        if show_progress and (t % 10 == 0 or t == episode_len - 1):
            action_str = ", ".join([f"{v:.1f}" for v in action])
            print(f"  [{t+1}/{episode_len}] Action: [{action_str}]")
        
        # 控制执行频率
        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"重播完成!")
    print(f"  总用时: {total_time:.2f}s")
    print(f"  实际帧率: {episode_len/total_time:.1f} FPS")
    
    # 打印误差统计
    if len(errors) > 0:
        errors = np.array(errors)
        mean_error = errors.mean(axis=0)
        max_error = errors.max(axis=0)
        
        print(f"\n  误差统计:")
        for i in range(7):
            print(f"    关节{i+1}: 平均 {mean_error[i]:.2f}°, 最大 {max_error[i]:.2f}°")
        print(f"    夹爪: 平均 {mean_error[7]:.1f}, 最大 {max_error[7]:.1f}")


def main(args):
    """主函数"""
    
    # 加载数据
    actions, qpos = load_episode_data(args['dataset_path'], args.get('episode_idx'))
    
    # 初始化机械臂
    robot = init_robot_arm(args['robot_ip'], args['robot_port'])
    
    try:
        # 移动到初始位置 (如果需要)
        if args.get('move_to_start') and qpos is not None:
            # print(f"\n移动到初始位置...")
            # initial_qpos = qpos[0]
            # initial_joints = initial_qpos[:7].tolist()
            
            # print(f"  初始关节: {initial_joints}")
            # result = robot.rm_movej(initial_joints, 20, 0, 0, 1)
            if result != 0:
                print(f"  警告: 移动到初始位置失败，错误码: {result}")
            time.sleep(2)
        
        # 等待用户确认
        if not args.get('skip_confirm'):
            input("\n按 Enter 键开始重播 (Ctrl+C 取消)...")
        
        # 重播数据
        replay_episode(
            robot=robot,
            actions=actions,
            qpos=qpos,
            speed_factor=args.get('speed_factor', 1.0),
            execute=args.get('execute', True),
            show_progress=args.get('verbose', True)
        )
        
        # 结束后是否复位
        if args.get('reset_on_exit') and args.get('init_joints'):
            print(f"\n复位到初始位置...")
            result = robot.rm_movej(args['init_joints'], 15, 0, 0, 1)
            if result != 0:
                print(f"  警告: 复位失败，错误码: {result}")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n用户中断 (Ctrl+C)")
    
    finally:
        print("\n清理资源...")
        robot.rm_delete_robot_arm()
        print("程序结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="单臂+夹爪数据重播脚本")
    
    # 数据参数
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='数据集目录 (.h5 文件所在目录) 或单个 .h5 文件路径')
    parser.add_argument('--episode_idx', type=int, default=None,
                       help='Episode索引 (如 0, 1, 2...)，当dataset_path是目录时需要。文件名格式: 0000.h5, 0001.h5, ...')
    
    # 硬件参数
    parser.add_argument('--robot_ip', type=str, default='192.168.1.18',
                       help='机械臂IP地址')
    parser.add_argument('--robot_port', type=int, default=8080,
                       help='机械臂端口')
    parser.add_argument('--init_joints', type=float, nargs=7,
                       default=[15.022, 17.552, -8.356, -112.476, 2.656, 2.709, -11.533],
                       help='初始关节角度 (用于复位)')
    
    # 控制参数
    parser.add_argument('--speed_factor', type=float, default=1.0,
                       help='速度因子 (>1更快, <1更慢)')
    parser.add_argument('--execute', action='store_true', default=True,
                       help='是否真正执行动作 (默认True)')
    parser.add_argument('--no-execute', action='store_false', dest='execute',
                       help='只打印不执行')
    parser.add_argument('--move_to_start', action='store_true',
                       help='开始前移动到数据初始位置')
    parser.add_argument('--reset_on_exit', action='store_true',
                       help='退出时复位到初始位置')
    parser.add_argument('--skip_confirm', action='store_true',
                       help='跳过确认提示')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细信息')
    
    args = parser.parse_args()
    main(vars(args))
