#!/usr/bin/env python3
"""
转换自定义H5格式到ACT训练格式 (单臂+夹爪)

输入字段:
  - observations/left_arm_joint_status (T, 7) - 7个关节角度 (度)
  - observations/left_arm_gripper_width (T, 1) - 夹爪宽度
  - observations/images/{cam_name} (T,) - 压缩图像

输出字段 (ACT格式):
  - /action (T, 8) - [7关节 + 1夹爪]
  - /observations/qpos (T, 8) - [7关节 + 1夹爪]
  - /observations/images/{cam_name} (T, H, W, 3) - RGB图像 (已从BGR转换)
"""

import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse


def decompress_image(compressed_data):
    """解压JPEG/PNG压缩图像并转换为RGB格式"""
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 返回BGR格式
    if img is None:
        raise ValueError("图像解压失败")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式(ResNet预训练权重需要)
    return img


def convert_episode(input_path, output_path, camera_names=None):
    """
    转换单个episode文件

    Args:
        input_path: 输入h5文件路径
        output_path: 输出h5文件路径
        camera_names: 要使用的相机列表，默认使用所有可用相机
    """
    if camera_names is None:
        camera_names = ['camera_head_image', 'camera_left_image']

    with h5py.File(input_path, 'r') as f_in:
        # 读取机械臂关节数据
        if 'observations/left_arm_joint_status' not in f_in:
            print(f"  错误: 缺少 left_arm_joint_status")
            return False

        joint_status = f_in['observations/left_arm_joint_status'][()]  # (T, 7)

        if joint_status.shape[0] == 0:
            print(f"  跳过: left_arm_joint_status 为空")
            return False

        # 读取夹爪数据
        if 'observations/left_arm_gripper_width' not in f_in:
            print(f"  错误: 缺少 left_arm_gripper_width")
            return False

        gripper_width = f_in['observations/left_arm_gripper_width'][()]  # (T, 1)

        # 确保长度一致
        episode_len = joint_status.shape[0]
        gripper_width = gripper_width[:episode_len]

        # 组合 qpos: [7关节角度 + 1夹爪] = 8维
        # 注意：ACT期望弧度制，但Realman使用角度制，这里保持角度制，后续训练时会归一化
        qpos = np.concatenate([joint_status, gripper_width], axis=-1).astype(np.float32)

        # Action = 下一时刻的qpos (teacher forcing)
        action = np.zeros_like(qpos)
        action[:-1] = qpos[1:]  # 向前shift 1步
        action[-1] = qpos[-1]   # 最后一帧保持不变

        # 读取并解压图像
        images = {}
        available_cams = []

        for cam_name in camera_names:
            cam_path = f'observations/images/{cam_name}'
            if cam_path not in f_in:
                print(f"  警告: 相机 {cam_name} 不存在，跳过")
                continue

            compressed_imgs = f_in[cam_path]

            # 检查是否是压缩格式
            if compressed_imgs.dtype == object:
                decompressed = []
                for i in range(episode_len):
                    try:
                        img = decompress_image(compressed_imgs[i])
                        decompressed.append(img)
                    except Exception as e:
                        print(f"  错误: 解压图像 {cam_name}[{i}] 失败: {e}")
                        return False

                images[cam_name] = np.array(decompressed, dtype=np.uint8)  # (T, H, W, 3)
            else:
                # 已经是解压格式
                images[cam_name] = compressed_imgs[:episode_len]

            available_cams.append(cam_name)

        if len(available_cams) == 0:
            print(f"  错误: 没有可用的相机数据")
            return False

    # 写入ACT格式文件
    with h5py.File(output_path, 'w') as f_out:
        # 核心数据
        f_out.create_dataset('/action', data=action, dtype=np.float32)
        f_out.create_dataset('/observations/qpos', data=qpos, dtype=np.float32)

        # 图像数据
        for cam_name, img_data in images.items():
            f_out.create_dataset(f'/observations/images/{cam_name}',
                               data=img_data, dtype=np.uint8)

        # 元数据
        f_out.attrs['sim'] = False
        f_out.attrs['compress'] = False

    print(f"  ✓ 成功: {episode_len} 帧, {len(available_cams)} 相机, qpos/action shape: {qpos.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description='转换数据到ACT格式')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出数据目录')
    parser.add_argument('--cameras', type=str, nargs='+',
                       default=['camera_head_image', 'camera_left_image'],
                       help='使用的相机名称列表')
    parser.add_argument('--pattern', type=str, default='*.h5',
                       help='文件匹配模式')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有h5文件
    import glob
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    if len(input_files) == 0:
        print(f"错误: 在 {args.input_dir} 中没有找到 {args.pattern} 文件")
        return

    print(f"找到 {len(input_files)} 个文件")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用相机: {args.cameras}")
    print("=" * 80)

    success_count = 0
    failed_files = []

    for input_path in tqdm(input_files, desc="转换进度"):
        filename = os.path.basename(input_path)
        # 生成输出文件名: 0001.h5 -> episode_1.hdf5
        if filename.startswith('0'):
            episode_num = int(filename.split('.')[0])
            output_filename = f'episode_{episode_num}.hdf5'
        else:
            output_filename = filename.replace('.h5', '.hdf5')

        output_path = os.path.join(args.output_dir, output_filename)

        print(f"\n{filename} -> {output_filename}")

        if convert_episode(input_path, output_path, args.cameras):
            success_count += 1
        else:
            failed_files.append(filename)

    print("\n" + "=" * 80)
    print(f"转换完成: {success_count}/{len(input_files)} 个文件成功")

    if failed_files:
        print(f"\n失败的文件 ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")

    print(f"\n输出目录: {args.output_dir}")
    print("\n下一步:")
    print(f"  1. 检查转换后的数据: python visualize_episodes.py --dataset_dir {args.output_dir}")
    print(f"  2. 开始训练: bash single_arm_pipeline/train.sh")


if __name__ == "__main__":
    main()
