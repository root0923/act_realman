#!/usr/bin/env python3
import h5py
import numpy as np

filepath = "/home/ysy/data/aloha/0001.h5"

with h5py.File(filepath, 'r') as f:
    print("=" * 80)
    print("关键数据字段:")
    print("=" * 80)

    # 左臂关节
    if 'observations/left_arm_joint_status' in f:
        data = f['observations/left_arm_joint_status'][()]
        print(f"\nleft_arm_joint_status: shape={data.shape}, dtype={data.dtype}")
        print(f"  范围: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  示例: {data[0]}")

    # 左臂夹爪
    if 'observations/left_arm_gripper_width' in f:
        data = f['observations/left_arm_gripper_width'][()]
        print(f"\nleft_arm_gripper_width: shape={data.shape}, dtype={data.dtype}")
        print(f"  范围: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  示例: {data[0]}")

    # 图像
    print(f"\n图像数据:")
    for cam in ['camera_chest_image', 'camera_left_image', 'camera_right_image']:
        if f'observations/images/{cam}' in f:
            img = f[f'observations/images/{cam}']
            print(f"  {cam}: shape={img.shape}, dtype={img.dtype}")
            # 检查是否压缩
            if img.dtype == object:
                sample = img[0]
                print(f"    -> 压缩格式, 字节数: {len(sample)}")

    # 时间戳
    if 'observations/timestamp' in f:
        ts = f['observations/timestamp'][()]
        print(f"\ntimestamp: shape={ts.shape}")
        if len(ts) > 1:
            fps = 1.0 / np.mean(np.diff(ts.flatten()))
            print(f"  平均帧率: {fps:.1f} Hz")
