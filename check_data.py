#!/usr/bin/env python3
"""
验证HDF5数据格式是否符合要求
"""
import h5py
import numpy as np
import sys

def check_hdf5_format(file_path):
    """检查HDF5文件格式"""
    print(f"检查文件: {file_path}")
    print("=" * 60)

    try:
        with h5py.File(file_path, 'r') as f:
            # 检查必需的数据集
            required_keys = {
                '/action': None,
                '/observations/qpos': None,
            }

            # 检查action
            if 'action' in f:
                action = f['action'][:]
                print(f"✓ /action 存在")
                print(f"  - Shape: {action.shape}")
                print(f"  - Dtype: {action.dtype}")
                print(f"  - 第一帧: {action[0]}")
                required_keys['/action'] = action.shape
            else:
                print("✗ /action 缺失！")
                return False

            # 检查qpos
            if 'observations' in f and 'qpos' in f['observations']:
                qpos = f['observations/qpos'][:]
                print(f"✓ /observations/qpos 存在")
                print(f"  - Shape: {qpos.shape}")
                print(f"  - Dtype: {qpos.dtype}")
                print(f"  - 第一帧: {qpos[0]}")
                required_keys['/observations/qpos'] = qpos.shape
            else:
                print("✗ /observations/qpos 缺失！")
                return False

            # 检查时间步是否一致
            if action.shape[0] != qpos.shape[0]:
                print(f"✗ 时间步不一致: action={action.shape[0]}, qpos={qpos.shape[0]}")
                return False
            else:
                print(f"✓ 时间步一致: T={action.shape[0]}")

            # 检查维度
            if action.shape[1] != qpos.shape[1]:
                print(f"⚠ 动作维度和状态维度不同: action_dim={action.shape[1]}, state_dim={qpos.shape[1]}")
            else:
                print(f"✓ 动作/状态维度: {action.shape[1]}")

            # 检查相机图像
            if 'observations' in f and 'images' in f['observations']:
                images_grp = f['observations/images']
                camera_names = list(images_grp.keys())
                print(f"✓ 相机数量: {len(camera_names)}")
                for cam_name in camera_names:
                    img = images_grp[cam_name]
                    print(f"  - {cam_name}: Shape={img.shape}, Dtype={img.dtype}")

                    # 检查图像时间步
                    if img.shape[0] != action.shape[0]:
                        print(f"    ✗ 时间步不匹配: {img.shape[0]} != {action.shape[0]}")
                        return False
            else:
                print("✗ /observations/images 缺失！")
                return False

            print("\n" + "=" * 60)
            print("✓ 数据格式验证通过！")
            return True

    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_data.py <hdf5文件路径>")
        print("示例: python check_data.py data/episode_0.hdf5")
        sys.exit(1)

    file_path = sys.argv[1]
    check_hdf5_format(file_path)
