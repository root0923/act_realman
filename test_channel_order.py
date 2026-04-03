#!/usr/bin/env python3
"""
测试HDF5数据的通道顺序
"""
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def test_channel_order(hdf5_path, output_path='channel_test.png'):
    """
    测试并可视化图像通道顺序

    显示三种情况：
    1. 原始数据（假设是RGB）
    2. 原始数据（假设是BGR）
    3. 通道转换后的数据
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 获取第一个相机的第一帧
        cam_names = list(f['observations/images'].keys())
        print(f"可用相机: {cam_names}")

        cam_name = cam_names[0]
        img = f[f'observations/images/{cam_name}'][0]
        print(f"图像形状: {img.shape}, dtype: {img.dtype}")

        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：原始数据的不同解释
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('原始数据 (假设RGB)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img[:, :, [2, 1, 0]])
        axes[0, 1].set_title('原始数据 BGR→RGB转换')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('cv2.cvtColor(BGR→RGB)')
        axes[0, 2].axis('off')

        # 第二行：各通道分布
        axes[1, 0].hist(img[:, :, 0].ravel(), bins=50, color='red', alpha=0.5, label='通道0')
        axes[1, 0].set_title('通道0分布')
        axes[1, 0].legend()

        axes[1, 1].hist(img[:, :, 1].ravel(), bins=50, color='green', alpha=0.5, label='通道1')
        axes[1, 1].set_title('通道1分布')
        axes[1, 1].legend()

        axes[1, 2].hist(img[:, :, 2].ravel(), bins=50, color='blue', alpha=0.5, label='通道2')
        axes[1, 2].set_title('通道2分布')
        axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n对比图已保存到: {output_path}")
        print("\n判断方法:")
        print("1. 如果'原始数据(假设RGB)'看起来颜色正常 → 数据是RGB格式")
        print("2. 如果'原始数据 BGR→RGB转换'看起来颜色正常 → 数据是BGR格式")
        print("3. 观察图像中明显的红色或蓝色物体来判断")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试HDF5数据的通道顺序')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='HDF5文件路径')
    parser.add_argument('--output', type=str, default='channel_test.png',
                       help='输出图像路径')

    args = parser.parse_args()
    test_channel_order(args.hdf5_path, args.output)
