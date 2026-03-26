import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import argparse

def print_joint_angles(csv_file_path, plot=True, save_fig=False, output_dir=None):
    """
    解析并打印CSV文件中的关节角数据（12关节版，分别绘制前六和后六关节）
    """
    try:
        # 读取CSV文件，不使用表头
        df = pd.read_csv(csv_file_path, header=None)
        # 为列分配名称
        df.columns = ['master_joints', 'slave_joints', 'timestamp', 'index']
        # 数据预处理 - 将字符串转换为Python列表
        df['master_joints'] = df['master_joints'].apply(ast.literal_eval)
        df['slave_joints'] = df['slave_joints'].apply(ast.literal_eval)
        df['index'] = df['index'].apply(ast.literal_eval).apply(lambda x: x[0])

        # 12关节名
        joint_names = [f'joint{i+1}' for i in range(12)]

        # 为主臂创建单独的列
        for i, name in enumerate(joint_names):
            df[f'master_{name}'] = df['master_joints'].apply(lambda x: x[i])
        # 为从臂创建单独的列
        for i, name in enumerate(joint_names):
            df[f'slave_{name}'] = df['slave_joints'].apply(lambda x: x[i])

        # 打印数据统计信息
        print(f"\n数据总帧数: {len(df)}")
        print("\n==== 主臂关节角度统计 ====")
        for i, name in enumerate(joint_names):
            col = f'master_{name}'
            print(f"{name}: 最小值 = {df[col].min():.2f}, 最大值 = {df[col].max():.2f}, 平均值 = {df[col].mean():.2f}, 标准差 = {df[col].std():.2f}")

        print("\n==== 从臂关节角度统计 ====")
        for i, name in enumerate(joint_names):
            col = f'slave_{name}'
            print(f"{name}: 最小值 = {df[col].min():.2f}, 最大值 = {df[col].max():.2f}, 平均值 = {df[col].mean():.2f}, 标准差 = {df[col].std():.2f}")

        if plot:
            if save_fig and output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 主臂前六关节
            plt.figure(figsize=(12, 8))
            for i in range(6):
                plt.subplot(3, 2, i+1)
                plt.plot(df.index, df[f'master_joint{i+1}'])
                plt.title(f'主臂 joint{i+1}')
                plt.ylabel('角度')
                plt.grid(True)
            plt.tight_layout()
            if save_fig and output_dir:
                plt.savefig(os.path.join(output_dir, 'master_joints_1-6.png'))
            plt.show()

            # 主臂后六关节
            plt.figure(figsize=(12, 8))
            for i in range(6, 12):
                plt.subplot(3, 2, i-5)
                plt.plot(df.index, df[f'master_joint{i+1}'])
                plt.title(f'主臂 joint{i+1}')
                plt.ylabel('角度')
                plt.grid(True)
            plt.tight_layout()
            if save_fig and output_dir:
                plt.savefig(os.path.join(output_dir, 'master_joints_7-12.png'))
            plt.show()

            # 从臂前六关节
            plt.figure(figsize=(12, 8))
            for i in range(6):
                plt.subplot(3, 2, i+1)
                plt.plot(df.index, df[f'slave_joint{i+1}'])
                plt.title(f'从臂 joint{i+1}')
                plt.ylabel('角度')
                plt.grid(True)
            plt.tight_layout()
            if save_fig and output_dir:
                plt.savefig(os.path.join(output_dir, 'slave_joints_1-6.png'))
            plt.show()

            # 从臂后六关节
            plt.figure(figsize=(12, 8))
            for i in range(6, 12):
                plt.subplot(3, 2, i-5)
                plt.plot(df.index, df[f'slave_joint{i+1}'])
                plt.title(f'从臂 joint{i+1}')
                plt.ylabel('角度')
                plt.grid(True)
            plt.tight_layout()
            if save_fig and output_dir:
                plt.savefig(os.path.join(output_dir, 'slave_joints_7-12.png'))
            plt.show()

            # 主从臂对比图（每个关节一张图）
            for i, name in enumerate(joint_names):
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df[f'master_{name}'], label='主臂')
                plt.plot(df.index, df[f'slave_{name}'], label='从臂')
                plt.title(f'{name} - 主从臂对比')
                plt.ylabel('角度')
                plt.xlabel('数据点')
                plt.legend()
                plt.grid(True)
                if save_fig and output_dir:
                    plt.savefig(os.path.join(output_dir, f'compare_{name}.png'))
                plt.show()

        return df
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='分析并可视化机器人关节角度数据（12关节版）')
    parser.add_argument('csv_file', help='CSV文件路径')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='不显示图表')
    parser.add_argument('--save', action='store_true', help='保存图表')
    parser.add_argument('--output', default='./joint_plots', help='图表保存目录')
    args = parser.parse_args()

    print(f"正在分析文件: {args.csv_file}")
    print_joint_angles(args.csv_file, plot=args.plot, save_fig=args.save, output_dir=args.output)

if __name__ == "__main__":
    main()