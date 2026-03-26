import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_hand_joints(h5_path):
    with h5py.File(h5_path, 'r') as f:
        # 读取 action 和 qpos
        action = f['action'][:]
        qpos = f['observations/qpos'][:]
        # 只取后6维
        action_hand = action[:, 6:12]
        qpos_hand = qpos[:, 6:12]
        steps = np.arange(action_hand.shape[0])

        # 打印前10行数据
        print("action 后6维（前10行）：")
        print(action_hand[:200])
        print("\nqpos 后6维（前10行）：")
        print(qpos_hand[:200])

        plt.figure(figsize=(12, 5))
        for i in range(6):
            plt.plot(steps, action_hand[:, i], label=f'action_hand_{i+1}')
        plt.title('Action 后6维（灵巧手）')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        for i in range(6):
            plt.plot(steps, qpos_hand[:, i], label=f'qpos_hand_{i+1}')
        plt.title('Qpos 后6维（灵巧手）')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    h5_path = "/home/shuxiangzhang/act_realman/data_12_grasp/data/real_test01/episode_1.hdf5"
    plot_hand_joints(h5_path)