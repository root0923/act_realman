import os
import h5py
import numpy as np

def quantize_hand_joints(arr):
    # arr: shape (N, 12) 或 (N, 8) 或 (N, 6+hand_dim)
    arr = np.array(arr)
    if arr.shape[1] < 12:
        print(f"警告：数据维度为{arr.shape[1]}，不是12维，跳过。")
        return arr
    arr_new = arr.copy()
    # 只处理第7~11个关节（Python索引6~10），第12个关节（索引11）不处理
    arr_new[:, 6:11] = np.round(arr[:, 6:11] / 655).astype(int) * 655
    return arr_new

def process_h5_file(h5_path):
    with h5py.File(h5_path, 'r+') as f:
        # 处理 action
        if 'action' in f:
            action = f['action'][:]
            action_new = quantize_hand_joints(action)
            del f['action']
            f.create_dataset('action', data=action_new)
            print(f"{h5_path}: action已处理")
        # 处理 observations/qpos
        if 'observations' in f and 'qpos' in f['observations']:
            qpos = f['observations/qpos'][:]
            qpos_new = quantize_hand_joints(qpos)
            del f['observations/qpos']
            f['observations'].create_dataset('qpos', data=qpos_new)
            print(f"{h5_path}: qpos已处理")

if __name__ == "__main__":
    data_dir = "/home/shuxiangzhang/act_realman/data_12_grasp/data/real_test01"
    for fname in os.listdir(data_dir):
        if fname.endswith('.hdf5') or fname.endswith('.h5'):
            h5_path = os.path.join(data_dir, fname)
            process_h5_file(h5_path)