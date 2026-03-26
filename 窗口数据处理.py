import h5py
import numpy as np
import glob
import os
import shutil

# 输入输出目录
rdir = 'data_lap_hand/data/real_test01'
wdir = 'data_lap_hand/data/real_test01p'
os.makedirs(wdir, exist_ok=True)
fname = glob.glob(rdir + '/*.hdf5')

# 定义加权滑动平均函数
def weighted_moving_average(data, window=8, weights=None):
    if weights is None:
        weights = np.ones(window) / window
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    pad = window // 2
    padded = np.pad(data, ((pad, pad), (0, 0)), mode='edge')
    smoothed = np.zeros_like(data)
    for t in range(data.shape[0]):
        window_data = padded[t:t+window]
        smoothed[t] = np.sum(window_data * weights[:, None], axis=0)
    return smoothed

# 你可以自定义权重，比如最近的帧权重大
weights = [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]

for fpath in fname:
    # 先复制原文件到新目录
    wname = os.path.join(wdir, os.path.basename(fpath))
    shutil.copy(fpath, wname)
    # 再只修改新文件中的 action 和 observations/qpos
    with h5py.File(wname, 'r+') as f:
        action = f['action'][:]
        qpos = f['observations/qpos'][:]
        action_smoothed = action.copy()
        qpos_smoothed = qpos.copy()
        action_smoothed[:, 6:12] = weighted_moving_average(action[:, 6:12], window=8, weights=weights)
        qpos_smoothed[:, 6:12] = weighted_moving_average(qpos[:, 6:12], window=8, weights=weights)
        del f['action']
        del f['observations/qpos']
        f.create_dataset('action', data=action_smoothed)
        f['observations'].create_dataset('qpos', data=qpos_smoothed)
    print(f"新文件已生成: {wname}")