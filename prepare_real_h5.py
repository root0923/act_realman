# prepare_real_h5.py
import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm

src_dir = "raw_h5"
dst_dir = "data/real_episodes_prepared"
os.makedirs(dst_dir, exist_ok=True)

TARGET_H, TARGET_W = 256, 256  # 改成 act-plus-plus 默认的输入尺寸

for fn in tqdm(sorted(os.listdir(src_dir))):
    if not fn.endswith(".h5"):
        continue

    src_path = os.path.join(src_dir, fn)
    dst_path = os.path.join(dst_dir, fn)

    with h5py.File(src_path, "r") as fr, h5py.File(dst_path, "w") as fw:
        # 复制 action, qpos, timestamps
        fw.create_dataset("action", data=fr["action"][:])
        fw.create_dataset("observations/qpos", data=fr["observations/qpos"][:])
        fw.create_dataset("observations/timestamps", data=fr["observations/timestamps"][:])

        # 处理图像 cam01 -> cam0
        images = fr["observations/images/cam01"][:]  # [N,H,W,3]
        N = images.shape[0]
        resized = np.zeros((N, TARGET_H, TARGET_W, 3), dtype=np.uint8)

        for i in range(N):
            img = images[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (TARGET_W, TARGET_H))
            resized[i] = img

        fw.create_dataset("observations/images/cam0", data=resized, compression="gzip")

print("✅ 转换完成，已保存到", dst_dir)
