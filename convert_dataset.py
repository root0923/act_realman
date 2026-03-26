# #!/usr/bin/env python3
#!/usr/bin/env python3
import os
import h5py
import numpy as np

def make_chunked_dataset(dst_file, name, data):
    """把原始 numpy 数据写成 chunked dataset"""
    dst_file.create_dataset(name, data=data, chunks=True, compression="gzip")

def convert_h5_file(src_path, dst_path):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # 读取原始数据
        qpos = src["observations/qpos"][:]           # (T, D)
        timestamps = src["observations/timestamps"][:]  # (T,)
        cam01 = src["observations/images/cam01"][:]     # (T, H, W, C)

        # 构造新的 action = 下一帧的 qpos
        actions = qpos[1:]          # (T-1, D)
        qpos = qpos[:-1]            # (T-1, D)
        timestamps = timestamps[:-1]  # (T-1,)
        cam01 = cam01[:-1]            # (T-1, H, W, C)

        # 保存 action
        make_chunked_dataset(dst, "action", actions)

        # 保存 observations
        obs_grp = dst.create_group("observations")
        make_chunked_dataset(obs_grp, "qpos", qpos)
        make_chunked_dataset(obs_grp, "timestamps", timestamps)

        img_grp = obs_grp.create_group("images")
        make_chunked_dataset(img_grp, "cam01", cam01)

    print(f"✅ Converted: {src_path} -> {dst_path}")

def convert_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith(".h5") or fname.endswith(".hdf5"):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            convert_h5_file(src_path, dst_path)

if __name__ == "__main__":
    src_dir = "raw_h5"                 # 原始 HDF5
    dst_dir = "data/real_episodes_prepared"   # ACT++ 数据输出目录
    convert_dataset(src_dir, dst_dir)

# import os
# import h5py
# import numpy as np

# def make_chunked_dataset(dst_file, name, data):
#     """把原始 numpy 数据写成 chunked dataset"""
#     dst_file.create_dataset(name, data=data, chunks=True, compression="gzip")

# def convert_h5_file(src_path, dst_path):
#     with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
#         # action
#         make_chunked_dataset(dst, "action", src["action"][:])
#         # observations
#         obs_grp = dst.create_group("observations")
#         make_chunked_dataset(obs_grp, "qpos", src["observations/qpos"][:])
#         make_chunked_dataset(obs_grp, "timestamps", src["observations/timestamps"][:])
#         img_grp = obs_grp.create_group("images")
#         make_chunked_dataset(img_grp, "cam01", src["observations/images/cam01"][:])
#     print(f"✅ Converted: {src_path} -> {dst_path}")

# def convert_dataset(src_dir, dst_dir):
#     os.makedirs(dst_dir, exist_ok=True)
#     for fname in os.listdir(src_dir):
#         if fname.endswith(".h5") or fname.endswith(".hdf5"):
#             src_path = os.path.join(src_dir, fname)
#             dst_path = os.path.join(dst_dir, fname)
#             convert_h5_file(src_path, dst_path)

# if __name__ == "__main__":
#     src_dir = "raw_h5"                 # 原始 HDF5
#     dst_dir = "data/real_episodes_prepared"   # ACT++ 数据输出目录
#     convert_dataset(src_dir, dst_dir)
