import h5py
import os

dataset_dir = '/home/ysy/data/teleop_data/rotate_hdf5'
files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])[:1]

for file in files:
    file_path = os.path.join(dataset_dir, file)
    print(f"文件: {file}")

    with h5py.File(file_path, 'r') as f:
        if 'observations' in f and 'images' in f['observations']:
            images_group = f['observations']['images']
            print("\n图像尺寸:")
            for cam_name in images_group.keys():
                shape = images_group[cam_name].shape
                print(f"  {cam_name}: {shape}")
