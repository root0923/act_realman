import os
import h5py
import cv2
import numpy as np
import argparse

def get_fps_from_hdf5(hdf5_path):
    """从HDF5文件计算实际FPS"""
    with h5py.File(hdf5_path, 'r') as f:
        if 'timestamps' in f:
            timestamps = f['timestamps'][()]
            if len(timestamps) > 1:
                dt = np.mean(np.diff(timestamps))
                return 1.0 / dt
        # 如果没有时间戳,尝试从属性获取
        if 'fps' in f.attrs:
            return f.attrs['fps']
    return None

def visualize_episode(hdf5_path, output_dir, default_fps=30):
    """将单个HDF5文件的图像转换为视频"""
    with h5py.File(hdf5_path, 'r') as f:
        cam_names = sorted(list(f['observations/images'].keys()))
        first_frame = f[f'observations/images/{cam_names[0]}'][0]
        h, w = first_frame.shape[:2]
        total_w = w * len(cam_names)
        n_frames = f[f'observations/images/{cam_names[0]}'].shape[0]

        fps = get_fps_from_hdf5(hdf5_path) or default_fps
        duration = n_frames / fps

        episode_name = os.path.basename(hdf5_path).replace('.hdf5', '')
        video_path = os.path.join(output_dir, f'{episode_name}.mp4')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (total_w, h))

        for i in range(n_frames):
            frames = []
            for cam_name in cam_names:
                frame = f[f'observations/images/{cam_name}'][i]
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
            out.write(np.concatenate(frames, axis=1))

        out.release()
        print(f'{episode_name}: {n_frames}帧, {duration:.2f}秒, {fps:.1f}FPS -> {video_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fps', type=float, default=30, help='默认FPS(如果文件中没有时间戳)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    hdf5_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.hdf5')])
    print(f'找到 {len(hdf5_files)} 个episode\n')

    for hdf5_file in hdf5_files:
        visualize_episode(os.path.join(args.data_dir, hdf5_file), output_dir, args.fps)

if __name__ == '__main__':
    main()
