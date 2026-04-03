import h5py
import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt

def replay_episode(h5_path, wait_ms=33, save_video=True):
    with h5py.File(h5_path, 'r') as f:
        cam01 = f['observations/images/camera_head_image'][:]
        cam02 = f['observations/images/camera_left_image'][:]
        actions = f['action'][:]
        qpos = f['observations/qpos'][:]
        n = cam01.shape[0]
        assert cam01.shape[0] == cam02.shape[0] == actions.shape[0] == qpos.shape[0], "帧数不一致"

        # 视频保存设置
        if save_video:
            out_path = h5_path.replace('.hdf5', '_replay.avi')
            h, w, c = cam01[0].shape
            video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w*2, h)
            )
            print(f"保存视频到: {out_path}")

        # 实时绘制关节角度曲线
        plt.ion()
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        action_lines_1 = [ax[0,0].plot([], [], label=f'Action {i+1}')[0] for i in range(6)]
        qpos_lines_1 = [ax[1,0].plot([], [], label=f'Qpos {i+1}')[0] for i in range(6)]
        ax[0,0].set_title('Action 前六关节')
        ax[1,0].set_title('Qpos 前六关节')
        for a in ax.flatten():
            a.set_xlim(0, n)
            a.set_ylim(np.min(qpos)-5, np.max(qpos)+5)
            a.legend()
            a.grid(True)
        ax[0,0].set_ylim(np.min(actions[:, :6])-5, np.max(actions[:, :6])+5)
        ax[1,0].set_ylim(np.min(qpos[:, :6])-5, np.max(qpos[:, :6])+5)
        action_hist = np.zeros((0, actions.shape[1]))
        qpos_hist = np.zeros((0, qpos.shape[1]))

        for i in range(n):
            img1 = cam01[i]
            img2 = cam02[i]
            # 转换RGB到BGR (cv2.imshow和VideoWriter需要BGR格式)
            img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img = np.concatenate([img1_bgr, img2_bgr], axis=1)
            txt1 = f"Action: " + np.array2string(actions[i], precision=2, separator=',')
            txt2 = f"Qpos:   " + np.array2string(qpos[i], precision=2, separator=',')
            cv2.putText(img, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Replay (cam01 | cam02)', img)

            # 保存视频帧
            if save_video:
                video_writer.write(img)

            # 更新曲线
            action_hist = np.vstack([action_hist, actions[i:i+1]])
            qpos_hist = np.vstack([qpos_hist, qpos[i:i+1]])
            x = np.arange(action_hist.shape[0])
            for j, line in enumerate(action_lines_1):
                line.set_data(x, action_hist[:, j])
            for j, line in enumerate(qpos_lines_1):
                line.set_data(x, qpos_hist[:, j])
            for a in ax.flatten():
                a.set_xlim(0, n)
            plt.pause(0.001)

            key = cv2.waitKey(wait_ms)
            if key == 27:
                break
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
        if save_video:
            video_writer.release()
            print(f"视频已保存到: {out_path}")

if __name__ == "__main__":
    files = sorted(glob.glob('/home/ysy/data/teleop_data/20260402_1_hdf5/*.hdf5'))
    print("可用文件：")
    for idx, f in enumerate(files):
        print(f"[{idx}] {f}")
    sel = input("输入要重播的文件编号（回车默认0）：")
    sel = int(sel) if sel.strip() else 0
    replay_episode(files[sel])