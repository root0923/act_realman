import time
import os
import numpy as np
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import cv2
import datetime
from tqdm import tqdm
import csv
import h5py
# 假设这里导入读取额外关节的函数1
# from extra_joints import get_extra_joints_angles

import threading
import socket
import json
import copy
import logging
import multiprocessing
from multiprocessing import Process, Manager, freeze_support
#from gui import gui  
#from gui_key import HandGUI
from usb_glove import OGlove
MASTER_IP = "192.168.110.119"
SLAVE_IP = "192.168.110.118" # 改成从臂的IP地址
LOCAL_IP = "192.168.110.164" # 改成自己的IP地址
PORT=8085
class Listener():
    def __init__(self, local_ip, remote_ip, port=8085, recv_time=2, max_retries=10):
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.port = port
        self.orig_port = port  # 保存原始端口号
        self.socket = None
        self.config_setting = None
        self.config = None
        self.atom_data = None
        self.recv_time = recv_time
        self.is_close = False  # 用于控制接收线程的运行状态
        # 尝试绑定端口，如果失败则尝试其他端口
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # 设置 SO_REUSEADDR 选项
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((self.local_ip, self.port))
                print(f"UDP Listener started on {self.local_ip}:{self.port}")
                break
            except OSError as e:
                retry_count += 1
                print(f"端口 {self.port} 被占用，尝试端口 {self.port + 1}...")
                self.port += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"无法绑定UDP端口，已尝试 {max_retries} 次")

    def start(self,arm,config=None):
        # 根据config参数设置实时推送配置
        self.is_close = False
        if config is None:
            self.config = rm_udp_custom_config_t()
        config_setting = rm_realtime_push_config_t(self.recv_time, True, self.port, 0, self.local_ip, self.config)
        print(arm.rm_set_realtime_push(config_setting))

    def receive_data(self):
        print("开始接收数据...")
        while not self.is_close:
            try:
                data, addr = self.socket.recvfrom(4096)
                try:
                    # 尝试将十六进制解码为字符串
                    text = data.decode("utf-8")
                    # 尝试将其转为 JSON
                    json_data = json.loads(text)
                    with threading.Lock():
                        self.atom_data=json_data
                except Exception as e:
                    logging.error("❌ JSON 解码失败:", e)
            except KeyboardInterrupt:
                pass
    
    def get_data(self):
        with threading.Lock():
            data=copy.deepcopy(self.atom_data)        
        return data
    def close(self):
        if self.config is not None:
            # 关闭实时推送
            pass
            # 在这里关闭会导[36.43, -3.88, 82.67, 5.19, 77.72, -3.09]etting = rm_realtime_push_config_t(self.recv_time, False, self.port, 0, LOCAL_IP, self.config)
            # self.arm.rm_set_realtime_push(config_setting)
        self.is_close = True
        time.sleep(0.5)
        self.socket.close()

def listener(shared_array, event,now_pos):
    """监听共享数据变化的进程函数"""

    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_handle = robot.rm_create_robot_arm(SLAVE_IP, 8080)
    # 获得初始时间戳
    print("手部控制进程启动，等待数据更新...")
    while True:
        if event.wait(1):  # 等待事件触发（最多0.1秒）
            # 获取共享数组的当前值
            values = list(shared_array[:])
            event.clear()  # 重置事件
            # 更新机器人手部位置            
            hand_pos = [int(value) for value in values]
            robot.rm_set_hand_follow_pos(hand_pos, 0)

def get_extra_joints_angles(listener_instance):
    """
    获取额外两个关节的角度。
    假设从臂的额外关节数据可以通过监听实例获取。
    """
    data = listener_instance.get_data()
    return data['hand']['hand_pos'][:]        

def collect_process():
    master_trajectory = []
    slave_trajectory = []
    save_images1 = []
    save_images2 = []
    image_time = []
    #init_pose = [36.43, -3.88, 82.67, 5.19, 77.72, -3.09]  # 主臂从臂的初始位姿
    #init_pose = [0, 0, 0, 0, 0, 0]  # 主臂从臂的初始位姿
    init_pose = [-18, 18, -90, -33, -75, -3]  # 主臂从臂的初始位姿
    original_pose = [0, 0, 0, 0, 0, 0]  # 主臂从臂的原始位姿
    # 初始化主臂
    try:
        master_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        slave_arm = RoboticArm()
        handle1 = master_arm.rm_create_robot_arm(MASTER_IP, 8080)
        handle2 = slave_arm.rm_create_robot_arm(SLAVE_IP, 8080)
    except Exception as e:
        print(f"创建主臂从臂失败: {e}")
        return
    print("创建主臂从臂成功", handle1.id, handle2.id)

    #--------------------------------------------------------------------------------------------------
    ### 启动手控制类
    #freeze_support()  # 必需用于Windows下的多进程支持
    multiprocessing.set_start_method("spawn", force=True)
    
    # 创建Manager对象和共享资源
    mgr = Manager()
    shared_values = mgr.Array('i', [0]*6)  # 初始化为6个0
    now_pos = mgr.Array('i', [0]*6)  # 用于存储当前手部位置的共享数组
    update_event = mgr.Event()  # 用于通知数据更新的事件
    
    # 启动监听并传递共享对象，监听gui的变化，用于控制手部位置
    p = Process(target=listener, args=(shared_values, update_event,now_pos))
    p.daemon = True  # 设置为守护进程，主进程结束时自动终止
    p.start()

    # 创建GUI实例并传递共享对象
    #mygui = HandGUI(shared_values, update_event,now_pos)
    mygui = OGlove(shared_values, update_event, now_pos, serial=None, timeout=2000)
    pgui = Process(target=mygui.launch_gui, args=())
    pgui.daemon = True  # 设置为守护进程，主进程结束时自动终止
    pgui.start()
    
    
    ### listerner的启动，需要借助一个手部控制实例
    # 手初始化，用于启动手部传输
    original_pose = [0, 0, 0, 0, 0, 0]  # 手的原始位姿
    slave_arm.rm_set_hand_follow_pos(original_pose, 0)
    listener_instance = Listener(LOCAL_IP, SLAVE_IP, PORT,recv_time=1)
    
    # 启动数据传输
    config = rm_udp_custom_config_t()
    config.joint_speed = 0
    config.lift_state = 0
    config.expand_state = 0
    config.hand_state = 1
    config.arm_current_status = 0
    config.aloha_state = 0
    listener_instance.start(arm=slave_arm,config=config)
    
    # 启动数据接收线程
    receive_data_thread = threading.Thread(target=listener_instance.receive_data)
    receive_data_thread.daemon = True  # 设置为守护线程，主线程结束时自动终止
    receive_data_thread.start()

    # ----------------------------------------------------------------------------
    # 确定图像的输入分辨率与帧率
    resolution_width = 640  # pixels
    resolution_height = 480  # pixels
    frame_rate = 60  # fps

    # 注册数据流，并对其图像
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
    # check相机是不是进来了
    connect_device = []
    for d in rs.context().devices:
        print('Found device: ',
              d.get_info(rs.camera_info.name), ' ',
              d.get_info(rs.camera_info.serial_number))
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))

    if len(connect_device) < 2:
        print('Registrition needs two camera connected.But got one.')
        exit()
                                 
    # 确认相机并获取相机的内部参数
    pipeline1 = rs.pipeline()
    rs_config.enable_device(connect_device[0])
    pipeline1.start(rs_config)

    pipeline2 = rs.pipeline()
    rs_config.enable_device(connect_device[1])
    pipeline2.start(rs_config)

    frames1 = pipeline1.wait_for_frames(timeout_ms=1000)
    frames2 = pipeline2.wait_for_frames(timeout_ms=1000)
    old_frame1 = None
    old_frame2 = None

    # print("相机连接成功，开始数据采集")

    try:
        print('try process start')
        print('master goes to 0', master_arm.rm_movej(init_pose, 30, 0, 0, 1))
        print('slave goes to 0', slave_arm.rm_movej(init_pose, 30, 0, 0, 1))
        key_input = input("请确认主从臂都在0位姿，按 s 回车继续..., \n ")
        while key_input != 's':
            key_input = input("请确认主从臂都在0位姿，按 s 回车继续..., \n ")
        print("开始数据采集")
        last_point = None
        while len(master_trajectory) < 500:
            start_time = time.time()
            frames1 = pipeline1.wait_for_frames(timeout_ms=1000)
            frames2 = pipeline2.wait_for_frames(timeout_ms=1000)
            if not frames1 or not frames2:
                print("未获取到相机帧数据，尝试重新获取...")
                continue

            # 在采集循环里
            frame1_timestamp = frames1.get_timestamp()
            frame2_timestamp = frames2.get_timestamp()
            arm_status = master_arm.rm_get_current_arm_state()
            slave_status = slave_arm.rm_get_current_arm_state()
            
            # 获取额外两个关节的角度
            try:
                # 假设你有一个函数叫get_extra_joints_angles()来获取额外关节的角度
                extra_angles = get_extra_joints_angles(listener_instance)  # 替换成你实际的函数
            except Exception as e:
                print(f"获取额外关节数据失败: {e}")
                extra_angles = [0.0, 0.0,0,0,0,0]  # 失败时使用默认值

            if slave_status[0] == 0 and arm_status[0] == 0:
                
                if frame1_timestamp != old_frame1 and frame2_timestamp != old_frame2:
                    old_frame1, old_frame2 = frame1_timestamp, frame2_timestamp
                    
                    # 将额外两个关节的角度添加到现有的关节数据中
                    master_joints = arm_status[1]['joint'].copy()
                    slave_joints = slave_status[1]['joint'].copy()
                    
                    # 扩展关节数据（6自由度 -> 8自由度）
                    master_joints.extend(extra_angles)
                    slave_joints.extend(extra_angles)  # 从臂可能也需要这两个额外关节
                    
                    master_trajectory.append(master_joints)
                    slave_trajectory.append(slave_joints)
                    
                    color_frame1, color_frame2 = frames1.get_color_frame(), frames2.get_color_frame()
                    color_image1, color_image2 = np.asanyarray(color_frame1.get_data()), np.asanyarray(color_frame2.get_data())
                    save_images1.append(color_image1.copy())
                    save_images2.append(color_image2.copy())
                    image_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
                
                if last_point is None:
                    last_point = arm_status[1]['joint'].copy()
                else:
                    if master_trajectory[-1][:6] != last_point:  # 只比较前6个关节
                        last_point = arm_status[1]['joint'].copy()
                        # result = slave_arm.rm_movej_follow(last_point)
                        # 注意：这里只使用前6个关节控制从臂，因为从臂可能不接受8个关节输入
                        result = slave_arm.rm_movej_canfd(last_point, False, 0, 1, 50)
                        if result != 0:
                            print(f"从臂跟随移动失败，错误码: {result}")
                            break

            else:
                print("从臂或主臂状态异常，无法采集数据")
                continue
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(elapsed_time)
            if elapsed_time < 1 / frame_rate:
                time.sleep(1 / frame_rate - elapsed_time)

    except KeyboardInterrupt:
        print("数据采集被中断")

    # 在 finally 块中修改复位逻辑
    finally:
        print("\n数据采集完成或被中断")
        print("======================================")
        print("按下 's' 键并回车，将机械臂复位到初始位置...")
        print("======================================")
        
        # 等待用户按下 's' 键
        user_input = input("请按 's' 键并回车复位机械臂: ")
        while user_input.lower() != 's':
            user_input = input("请按 's' 键并回车复位机械臂: ")
        
        print("正在将机械臂移回初始位置...")
        # 使用较低的速度确保安全复位
        master_arm.rm_movej(init_pose, 20, 0, 0, 1)
        slave_arm.rm_movej(init_pose, 20, 0, 0, 1)
        
        # 等待机械臂移动完成
        time.sleep(2)
        print("机械臂已复位到初始位置")
        
        # 释放资源
        
        pipeline1.stop()
        pipeline2.stop()
        
        pgui.terminate()  # 终止GUI进程
        p.terminate()  # 终止监听进程
        listener_instance.close()  # 关闭监听实例
        config = rm_udp_custom_config_t()
        config_settings = rm_realtime_push_config_t(100, False, PORT, 0, LOCAL_IP, config)
        slave_arm.rm_set_realtime_push(config_settings)  # 停止实时推送
        master_arm.rm_delete_robot_arm()
        slave_arm.rm_delete_robot_arm()

        print("所有资源已释放，程序结束")

    return master_trajectory, slave_trajectory, save_images1, save_images2, image_time

def create_video(images, video_path, index_video, frame_rate=60):
    # 保持原有实现不变...
    if not images:
        print("没有图像数据，无法创建视频")
        return

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    for img in tqdm(images, desc="创建视频{}".format(index_video)):
        video_writer.write(img)

    video_writer.release()
    print(f"视频已保存到 {video_path}")

def save_data_to_h5(h5_path, action_list, qpos_list, images1, images2):
    # 处理8自由度数据
    action_arr = np.array(action_list, dtype=np.float32)  # 现在是8维
    qpos_arr = np.array(qpos_list, dtype=np.float32)  # 现在是8维1
    images1_arr = np.stack(images1, axis=0).astype(np.uint8)
    images2_arr = np.stack(images2, axis=0).astype(np.uint8)
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('action', data=action_arr)  # 8维动作
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos_arr)  # 8维状态
        images_grp = obs.create_group("images")
        images_grp.create_dataset("cam01", data=images1_arr)
        images_grp.create_dataset("cam02", data=images2_arr)
    print(f"数据已保存到 {h5_path}")

if __name__ == "__main__":
    print("程序开始")
    
    # 询问要收集的组数
    num_episodes = int(input("请输入要收集的数据组数: "))
    
    for episode in range(1, num_episodes + 1):
        print(f"\n=== 开始收集第 {episode}/{num_episodes} 组数据 ===")
        
        # 收集当前组数据
        master_trajectory, slave_trajectory, save_images1, save_images2, image_time = collect_process()
        
        # 保存当前组数据
        data_path = os.path.join(os.getcwd(), 'data_12(grasp1)', f'b{episode}')
        hdf5datapath = os.path.join(os.getcwd(), 'data_12(grasp1)', 'data')
        h55_path = os.path.join(hdf5datapath, f'episode_{episode}.hdf5')
        
        position_file = f'action_{episode}.csv'
        video_path1 = f'video1_{episode}.mp4'
        video_path2 = f'video2_{episode}.mp4'
        20
        master_pos_path = os.path.join(data_path, position_file)
        video_p1 = os.path.join(data_path, video_path1)
        video_p2 = os.path.join(data_path, video_path2)
        
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(hdf5datapath, exist_ok=True)
        
        length_data = len(master_trajectory)
        
        with open(master_pos_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for i in tqdm(range(length_data), total=length_data, desc=f"保存第{episode}组数据"):
                row = [master_trajectory[i], slave_trajectory[i], image_time[i], [i]]
                writer.writerow(row)
        
        create_video(save_images1, video_p1, episode)
        create_video(save_images2, video_p2, episode)
        
        print(f'数据{episode}长度:', len(master_trajectory), len(slave_trajectory), len(save_images1), len(save_images2))
        print(f"数据{episode}维度:", len(master_trajectory[0]) if master_trajectory else 0)
        
        save_data_to_h5(h55_path, master_trajectory, slave_trajectory, save_images1, save_images2)
        
        # 如果不是最后一组，等待按下g键开始下一组
        if episode < num_episodes:
            print("\n======================================")
            print(f"第 {episode} 组数据收集完成！")
            print("按下 'g' 键并回车，开始下一组数据收集...")
            print("======================================")
            
            user_input = input("请按 'g' 键并回车继续: ")
            while user_input.lower() != 'g':
                user_input = input("请按 'g' 键并回车继续: ")
    
    print("\n所有数据组收集完成！程序结束")