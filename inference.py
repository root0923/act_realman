import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms
from Robotic_Arm import *
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs
import h5py
from tqdm import tqdm
import cv2

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']


    print('args: ', args)
    print('---------------------------------------')

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    # print teask name and config
    print('task_name: ', task_name)
    if is_sim or task_name == 'all':
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        # from aloha_scripts.constants import TASK_CONFIGS
        # task_config = TASK_CONFIGS[task_name]
        from constants import REAL_TASK_CONFIGS
        task_config = REAL_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 6
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
        'lr': 1e-5,
        'num_queries': args['chunk_size'],
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': camera_names,
        'vq': args['use_vq'],
        'vq_class': args['vq_class'],
        'vq_dim': args['vq_dim'],
        'state_dim': state_dim,
        'action_dim': 6,
        'no_encoder': args['no_encoder'],
    }


    config = {
        'num_steps': 20000,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'lr': 1e-5,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        
    }

    ckpt_names = 'policy_best.ckpt'
    ckpt_dir = config['ckpt_dir']
    state_dim = 6
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    vq = config['policy_config']['vq']
    real_robot = args['real_robot']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_names)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    try:
        # master_arm = RoboticArm()
        slave_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        # handle1 = master_arm.rm_create_robot_arm("192.168.110.119", 8080)
        handle2 = slave_arm.rm_create_robot_arm("192.168.110.118", 8080)
    except Exception as e:
        print(f"创建主臂从臂失败: {e}")
        return
    # print("创建主臂从臂成功", handle1.id, handle2.id)
    print("创建从臂成功", handle2.id)

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

    with torch.inference_mode():
        try:
            length = args['max_timesteps']
            init_pose = [36.43, -3.88, 82.67, 5.19, 77.72, -3.09]
            while True:

                if temporal_agg:
                    print('Using temporal aggregation')
                    num_queries = policy_config['num_queries']
                    all_time_actions = torch.zeros([length, length + num_queries, 6]).cuda()

                frames1, frames2 = pipeline1.wait_for_frames(timeout_ms=1000), pipeline2.wait_for_frames(timeout_ms=1000)
                if not frames1 or not frames2:
                    print('No frames received from cameras, retrying...')
                    continue

                arm_status = slave_arm.rm_get_current_arm_state()
                if arm_status[0] != 0:
                    print(f"从臂状态获取失败: {arm_status[1]}")
                    continue
                
                qpos = arm_status[1]['joint']
                color_frame1, color_frame2 = frames1.get_color_frame(), frames2.get_color_frame()
                image1, image2 = np.asanyarray(color_frame1.get_data()), np.asanyarray(color_frame2.get_data())

                # preprocess
                qpos = pre_process(np.array(qpos, dtype=np.float32))
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                curr_image = image_preprocess(image1, image2, rand_crop_resize=(config['policy_class'] == 'Diffusion'))

                all_action = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_action
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_action[:, 0]

                raw_action = raw_action.squeeze(0).detach().cpu().numpy()
                action = post_process(raw_action)

        except KeyboardInterrupt:
            print('KeyboardInterrupt received, exiting...')
        finally:
            pipeline1.stop()
            pipeline2.stop()
            slave_arm.rm_movej(init_pose, 20, 0, 0, 1)
            slave_arm.rm_delete_robot_arm()

    torch.cuda.empty_cache()
    

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def image_preprocess(image1, image2, rand_crop_resize=False):
    output = []
    for image in [image1, image2]:
        img = rearrange(image, 'h w c -> c h w')
        output.append(img)
    output = np.stack(output, axis=0)
    output = torch.from_numpy(output / 255.0).float().cuda().unsqueeze(0)
    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = output.shape[-2:]
        ratio = 0.95
        output = output[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        output = output.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        output = resize_transform(output)
        output = output.unsqueeze(0)
    return output


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='/home/arm_hao/act-plus-plus-main/ckpt/real_test01', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate', required=False)
    parser.add_argument('--policy_class', default='ACT', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', default='real_test01', type=str, help='task_name', required=False)
    parser.add_argument('--seed', default=0, type=int, help='seed', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', type=int)
    parser.add_argument('--future_len', type=int)
    parser.add_argument('--prediction_len', type=int)
    parser.add_argument('--num_steps', default=20000, type=int, help='num_steps', required=False)
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size', required=False)

    # for ACT
    parser.add_argument('--kl_weight', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', default=50, type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', default=True)
    parser.add_argument('--max_timesteps', default=2000, type=int, help='max_timesteps', required=False)
    parser.add_argument('--real_robot', default=True)
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', type=int, help='vq_class')
    parser.add_argument('--vq_dim', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    main(vars(parser.parse_args()))
