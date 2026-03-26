#!/usr/bin/env python3
import torch
import numpy as np
import os
import pickle
import argparse
import time
from einops import rearrange
from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs

from utils import set_seed
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy


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


def image_preprocess_single(image, rand_crop_resize=False):
    # image: HWC (uint8)
    img = rearrange(image, 'h w c -> c h w')
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, C, H, W)
    img = img.cuda()
    if rand_crop_resize:
        # keep behavior similar to original when requested
        original_size = img.shape[-2:]
        ratio = 0.95
        h0 = int(original_size[0] * (1 - ratio) / 2)
        h1 = int(original_size[0] * (1 + ratio) / 2)
        w0 = int(original_size[1] * (1 - ratio) / 2)
        w1 = int(original_size[1] * (1 + ratio) / 2)
        img = img[..., h0:h1, w0:w1]
        img = torch.nn.functional.interpolate(img.view(1, img.shape[2], img.shape[3], img.shape[4]),
                                              size=original_size, mode='bilinear', align_corners=False)
        img = img.unsqueeze(0)
    return img


def main(args):
    set_seed(args.seed)

    # load task / policy config
    ckpt_dir = args.ckpt_dir
    policy_class = args.policy_class
    task_name = args.task_name

    # single camera name
    camera_names = ['cam01']

    policy_config = {
        'lr': args.lr,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'vq': args.use_vq,
        'vq_class': args.vq_class,
        'vq_dim': args.vq_dim,
        'state_dim': 6,
        'action_dim': 6,
        'no_encoder': args.no_encoder,
    }

    # load policy
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    policy = make_policy(policy_class, policy_config)
    load_status = policy.deserialize(torch.load(ckpt_path))
    print('policy load status:', load_status)
    policy.cuda()
    policy.eval()

    # load stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f'stats not found: {stats_path}')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # initialize robotic arm
    try:
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(args.robot_ip, args.robot_port)
        print('created arm handle:', handle.id)
    except Exception as e:
        print('Failed to create robot arm:', e)
        return

    # initialize single RealSense camera
    ctx = rs.context()
    devices = [d for d in ctx.devices if d.get_info(rs.camera_info.name).lower() != 'platform camera']
    if len(devices) == 0:
        print('No RealSense devices found')
        arm.rm_delete_robot_arm()
        return

    serial = devices[0].get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(rs_config)
    print('started camera serial:', serial)

    init_pose = args.init_pose

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            if not frames:
                print('No frames, retrying...')
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                print('No color frame, retrying...')
                continue

            image = np.asanyarray(color_frame.get_data())

            arm_status = arm.rm_get_current_arm_state()
            if arm_status[0] != 0:
                print('Failed to get arm state:', arm_status[1])
                continue

            qpos = np.array(arm_status[1]['joint'], dtype=np.float32)
            qpos = torch.from_numpy(pre_process(qpos)).float().cuda().unsqueeze(0)

            curr_image = image_preprocess_single(image, rand_crop_resize=(policy_class == 'Diffusion'))

            with torch.inference_mode():
                # policy expected signature may vary; using policy(qpos, curr_image)
                out = policy(qpos, curr_image)
                # handle different types of outputs
                if isinstance(out, tuple) or isinstance(out, list):
                    all_action = out[0]
                else:
                    all_action = out

                raw_action = all_action[:, 0].squeeze(0).detach().cpu().numpy()
                action = post_process(raw_action)

            # send joint command
            # arm.rm_movej(action.tolist(), 20, 0, 0, 1)
            print("arm:", action.tolist())

    except KeyboardInterrupt:
        print('KeyboardInterrupt, exiting...')
    finally:
        pipeline.stop()
        try:
            arm.rm_movej(init_pose, 10, 0, 0, 1)
            arm.rm_delete_robot_arm()
        except Exception:
            pass

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='ckpts/real_single_arm', type=str)
    parser.add_argument('--policy_class', default='ACT', type=str)
    parser.add_argument('--task_name', default='real_single_arm', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--eval_every', default=500, type=int)
    parser.add_argument('--validate_every', default=500, type=int)
    parser.add_argument('--save_every', default=500, type=int)
    parser.add_argument('--kl_weight', default=10, type=float)
    parser.add_argument('--chunk_size', default=50, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=3200, type=int)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--max_timesteps', default=2000, type=int)
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', type=int)
    parser.add_argument('--vq_dim', type=int)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--robot_ip', default='192.168.110.118', type=str)
    parser.add_argument('--robot_port', default=8080, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)
    parser.add_argument('--fps', default=60, type=int)
    parser.add_argument('--init_pose', nargs='+', type=float, default=[0,0,0,0,0,0,0])

    args = parser.parse_args()
    main(args)
