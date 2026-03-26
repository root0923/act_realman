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
import pyrealsense2 as rs
import h5py
import cv2

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_box_pose, sample_insertion_pose
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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
    print('task_name: ', task_name)
    if is_sim or task_name == 'all':
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import REAL_TASK_CONFIGS
        task_config = REAL_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 12
    action_dim = 12
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
        'action_dim': action_dim,
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

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean'][:state_dim]) / stats['qpos_std'][:state_dim]
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'][:action_dim] - stats['action_min'][:action_dim]) + stats['action_min'][:action_dim]
    else:
        post_process = lambda a: a * stats['action_std'][:action_dim] + stats['action_mean'][:action_dim]

    with torch.inference_mode():
        for ak in range(1, 3):
            datapath = f'/home/shuxiangzhang/act-yang/data_12(grasp)/data/real_test01/episode_{ak}.hdf5'
            print(f'Loading data from {datapath}')
            with h5py.File(datapath, 'r') as f:
                all_images1 = f['/observations/images/cam01'][()]
                all_images2 = f['/observations/images/cam02'][()]
                all_qpos = f['/observations/qpos'][()]
                all_actions = f['/action'][()]

            length = len(all_images1)

            errors = []  # shape: [length, action_dim]
            if temporal_agg:
                print('Using temporal aggregation')
                num_queries = policy_config['num_queries']
                all_time_actions = torch.zeros([length, length + num_queries, action_dim]).cuda()

            for t in tqdm(range(length), desc='Processing timesteps {}/{}'.format(ak, length)):
                image1 = all_images1[t]
                image2 = all_images2[t]
                qpos_numpy = all_qpos[t]
                action_numpy = all_actions[t]

                # preprocess
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                curr_image = image_preprocess(image1, image2, rand_crop_resize=(config['policy_class'] == 'Diffusion'))

                all_action = policy(qpos, curr_image)
                if temporal_agg:
                    end_idx = min(t + num_queries, all_time_actions.shape[1])
                    available_slots = end_idx - t
                    if available_slots > 0:
                        all_time_actions[[t], t:end_idx] = all_action[:, :available_slots]
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

                # 记录每个关节的误差
                joint_error = action - action_numpy  # shape: [action_dim]
                errors.append(joint_error)

            errors = np.stack(errors, axis=0)  # shape: [length, action_dim]

            # 绘制每个关节的误差曲线
            plt.figure(figsize=(14, 8))
            x = np.arange(length)
            for i in range(errors.shape[1]):
                plt.plot(x, errors[:, i], label=f'Joint {i+1}')
            plt.xlabel('Timestep')
            plt.ylabel('Joint Error (Prediction - Ground Truth)')
            plt.title('Per-joint Prediction Error Curve')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()

            # 保存到指定路径
            save_path = f'/home/shuxiangzhang/act-yang/ckpt/error/error_time/prediction_joint_error_{ak}.png'
            plt.savefig(save_path)
            print(f'Error curve saved to {save_path}')

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
    return policy(qpos_data, image_data, action_data, is_pad)

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='/home/shuxiangzhang/act-yang/ckpt/real_test_grasp', type=str, help='ckpt_dir', required=False)
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
    parser.add_argument('--temporal_agg', action='store_true', default=False)
    parser.add_argument('--real_robot', default=True)
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', type=int, help='vq_class')
    parser.add_argument('--vq_dim', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    main(vars(parser.parse_args()))