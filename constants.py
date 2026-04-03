import pathlib
import os

### Task parameters
# DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else '/scr/tonyzhao/datasets'
# DATA_DIR = '/home/alvin/data/aloha/'
#DATA_DIR = '/home/shuxiangzhang/act-yang/data'



#DATA_DIR ='/home/shuxiangzhang/act-yang/data_all12/data'#开门+抓取
#DATA_DIR ='/home/shuxiangzhang/act-yang/data_12(11)/data'#开门
#DATA_DIR ='/home/shuxiangzhang/act_realman/data_12_grasp/data'#抓取
#DATA_DIR ='/home/shuxiangzhang/act_realman/data_lap_hand/data'#顶部相机
DATA_DIR = '/home/ysy/data/aloha'

REAL_TASK_CONFIGS = {
    'single_arm_left': {
        'dataset_dir': '/home/ysy/data/teleop_data/20260402_1_hdf5',  # 转换后的数据目录
        'episode_len': 500,
        'camera_names': ['camera_head_image', 'camera_left_image'],
        'action_dim': 8,   # 7 joints + 1 gripper
        'state_dim': 8     # 7 joints + 1 gripper
    },
    'aloha': {
        'dataset_dir': DATA_DIR,
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        'action_dim': 16,  # 14 joint actions + 2 base actions
        'state_dim': 14    # qpos dimension
    },
    # 'real_single_arm': {
    #     "dataset_dir": "data/real_episodes_prepared",  # 你 prepare_real_h5.py 生成的路径
    #     'num_episodes': 50,
    #     'episode_len': 500,
    #     'camera_names': ['cam01'],
    #     'action_dim': 12,  # 添加这一行，指定动作维度为8
    #     'state_dim': 12    # 添加这一行，指定状态维度为8
    # },
    'real_single_arm': {
        "dataset_dir": "data/real_episodes_prepared",  # 你 prepare_real_h5.py 生成的路径
        "history_len": 10,       # 可根据训练需求调整
        "future_len": 5,         # 预测长度
        "prediction_len": 5,     # 模型输出长度
        "action_dim": 8,       # 7关节 + 1夹爪
        "state_dim": 8,
        "camera_names": ["cam01"],  # 你采集的相机名称
        "use_images": True,      # 是否使用图像输入
        "image_size": (128, 128), # 根据你 h5 图像大小设置
        "normalize_qpos": True,  # 是否归一化关节角度
        "mirrored_data": False,   # 数据增强开关，根据需要设
        "episode_len": 500   # 必填字段，对应每条序列长度
    },
    'real_single_arm2': {
        "dataset_dir": "data/real_episodes_prepared",  # 你 prepare_real_h5.py 生成的路径
        "history_len": 10,       # 可根据训练需求调整
        "future_len": 5,         # 预测长度
        "prediction_len": 5,     # 模型输出长度
        "action_dim": 8,       # 7关节 + 1夹爪
        "state_dim": 8,
        "camera_names": ["cam01"],  # 你采集的相机名称
        "use_images": True,      # 是否使用图像输入
        "image_size": (128, 128), # 根据你 h5 图像大小设置
        "normalize_qpos": True,  # 是否归一化关节角度
        "mirrored_data": False,   # 数据增强开关，根据需要设
        "episode_len": 500   # 必填字段，对应每条序列长度
    }
}

SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}


### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
