# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements imitation learning algorithms (ACT, Diffusion Policy, VINN) for robotic manipulation tasks, based on Mobile ALOHA. It is primarily used for **training policies and inference** on collected robot demonstration data. The codebase supports dual-arm and single-arm configurations with Realman robotic arms.

**Note**: Data collection can be done with external platforms. This framework focuses on policy training and deployment.

## Key Commands

### Environment Setup
```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install -r requirements.txt
cd detr && pip install -e .
cd robomimic && pip install -v -e .
```

### Data Preparation (External Collection Platform)
If you collect data using an external platform, ensure the HDF5 files follow this structure:

```
episode_0.hdf5
├── /action              # Shape: (T, action_dim), dtype: float32
├── /observations
│   ├── /qpos           # Shape: (T, state_dim), dtype: float32
│   └── /images
│       ├── /cam01      # Shape: (T, H, W, 3), dtype: uint8
│       └── /cam02      # Shape: (T, H, W, 3), dtype: uint8
```

**For dual-arm + gripper setup:**
- `action_dim = 14` (left_arm: 7 joints + right_arm: 7 joints)
- Gripper commands are typically the 7th joint of each arm (as part of the 7 DOF)
- OR `action_dim = 16` if grippers are separate (left_arm: 7 + left_gripper: 1 + right_arm: 7 + right_gripper: 1)

Place your collected episodes in the dataset directory specified in `constants.py`.

### Training
```bash
# ACT policy training
python imitate_episodes.py \
    --task_name <task_name> \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_steps 20000 \
    --lr 1e-5 \
    --seed 0

# Using shell script
bash scripts/train.sh <task_name>

# Diffusion Policy training
python imitate_episodes.py \
    --task_name <task_name> \
    --ckpt_dir <checkpoint_dir> \
    --policy_class Diffusion \
    --chunk_size 32 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 0 \
    --num_steps 100000
```

### Evaluation
```bash
# Evaluate trained policy (add --eval flag)
python imitate_episodes.py \
    --task_name <task_name> \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --eval \
    --temporal_agg  # Enable temporal ensembling

# Real robot inference
python inference.py
python infer_single_arm.py
```

### Data Visualization
```bash
# Visualize episode from dataset
python visualize_episodes.py \
    --dataset_dir <data_dir> \
    --episode_idx 0

# Replay episode actions
python replay_episodes.py \
    --dataset_path <path_to_episode.hdf5>
```

### Simulation (if using sim environments)
```bash
# Generate scripted episodes
python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data_save_dir> \
    --num_episodes 50 \
    --onscreen_render  # Optional: visualize during collection
```

## Architecture

### Policy Classes
The codebase implements three main policy architectures in `policy.py`:

1. **ACT (Action Chunking with Transformers)**:
   - Uses DETR-based encoder-decoder transformer architecture
   - Implements CVAE (Conditional VAE) for action prediction
   - Defined in `detr/models/detr_vae.py`
   - Key hyperparameters: `chunk_size`, `kl_weight`, `hidden_dim`, `dim_feedforward`

2. **DiffusionPolicy**:
   - Vision-based diffusion model for action prediction
   - Uses ResNet18 backbones with spatial softmax pooling
   - Conditional U-Net for noise prediction
   - Implements DDIM scheduler with EMA (Exponential Moving Average)
   - Key hyperparameters: `observation_horizon`, `action_horizon`, `prediction_horizon`

3. **CNNMLPPolicy**:
   - Simple baseline with CNN vision encoder + MLP action decoder

### Data Pipeline
- **Dataset Format**: HDF5 files with structure defined in `structure_hdf5.md`
  - `/action`: (T, action_dim) - joint commands
  - `/observations/qpos`: (T, state_dim) - joint positions
  - `/observations/images/{cam_name}`: (T, H, W, 3) - camera images
- **Data Loading**: `EpisodicDataset` class in `utils.py`
  - Handles multiple camera views
  - Supports data augmentation for Diffusion Policy
  - Implements chunked action sequences

### Model Components

#### DETR Backbone (`detr/models/`)
- `backbone.py`: ResNet18/34/50 vision encoders with positional encoding
- `transformer.py`: Multi-head attention encoder-decoder
- `detr_vae.py`: ACT model with CVAE latent action space
- `latent_model.py`: Latent dynamics model (optional)

#### RoboMimic Integration (`robomimic/`)
- Provides additional model components and utilities
- `robomimic/models/obs_nets.py`: Observation encoders
- `robomimic/algo/diffusion_policy.py`: Diffusion policy implementation
- `robomimic/utils/`: Training utilities, tensor operations, visualization

### Task Configuration
Tasks are configured in `constants.py` with two categories:

1. **Real Robot Tasks** (`REAL_TASK_CONFIGS`):
   - `real_test01`: Dual arm (12 DOF) with 2 cameras
   - `real_single_arm`: Single arm (8 DOF = 7 joints + gripper) with 1 camera
   - Key fields: `dataset_dir`, `camera_names`, `action_dim`, `state_dim`, `episode_len`

2. **Simulation Tasks** (`SIM_TASK_CONFIGS`):
   - `sim_transfer_cube_scripted`, `sim_insertion_scripted`, etc.
   - Used for testing and validation

**Important**: Update `DATA_DIR` in `constants.py` to point to your data directory before training or collecting data.

### Hardware Interfaces
- **Robotic Arms**: Realman robot interface in `Robotic_Arm/rm_robot_interface.py`
- **Glove Interface**: Oymotion glove handling in `usb_glove.py`
- **Cameras**: Intel RealSense integration via `pyrealsense2`
- **Data Collection Scripts**: `vla_double_arm_collect.py` for dual-arm teleoperation

## Training Notes

### ACT Tuning Tips
- If policy is jerky or pauses mid-episode, train longer - success rate and smoothness improve after loss plateaus
- For real-world data, train for at least 5000 epochs or 3-4x the length after loss plateau
- Typical success rates: ~90% for transfer cube, ~50% for insertion (sim)
- Enable `--temporal_agg` for temporal ensembling during evaluation
- See [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit)

### Diffusion Policy Notes
- Requires installing robomimic from r2d2 branch first
- Uses 50 training diffusion steps by default
- EMA improves stability and performance
- Data augmentation is automatically enabled for Diffusion Policy

### Co-training
- Mobile ALOHA's key finding: co-training with mixed sim+real data significantly improves success rates
- Specify multiple datasets or use `all` task config to enable co-training
- Use `name_filter` in task config to filter dataset files

### Wandb Logging
Training uses wandb for experiment tracking. To configure:
- Edit `imitate_episodes.py` main function
- Update `entity` and `project` parameters in `wandb.init()`
- Or comment out wandb calls if not needed

## Code Organization

### Core Training Loop
`imitate_episodes.py` contains the main training loop:
1. Load task config from `constants.py`
2. Initialize policy (ACT/Diffusion/CNNMLP)
3. Load datasets with `load_data()` from `utils.py`
4. Training loop with validation and checkpointing
5. Evaluation with `eval_bc()` (requires robot connection for real tasks)

### Policy Inference Flow
1. Load checkpoint and rebuild policy
2. Initialize robot connection (real) or simulation environment
3. Observation loop:
   - Capture images from cameras
   - Read joint positions (qpos)
   - Preprocess and normalize inputs
4. Policy forward pass to get action sequence
5. Execute actions with optional temporal ensembling

### Adding New Tasks
1. Add task config to `REAL_TASK_CONFIGS` or `SIM_TASK_CONFIGS` in `constants.py`
2. Specify: `dataset_dir`, `camera_names`, `action_dim`, `state_dim`, `episode_len`
3. Collect data in HDF5 format matching structure in `structure_hdf5.md`
4. Run training with `--task_name <new_task_name>`

## Common Issues

### Action/State Dimension Mismatch
- Ensure `action_dim` and `state_dim` in task config match your robot's DOF
- Default is 14 for dual-arm (2x7 joints), 8 for single arm (7 joints + 1 gripper)
- Check data collection scripts produce correct dimensions

### Camera Configuration
- Camera names must match between data collection and training
- Update `camera_names` list in task config
- For single camera setups, use `['cam01']`

### Real Robot Evaluation
- `eval_bc()` requires robot connection and ROS packages
- Comment out evaluation code if training without hardware access
- Validation loss can still be monitored via `--validate_every` flag

### Data Directory Paths
- Always use absolute paths for `dataset_dir` in task configs
- Verify `DATA_DIR` in `constants.py` points to correct location
- Data files are expected as `episode_*.hdf5` in the dataset directory
