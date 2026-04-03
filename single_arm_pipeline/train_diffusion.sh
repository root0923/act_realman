#!/bin/bash
# 单臂Diffusion Policy训练脚本

# 配置
TASK_NAME="single_arm_left"
CKPT_DIR="/home/ysy/data/weights/${TASK_NAME}_diffusion_0402"
POLICY_CLASS="Diffusion"

# Diffusion Policy 超参数
CHUNK_SIZE=32
BATCH_SIZE=32
NUM_STEPS=100000
LR=1e-4
SEED=0

# 训练参数
VALIDATE_EVERY=1000
SAVE_EVERY=1000

# 创建checkpoint目录
mkdir -p ${CKPT_DIR}

echo "=================================="
echo "开始训练单臂Diffusion Policy模型"
echo "任务: ${TASK_NAME}"
echo "Checkpoint目录: ${CKPT_DIR}"
echo "=================================="

# 运行训练
python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ${POLICY_CLASS} \
    --chunk_size ${CHUNK_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_steps ${NUM_STEPS} \
    --lr ${LR} \
    --seed ${SEED} \
    --validate_every ${VALIDATE_EVERY} \
    --save_every ${SAVE_EVERY}

echo "=================================="
echo "训练完成！"
echo "最佳模型保存在: ${CKPT_DIR}"
echo "=================================="
