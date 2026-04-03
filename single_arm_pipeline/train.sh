#!/bin/bash
# 单臂训练脚本

# 配置
TASK_NAME="single_arm_left"
CKPT_DIR="ckpt/${TASK_NAME}"
POLICY_CLASS="ACT"

# ACT 超参数
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
BATCH_SIZE=8
DIM_FEEDFORWARD=3200
NUM_STEPS=20000
LR=1e-5
SEED=0

# 训练参数
EVAL_EVERY=500
VALIDATE_EVERY=500
SAVE_EVERY=500

# 创建checkpoint目录
mkdir -p ${CKPT_DIR}

echo "=================================="
echo "开始训练单臂ACT模型"
echo "任务: ${TASK_NAME}"
echo "Checkpoint目录: ${CKPT_DIR}"
echo "=================================="

# 运行训练
python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight ${KL_WEIGHT} \
    --chunk_size ${CHUNK_SIZE} \
    --hidden_dim ${HIDDEN_DIM} \
    --batch_size ${BATCH_SIZE} \
    --dim_feedforward ${DIM_FEEDFORWARD} \
    --num_steps ${NUM_STEPS} \
    --lr ${LR} \
    --seed ${SEED} \
    --eval_every ${EVAL_EVERY} \
    --validate_every ${VALIDATE_EVERY} \
    --save_every ${SAVE_EVERY}

echo "=================================="
echo "训练完成！"
echo "最佳模型保存在: ${CKPT_DIR}"
echo "=================================="
