# if args less than 1, return
if [ $# -lt 1 ]; then
    echo "Usage: $0 task_name"
    exit 1
fi

# if exist second parameter, use it to decide run in debug mode
if [ $# -eq 2 ]; then
    if [ $2 == "debug" ]; then
        python3 imitate_episodes.py --task_name $1 --ckpt_dir ckpt/$1/ \
            --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
            --batch_size 4 --dim_feedforward 3200  --num_steps 20000 --lr 1e-5 --seed 0 \
            --eval_every 10 --validate_every 10
        exit 1
    fi
fi
python3 imitate_episodes.py --task_name $1 --ckpt_dir ckpt/$1/ \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 4 --dim_feedforward 3200  --num_steps 20000 --lr 1e-5 --seed 0 \
    