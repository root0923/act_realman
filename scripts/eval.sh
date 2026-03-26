# if args less than 1, exit
if [ $# -lt 1 ]; then
    echo "Usage: $0 task_name"
    exit
fi
python3 imitate_episodes.py --eval --task_name $1 --ckpt_dir ckpt/$1/ \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 4 --dim_feedforward 3200  --num_steps 2000 --lr 1e-5 --seed 0 --onscreen_render