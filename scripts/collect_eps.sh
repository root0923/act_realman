# if args less then 1, return
if [ $# -lt 1 ]; then
    echo "Usage: $0 task_name"
    exit 1
fi
# python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir ~/data/sim_transfer_cube_scripted --num_episodes 50 --onscreen_render
python3 record_sim_episodes.py --task_name $1 --dataset_dir ~/data/aloha/$1 --num_episodes 50 --onscreen_render