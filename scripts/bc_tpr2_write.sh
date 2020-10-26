#!/bin/bash


# 7/10 worked! (>100 is good) alignment is good 8/10

# start new tmux sesson
SESS_NAME="bc_tpr2_write"
DOC="empty"
VENV_DIR="../.virtualenv/cdil/bin/activate"
#VENV_DIR="~/Desktop/Research/2.DAIL/cdil/venv/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME

BEGIN=0
END=4
TOTAL_GPU=4

for ((i=BEGIN; i<=END; i++)); do
gpu_num=$((i % TOTAL_GPU))

PYTHON_CMD="source ${VENV_DIR} && python train.py --algo ddpg --agent_type bc --save_expert_dir ./target_expert/tp_write_reacher2/demo200_seed_${i} --load_dataset_dir ./target_demo/tp_write_reacher2 --edomain tp_write_reacher2 --ldomain write_reacher2 --seed 100${i} --doc tpr2_write_bc_${i} --n_demo 200 --gpu ${gpu_num}"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 0.5

tmux select-layout tiled
done
tmux a -t $SESS_NAME

