#!/bin/bash

# gen_weight = 0.001 with learning rate 1e-5
# 8/10 worked

# start new tmux sesson
SESS_NAME="push_expert"
DOC="empty"
VENV_DIR="~/Desktop/Research/2.DAIL/cdil/venv/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME

BEGIN=0
END=9
TOTAL_GPU=4

for ((i=BEGIN; i<=END; i++)); do
gpu_num=$((i % TOTAL_GPU))

PYTHON_CMD="source ${VENV_DIR} && python train.py --algo ddpg --agent_type rollout_expert --load_expert_dir ./saved_expert/reacher2_push/no_ctrl_${i} --edomain reacher2_push --ldomain reacher2 --eseed 100${i} --lseed 100${i} --doc reacher2_push"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 0.1

tmux select-layout tiled
done
tmux a -t $SESS_NAME

