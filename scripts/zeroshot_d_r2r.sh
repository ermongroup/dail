#!/bin/bash

# gen_weight = 0.01 and action_loss * 100 with learning rate 1e-5
# 7/10 worked! alignment is good 7/10

# start new tmux sesson
SESS_NAME="eval_d_r2r"
DOC="empty"
#VENV_DIR="../.virtualenv/cdil/bin/activate"
VENV_DIR="~/Desktop/Research/2.DAIL/cdil/venv/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME

BEGIN=0
END=9
TOTAL_GPU=4

for ((i=BEGIN; i<=END; i++)); do
gpu_num=$((i % TOTAL_GPU))

PYTHON_CMD="source ${VENV_DIR} && python train.py --algo ddpg --agent_type zeroshot --load_expert_dir ./target_expert/reacher2_corner/alldemo --load_learner_dir ./saved_alignments/dynamics/12goals/seed_${i} --edomain reacher2_corner --ldomain reacher2_act_corner --seed 100${i} --doc d_r2r"

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

