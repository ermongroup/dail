#!/bin/bash

# gen weight = 0.001, lr 1e-5
# 10/10 worked!

# start new tmux sesson
SESS_NAME="align_embodiment"
DOC="empty"
VENV_DIR="../.virtualenv/cdil/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME

BEGIN=0
END=9
TOTAL_GPU=4

for ((i=BEGIN; i<=END; i++)); do
gpu_num=$((i % TOTAL_GPU))

PYTHON_CMD="source ${VENV_DIR} && python train.py --algo ddpg --agent_type gama --load_dataset_dir ./alignment_taskset/embodiment.pickle --load_expert_dir ./alignment_expert/reacher3_wall/12goals --save_learner_dir ./saved_alignments/embodiment/12goals/seed_${i} --logdir ./logs/embodiment/12goals/seed_${i} --edomain reacher3_wall --ldomain reacher2_wall --seed ${i} --gpu ${gpu_num}"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 15

tmux select-layout tiled
done
tmux a -t $SESS_NAME

