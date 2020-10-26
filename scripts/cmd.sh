#!/bin/bash



#== dynamics mismatch (12 goals)
#python train.py --algo ddpg --agent_type zeroshot --save_dataset_dir ./temp/d_r2r_12goals_noise_05_exp.pickle --load_expert_dir ./saved_expert/reacher2/allgoals/3E2L_all_start_2 --edomain reacher2 --ldomain reacher2_act --eseed 0 --lseed 0 --gpu 0

#== embodiment mismatch (12 goals)
#python train.py --algo ddpg --agent_type zeroshot --save_dataset_dir ./temp/e_r2r_noise_05_exp.pickle --load_expert_dir ./saved_expert/reacher3/12goals/3E2L_all_start_4 --load_learner_dir ./saved_learner/reacher3E2L/12goals/3E2L_wall_4_3 --edomain reacher3 --ldomain reacher2 --eseed 0 --lseed 0 --gpu 0

#== viewpoint mismatch (all goals)
python train.py --algo ddpg --agent_type zeroshot --save_dataset_dir ./temp/v_r2w_noise_05_exp.pickle --load_expert_dir ./saved_expert/tp_reacher/allgoals/div_offset_180_3 --load_learner_dir ./saved_learner/tp_reacher/allgoals/div_offset_180_2_5 --edomain tp_reacher2 --ldomain reacher2 --eseed 0 --lseed 0 --gpu 0

#python train.py --algo ddpg --agent_type zeroshot --save_dataset_dir ./temp/temp.pickle --load_expert_dir ./saved_expert/tp_write_reacher/test/offset_180_1 --load_learner_dir ./saved_learner/tp_reacher/allgoals/div_offset_180_2_5 --edomain tp_write_reacher2 --ldomain write_reacher2 --eseed 0 --lseed 0 --gpu 0

#python train.py --algo ddpg --agent_type zeroshot --save_dataset_dir ./temp/v_r2w_noise_05_exp.pickle --load_expert_dir ./saved_expert/tp_reacher/allgoals/div_offset_180_3 --load_learner_dir ./saved_learner/tp_reacher/allgoals/div_offset_180_2_5 --edomain tp_write_reacher2 --ldomain write_reacher2 --eseed 0 --lseed 0 --gpu 0
