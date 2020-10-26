# Domain Adaptive Imitation Learning (DAIL)

This repo contains the official implementation for the paper [Domain Adaptive Imitation Learning](https://arxiv.org/abs/1910.00105). 

Authors: Kuno Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, Stefano Ermon. 

For any questions, please correspond with Kuno Kim (khkim@cs.stanford.edu)

-----------------------------------------------------------------------------------------

### Dependencies

Core python dependencies are `tensorflow==1.15.0, mujoco-py==0.5.7, gym==0.7.4`. Also be sure to have [tmux](https://github.com/tmux/tmux) installed to run the provided shell scripts and virtualenv for package management. On a GPU machine, run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

For CPU machines, replace the `tensorflow-gpu==1.15.0` package with `tensorflow==1.15.0`


### Overall Pipeline ### 

```
-> Learn state/action alignments via GAMA 
-> Train expert domain policy on new task via BC 
-> Compose alignments with new expert domain policy for zeroshot imitation
```
[Videos](https://www.youtube.com/watch?v=Ue-QhWbUr5s&feature=youtu.be&ab_channel=KunoKim) of learned alignments

### Generative Adversarial MDP Alignment (GAMA)

In the GAMA step, we learn state/action alignments between expert and self domain MDPs by minimizing the GAMA loss which consists of a distribution matching loss in the expert domain and a behavioral cloning (imitation) loss in the self domain. Prior to minimizing the GAMA loss, we first need to learn the expert policy for the alignment tasks as well as a dynamics model in the self domain. To reduce execution time, we provide expert domain policies for the alignment tasks pretrained via BC in `dail/alignment_expert/` and load these pretrained experts during the GAMA step. All MDP alignment commands can be found in the `scripts/align_*.sh` shell scripts. Before running: 

1. First appropriately set `VENV_DIR` in the shell script to your virtualenv directory for this project. Furthermore, set the `GPU_NUM` to the number of GPUs available on your machine. For CPU runs, simply set the gpu_num variable to -1. Finally, set the the number of seeds you want to run by changing the `END` variable. (default = 9) 

2. Download the alignment taskset from the provided link below and place in `dail/alignment_taskset`. 

3. Make sure to have [tmux](https://github.com/tmux/tmux) installed. 

After these steps, from the root folder `dail/` run `scripts/align_*.sh`. For example, `scripts/align_dynamics.sh` will train alignments between MDPs with mismatched dynamics. Similarly run `scripts/align_embodiment, scripts/align_viewpoint` for different types of domain mismatch. The script will automatically run many seeds of GAMA training and, for each seed, save the alignment that achieve the lowest GAMA loss to `saved_alignments/`. One procedure is to try multiple seeds, and pick the one with the lowest shown "BEST GAMA LOSS" for zeroshot evaluation. You can monitor GAMA training via tensorboard files in `logs/`. At any point during the code execution you can use `ctrl + c` to bring up a selection menu which allows you to visualize, save the current model, or terminate execution. 


### Zeroshot evaluation

After the self and expert domains are aligned via GAMA, you can leverage cross domain demonstrations by 

1. Train an expert domain policy for a new task via Behavioral Cloning. To do so, first download demonstrations for the target tasks from the link below and place in `dail/target_demo`. Then, execute `scripts/bc_*.sh`. Follow the same steps as for GAMA to set `VENV_DIR, GPU_NUM, END, gpu_num`. For example, `scripts/bc_r2_corner.sh` trains a 2-link reacher that reaches for goals near the arena corners. Change the `--n_demo` argument to change the number of demonstrations used for BC. This script will run BC for 100 epochs and save the last model. 

2. Compose the new expert policy with previously learned alignments. For example, execute `scripts/zeroshot_d_r2r.sh` to use adapt the 2-link reacher corner expert policy into an 2-link reacher in a different dynamics environment reaching for corner goals. Follow the same steps as for GAMA to set `VENV_DIR, GPU_NUM, END, gpu_num`. Appropriately change the `--load_expert_dir` to point to the expert checkpoint directory you have used in the previous BC step and `--load_learner_dir` to point to the MDP alignment that you would like to use from the GAMA step. You can see videos of the task executed in the self domain side by side with the output of the learned statemap in `dail/cached_videos/`. 


### Alignment Taskset and Target Demonstration data

Link: https://drive.google.com/drive/folders/1Cl84d0-vYs3bDClp7YGrpgxrsylD8-RR?usp=sharing


## References

If you find the code/idea useful for your research, please consider citing

```bib
@article{kim2020dail,
  title={Domain Adaptive Imitation Learning},
  author={Kim, Kuno and Gu, Yihong and Song, Jiaming and Zhao, Shengjia and Ermon, Stefano},
  journal={arXiv preprint arXiv:1910.00105},
  year={2020}
}
```
