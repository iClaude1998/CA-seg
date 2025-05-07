#!/bin/bash
# Configure the resources required
#SBATCH --job-name=0200574 # job name
#SBATCH -p a100
#SBATCH -N 1 # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH --ntasks=1          # number of tasks (multi-thread job starts 4 tasks)
#SBATCH --mem=32G              # memory required by the job (if above 64G, use --mem=128G)
#SBATCH -c 8                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=00:40:00         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1            # generic resource required (here requires 4 GPUs)
#SBATCH --chdir=/gpfs/users/a1233646/myprojects/clipflow2 # set the working directory

# Configure notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yunxiang.liu@adelaide.edu.au

# module load NCCL/2.12.12-GCCcore-11.2.0-CUDA-11.6.2
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

conda info --envs
nvcc -V


export TRANSFORMERS_CACHE=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

<<<<<<< HEAD
exp_name=pmc_camus_4ch22ch-left+heart+ventricle_aug
config_file=configs/cbm_pmc/pmc_camus_4ch22ch-left+heart+ventricle_aug.yaml
test_type=train
=======

exp_name=camus_2t4ch_aug
config=configs/flowmatch/bioparse/camus_2t4ch_aug.yaml
test_type=test
>>>>>>> 115363385004710b0254bbc523d14ecdae6f7efc
learn_obj=recflow

python main.py --task test \
       --exp_name ${exp_name} \
       --config ${config} \
       --num_workers 8 \
       --test_type ${test_type} \
       --learn_obj ${learn_obj} \
       --load_checkpoint