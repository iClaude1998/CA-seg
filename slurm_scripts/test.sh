#!/bin/bash
# Configure the resources required
#SBATCH --job-name=jiduanzisi # job name
#SBATCH -p a100
#SBATCH -N 1 # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH --ntasks=1          # number of tasks (multi-thread job starts 4 tasks)
#SBATCH --mem=32G              # memory required by the job (if above 64G, use --mem=128G)
#SBATCH -c 8                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=00:10:00         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1            # generic resource required (here requires 4 GPUs)
#SBATCH --chdir=/gpfs/users/a1233646/myprojects/clipflow2 # set the working directory

# Configure notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yunxiang.liu@adelaide.edu.au

module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

conda info --envs
nvcc -V

export TRANSFORMERS_CACHE=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=amos22_liver
config_file=configs/flowmatch/bioparse/amos22_liver.yaml
test_type=test
learn_obj=recflow


python main.py --task test \
               --exp_name ${exp_name} \
               --config ${config_file} \
               --device cuda \
               --test_type ${test_type} \
               --learn_obj ${learn_obj} \
               --num_workers 8 \
               --load_checkpoint
