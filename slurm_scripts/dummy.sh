#!/bin/bash
# Configure the resources required
#SBATCH -p a100
#SBATCH -N 1 # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 8                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=00:20:00         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1            # generic resource required (here requires 4 GPUs)

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

echo $TRANSFORMERS_CACHE
echo $HUGGINGFACE_HUB_CACHE
echo $XDG_CACHE_HOME

python dummy.py
