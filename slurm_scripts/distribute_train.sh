#!/bin/bash
# Configure the resources required
#SBATCH --job-name=yliusb-rlpclip22 # job name
#SBATCH -p a100
#SBATCH -N 1 # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH --ntasks=1          # number of tasks (multi-thread job starts 4 tasks)
#SBATCH --mem=32G              # memory required by the job (if above 64G, use --mem=128G)
#SBATCH -c 8                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=1-00:00:00         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:4            # generic resource required (here requires 4 GPUs)

# Configure notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yunxiang.liu@adelaide.edu.au

# module load NCCL/2.12.12-GCCcore-11.2.0-CUDA-11.6.2
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

conda info --envs
nvcc -V

# export MASTER_PORT=12340
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr-ib
# echo "MASTER_ADDR="$MASTER_ADDR
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_P2P_DISBLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_LL_THRESHOLD=0
# export NCCL_DEBUG=info

export TRANSFORMERS_CACHE=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

accelerate launch --multi-gpu \
                  --num_processes=4 \
                  --num_machines=1 \
                  --mixed-precision=no \
                  --dynamo_backend=no \
                   main.py --task train \
                   --exp_name rlp_clip \
                   --config configs/ddpmpp/isic/rlp_clip.yaml \
                   --num_workers 8 \
                   --learn_obj ddpmpp \
                   --distribution_training


# accelerate launch --multi-gpu \
#                   --main_process_ip=$MASTER_ADDR \
#                   --main_process_port=$MASTER_PORT \
#                   --num_processes=4 \
#                   --num_machines=1 \
#                   --mixed-precision=no \
#                   --dynamo_backend=no \
#                    main.py --task train \
#                    --exp_name flow_with_clip \
#                    --config configs/isic_clipcam_flow.yaml \
#                    --num_workers 4 \
#                    --distribution_training
