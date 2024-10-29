#!/bin/bash

module load NCCL/2.12.12-GCCcore-11.2.0-CUDA-11.6.2
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0


conda info --envs
nvcc -V

export MASTER_PORT=12340
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr-ib
echo "MASTER_ADDR="$MASTER_ADDR
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISBLE=1
export NCCL_IB_DISABLE=1
export NCCL_LL_THRESHOLD=0
export NCCL_DEBUG=info


export TRANSFORMERS_CACHE=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

accelerate launch --multi-gpu \
                  --main_process_ip=$MASTER_ADDR \
                  --main_process_port=$MASTER_PORT \
                  --num_processes=2 \
                  --num_machines=1 \
                  --mixed-precision=no \
                  --dynamo_backend=no \
                   main.py --task train \
                   --exp_name flow_with_clip \
                   --config configs/flow_noise.yaml \
                   --num_workers 4 \
                   --distribution_training