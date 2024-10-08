#!/bin/bash

export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=$1
config_file=$2


# Ask the user if they want to enable distributed training
read -p "Enable distributed training? (y/n): " DISTRIBUTED_TRAINING

# Check if distributed training is enabled
if [ "$DISTRIBUTED_TRAINING" == "y" ]; then
    echo "Distributed training enabled. Running distributed training commands..."
    accelerate launch --multi-gpu --num_processes=2 --num_machines=1 --mixed-precision=no --dynamo-backend=no  main.py --distribution_training --task train --config configs/${config_file} --exp_name ${exp_name}
else
    echo "Distributed training not enabled. Running single-node training commands..."
    export CUDA_VISIBLE_DEVICES=1
    python main.py --task train --config configs/${config_file} --exp_name ${exp_name} --device cuda
fi