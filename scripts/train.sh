#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=$1
config_file=$2
learn_obj=$3


# Ask the user if they want to enable distributed training
read -p "Enable distributed training? (y/n): " DISTRIBUTED_TRAINING

# Check if distributed training is enabled
if [ "$DISTRIBUTED_TRAINING" == "y" ]; then
    echo "Distributed training enabled. Running distributed training commands..."
    accelerate launch --multi-gpu --num_processes=2 --num_machines=1 --mixed-precision=no --dynamo_backend=no main.py --exp_name ${exp_name} --task train --config ${config_file} --num_workers 4 --learn_obj ${learn_obj} --distribution_training
else
    echo "Distributed training not enabled. Running single-node training commands..."
    python main.py --exp_name ${exp_name} --task train --config ${config_file} --num_workers 4 --learn_obj ${learn_obj}
fi