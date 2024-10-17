#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=$1
config_file=$2

python main.py --task inf --config configs/${config_file} --exp_name ${exp_name} --device cuda --load_checkpoint


