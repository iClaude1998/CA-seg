#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=$1
config_file=$2
learn_obj=$3

python main.py --task inf \
               --exp_name ${exp_name} \
               --config configs/${config_file} \
               --device cuda \
               --learn_obj ${learn_obj} \
               --load_checkpoint


