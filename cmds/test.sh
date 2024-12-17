#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

exp_name=$1
config_file=$2
test_type=$3
learn_obj=$4


python main.py --task test \
               --exp_name ${exp_name} \
               --config ${config_file} \
               --device cuda \
               --test_type ${test_type} \
               --learn_obj ${learn_obj} \
               --load_checkpoint