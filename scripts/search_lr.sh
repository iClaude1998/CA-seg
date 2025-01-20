#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips


config_file=$1
learn_obj=$2


python main.py --task lr_search \
               --config ${config_file} \
               --device cuda \
               --learn_obj ${learn_obj} \
               --load_checkpoint