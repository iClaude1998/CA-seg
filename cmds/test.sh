#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

python main.py --task inf --config configs/isic_clip.yaml --exp_name imagediffuse --device cuda 


