#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips


scripts/produce_cam.sh amos22_aorta configs/cbm_amos22/amos22_aorta.yaml cbm
scripts/produce_cam.sh amos22_duodenum configs/cbm_amos22/amos22_duodenum.yaml cbm
scripts/produce_cam.sh amos22_esophagus configs/cbm_amos22/amos22_esophagus.yaml cbm
scripts/produce_cam.sh amos22_gallbladder configs/cbm_amos22/amos22_gallbladder.yaml cbm
scripts/produce_cam.sh amos22_left+adrenal+gland configs/cbm_amos22/amos22_left+adrenal+gland.yaml cbm
scripts/produce_cam.sh amos22_left+kidney configs/cbm_amos22/amos22_left+kidney.yaml cbm
scripts/produce_cam.sh amos22_liver configs/cbm_amos22/amos22_liver.yaml cbm
scripts/produce_cam.sh amos22_pancreas configs/cbm_amos22/amos22_pancreas.yaml cbm
scripts/produce_cam.sh amos22_postcava configs/cbm_amos22/amos22_postcava.yaml cbm
scripts/produce_cam.sh amos22_right+adrenal+gland configs/cbm_amos22/amos22_right+adrenal+gland.yaml cbm
scripts/produce_cam.sh amos22_right+kidney configs/cbm_amos22/amos22_right+kidney.yaml cbm
scripts/produce_cam.sh amos22_spleen configs/cbm_amos22/amos22_spleen.yaml cbm
scripts/produce_cam.sh amos22_stomach configs/cbm_amos22/amos22_stomach.yaml cbm