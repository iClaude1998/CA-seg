#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

python main.py --exp_name amos22_aorta --task train --config configs/cbm_bioparse/amos22_aorta.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_duodenum --task train --config configs/cbm_bioparse/amos22_duodenum.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_esophagus --task train --config configs/cbm_bioparse/amos22_esophagus.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_gallbladder --task train --config configs/cbm_bioparse/amos22_gallbladder.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_left+adrenal+gland --task train --config configs/cbm_bioparse/amos22_left+adrenal+gland.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_left+kidney --task train --config configs/cbm_bioparse/amos22_left+kidney.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_liver --task train --config configs/cbm_bioparse/amos22_liver.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_pancreas --task train --config configs/cbm_bioparse/amos22_pancreas.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_postcava --task train --config configs/cbm_bioparse/amos22_postcava.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_right+adrenal+gland --task train --config configs/cbm_bioparse/amos22_right+adrenal+gland.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_right+kidney --task train --config configs/cbm_bioparse/amos22_right+kidney.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_spleen --task train --config configs/cbm_bioparse/amos22_spleen.yaml --num_workers 4 --learn_obj cbm
python main.py --exp_name amos22_stomach --task train --config configs/cbm_bioparse/amos22_stomach.yaml --num_workers 4 --learn_obj cbm
