#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips


scripts/test.sh pmc_amos22_aorta configs/cbm_amos22/pmc_amos22_aorta.yaml test cbm
scripts/test.sh pmc_amos22_duodenum configs/cbm_amos22/pmc_amos22_duodenum.yaml test cbm
scripts/test.sh pmc_amos22_esophagus configs/cbm_amos22/pmc_amos22_esophagus.yaml test cbm
scripts/test.sh pmc_amos22_gallbladder configs/cbm_amos22/pmc_amos22_gallbladder.yaml test cbm
scripts/test.sh pmc_amos22_left+adrenal+gland configs/cbm_amos22/pmc_amos22_left+adrenal+gland.yaml test cbm
scripts/test.sh pmc_amos22_left+kidney configs/cbm_amos22/pmc_amos22_left+kidney.yaml test cbm
scripts/test.sh pmc_amos22_liver configs/cbm_amos22/pmc_amos22_liver.yaml test cbm
scripts/test.sh pmc_amos22_pancreas configs/cbm_amos22/pmc_amos22_pancreas.yaml test cbm
scripts/test.sh pmc_amos22_postcava configs/cbm_amos22/pmc_amos22_postcava.yaml test cbm
scripts/test.sh pmc_amos22_right+adrenal+gland configs/cbm_amos22/pmc_amos22_right+adrenal+gland.yaml test cbm
scripts/test.sh pmc_amos22_right+kidney configs/cbm_amos22/pmc_amos22_right+kidney.yaml test cbm
scripts/test.sh pmc_amos22_spleen configs/cbm_amos22/pmc_amos22_spleen.yaml test cbm
scripts/test.sh pmc_amos22_stomach configs/cbm_amos22/pmc_amos22_stomach.yaml test cbm