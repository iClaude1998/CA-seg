#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=$(pwd)/pretrained/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/pretrained/huggingface_hub
export XDG_CACHE_HOME=$(pwd)/pretrained/clips

# for vis_layer in {0..11}
# do
#     if ! python dummy.py --task cam --vis_layer ${vis_layer} --config configs/vis_cam.yaml --exp_name cam_test_${vis_layer} --device cuda; then
#         echo "Error: Command failed for vis_layer ${vis_layer}"
#         exit 1
#     fi
# done

vis_layer=11
python dummy.py --task rlp \
                --vis_layer ${vis_layer} \
                --config configs/isic_clipcam_highres.yaml \
                --exp_name cam_test_${vis_layer} \
                --device cuda \
                # --run_diffusion