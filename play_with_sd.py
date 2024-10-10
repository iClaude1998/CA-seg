import os 
import yaml
import torch
import base64
import requests
import open_clip

from PIL import Image 
from datasets import build_dataset
from easydict import EasyDict as edict
from torch.utils.data import DataLoader



if __name__ == "__main__":
    config_file = "configs/isic_clip.yaml"
    save_dir = "experiments"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) 
        
    os.makedirs(save_dir, exist_ok=True)
    
    edict_config = edict(config_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    dataset = build_dataset(edict_config.datasets.val, [preprocess, tokenizer])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    example = dataset[0]
    path = example['img_path']
    sentence = example['sentence']
    
    with open(path, "rb") as image:
        files = {"image": image}
        data = {"prompt": f"The segmentation mask of {sentence}", "output_format": "jpeg", "mode": "image-to-image", "strength": 0}
        headers = {"authorization": f"Bearer sk-GLIWRJzpsbbCK8q7kmtJBKN8sahU3TmwRTSidO1neYfBzQmL",
                   "accept": "image/*"}
        response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers=headers,
        files=files,
        data=data,
        )

    if response.status_code == 200:
        with open("./lighthouse.jpeg", 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(str(response.json()))
