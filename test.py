import os
import yaml
import torch 
import open_clip 

from tqdm import tqdm 
from datasets import build_dataset
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from utils.vis import vis_batch
from models.clips import CLIPWrapper, CLIPLRP





if __name__ == '__main__':
    
    config_file = "configs/isic_clip.yaml"
    save_dir = "outputs"
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    edict_config = edict(config_dict)
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model, _ , preprocess = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
    tokenizer = open_clip.get_tokenizer('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPLRP(CLIPWrapper(model), device)
    
    val_dataset = build_dataset(edict_config.datasets.val, [preprocess, tokenizer])
    dataloader = DataLoader(val_dataset, batch_size=edict_config.datasets.batch_size, shuffle=False)
    
    for batch in tqdm(dataloader):
        image = batch['pixel_values']
        text_ids = batch['input_ids']
        Rs = model(image.to(device), text_ids.to(device))
        bz = Rs.size(0)
        R_h = int(Rs[0].numel() ** 0.5)
        Rs = Rs.view(bz, R_h, R_h)
        vis_batch(batch, save_dir, Rs)
    
    
    
    

        
        
    
    

    





