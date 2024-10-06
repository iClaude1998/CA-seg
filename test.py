import os
import yaml
import torch 
import clip 
import open_clip 

from tqdm import tqdm 
from datasets import build_dataset
from torch.nn import functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

from utils.vis import vis_batch
from utils.img_process import interpolate_cam
from models.diffusion import UNetModel_v1preview
from models.build_models import load_clip_and_tokenizer, create_diffusion


# Set the custom cache directory for pretrained models, tokenziers, etc.
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "pretrained", "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.getcwd(), "pretrained", "huggingface_hub")



if __name__ == '__main__':
    
    config_file = "configs/isic_clip.yaml"
    save_dir = "experiments"
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    config = edict(config_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(config.model.clip, device)
    diffusion_model = create_diffusion(config.model.diffusion).to(device)

    # unet = UNetModel_v1preview(config.model.unet)
    val_dataset = build_dataset(config.datasets.val, [preprocess, tokenizer, resolution])
    dataloader = DataLoader(val_dataset, batch_size=config.datasets.batch_size, shuffle=False)
    
    for batch in tqdm(dataloader):
        image = batch['pixel_values'].to(device)
        text_ids = batch['input_ids'].to(device)
        sdf_mask = batch['sdf_map'].to(device)
        mask = batch['mask'].to(device)
        bz = image.size(0)
        h, w = image.size(-2), image.size(-1)
        timesteps = torch.randint(0, 1000, (bz,)).to(device)
        
        Rs, intermediate = cliprlp(image, text_ids)
        bz = Rs.size(0)
        R_h = int(Rs[0].numel() ** 0.5)
        Rs = Rs.view(bz, 1, R_h, R_h)
        Rs = F.interpolate(Rs, (h, w), mode='bilinear', align_corners=False)
        
        # x = torch.cat([image, Rs], dim=1)
        # out = diffusion_model(x, timesteps, y=None)
        
        break


        # vis_batch(batch, save_dir, Rs)
    
    
    
    

        
        
    
    

    





