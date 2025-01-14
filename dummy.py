import os
import yaml
import torch 

from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import build_dataset
from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from models.build_models import load_clipcbn_preprocessor



def parse_args():
    parser = ArgumentParser(description='Reflow')
    parser.add_argument('--config', type=str, default='configs/isic_clip.yaml', help='path to config file')
    parser.add_argument('--exp_name', type=str, default='debug', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--run_diffusion', action="store_true", help='whether run diffusion')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    config_file = args.config
    
    cfgs = OmegaConf.load(config_file)
    cfgs = edict(cfgs)
    
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:  # Update only if argument is provided
            cfgs[key] = value
    
    save_dir = os.path.join("experiments", cfgs.exp_name, "visualizations") # "experiments/cam_test/visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    preprocess, model, tokenizer, resolution = load_clipcbn_preprocessor(cfgs.model)
    model = model.to(args.device)
    dataset = build_dataset(cfgs.datasets.test, [preprocess, tokenizer, resolution])
    dataloader = DataLoader(dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    abatch = next(iter(dataloader))  
    
    features = model(abatch['pixel_values'].to(args.device))
    print(features.shape)
    
    

        

    
    
    
    

        
        
    
    

    





