import os
import yaml
import torch 

from tqdm import tqdm
from datasets import build_dataset
from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from models.clips import ClipCBN
from models.build_models import load_clip_and_tokenizer, create_diffusion



def parse_args():
    parser = ArgumentParser(description='Reflow')
    parser.add_argument('--task', type=str, default='rlp', help='the task to performs', choices=['rlp', 'cam'])
    parser.add_argument('--vis_layer', type=int, default=-1, help='visualization layer')
    parser.add_argument('--config', type=str, default='configs/isic_clip.yaml', help='path to config file')
    parser.add_argument('--exp_name', type=str, default='debug', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--run_diffusion', action="store_true", help='whether run diffusion')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    config_file = args.config
    save_dir = os.path.join("experiments", args.exp_name, "visualizations") # "experiments/cam_test/visualizations"
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    model = ClipCBN("PubMedCLIP", 45).to("cuda")
    dummy = torch.randn(4, 3, 224, 224).to("cuda")
    
    out = model(dummy)
    print(out.size())
    
    
    
    # config = edict(config_dict)
    
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(config.model.clip, device)
    # diffusion_model = create_diffusion(config.model.diffusion).to(device)

    # # unet = UNetModel_v1preview(config.model.unet)
    # dataset = build_dataset(config.datasets.test, [preprocess, tokenizer, resolution])
    # dataloader = DataLoader(dataset, batch_size=config.datasets.batch_size, shuffle=False)
    
    # for batch in tqdm(dataloader):
    #     image = batch['pixel_values'].to(device)
    #     text_ids = batch['input_ids'].to(device)
    #     sdf_mask = batch['sdf_map'].to(device)
    #     mask = batch['mask'].to(device)
    #     Rs = batch.get("inter_map", None)
    #     bz = image.size(0)
    #     h, w = image.size(-2), image.size(-1)

        

    
    
    
    

        
        
    
    

    





