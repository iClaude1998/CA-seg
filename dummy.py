import os
import yaml
import torch 

from tqdm import tqdm 
from datasets import build_dataset
from torch.nn import functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader


from utils.vis import vis_batch
from argparse import ArgumentParser
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
    
    config = edict(config_dict)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
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

        if args.task == 'rlp':
            Rs, intermediate = cliprlp(image, text_ids)
            bz = Rs.size(0)    
        elif args.task == 'cam':
            Rs = cliprlp.clip_model.produce_cam(image, text_ids, vis_layer=args.vis_layer)
        
        R_h = int(Rs[0].numel() ** 0.5)
        Rs = Rs.view(bz, 1, R_h, R_h)
        Rs = F.interpolate(Rs, (h, w), mode='bilinear', align_corners=False)
        vis_batch(batch, save_dir, Rs)   
        
        # layer 3 7 11
        if args.run_diffusion:
            ts = torch.randint(1, 1000, (bz,), device=device).long()
            if config.model.diffusion.version == 'v1':
                x = torch.cat([image, Rs], dim=1)
                out = diffusion_model(x, ts, y=None)
            elif config.model.diffusion.version == 'v2':
                out = diffusion_model(Rs, ts, intermediate.detach())
        

    
    
    
    

        
        
    
    

    





