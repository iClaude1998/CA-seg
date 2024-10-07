import os
import yaml
import torch
import random
import numpy as np

from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from datasets import build_dataset
from trainer import Reflow_ControlLDM
from models.build_models import load_clip_and_tokenizer, create_diffusion


# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = ArgumentParser(description='Reflow')
    parser.add_argument('--task', type=str, default='train', help='the task to performs', choices=['train', 'inf', 'test'])
    parser.add_argument('--config', type=str, default='configs/isic_clip.yaml', help='path to config file')
    parser.add_argument('--exp_name', type=str, default='debug', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--distribution_training', type=bool, default=False, help='whether enable distribution training')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    config_file = args.config
    
    with open(config_file, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
    
    cfgs = edict(cfgs)
    
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:  # Update only if argument is provided
            cfgs[key] = value
    
    os.makedirs(os.path.join('experiments', cfgs.exp_name), exist_ok=True)
    
    cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(cfgs.model.clip, 'cpu')
    diffusion_model = create_diffusion(cfgs.model.diffusion)
    
    train_dataset = build_dataset(cfgs.datasets.train, [preprocess, tokenizer, resolution])
    val_dataset = build_dataset(cfgs.datasets.val, [preprocess, tokenizer, resolution])
    test_dataset = build_dataset(cfgs.datasets.test, [preprocess, tokenizer, resolution])
    
    train_dl = DataLoader(train_dataset, batch_size=cfgs.datasets.batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    
    dataloader_pakages = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    
    distribution_training = cfgs.distribution_training and cfgs.task == 'train'
    trainer = Reflow_ControlLDM(cfgs.task,
                                cfgs.exp_name, 
                                cliprlp, 
                                diffusion_model,
                                dataloader_pakages,
                                cfgs.trainer.learning_rate,
                                cfgs.device,
                                cfgs.trainer.use_ema,
                                cfgs.trainer.checkpoint_name,
                                cfgs.trainer.num_timesteps,
                                cfgs.trainer.num_iterations,
                                cfgs.trainer.save_interval,
                                distribution_training,
                                cfgs.log_method)
    
    if cfgs.task == 'train':
        if cfgs.distribution_training:
            trainer.distribution_train()
        else:
            trainer.train()
    elif cfgs.task == 'inf':
        trainer.inference() # inference the results for visualization
    elif cfgs.task == 'test':
        pass #TODO: implement test for quantitive evaluation 
    else:
        raise ValueError(f"Unsupported task: {cfgs.task}, what do you wanna do ???")
        
    
    
    
     
    
    
    