import os
import torch
import random
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from argparse import ArgumentParser
from easydict import EasyDict as edict



from trainers import build_trainer
from utils import build_dataloaders
from models.build_models import load_clipcbn_preprocessor


# Setting reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = ArgumentParser(description='Reflow')
    parser.add_argument('--config', type=str, default='configs/cbm_amos22/amos22_stomach.yaml', help='path to config file')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='amos22_stomach', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    return parser.parse_args()


if __name__ == '__main__':
    
    task = 1
    if task == 0:
    
        args = parse_args()
        config_file = args.config
        
        cfgs = OmegaConf.load(config_file)
        cfgs = edict(cfgs)
        
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None:  # Update only if argument is provided
                cfgs[key] = value
        cfgs.learn_obj = 'cbm'
        cfgs.load_checkpoint = True
        cfgs.task = 'test'
        output_dir = os.path.join('experiments', cfgs.learn_obj, cfgs.datasets.train.name ,cfgs.exp_name)   
        
        cliprlp, tokenizer, preprocess, resolution = load_clipcbn_preprocessor(cfgs.model.clip)
        train_dl, val_dl, test_dl, num_training_samples, num_val_samples, num_test_samples = build_dataloaders(cfgs, preprocess, tokenizer, resolution, 1)
        
        dataloader_pakages = {'train': train_dl, 'val': val_dl, 'test': test_dl}
        
        trainer = build_trainer(cfgs, output_dir, cliprlp, None, dataloader_pakages, None, cfgs.device)
        # outcomes = trainer.test('train')
        # thresh = outcomes['dice_II'].mean()
        
        # print(f'Threshold: {thresh}')
        thresh = 0.3

        trainer.filter_data(thresh)
        
    elif task == 1:
        
        exps = ["amos22_aorta", "amos22_duodenum", "amos22_esophagus", "amos22_gallbladder", "amos22_left+adrenal+gland", 
                "amos22_left+kidney", "amos22_liver", "amos22_pancreas", "amos22_postcava", "amos22_right+adrenal+gland", 
                "amos22_right+kidney", "amos22_spleen", "amos22_stomach"]
        csv_files = [f'experiments/cbm/bioparse_amos22/{exp}/output_logs/filtered_infos.csv' for exp in exps]
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv("annotation.csv", index=False)
        print(df.head())
        
        
        
        
    
    
    
    
    
    
    
    
