import os
import torch
import random
import numpy as np
import pandas as pd 

from omegaconf import OmegaConf
from torch.backends import cudnn
from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

from datasets import build_dataset
from trainers import build_trainer
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
    parser.add_argument('--infer_algo', type=str, default='ddpm', help='path to config file')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='debug', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--learn_obj', type=str, default='recflow', choices=['recflow', 'ddpm', 'ddpmpp', 'recflowturb'], help='the learning objective')
    parser.add_argument('--distribution_training', action="store_true", help='whether enable distribution training')
    parser.add_argument('--load_checkpoint', action="store_true", help='whether to load checkpoint')
    parser.add_argument('--test_type', type=str, default='test', help='whether to load checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    config_file = args.config
    
    # with open(config_file, 'r', encoding='utf-8') as f:
    #     cfgs = yaml.safe_load(f)
    
    cfgs = OmegaConf.load(config_file)
    cfgs = edict(cfgs)
    
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:  # Update only if argument is provided
            cfgs[key] = value
    
    
    output_dir = os.path.join('experiments', cfgs.learn_obj, cfgs.datasets.test.name ,cfgs.exp_name)        
    if cfgs.distribution_training:
        cudnn.benchmark = True
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=cfgs.trainer.gradient_accumulation_steps)
        device = accelerator.device
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
    else:
        os.makedirs(output_dir, exist_ok=True)
        device = cfgs.device
        accelerator = None
    
    cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(cfgs.model.clip, 'cpu')
    diffusion_model = create_diffusion(cfgs.model.diffusion)
    
    train_dataset = build_dataset(cfgs.datasets.train, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
    val_dataset = build_dataset(cfgs.datasets.val, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
    test_dataset = build_dataset(cfgs.datasets.test, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
    
    train_dl = DataLoader(train_dataset, batch_size=cfgs.datasets.batch_size, num_workers=cfgs.num_workers, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    
    dataloader_pakages = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    
    distribution_training = cfgs.distribution_training and cfgs.task == 'train'
    
    trainer = build_trainer(cfgs, output_dir, cliprlp, diffusion_model, dataloader_pakages, accelerator, device)
    
    if cfgs.task == 'train':
        if cfgs.distribution_training:
            trainer.distribution_train()
        else:
            trainer.train(cfgs.trainer.gradient_accumulation_steps)
    elif cfgs.task == 'inf':
        trainer.inference() # inference the results for visualization
    elif cfgs.task == 'test':
        outcomes = trainer.test(cfgs.test_type) # test the model on the test set /validation set
        numeric_outcomes = outcomes.select_dtypes(include='number')
        columns_means = numeric_outcomes.mean()
        res = (columns_means * 100).round(2)
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_dir, 'output_logs', f'{cfgs.test_type}_results.csv'))
<<<<<<< HEAD

=======
>>>>>>> 746be98a9cca6855a5cd96a9fdde8a41a40643bf
    else:
        raise ValueError(f"Unsupported task: {cfgs.task}, what do you wanna do ???")
        
    
    
    
     
    
    
    