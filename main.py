import os
import yaml
import torch
import random
import numpy as np

from torch.backends import cudnn
from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from datasets import build_dataset
from trainers import Reflow_Trainer
from accelerate import Accelerator, DistributedDataParallelKwargs
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
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='debug', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--learn_obj', type=str, default='recflow', choices=['recflow', 'ddpm'], help='the learning objective')
    parser.add_argument('--distribution_training', action="store_true", help='whether enable distribution training')
    parser.add_argument('--load_checkpoint', action="store_true", help='whether to load checkpoint')
    parser.add_argument('--test_type', type=str, default='test', help='whether to load checkpoint')
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
    
            
    if cfgs.distribution_training:
        cudnn.benchmark = True
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        device = accelerator.device
        if accelerator.is_main_process:
            os.makedirs(os.path.join('experiments', cfgs.exp_name), exist_ok=True)
        
    else:
        os.makedirs(os.path.join('experiments', cfgs.exp_name), exist_ok=True)
        device = cfgs.device
        accelerator = None
    
    cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(cfgs.model.clip, 'cpu')
    diffusion_model = create_diffusion(cfgs.model.diffusion)
    
    train_dataset = build_dataset(cfgs.datasets.train, [preprocess, tokenizer, resolution])
    val_dataset = build_dataset(cfgs.datasets.val, [preprocess, tokenizer, resolution])
    test_dataset = build_dataset(cfgs.datasets.test, [preprocess, tokenizer, resolution])
    
    train_dl = DataLoader(train_dataset, batch_size=cfgs.datasets.batch_size, num_workers=cfgs.num_workers, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
    
    dataloader_pakages = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    
    distribution_training = cfgs.distribution_training and cfgs.task == 'train'
    if cfgs.learn_obj == 'recflow':
        trainer = Reflow_Trainer(cfgs.model.diffusion.version,
                                cfgs.task,
                                cfgs.exp_name, 
                                cliprlp, 
                                diffusion_model,
                                dataloader_pakages,
                                cfgs.trainer.learning_rate,
                                cfgs.trainer.gt_type,
                                device,
                                cfgs.trainer.use_ema,
                                cfgs.load_checkpoint,
                                cfgs.trainer.checkpoint_name,
                                cfgs.trainer.num_timesteps,
                                cfgs.trainer.num_iterations,
                                cfgs.trainer.save_interval,
                                accelerator,
                                cfgs.log_method,
                                cfgs.trainer.start_point)
    elif cfgs.learn_obj == 'ddpm':
        raise NotImplementedError("DDPM is not implemented yet, let's do it")
    else:
        raise ValueError(f"Unsupported learning objective: {cfgs.learn_obj}, what do you wanna do ???")
    
    if cfgs.task == 'train':
        if cfgs.distribution_training:
            trainer.distribution_train()
        else:
            trainer.train()
    elif cfgs.task == 'inf':
        trainer.inference() # inference the results for visualization
    elif cfgs.task == 'test':
        outcomes = trainer.test(cfgs.test_type) # test the model on the test set /validation set
        numeric_outcomes = outcomes.select_dtypes(include='number')
        columns_means = numeric_outcomes.mean()
        print((columns_means * 100).round(2))
    else:
        raise ValueError(f"Unsupported task: {cfgs.task}, what do you wanna do ???")
        
    
    
    
     
    
    
    