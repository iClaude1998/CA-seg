import os
import torch
import random
import numpy as np
import pandas as pd 

from torch.optim import Adam
from omegaconf import OmegaConf
from torch.backends import cudnn
from argparse import ArgumentParser
from easydict import EasyDict as edict
from trainers.cbm_trainer import DiceLosswithRegularizer
from accelerate import Accelerator, DistributedDataParallelKwargs

from datasets import build_dataset
from trainers import build_trainer
from utils import LearningRateFinder, build_dataloaders
from models.build_models import load_clip_and_tokenizer, create_diffusion, load_clipcbn_preprocessor


# Setting reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = ArgumentParser(description='Reflow')
    parser.add_argument('--task', type=str, default='train', help='the task to performs', 
                        choices=['train', 'inf', 'test', 'vis_process', 'thresh_search', 'lr_search', 'produce_cam'])
    parser.add_argument('--config', type=str, default='configs/cbm_bioparse/covid_left+lung_1l.yaml', help='path to config file')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='covid_left+lung_1l', help='the name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='experiment device')
    parser.add_argument('--learn_obj', type=str, default='cbm', choices=['recflow', 'ddpm', 'ddpmpp', 'recflowturb', 'cbm'], help='the learning objective')
    parser.add_argument('--distribution_training', action="store_true", help='whether enable distribution training')
    parser.add_argument('--load_checkpoint', action="store_true", help='whether to load checkpoint')
    parser.add_argument('--test_type', type=str, default='test', help='The test dataset')
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
    
    
    output_dir = os.path.join('experiments', cfgs.learn_obj, cfgs.datasets.train.name ,cfgs.exp_name)        
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
    
    if cfgs.learn_obj != 'cbm':
        cliprlp, tokenizer, preprocess, resolution = load_clip_and_tokenizer(cfgs.model.clip, 'cpu')
        diffusion_model = create_diffusion(cfgs.model.diffusion)
    else:
        cliprlp, tokenizer, preprocess, resolution = load_clipcbn_preprocessor(cfgs.model.clip)
        diffusion_model = None
    
    train_dl, val_dl, test_dl, num_training_samples, num_val_samples, num_test_samples = build_dataloaders(cfgs, preprocess, tokenizer, resolution)
    
    dataloader_pakages = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    
    distribution_training = cfgs.distribution_training and cfgs.task == 'train'
    
    if cfgs.task == 'lr_search':
        params = list(cliprlp.concept_head.parameters())
        lr = cfgs.trainer.learning_rate
        optimizer = Adam(params, lr=lr)

        criterion = DiceLosswithRegularizer(0.33, 0.33, reduction='mean', with_sigmoid=cfgs.trainer.with_sigmoid)
        
        lr_searcher = LearningRateFinder(cliprlp, optimizer, criterion, device=cfgs.device)
        lr_searcher.range_test(train_dl, val_dl, num_training_samples, num_val_samples, start_lr=1e-5, end_lr=1e-3, num_iter=1000, step_mode="linear")
        best_lr, lossval = lr_searcher.get_steepest_gradient()
        print(f"best learning rate: {best_lr} in loss: {lossval}")
    else:
        trainer = build_trainer(cfgs, output_dir, cliprlp, diffusion_model, dataloader_pakages, accelerator, device)
        
        if cfgs.task == 'train':
            if cfgs.distribution_training:
                trainer.distribution_train()
            else:
                trainer.train(cfgs.trainer.gradient_accumulation_steps)
                
        elif cfgs.task == 'inf':
            trainer.inference() # inference the results for visualization
            
        elif cfgs.task == 'test':
            outcomes = trainer.test(cfgs.test_type) # test the model on the test set / validation set
            numeric_outcomes = outcomes.select_dtypes(include='number')
            columns_means = numeric_outcomes.mean()
            res = (columns_means * 100).round(2)
            print(res)
            res = pd.DataFrame(res)
            res.to_csv(os.path.join(output_dir, 'output_logs', f'{cfgs.test_type}_results.csv'))
            
        elif cfgs.task == 'vis_process':
            trainer.random_inference_process()
            
        elif cfgs.task == 'thresh_search' and hasattr(trainer, 'thresh_search'):
            best_thresh, best_outcome = trainer.thresh_search('dice')
            print(f'Best Threshold: {best_thresh} with Dice: {best_outcome}')
        elif cfgs.task == 'produce_cam':
            if not hasattr(trainer, 'produce_cam'):
                raise ValueError(f"Unsupported task: {cfgs.task} due to the trainer doesn't have the attribute ???")
            train_outdir, val_outdir, test_outdir = cfgs.cbm_produce.train_dir, cfgs.cbm_produce.val_dir, cfgs.cbm_produce.test_dir
            if train_outdir is not None:
                os.makedirs(train_outdir, exist_ok=True)
            if val_outdir is not None:
                os.makedirs(val_outdir, exist_ok=True)
            if test_outdir is not None:
                os.makedirs(test_outdir, exist_ok=True)
            trainer.produce_cam(train_outdir, val_outdir, test_outdir, False)
        else:
            raise ValueError(f"Unsupported task: {cfgs.task}, what do you wanna do ???")
        
    
    
    
     
    
    
    