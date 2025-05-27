import os
import sys
sys.path.append('..')
import time
import statistics
import torch
import random
import logging
import torchvision
import numpy as np
import pandas as pd
import importlib.util
import torch.distributed as dist
if importlib.util.find_spec('wandb') is not None:
    import wandb

from tqdm import tqdm
from torch.optim import AdamW
from monai.losses import DiceLoss
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from torch.nn import functional as F
from contextlib import contextmanager
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict as dedict


from torch.utils.tensorboard import SummaryWriter


from models.diffusion import LitEma
from utils import process_checkpoints, mix_images_with_masks, save_batch, compute_metrics, import_or_skip
wandb = import_or_skip('wandb')


to_pil = transforms.ToPILImage()

class Dice_Trainer(object):
    
    def __init__(
        self,
        task,
        output_dir,
        model,
        dataloaders,
        learning_rate,
        gt_type='sdf_map',
        device='cuda',
        use_ema=False,
        load_checkpoint=False,
        checkpoint_name=None,
        num_epochs=100,
        save_interval=100,
        accelerator=None,
        log_method='wandb',
        clip_grads=None,
        with_codition=False,
    ):
        # instantiate control module
        self.output_dir = output_dir
        self.task = task
        self.model = model
        self.learning_rate = learning_rate
        self.gt_type = gt_type
        self.log_method = log_method
        self.criterion = DiceLoss(reduction='mean', sigmoid=True)
        self.device = device
        self.use_ema = use_ema
        
        self.with_codition = with_codition
        self.start_epoch = 0
        self.clip_grads = clip_grads

        
        self.create_exp_name()
        self.log_path = os.path.join(output_dir, 'output_logs')
        self.checkpoint_path = os.path.join(output_dir, 'checkpoints')
        self.vis_path = os.path.join(output_dir, 'visualizations') 
        self.vis_process_path = os.path.join(output_dir, 'vis_process') 
        
        # I have to seperate the branches
        if accelerator is None:
            self.create_output_dirs()
            self.init_loggers()
            
        if use_ema:
            self.model_ema = LitEma(self.model)
            
        if checkpoint_name is not None and load_checkpoint:
            self.load_succeed = self.load_checkpoint(checkpoint_name)
            self.checkpoint_name = os.path.splitext(checkpoint_name)[0]
        else:
            self.load_succeed = False
        self.unzip_dataloaders(dataloaders)
        
        self.accelerator = accelerator
        
        self.models_to_device()
        self.optimizer = self.configure_optimizers()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        
        # I have to seperate the branches
        if self.accelerator is not None:
            self.distribution_init()
            if self.accelerator.is_local_main_process:
                self.create_output_dirs()
                self.init_loggers()
            
    
    def create_output_dirs(self):

        os.makedirs(self.log_path, exist_ok=True)  
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        os.makedirs(self.vis_process_path, exist_ok=True)
    
    def create_exp_name(self):
        _, learn_obj, dataset_name, exp_name = self.output_dir.split('/')
        self.exp_name = f"{learn_obj}-{dataset_name}-{exp_name}"
    
    
    def train(self, gradient_accumulation_steps=1):
        
        self.model.train()
        best_metrics = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            losses = []
            iter_id = 0
            for batch in tqdm(self.train_dataloader):

                loss = self.training_step(batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if iter_id % gradient_accumulation_steps == 0:
                    if self.clip_grads is not None:
                        clip_grad_norm_(self.model.parameters(), self.clip_grads)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if self.use_ema:
                    self.model_ema(self.model)
                
                iter_id += 1
                losses.append(loss.detach().cpu().item())
            
            # after each epoch, do evaluation
            self.logger.info(f'Epoch [{epoch}/{self.num_epochs}], Loss: {statistics.mean(losses):.4f}')
            evaluation_outcomes = self.evaluation()
            # log infos
            if self.log_method == 'wandb':
                wandb.log({'Training Loss': statistics.mean(losses), 'Epoch': epoch})
                for key, value in evaluation_outcomes.items():
                    wandb.log({f'Evaluation {key}': value, 'Epoch': epoch})
            elif self.log_method == 'tensorboard':
                self.writer.add_scalar('Training Loss', statistics.mean(losses), epoch)
                for key, value in evaluation_outcomes.items():
                    self.writer.add_scalar(key, value, epoch)
                
            # save the best checkpoints        
            if evaluation_outcomes['dice_II'] > best_metrics:
                best_metrics = evaluation_outcomes['dice_II']
                self.save_checkpoints(epoch, 'best')
                
            self.scheduler.step(evaluation_outcomes['dice_II'])    
            print(f"\rEpoch: {epoch}", end='', flush=True)
        
        # let's do the last inference and log
        self.logger.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        preds, random_batch = self.random_inference()
        self.visualize(preds, random_batch, epoch + 1)
        self.save_checkpoints(epoch + 1, 'last')
        
        if self.log_method == 'wandb':
            wandb.finish()
        elif self.log_method == 'tensorboard':
            self.writer.close()
    
    
    def distribution_train(self):
        
        if self.accelerator.is_local_main_process:
            best_metrics = 0
            if self.log_method == 'wandb':
                wandb.init(project='clipflow2', name=self.exp_name)
        self.model.train()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            losses = []
            if self.accelerator.is_local_main_process:
                pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch}/{self.num_epochs}")
                
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)
                    if self.clip_grads is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # gather loss from all processes for display
                gathered_loss = self.accelerator.gather(loss)
                mean_loss = torch.mean(gathered_loss)
                losses.append(mean_loss.detach().cpu().item())
                if self.use_ema:
                    self.model_ema(self.model)
                if self.accelerator.is_local_main_process:
                    pbar.update(1)
            
            self.accelerator.wait_for_everyone()
            evaluation_outcomes = self.evaluation()
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_local_main_process:
                dice = evaluation_outcomes['dice_II']
                dice = torch.tensor(dice, device=self.accelerator.device) 
            else:
                dice = torch.tensor(0.0, device=self.accelerator.device) 
            
            self.accelerator.wait_for_everyone()
            dist.broadcast(dice, src=0)
            self.scheduler.step(dice) 
                 
            if self.accelerator.is_local_main_process:
                self.logger.info(f'Epoch [{epoch}/{self.num_epochs}], Loss: {statistics.mean(losses):.4f}')
                if self.log_method == 'wandb':
                    wandb.log({'Training Loss': statistics.mean(losses), 'epoch': epoch})
                    for key, value in evaluation_outcomes.items():
                        wandb.log({f'Training {key}': value, 'epoch': epoch})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Training Loss', statistics.mean(losses), epoch)
                    for key, value in evaluation_outcomes.items():
                        self.writer.add_scalar(f'Training {key}', value, epoch)
                if evaluation_outcomes['dice_II'] > best_metrics:
                    best_metrics = evaluation_outcomes['dice_II']
                    self.save_checkpoints(epoch, name='best')
            
            self.accelerator.wait_for_everyone()
                
            if self.accelerator.is_local_main_process:
                pbar.close()
                print(f"\rEpoch: {epoch}", end='', flush=True)
                
        # perfect ending
        preds, random_batch = self.random_inference()
        self.save_checkpoints(epoch + 1, 'last')
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_local_main_process:
            self.logger.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {statistics.mean(losses):.4f}')
            self.visualize(preds, random_batch, epoch + 1)
            if self.log_method == 'wandb':
                wandb.finish() 
            elif self.log_method == 'tensorboard':
                self.writer.close()   
    
    
    def training_step(self, batch):
       
        images, gt, Rs = self.get_input(batch)
        if self.with_codition:
            preds = self.model(images, Rs)
        else:
            preds = self.model(images)
        loss = self.criterion(preds, gt)
        return loss   


    def inference(self):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        self.model.eval()
        for batch in tqdm(self.test_dataloader):
            preds, Rs = self.test_step(batch)
            images = batch['pixel_values']
            gts = batch["mask"]
            mask_names = batch['mask_name']
            # gt -> [B, H, W, C]
            with torch.no_grad():
                mixed_img_predits_I = mix_images_with_masks(images, Rs)
                mixed_img_predits_II = mix_images_with_masks(images, preds)
                mixed_img_gts = mix_images_with_masks(images, gts) 
                
                save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, self.vis_path)
                
    
    @torch.no_grad()
    def evaluation(self):

        self.model.eval()
        outcomes = dedict(list)

        dl = self.val_dataloader
        for batch in tqdm(dl):
            mask_name = batch['mask_name']
            gts = batch['mask']
            preds, _ = self.test_step(batch)
            
            if self.accelerator is not None:
                preds = self.accelerator.gather(preds)
                gts = self.accelerator.gather(gts)
            
            if self.accelerator is None:
                iou_batch_II = compute_metrics(preds, gts, mask_name, metric='iou') # stage II
                dice_batch_II = compute_metrics(preds, gts, mask_name, metric='dice') # stage II
                outcomes['iou_II'].extend(iou_batch_II)
                outcomes['dice_II'].extend(dice_batch_II)
            else:
                if self.accelerator.is_main_process:
                    iou_batch_II = compute_metrics(preds, gts, mask_name, metric='iou') # stage II
                    dice_batch_II = compute_metrics(preds, gts, mask_name, metric='dice') # stage II
                    outcomes['iou_II'].extend(iou_batch_II) 
                    outcomes['dice_II'].extend(dice_batch_II)
                
        
        if self.accelerator is not None:
            if self.accelerator.is_main_process:
                for key in outcomes.keys():
                    outcomes[key] = np.mean(outcomes[key])
        else:
            for key in outcomes.keys():
                outcomes[key] = np.mean(outcomes[key])
        self.model.train()
        return outcomes
    
    
    
    @torch.no_grad()
    def test(self, testset='test'):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        self.model.eval()
        outcomes = dedict(list)
        if testset == 'test':
            dl = self.test_dataloader
        elif testset == 'val':
            dl = self.val_dataloader
        else:
            dl = self.train_dataloader
        for batch in tqdm(dl):
            preds, Rs = self.test_step(batch)
            
            mask_name = batch['mask_name']
            gts = batch['mask']
            if Rs.shape[2:] != gts.shape[2:]:
                if len(Rs.shape) == 3:
                    Rs = Rs.unsqueeze(1)
                Rs = F.interpolate(Rs, gts.shape[-2:], mode='bilinear', align_corners=False)
            if Rs.shape[1] != gts.shape[1]:
                Rs = torch.repeat_interleave(Rs, gts.shape[1], dim=1)
            iou_batch_I = compute_metrics(Rs, gts, mask_name, metric='iou') # stage I
            iou_batch_II = compute_metrics(preds, gts, mask_name, metric='iou') # stage II
            
            dice_batch_I = compute_metrics(Rs, gts, mask_name, metric='dice') # stage I
            dice_batch_II = compute_metrics(preds, gts, mask_name, metric='dice') # stage II
            
            outcomes['mask_name'].extend(mask_name)
            outcomes['iou_I'].extend(iou_batch_I)
            outcomes['iou_II'].extend(iou_batch_II)
            outcomes['dice_I'].extend(dice_batch_I)
            outcomes['dice_II'].extend(dice_batch_II)
        outcomes = pd.DataFrame(outcomes)
        outcomes.to_csv(os.path.join(self.log_path, f'outcomes_{testset}_out.csv'), index=False)
        return outcomes
    
                    
    
    def test_step(self, batch):

        images, _, Rs = self.get_input(batch)            
        
        if self.with_codition:
            preds = self.model(images, Rs)
        else:
            preds = self.model(images)
        
        return preds, Rs 
    
    
    
    def distribution_init(self):
        
        self.accelerator.print(f"Total CUDA devices: {torch.cuda.device_count()}")
        
        # init model, optimizers, and dataloaders
        if self.train_dataloader is not None:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
    
    
    def save_checkpoints(self, epoch, name='best'):
        checkpoint = {
            'model': self.model.state_dict(),
            'model_ema': self.model_ema.state_dict() if self.use_ema else None,
            'learning_rate': self.learning_rate,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        if self.accelerator is not None:
            self.accelerator.save(checkpoint, os.path.join(self.checkpoint_path, f'{name}.pth'))
            if self.accelerator.is_local_main_process:
                self.logger.info(f"Saved checkpoint at epoch {epoch}")
        else:
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f'{name}.pth'))
            self.logger.info(f"Saved checkpoint at epoch {epoch}")
    
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            # ema <---- model
            self.model_ema.store(self.model.parameters())
            # ema ----> model
            self.model_ema.copy_to(self.model)
            if context is not None:
                self.logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema: 
                # ema ----> model
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    self.logger.info(f"{context}: Restored training weights")


    def models_to_device(self): 
        self.model = self.model.to(self.device)   
        if self.use_ema:
            self.model_ema = self.model_ema.to(self.device) 
            
            
    def load_checkpoint(self, checkpoint_path):
        # check whether the path exist, if not, find the ckpt according to the exp_name
        print(f"Load checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint_path = checkpoint_path
        elif os.path.exists(os.path.join(self.checkpoint_path, checkpoint_path)):
            checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_path)
        else:
            self.logger.info(f"fail loading checkpoint from {checkpoint_path}, please check it, and try again")
            return False
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Load checkpoint from epoch: ", checkpoint['epoch'])
        
        checkpoint = process_checkpoints(checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        if self.use_ema and checkpoint['model_ema'] is not None:
            self.model_ema.load_state_dict(checkpoint['model_ema'])
        self.learning_rate = checkpoint['learning_rate']
        self.start_epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
        
    
    
    def unzip_dataloaders(self, dataloaders):
        self.train_dataloader = dataloaders.get('train', None)
        self.val_dataloader = dataloaders.get('val', None)
        self.test_dataloader = dataloaders.get('test', None)
    
    
     
    @torch.no_grad()
    def get_input(self, batch):
        if self.accelerator is not None:
            image = batch['pixel_values']
            gt = batch['mask']
            Rs = batch.get('inter_map', None)
        else:
            image = batch['pixel_values'].to(self.device)
            gt = batch['mask'].to(self.device)
            Rs = batch.get('inter_map', None)
            if Rs is not None:
                Rs = Rs.to(self.device)
            
        return image, gt, Rs



    def configure_optimizers(self):
        
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = AdamW(params, lr=lr)
        return opt
    
    
    
    def random_inference(self):
        self.model.eval()
        loader_list = list(self.val_dataloader)
        random_batch = random.choice(loader_list)
        preds, _ = self.test_step(random_batch)
        self.model.train()
        return preds.sigmoid().detach().cpu(), random_batch
    
      
    
    def init_loggers(self):
        log_file_path = os.path.join(self.log_path, "training.log")
        logging.basicConfig(
                filename=log_file_path, 
                filemode='a',
                level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(message)s',
                force=True)
        self.logger = logging.getLogger()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        if self.task == 'train':
            if self.log_method == "tensorboard":
                self.writer = SummaryWriter(self.log_path)
    
    def inference_speed_test(self):

        # images, text_ids, gt, Rs = self.get_input(batch)
        images = torch.randn(1, 3, 224, 224).to(self.device) 
        Rs = torch.randn(1, 1, 224, 224).to(self.device)      
        for _ in range(10):
            preds = self.model(images, Rs)
            # if self.with_codition:
            #     preds = self.model(images, Rs)
            # else:
            #     preds = self.model(images)
        
        # speed test
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                preds = self.model(images, Rs)
                # if self.with_codition:
                #     preds = self.model(images, Rs)
                # else:
                #     preds = self.model(images)
        
        torch.cuda.synchronize()
        end_time = time.time()
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        self.logger.info(f"Average inference time per image: {avg_time:.2f} ms")
        self.logger.info(f"Throughput: {1000 / avg_time:.2f} FPS")      
    
    
    
    def visualize(self, preds, random_batch, epoch):
        # it is move effectively
        images = random_batch['pixel_values']
        gts = random_batch[self.gt_type]
        # gt -> [B, H, W, C]
        mixed_img_gts = mix_images_with_masks(images, gts) 
        mixed_img_predits = mix_images_with_masks(images, preds)
        if self.log_method == "wandb":
            wandb.log({f"Predictions on epoch {epoch}": [wandb.Image(img, caption=f"predictions {i}") for i, img in enumerate(mixed_img_predits)]})
            wandb.log({f"GTs on epoch {epoch}": [wandb.Image(img, caption=f"gts {i}") for i, img in enumerate(mixed_img_gts)]})
        elif self.log_method == "tensorboard":
            B = mixed_img_gts.shape[0]
            pred_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_predits).permute(0, 3, 1, 2), nrow=B)
            gt_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_gts).permute(0, 3, 1, 2), nrow=B)
            self.writer.add_image(f'Predictions on epoch {epoch}', pred_grids)
            self.writer.add_image(f'Ground Truths on epoch {epoch}', gt_grids)
    
    
    
   
                
            
            
        
    
    

        
        
    

        
        




        


    
    
    

            
            