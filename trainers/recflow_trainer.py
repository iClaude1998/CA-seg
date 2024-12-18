import os
import sys
sys.path.append('..')
import torch
import random
import logging
import torchvision
import pandas as pd

import torch.nn as nn
import importlib.util
if importlib.util.find_spec('wandb') is not None:
    import wandb

from tqdm import tqdm
from torch.optim import AdamW
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.nn import functional as F
from contextlib import contextmanager
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict as dedict


from torch.utils.tensorboard import SummaryWriter


from models.diffusion import LitEma
from utils import process_Relevant_score_batch, process_checkpoints, mix_images_with_masks, save_batch, compute_metrics, import_or_skip
wandb = import_or_skip('wandb')


to_pil = transforms.ToPILImage()

class Reflow_Trainer(object):
    
    def __init__(
        self,
        diffusion_version,
        task,
        output_dir,
        clip_model,
        diffusion_model,
        dataloaders,
        learning_rate,
        gt_type='sdf_map',
        device='cuda',
        use_ema=False,
        load_checkpoint=False,
        checkpoint_name=None,
        num_timesteps=1000,
        num_iterations=100000,
        save_interval=100,
        accelerator=None,
        log_method='wandb',
        start_point="LRP",
        clip_grads=None,
    ):
        # instantiate control module
        self.diffusion_version = diffusion_version
        self.output_dir = output_dir
        self.task = task
        self.num_timesteps = num_timesteps
        self.clip_model = clip_model
        self.diffusion_model = diffusion_model
        self.learning_rate = learning_rate
        self.gt_type = gt_type
        self.log_method = log_method
        self.criterion = nn.MSELoss(reduction='mean')
        self.device = device
        self.use_ema = use_ema
        self.start_iteration = 0
        self.start_point = start_point
        self.clip_grads = clip_grads
        self.inter_mode = self.clip_model.inter_mode
        
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
            self.model_ema = LitEma(self.diffusion_model)
            
        if checkpoint_name is not None and load_checkpoint:
            self.load_succeed = self.load_checkpoint(checkpoint_name)
            self.checkpoint_name = os.path.splitext(checkpoint_name)[0]
        else:
            self.load_succeed = False
        self.unzip_dataloaders(dataloaders)
        
        self.accelerator = accelerator
        
        self.models_to_device()
        self.optimizer = self.configure_optimizers()
        self.num_iterations = num_iterations
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
        
        if self.log_method == 'wandb':
            wandb.init(project='clipflow2', name=self.exp_name)
        self.diffusion_model.train()
        iter_id = self.start_iteration
        data_iter = iter(self.train_dataloader)
        
        while (iter_id < self.num_iterations):

            try:
                batch = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            loss_mse = self.training_step(batch)
            loss_mse = loss_mse / gradient_accumulation_steps
            loss_mse.backward()
            
            if iter_id % gradient_accumulation_steps == 0:
                if self.clip_grads is not None:
                    clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grads)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.use_ema:
                self.model_ema(self.model)
            
            if iter_id % self.save_interval == 0:
                self.logger.info(f'Step [{iter_id}/{self.num_iterations}], Loss: {loss_mse.item():.4f}')
                # log infos
                if self.log_method == 'wandb':
                    wandb.log({'Training Loss': loss_mse.detach().cpu().numpy(), 'iteration': iter_id})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Training Loss', loss_mse.detach().cpu().numpy(), iter_id)
            
            if iter_id % (self.save_interval * 500) == 0:
                vts, random_batch = self.random_inference()
                self.visualize(vts, random_batch, iter_id)
                self.save_checkpoints(iter_id)
            print(f"\rIter: {iter_id}", end='', flush=True)
            iter_id += 1
        
        # let's do the last inference and log
        self.logger.info(f'Step [{iter_id}/{self.num_iterations}], Loss: {loss_mse.item():.4f}')
        vts, random_batch = self.random_inference()
        self.visualize(vts, random_batch, iter_id)
        self.save_checkpoints(iter_id)
        
        if self.log_method == 'wandb':
            wandb.finish()
        elif self.log_method == 'tensorboard':
            self.writer.close()
    
    
    def distribution_train(self):
        
        if self.accelerator.is_local_main_process:
            if self.log_method == 'wandb':
                wandb.init(project='clipflow2', name=self.exp_name)
        self.diffusion_model.train()
        iter_id = self.start_iteration
        data_iter = iter(self.train_dataloader)
        
        while (iter_id < self.num_iterations):

            try:
                batch = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            with self.accelerator.accumulate(self.diffusion_model):
                loss_mse = self.training_step(batch)
                self.accelerator.backward(loss_mse)
                if self.clip_grads is not None:
                    self.accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grads)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # gather loss from all processes for display
            gathered_loss = self.accelerator.gather(loss_mse)
            mean_loss = torch.mean(gathered_loss)
            if self.use_ema:
                self.model_ema(self.model)
            
            if iter_id % self.save_interval == 0 and self.accelerator.is_local_main_process:
                self.logger.info(f'Step [{iter_id}/{self.num_iterations}], Loss: {mean_loss.detach().cpu().item():.4f}')
                if self.log_method == 'wandb':
                    wandb.log({'Training Loss': mean_loss.detach().cpu().numpy(), 'iteration': iter_id})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Training Loss', mean_loss.detach().cpu().numpy(), iter_id)
            
            if iter_id % (self.save_interval * 100) == 0:
                vts, random_batch = self.random_inference()
                if self.accelerator.is_local_main_process:
                    self.visualize(vts, random_batch, iter_id)
                self.save_checkpoints(iter_id)
            if self.accelerator.is_local_main_process:
                print(f"\rIter: {iter_id}", end='', flush=True)
            iter_id += 1
        
        # perfect ending
        vts, random_batch = self.random_inference()
        self.save_checkpoints(iter_id)
        if self.accelerator.is_local_main_process:
            self.logger.info(f'Step [{iter_id}/{self.num_iterations}], Loss: {mean_loss.detach().cpu().item():.4f}')
            self.visualize(vts, random_batch, iter_id)
            if self.log_method == 'wandb':
                wandb.finish() 
            elif self.log_method == 'tensorboard':
                self.writer.close()   
    
    
    def training_step(self, batch):
       
        images, text_ids, gt, Rs = self.get_input(batch)
        B = gt.shape[0]
        # zT = torch.randn_like(sdf_map, device=self.device)
        
        z0, conditions, _, intermediate = self.get_conditions(images, text_ids, Rs=Rs)
        
        t = torch.randint(1, self.num_timesteps, (B,), device=self.device).long()
        t_norm = t.float() / (self.num_timesteps - 1)
        t_norm = t_norm.view(B ,1, 1, 1)

        # TODO: add noise maybe, (hope not)
        zt = t_norm * z0 + (1 - t_norm) * gt
        
        x = torch.cat([conditions, zt], dim=1)
        if self.diffusion_version == 'v1' or self.diffusion_version == 'v1p':
            v = self.diffusion_model(x, t, y=None)
        elif self.diffusion_version == 'v2' or self.diffusion_version == 'v2p':
            v = self.diffusion_model(x, t, intermediate.detach())
        loss_mse = self.criterion(gt - z0, v)
      
        return loss_mse   #+loss_perc


    def inference(self):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        self.diffusion_model.eval()
        for batch in tqdm(self.test_dataloader):
            vts, Rs = self.test_step(batch)
            images = batch['pixel_values']
            gts = batch[self.gt_type]
            mask_names = batch['mask_name']
            # gt -> [B, H, W, C]
            with torch.no_grad():
                mixed_img_predits_I = mix_images_with_masks(images, Rs)
                mixed_img_predits_II = mix_images_with_masks(images, vts)
                mixed_img_gts = mix_images_with_masks(images, gts) 
                
                save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, self.vis_path)
    
    
    def test(self, testset='test'):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        self.diffusion_model.eval()
        outcomes = dedict(list)
        if testset == 'test':
            dl = self.test_dataloader
        elif testset == 'val':
            dl = self.val_dataloader
        else:
            raise ValueError(f"Unsupported testset: {testset}")
        for batch in tqdm(dl):
            vts, Rs = self.test_step(batch)
            mask_name = batch['mask_name']
            gts = batch[self.gt_type]
            onehot_mask = batch['mask']
            with torch.no_grad():
                # iou_batch_I = compute_metrics(Rs, gts, mask_name, metric='iou', thresh=66, gt_type=self.gt_type) # stage I
                # iou_batch_II = compute_metrics(vts, gts, mask_name, metric='iou', thresh=33, gt_type=self.gt_type) # stage II
                
                # dice_batch_I = compute_metrics(Rs, gts, mask_name, metric='dice', thresh=66, gt_type=self.gt_type) # stage I
                # dice_batch_II = compute_metrics(vts, gts, mask_name, metric='dice', thresh=33, gt_type=self.gt_type) # stage II
                
                iou_batch_I = compute_metrics(Rs, onehot_mask, mask_name, metric='iou', thresh=66, gt_type=self.gt_type) # stage I
                iou_batch_II = compute_metrics(vts, onehot_mask, mask_name, metric='iou', thresh=33, gt_type=self.gt_type) # stage II
                
                dice_batch_I = compute_metrics(Rs, onehot_mask, mask_name, metric='dice', thresh=66, gt_type=self.gt_type) # stage I
                dice_batch_II = compute_metrics(vts, onehot_mask, mask_name, metric='dice', thresh=33, gt_type=self.gt_type) # stage II
                
                outcomes['mask_name'].extend(mask_name)
                outcomes['iou_I'].extend(iou_batch_I)
                outcomes['iou_II'].extend(iou_batch_II)
                outcomes['dice_I'].extend(dice_batch_I)
                outcomes['dice_II'].extend(dice_batch_II)
        outcomes = pd.DataFrame(outcomes)
        outcomes.to_csv(os.path.join(self.log_path, f'outcomes_{testset}_{self.checkpoint_name}.csv'), index=False)
        return outcomes
    
    
    def test_step(self, batch):

        images, text_ids, gt, Rs = self.get_input(batch)            
        B = gt.shape[0]
        zt, conditions, Rs, intermediate = self.get_conditions(images, text_ids, Rs=Rs)
        eular_steps = [999, 749, 499, 249]            
        # eular_steps = [999,899,799,699,599,499,399,299,199,99]
        # eular_steps = list(range(1000))[::-1]
        for i, step in enumerate(eular_steps):
            ts = torch.ones(B, device=self.device) * step
            x = torch.cat([conditions, zt], dim=1)
            if self.diffusion_version == 'v1' or self.diffusion_version == 'v1p':
                v = self.diffusion_model(x, ts, y=None)
            elif self.diffusion_version == 'v2' or self.diffusion_version == 'v2p':
                v = self.diffusion_model(x, ts, intermediate.detach())
            zt = zt + v / len(eular_steps)
        return zt, Rs   
    
    
    def test_step_process(self, batch):
        images, text_ids, gt, Rs = self.get_input(batch)            
        B = gt.shape[0]
        zt, conditions, Rs, intermediate = self.get_conditions(images, text_ids, Rs=Rs)
        eular_steps = list(range(1000))[::-1]
        with torch.no_grad():           
            for i, step in enumerate(eular_steps):
                ts = torch.ones(B, device=self.device) * step
                x = torch.cat([conditions, zt], dim=1)
                if self.diffusion_version == 'v1' or self.diffusion_version == 'v1p':
                    v = self.diffusion_model(x, ts, y=None)
                elif self.diffusion_version == 'v2' or self.diffusion_version == 'v2p':
                    v = self.diffusion_model(x, ts, intermediate.detach())
                zt = zt + v / len(eular_steps)
                yield zt
        
    
    
    def get_conditions(self, images, text_ids, Rs=None):
        if Rs is None and self.inter_mode:
            Rs, intermediate = self.clip_model(images, text_ids)
            B = Rs.shape[0]
            R_h = int(Rs[0].numel() ** 0.5)
            Rs = Rs.view(B, 1, R_h, R_h)
            Rs = F.interpolate(Rs, images.shape[-2:], mode='bilinear', align_corners=False)
            # normalize Rs
            Rs = process_Relevant_score_batch(Rs, images.shape[-2:])
        else:
            with torch.no_grad():
                _, _, intermediate = self.clip_model.clip_model(images, text_ids)
        if self.start_point == "LRP":
            z0 = Rs
            conditions = images
        elif self.start_point == "guassian":
            z0 = torch.randn_like(images[:, 0:1], device=images.device)
            conditions = images
        elif self.start_point == "all":
            z0 = torch.randn_like(images[:, 0:1], device=images.device)
            conditions = torch.cat([images, Rs], dim=1)
        else:
            raise ValueError(f"Unsupported start_point: {self.start_point}")
        
        return z0, conditions, Rs, intermediate
    
    
    def distribution_init(self):
        
        self.accelerator.print(f"Total CUDA devices: {torch.cuda.device_count()}")
        
        # init model, optimizers, and dataloaders
        if self.train_dataloader is not None:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        self.diffusion_model, self.optimizer = self.accelerator.prepare(self.diffusion_model, self.optimizer)
        self.clip_model.to(self.device)
        
    
    
    def save_checkpoints(self, iteration):
        checkpoint = {
            'model': self.diffusion_model.state_dict(),
            'model_ema': self.model_ema.state_dict() if self.use_ema else None,
            'learning_rate': self.learning_rate,
            'iteration': iteration,
            'optimizer': self.optimizer.state_dict()
        }
        if self.accelerator is not None:
            self.accelerator.save(checkpoint, os.path.join(self.checkpoint_path, f'checkpoint_iter{iteration}.pth'))
            if self.accelerator.is_local_main_process:
                self.logger.info(f"Saved checkpoint at iteration {iteration}")
        else:
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f'checkpoint_iter{iteration}.pth'))
            self.logger.info(f"Saved checkpoint at iteration {iteration}")
    
    
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
        self.clip_model.to(self.device) # the real clip model params is wrapped in the LRP model
        self.diffusion_model = self.diffusion_model.to(self.device)   
        if self.use_ema:
            self.model_ema = self.model_ema.to(self.device) 
            
            
    def load_checkpoint(self, checkpoint_path):
        # check whether the path exist, if not, find the ckpt according to the exp_name
        if os.path.exists(checkpoint_path):
            checkpoint_path = checkpoint_path
        elif os.path.exists(os.path.join(self.checkpoint_path, checkpoint_path)):
            checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_path)
        else:
            self.logger.info(f"fail loading checkpoint from {checkpoint_path}, please check it, and try again")
            return False
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = process_checkpoints(checkpoint)
        self.diffusion_model.load_state_dict(checkpoint['model'])
        if self.use_ema and checkpoint['model_ema'] is not None:
            self.model_ema.load_state_dict(checkpoint['model_ema'])
        self.learning_rate = checkpoint['learning_rate']
        self.start_iteration = checkpoint['iteration']
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
            text_ids = batch['input_ids']
            gt = batch[self.gt_type]
            Rs = batch.get('inter_map', None)
        else:
            image = batch['pixel_values'].to(self.device)
            text_ids = batch['input_ids'].to(self.device)
            gt = batch[self.gt_type].to(self.device)
            Rs = batch.get('inter_map', None)
            if Rs is not None:
                Rs = Rs.to(self.device)
            
        return image, text_ids, gt, Rs


    def configure_optimizers(self):
        
        lr = self.learning_rate
        params = list(self.diffusion_model.parameters())
        opt = AdamW(params, lr=lr)
        return opt
    
    
    def random_inference(self):
        self.diffusion_model.eval()
        loader_list = list(self.val_dataloader)
        random_batch = random.choice(loader_list)
        vts, _ = self.test_step(random_batch)
        self.diffusion_model.train()
        return vts.detach().cpu(), random_batch
    
    
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
    
    
    def visualize(self, vts, random_batch, step):
        # it is move effectively
        images = random_batch['pixel_values']
        gts = random_batch[self.gt_type]
        # gt -> [B, H, W, C]
        mixed_img_gts = mix_images_with_masks(images, gts) 
        mixed_img_predits = mix_images_with_masks(images, vts)
        if self.log_method == "wandb":
            wandb.log({f"Predictions on iter {step}": [wandb.Image(img, caption=f"predictions {i}") for i, img in enumerate(mixed_img_predits)]})
            wandb.log({f"GTs on iter {step}": [wandb.Image(img, caption=f"gts {i}") for i, img in enumerate(mixed_img_gts)]})
        elif self.log_method == "tensorboard":
            B = mixed_img_gts.shape[0]
            pred_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_predits).permute(0, 3, 1, 2), nrow=B)
            gt_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_gts).permute(0, 3, 1, 2), nrow=B)
            self.writer.add_image(f'Predictions on iter {step}', pred_grids)
            self.writer.add_image(f'Ground Truths on iter {step}', gt_grids)
    
    
    def random_inference_process(self):
        self.diffusion_model.eval()
        loader_list = list(self.val_dataloader)
        random_batch = random.choice(loader_list)
        images = random_batch['pixel_values']
        gts = random_batch[self.gt_type]
        mask_names = random_batch['mask_name']
        B = images.shape[0]
        mixed_img_gts = mix_images_with_masks(images, gts) 
        for i, vt in enumerate(tqdm(self.test_step_process(random_batch))):
            mixed_img_predit = mix_images_with_masks(images, vt)
            for b in range(B):
                prefix = os.path.splitext(mask_names[b])[0]
                save_dir = os.path.join(self.vis_process_path, f'{prefix}')
                os.makedirs(save_dir, exist_ok=True)
                plt.figure(figsize=(10, 5))
                plt.imshow(mixed_img_predit[b])
                plt.axis('off')
                plt.title(f'step {i}')
                plt.savefig(os.path.join(save_dir, f"step_{i}.png"))
                plt.close()
        
        for b in range(B):
            prefix = os.path.splitext(mask_names[b])[0]
            save_dir = os.path.join(self.vis_process_path, f'{prefix}')
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.imshow(mixed_img_gts[b])
            plt.axis('off')
            plt.title(f'GT')
            plt.savefig(os.path.join(save_dir, f"GT.png")) 
            plt.close()
                
                
            
            
        
    
    

        
        
    

        
        




        


    
    
    

            
            