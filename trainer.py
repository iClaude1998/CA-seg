import os
import io
import torch
import wandb 
import random
import logging
import matplotlib
import numpy as np
import torchvision
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW
from torch.backends import cudnn
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.diffusion import LitEma
from utils import process_Relevant_score_batch, process_checkpoints


to_pil = transforms.ToPILImage()

class Reflow_ControlLDM(object):
    
    def __init__(
        self,
        task,
        exp_name,
        clip_model,
        diffusion_model,
        dataloaders,
        learning_rate,
        device='cuda',
        use_ema=False,
        checkpoint_name=None,
        num_timesteps=1000,
        num_iterations=100000,
        save_interval=100,
        accelerator=None,
        log_method='wandb'
    ):
        # instantiate control module
        self.exp_name = exp_name
        self.task = task
        self.num_timesteps = num_timesteps
        self.clip_model = clip_model
        self.diffusion_model = diffusion_model
        self.learning_rate = learning_rate
        self.log_method = log_method
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.device = device
        self.use_ema = use_ema
        self.start_iteration = 0
        if use_ema:
            self.model_ema = LitEma(self.diffusion_model)
        if checkpoint_name is not None:
            self.load_succeed = self.load_checkpoint(checkpoint_name)
        else:
            self.load_succeed = False
        self.unzip_dataloaders(dataloaders)
        
        self.accelerator = accelerator
        
        self.models_to_device()
        self.optimizer = self.configure_optimizers()
        self.num_iterations = num_iterations
        self.save_interval = save_interval
        if self.accelerator is not None:
            self.distribution_init()
            self.log_path = os.path.join('experiments', self.exp_name, 'output_logs')
            self.checkpoint_path = os.path.join('experiments', self.exp_name, 'checkpoints')
            self.vis_path = os.path.join('experiments', self.exp_name, 'visualizations') 
            if self.accelerator.is_local_main_process:
                self.create_output_dirs()
                self.init_loggers()
        else:
            self.create_output_dirs()
            self.init_loggers()
            
    
    
    
    def create_output_dirs(self):

        os.makedirs(self.log_path, exist_ok=True)  
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
    
    
    
    def train(self):
        
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
            
            self.optimizer.zero_grad()
            loss_mse.backward()
            self.optimizer.step()
            
            if self.use_ema:
                self.model_ema(self.model)
            
            if iter_id % self.save_interval == 0:
                self.logger.info(f'Step [{iter_id}/{self.num_iterations}], Loss: {loss_mse.item():.4f}')
                # log infos
                if self.log_method == 'wandb':
                    wandb.log({'Training Loss': loss_mse.detach().cpu().numpy(), 'iteration': iter_id})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Training Loss', loss_mse.detach().cpu().numpy(), iter_id)
            
            if iter_id % (self.save_interval * 100) == 0:
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
            loss_mse = self.training_step(batch)
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss_mse)
            self.optimizer.step()
            
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
       
        images, text_ids, sdf_map = self.get_input(batch)
        B = sdf_map.shape[0]
        # zT = torch.randn_like(sdf_map, device=self.device)
        Rs, intermediate = self.clip_model(images, text_ids)
        R_h = int(Rs[0].numel() ** 0.5)
        Rs = Rs.view(B, 1, R_h, R_h)
        Rs = F.interpolate(Rs, images.shape[-2:], mode='bilinear', align_corners=False)
        
        # normalize Rs
        Rs = process_Relevant_score_batch(Rs, images.shape[-2:])
        
        t = torch.randint(1, self.num_timesteps, (B,), device=self.device).long()
        t_norm = t.float() / (self.num_timesteps - 1)
        t_norm = t_norm.view(B ,1, 1, 1)

        # TODO: add noise maybe, (hope not)
        zt = t_norm * Rs + (1 - t_norm) * sdf_map
        x = torch.cat([images, zt], dim=1)
        
        v = self.diffusion_model(x, t, y=None)
        loss_mse = self.criterion(sdf_map - Rs, v)
      
        return loss_mse   #+loss_perc


    def inference(self):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        self.diffusion_model.eval()
        for batch in tqdm(self.test_dataloader):
            vts, Rs = self.test_step(batch)
            images = batch['pixel_values']
            usdf_gts = batch['sdf_map']
            mask_names = batch['mask_name']
            # gt -> [B, H, W, C]
            with torch.no_grad():
                mixed_img_predits_I = mix_images_with_sdfs(images, Rs)
                mixed_img_predits_II = mix_images_with_sdfs(images, vts)
                mixed_img_gts = mix_images_with_sdfs(images, usdf_gts) 
                
                save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, self.vis_path)

    
    
    def test_step(self, batch):

        images, text_ids, sdf_map = self.get_input(batch)
        B = sdf_map.shape[0]
        Rs, intermediate = self.clip_model(images, text_ids)
        with torch.no_grad():
            R_h = int(Rs[0].numel() ** 0.5)
            Rs = Rs.view(B, 1, R_h, R_h)
            Rs = F.interpolate(Rs, images.shape[-2:], mode='bilinear', align_corners=False)
            
            # normalize Rs
            Rs = process_Relevant_score_batch(Rs, images.shape[-2:])
            zt = Rs
            eular_steps = [999, 749, 499, 249]
            #eular_steps = [999,899,799,699,599,499,399,299,199,99]
            for i, step in enumerate(eular_steps):
                ts = torch.ones(B, device=self.device) * step
                x = torch.cat([images, zt], dim=1)
                v = self.diffusion_model(x, ts, y=None)
                zt = zt + v / len(eular_steps)
        return zt, Rs   
    
    
    def distribution_init(self):
        
        # if self.accelerator.is_local_main_process:
        #     self.logger.info(f"Total CUDA devices: {torch.cuda.device_count()}")
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
            sdf_map = batch['sdf_map']
        else:
            image = batch['pixel_values'].to(self.device)
            text_ids = batch['input_ids'].to(self.device)
            sdf_map = batch['sdf_map'].to(self.device)
        return image, text_ids, sdf_map


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
        usdf_gts = random_batch['sdf_map']
        # gt -> [B, H, W, C]
        mixed_img_gts = mix_images_with_sdfs(images, usdf_gts) 
        mixed_img_predits = mix_images_with_sdfs(images, vts)
        if self.log_method == "wandb":
            wandb.log({f"Predictions on iter {step}": [wandb.Image(img, caption=f"predictions {i}") for i, img in enumerate(mixed_img_predits)]})
            wandb.log({f"GTs on iter {step}": [wandb.Image(img, caption=f"gts {i}") for i, img in enumerate(mixed_img_gts)]})
        elif self.log_method == "tensorboard":
            B = mixed_img_gts.shape[0]
            pred_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_predits).permute(0, 3, 1, 2), nrow=B)
            gt_grids = torchvision.utils.make_grid(torch.from_numpy(mixed_img_gts).permute(0, 3, 1, 2), nrow=B)
            self.writer.add_image(f'Predictions on iter {step}', pred_grids)
            self.writer.add_image(f'Ground Truths on iter {step}', gt_grids)
        
        



def mix_images_with_sdfs(images, usdfs, alpha_heatmap=0.5, colormap='jet'):
    """
    Mixes images with unsigned distance functions (USDFs) using a specified colormap and alpha blending.
    Args:
        images (torch.Tensor): A batch of images with shape (N, C, H, W).
        usdfs (torch.Tensor): A batch of unsigned SDFs with shape (N, 1, H, W).
        alpha_heatmap (float, optional): The blending factor for the heatmap overlay. Default is 0.5.
        colormap (str, optional): The colormap to use for the SDFs. Default is 'jet'.
    Returns:
        numpy.ndarray: The resulting images with the SDF heatmap overlay, with shape (N, H, W, C).
    """

    cmap = matplotlib.colormaps.get_cmap(colormap)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    usdfs = usdfs.squeeze(1).cpu().numpy()
    
    # normalize usdfs
    mim_usdfs = np.min(usdfs, axis=(1, 2), keepdims=True)
    max_usdfs = np.max(usdfs, axis=(1, 2), keepdims=True)
    nonzero_max = (max_usdfs > 0).astype('float32')
    max_usdfs = np.where(max_usdfs > 0, max_usdfs, 1e-6)
    usdfs = nonzero_max * (usdfs - mim_usdfs) / (max_usdfs - mim_usdfs)
    
    rgb_heatmaps_np = cmap(usdfs)[..., :3]
    overlayed_images = (1 - alpha_heatmap) * images + alpha_heatmap * rgb_heatmaps_np
    return np.clip(overlayed_images, a_min=0., a_max=1.)
    
    
def save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, vis_path):
    num_examples = mixed_img_predits_I.shape[0]
    for i in range(num_examples):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(mixed_img_predits_I[i])
        ax[0].axis('off')
        ax[0].set_title('Predictions I')   
        ax[1].imshow(mixed_img_predits_II[i])
        ax[1].axis('off')
        ax[1].set_title('Predictions II')   
        ax[2].imshow(mixed_img_gts[i])
        ax[2].axis('off')
        ax[2].set_title('Ground truths')   
        plt.savefig(os.path.join(vis_path, mask_names[i]))
        plt.close(fig)

            
            