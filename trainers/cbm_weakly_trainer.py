import os 
import sys
sys.path.append('..')
import torch
import logging
import statistics
import numpy as np
import pandas as pd
import importlib.util

# adapt for pheonix cluster
if importlib.util.find_spec('wandb') is not None:
    import wandb

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict as dedict
from torch.utils.tensorboard import SummaryWriter

from utils import process_checkpoints, compute_metrics, mix_images_with_masks, save_batch, WarmupExponentialLR





class CLIPCBM_Weakly_Trainer(object):
    
    def __init__(
        self,
        task,
        output_dir,
        model,
        dataloaders,
        learning_rate,
        device='cuda',
        load_checkpoint=False,
        checkpoint_name=None,
        num_epoch=100,
        save_interval=20,
        validation_interval=1,
        gamma = 0.98,
        with_sigmoid=True,
        accelerator=None,
        log_method='wandb',
        clip_grads=None,
    ):
        # instantiate control module
        self.task = task
        self.output_dir = output_dir
        self.model = model
        self.learning_rate = learning_rate
        self.log_method = log_method
        self.with_sigmoid = with_sigmoid  
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.spatial_loss = nn.MSELoss(reduction='mean')
        self.device = device
        self.start_epoch = 0
        self.gamma = gamma
        self.clip_grads = clip_grads
        self.num_epoch = num_epoch
        self.save_interval = save_interval
  
        self.create_exp_name()
        self.log_path = os.path.join(output_dir, 'output_logs')
        self.checkpoint_path = os.path.join(output_dir, 'checkpoints')
        self.vis_path = os.path.join(output_dir, 'visualizations') 
        self.validation_interval = validation_interval
        
        # I have to seperate the branches
        if accelerator is None:
            self.create_output_dirs()
            self.init_loggers()
            
        if checkpoint_name is not None and load_checkpoint:
            self.load_succeed = self.load_checkpoint(checkpoint_name)
            self.checkpoint_name = os.path.splitext(checkpoint_name)[0]
        else:
            self.load_succeed = False
        self.unzip_dataloaders(dataloaders)
        
        self.accelerator = accelerator
        
        self.models_to_device()
        self.optimizer, self.scheduler = self.configure_optimizers()
        
        
        # I have to seperate the branches
        if self.accelerator is not None:
            self.distribution_init()
            if self.accelerator.is_local_main_process:
                self.create_output_dirs()
                self.init_loggers()


    
    
    def train(self, gradient_accumulation_steps=1):
        
        if hasattr(self.model, 'stage'):
            self.model.stage = 'train'
        
        if self.log_method == 'wandb':
            wandb.init(project='clipflow2', name=self.exp_name,
                       config={"learning_rate": self.learning_rate,
                               "batch_size": self.train_dataloader.batch_size,
                               "num_epoch": self.num_epoch}
                      )
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.train()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.train()
        
        num_batches = len(self.train_dataloader)
        best_iou, best_dice = 0, 0
        for epoch in range(self.start_epoch, self.num_epoch):
            training_loss_temp = []
            
            for iter_id, batch in tqdm(enumerate(self.train_dataloader),desc=f'Epoch {epoch}'):

                images = batch['pixel_values'].to(self.device)
                cams = batch['inter_map'].to(self.device)
                # sdf_maps = batch['sdf_map'].to(self.device)
                _ = batch['mask'].to(self.device)
                gt = batch['label'].to(self.device)
                if hasattr(self.model, 'stage'):
                    scores, preds, scores_aug, preds_aug = self.model(images, cams)
                else:
                    concept_score, pred = self.model(images, cams)
                
                # preds = postprocess_pred(preds)
                # loss = self.criterion(preds, onehot_maps) + self.regularizer(concept_weights)
                
                
                if hasattr(self.model, 'stage'):
                    loss = self.criterion(scores, gt[:, None].to(torch.float32))
                    loss += self.criterion(scores_aug, gt[:, None].to(torch.float32))
                    loss += self.spatial_loss(preds, preds_aug)
                else:
                    loss = self.criterion(concept_score, gt[:, None].to(torch.float32))
            
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
                if iter_id % gradient_accumulation_steps == 0:
                    if self.clip_grads is not None:
                        clip_grad_norm_(self.model.parameters(), self.clip_grads)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.scheduler.step()
            
                self.logger.info(f'Step [{iter_id}/{num_batches}], Loss: {loss.item():.4f}')
                training_loss_temp.append(loss.item())
            
            self.scheduler.step()
                    
            # log infos
            if self.log_method == 'wandb':
                wandb.log({'Training Loss': statistics.mean(training_loss_temp), 'epoch': epoch})
                wandb.log({'Learning Rate': self.scheduler.get_last_lr()[0], 'epoch': epoch})
            elif self.log_method == 'tensorboard':
                self.writer.add_scalar('Training Loss', statistics.mean(training_loss_temp), epoch)
                self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
            
            if self.validation_interval is not None and epoch % self.validation_interval  == 0:
                outcomes = self.validation()
                if self.log_method == 'wandb':
                    wandb.log({'IoU': outcomes['iou'], 'epoch': epoch})
                    wandb.log({'Dice': outcomes['dice'], 'epoch': epoch})
                    wandb.log({'-IoU': outcomes['neg_iou'], 'epoch': epoch})
                    wandb.log({'-Dice': outcomes['neg_dice'], 'epoch': epoch})
                    wandb.log({'Validation Loss': outcomes['loss'], 'epoch': epoch})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Validation Loss', outcomes['loss'], epoch)
                    self.writer.add_scalar('Dice', outcomes['dice'], epoch)
                    self.writer.add_scalar('IoU', outcomes['iou'], epoch)
                    self.writer.add_scalar('-Dice', outcomes['dice'], epoch)
                    self.writer.add_scalar('-IoU', outcomes['iou'], epoch)
                
                iou = max(outcomes['iou'], outcomes['neg_iou'])
                dice = max(outcomes['dice'], outcomes['neg_dice'])
                
                if iou > best_iou and dice > best_dice:
                    checkpoint = {'model': self.model.state_dict()}
                    torch.save(checkpoint, os.path.join(self.checkpoint_path, 'best.pth'))
                    
                if iou > best_iou:
                    best_iou = iou
                    self.logger.info(f'Achieves the best iou {round(100 * best_iou, 2)} at epoch {epoch}')
                
                if dice > best_dice:
                    best_dice = dice
                    self.logger.info(f'Achieves the best dice {round(100 * best_dice, 2)} at epoch {epoch}')
                    
            
            if self.save_interval is not None and epoch % self.save_interval == 0:
                self.save_checkpoints(epoch)
        
        # save the final checkpoint
        self.save_checkpoints(epoch)

    
    def distribution_train(self):
        
        if hasattr(self.model, 'stage'):
            self.model.stage = 'train'
        
        if self.accelerator.is_local_main_process:
            if self.log_method == 'wandb':
                wandb.init(project='clipflow2', name=self.exp_name)
                
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.train()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.train()
        
        num_batches = len(self.train_dataloader)
        for epoch in range(self.start_epoch, self.num_epoch):
            training_loss_temp = []
            best_iou, best_dice = 0, 0
            for iter_id, batch in tqdm(enumerate(self.train_dataloader, desc=f'Epoch {epoch}')):

                images = batch['pixel_values'].to(self.device)
                cams = batch['inter_map'].to(self.device)
                # sdf_maps = batch['sdf_map'].to(self.device)
                onehot_maps = batch['mask'].to(self.device)
                gt = batch['label'].to(self.device)
                B, _, H, W = cams.shape
                C = onehot_maps.shape[1]
                
                with self.accelerator.accumulate(self.model):
                    
                    
                    
                    if hasattr(self.model, 'stage'):
                        scores, preds, scores_aug, preds_aug = self.model(images, cams)
                    else:
                        concept_score, pred = self.model(images, cams)

                    if hasattr(self.model, 'stage'):
                        loss = self.criterion(scores, gt[:, None].to(torch.float32))
                        loss += self.criterion(scores_aug, gt[:, None].to(torch.float32))
                        loss += self.spatial_loss(preds, preds_aug)
                    else:
                        loss = self.criterion(concept_score, gt[:, None].to(torch.float32))
                    self.accelerator.backward(loss)
                    if self.clip_grads is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                gathered_loss = self.accelerator.gather(loss)
                loss = torch.mean(gathered_loss)

                if self.accelerator.is_local_main_process:
                    self.logger.info(f'Step [{iter_id}/{num_batches}], Loss: {loss.item():.4f}')
                training_loss_temp.append(loss.item())
            
            if self.accelerator.is_local_main_process:
                if self.log_method == 'wandb':
                    wandb.log({'Training Loss': statistics.mean(training_loss_temp), 'epoch': epoch})
                    wandb.log({'Learning Rate': self.scheduler.get_last_lr()[0], 'epoch': epoch})
                elif self.log_method == 'tensorboard':
                    self.writer.add_scalar('Training Loss', statistics.mean(training_loss_temp), epoch)
                    self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
            
            if self.validation_interval is not None and epoch % self.validation_interval  == 0:
                outcomes = self.validation()
                if self.accelerator.is_local_main_process:
                    if self.log_method == 'wandb':
                        wandb.log({'IoU': outcomes['iou'], 'epoch': epoch})
                        wandb.log({'Dice': outcomes['dice'], 'epoch': epoch})
                        wandb.log({'Validation Loss': outcomes['loss'], 'epoch': epoch})
                    elif self.log_method == 'tensorboard':
                        self.writer.add_scalar('Validation Loss', outcomes['loss'], epoch)
                        self.writer.add_scalar('Dice', outcomes['dice'], epoch)
                        self.writer.add_scalar('IoU', outcomes['iou'], epoch)
                
                if outcomes['iou'] > best_iou and outcomes['dice'] > best_dice:
                    checkpoint = {'model': self.model.state_dict()}
                    self.accelerator.save(checkpoint, os.path.join(self.checkpoint_path, 'best.pth'))
                    
                if outcomes['iou'] > best_iou:
                    best_iou = outcomes['iou']
                    if self.accelerator.is_local_main_process:
                        self.logger.info(f'Achieves the best iou {round(100 * best_iou, 2)} at epoch {epoch}')
                        
                if outcomes['dice'] > best_dice:
                    best_dice = outcomes['dice']
                    if self.accelerator.is_local_main_process:
                        self.logger.info(f'Achieves the best dice {round(100 * best_dice, 2)} at epoch {epoch}')
            
            if self.save_interval is not None and epoch % self.save_interval == 0:
                self.save_checkpoints(epoch)
            
        # save the final checkpoint
        self.save_checkpoints(epoch)
        
    
    def validation(self):
        
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.eval()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.eval()
        if hasattr(self.model, 'stage'):
            self.model.stage = 'test'
        outcomes = dedict(list)
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['pixel_values'].to(self.device)
            cams = batch['inter_map'].to(self.device)
            # sdf_maps = batch['sdf_map'].to(self.device)
            onehot_maps = batch['mask'].to(self.device)
            gts = batch['label'].to(self.device)
            mask_name = batch['mask_name']

            with torch.no_grad():
                concept_score, preds = self.model(images, cams)

                preds = postprocess_pred(preds, self.with_sigmoid)
                neg_preds = postprocess_pred(-preds, self.with_sigmoid)
                loss = self.criterion(concept_score, gts[:, None].to(torch.float32))

                iou_batch = compute_metrics(preds, onehot_maps, mask_name, metric='iou', thresh=17)
                dice_batch = compute_metrics(preds, onehot_maps, mask_name, metric='dice', thresh=17)
                
                neg_iou_batch = compute_metrics(neg_preds, onehot_maps, mask_name, metric='iou', thresh=17)
                neg_dice_batch = compute_metrics(neg_preds, onehot_maps, mask_name, metric='dice', thresh=17)
                outcomes['iou'].extend(iou_batch)
                outcomes['dice'].extend(dice_batch)
                outcomes['neg_iou'].extend(neg_iou_batch)
                outcomes['neg_dice'].extend(neg_dice_batch)
                outcomes['loss'].append(loss.item())
                outcomes['label'].extend(gts.cpu().numpy().tolist())
            
        outcomes['iou'] = (np.array(outcomes['iou']) * np.array(outcomes['label'])).sum() / np.array(outcomes['label']).sum()
        outcomes['dice'] = (np.array(outcomes['dice']) * np.array(outcomes['label'])).sum() / np.array(outcomes['label']).sum()
        
        outcomes['neg_iou'] = (np.array(outcomes['neg_iou']) * np.array(outcomes['label'])).sum() / np.array(outcomes['label']).sum()
        outcomes['neg_dice'] = (np.array(outcomes['neg_dice']) * np.array(outcomes['label'])).sum() / np.array(outcomes['label']).sum()
        outcomes['loss'] = statistics.mean(outcomes['loss'])
            
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.train()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.train()
        if hasattr(self.model, 'stage'):
            self.model.stage = 'train'
        return outcomes
    
    
    
    def test(self, testset='test'):
        
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.eval()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.eval()
        if hasattr(self.model, 'stage'):
            self.model.stage = 'test'
        
        if testset == 'test':
            dl = self.test_dataloader
        elif testset == 'val':
            dl = self.val_dataloader
        else:
            dl = self.train_dataloader
        
        outcomes = dedict(list)
        for batch in tqdm(dl, desc='Test'):
            images = batch['pixel_values'].to(self.device)
            cams = batch['inter_map'].to(self.device)
            # sdf_maps = batch['sdf_map'].to(self.device)
            onehot_maps = batch['mask'].to(self.device)
            mask_name = batch['mask_name']
           
            with torch.no_grad():
                concept_score, preds = self.model(images, cams)
                untrained = torch.mean(cams, dim=1, keepdim=True)
                
                untrained = postprocess_pred(untrained, self.with_sigmoid)
                preds = postprocess_pred(preds, self.with_sigmoid)
                
                iou_batch_I = compute_metrics(untrained, onehot_maps, mask_name, metric='iou', thresh=17)
                dice_batch_I = compute_metrics(untrained, onehot_maps, mask_name, metric='dice', thresh=17)
                
                iou_batch_II = compute_metrics(preds, onehot_maps, mask_name, metric='iou', thresh=17)
                dice_batch_II = compute_metrics(preds, onehot_maps, mask_name, metric='dice', thresh=17)
                    
                outcomes['iou_I'].extend(iou_batch_I)
                outcomes['dice_I'].extend(dice_batch_I)
                
                outcomes['iou_II'].extend(iou_batch_II)
                outcomes['dice_II'].extend(dice_batch_II)
                
                outcomes['label'].extend(batch['label'].cpu().numpy().tolist())
            
        outcomes = pd.DataFrame(outcomes)
        outcomes.to_csv(os.path.join(self.log_path, f'outcomes_test_100.csv'), index=False)
        return outcomes
    
    
    
    def thresh_search(self, metric='dice'):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.eval()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.eval()
        if hasattr(self.model, 'stage'):
            self.model.stage = 'val'
        
        best_thresh, best_outcome = 0, 0
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        for thresh in range(1, 256):
            results = []
            for batch in tqdm(dataloader):
                images = batch['pixel_values'].to(self.device)
                mask_name = batch['mask_name']
                cams = batch['inter_map'].to(self.device)
                onehot_mask = batch['mask'].to(self.device)

                with torch.no_grad():
                    concept_score, preds = self.model(images, cams)
                    result_batch = compute_metrics(preds, onehot_mask, mask_name, metric=metric, thresh=thresh) # stage II
                    results.extend(result_batch)
            
            results = statistics.mean(results)
            if results > best_outcome:
                best_outcome = results
                best_thresh = thresh
        
        return best_thresh, best_outcome

    
    
    def inference(self):
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.eval()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.eval()
        if hasattr(self.model, 'stage'):
            self.model.stage = 'test'
        dataloader = self.test_dataloader if self.test_dataloader is not None else self.train_dataloader
        for batch in tqdm(dataloader):
            images = batch['pixel_values'].to(self.device)
            cams = batch['inter_map'].to(self.device)
            sdf_maps = batch['sdf_map'].to(self.device)
            mask_name = batch['mask_name']

            # gt -> [B, H, W, C]
            h, w = images.shape[-2:]
            with torch.no_grad():
                concept_score, preds = self.model(images, cams)
                untrained = torch.mean(cams, dim=1, keepdim=True)

                untrained = postprocess_pred(untrained, self.with_sigmoid)
                preds = postprocess_pred(preds, self.with_sigmoid)
                
                untrained = F.interpolate(untrained, size=(h, w), mode='bilinear', align_corners=False)
                preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=False)
                sdf_maps = F.interpolate(sdf_maps, size=(h, w), mode='bilinear', align_corners=False)
                
                mixed_img_predits_I = mix_images_with_masks(images, untrained)
                mixed_img_predits_II = mix_images_with_masks(images, preds)
                mixed_img_gts = mix_images_with_masks(images, sdf_maps) 
                    
                save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_name, self.vis_path)
    
    
    def produce_cam(self, train_outdir, val_outdir, test_outdir, interpolate=True):
        """
        Produce the results for stage II clipflow refinements
        """
        if not self.load_succeed:
            raise FileNotFoundError("No checkpoint found, please check the path (you don't wanna inference from scratch, right? ^ V ^)")
        if hasattr(self.model, 'concept_head'):
            self.model.concept_head.eval()
        if hasattr(self.model, 'clss_head'):
            self.model.clss_head.eval()
        
        dataloaders = [loader for loader in [self.train_dataloader, self.val_dataloader, self.test_dataloader] if loader is not None]
        outdirs = [outdir for outdir in [train_outdir, val_outdir, test_outdir] if outdir is not None]
        
        for dataloader, outdir in zip(dataloaders, outdirs):
            for batch in tqdm(dataloader):
                images = batch['pixel_values'].to(self.device)
                cams = batch['inter_map'].to(self.device)
                mask_names = batch['mask_name']
                # gt -> [B, H, W, C]
                h, w = images.shape[-2:]
                with torch.no_grad():
                    concept_score, preds = self.model(images, cams)
                    if interpolate:
                        preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=False).detach().cpu().numpy()
                    else:
                        preds = preds.detach().cpu().numpy()
                        
                    for i in range(preds.shape[0]):
                        cam = preds[i]
                        mask_name = mask_names[i]
                        mask_name = os.path.splitext(mask_name)[0]
                        # dataset_name, idx, _ = mask_name.split('_')
                        if interpolate:
                            np.save(os.path.join(outdir, f'{mask_name}_l.npy'), cam[0])
                        else:
                            np.save(os.path.join(outdir, f'{mask_name}.npy'), cam[0])
                            
                      
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
        self.model.load_state_dict(checkpoint['model'])
        
        self.learning_rate = checkpoint.get('learning_rate', 0) 
        self.start_epoch = checkpoint.get('epoch', 0) 
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True 
    
    
    
    def save_checkpoints(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'learning_rate': self.learning_rate,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        if self.accelerator is not None:
            self.accelerator.save(checkpoint, os.path.join(self.checkpoint_path, f'checkpoint_epoch{epoch}.pth'))
            if self.accelerator.is_local_main_process:
                self.logger.info(f"Saved checkpoint at epoch {epoch}")
        else:
            torch.save(checkpoint, os.path.join(self.checkpoint_path, f'checkpoint_epoch{epoch}.pth'))
            self.logger.info(f"Saved checkpoint at epoch {epoch}")         
    
    
    
    def models_to_device(self): 
        self.model = self.model.to(self.device)   

    
    
    def unzip_dataloaders(self, dataloaders):
        self.train_dataloader = dataloaders.get('train', None)
        self.val_dataloader = dataloaders.get('val', None)
        self.test_dataloader = dataloaders.get('test', None)
    
    
    
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
        self.model.to(self.device)
    
    

    def configure_optimizers(self):
        
        params = [param for name, param in self.model.named_parameters() if param.requires_grad]
        opt = Adam(params, lr=self.learning_rate)
        # we take the 5% of the total steps as the warmup steps
        # warmup_steps = int(len(self.train_dataloader) * self.num_epoch * 0.05)
        warmup_steps = int(self.num_epoch * 0.05)
        scheduler = WarmupExponentialLR(opt, warmup_steps, gamma=self.gamma) # gamma ranges from 0.9 to 0.99 normalily
        return opt, scheduler
    
    
    
    def init_loggers(self):
        log_file_path = os.path.join(self.log_path, "training.log")
        logging.basicConfig(
                filename=log_file_path, 
                filemode='a',
                level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(message)s',
                force=True)
        self.logger = logging.getLogger(__name__)
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        if self.task == 'train':
            if self.log_method == "tensorboard":
                self.writer = SummaryWriter(self.log_path)
                
    def filter_data(self, thresh):
        infos = dedict(list)
        filtered_infos = dedict(list)
        for dataloader in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            if dataloader is not None:
                with torch.no_grad():
                    for batch in tqdm(dataloader):
                        images = batch['pixel_values'].to(self.device)
                        cams = batch['inter_map'].to(self.device)
                        # sdf_maps = batch['sdf_map'].to(self.device)
                        onehot_maps = batch['mask'].to(self.device)
                        mask_name = batch['mask_path'][0]
                        img_name = batch['img_path'][0]
                        
                        mask_name = mask_name.split('/')[-2:]
                        mask_name = '/'.join(mask_name)
                        
                        img_name = img_name.split('/')[-2:]
                        img_name = '/'.join(img_name)
                        
                        concept_score, preds = self.model(images, cams)                
                        
                        preds = postprocess_pred(preds, self.with_sigmoid)
                        
                        dice_batch_II = compute_metrics(preds, onehot_maps, mask_name, metric='dice', thresh=17)[0]   
                        
                        infos['img_path'].append(img_name)
                        infos['mask_path'].append(mask_name)
                        infos['dice'].append(dice_batch_II)
                        if dice_batch_II > thresh:
                            filtered_infos['img_path'].append(img_name)
                            filtered_infos['mask_path'].append(mask_name)
                            filtered_infos['dice'].append(dice_batch_II)
        
        print(f"Total number of filtered images: {len(filtered_infos['img_path'])}")                
        infos = pd.DataFrame(infos)
        filtered_infos = pd.DataFrame(filtered_infos)
        infos.to_csv(os.path.join(self.log_path, 'infos.csv'), index=False)
        filtered_infos.to_csv(os.path.join(self.log_path, 'filtered_infos.csv'), index=False)
    
    
    def create_output_dirs(self):

        os.makedirs(self.log_path, exist_ok=True)  
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
    
    
    
    def create_exp_name(self):
        _, learn_obj, dataset_name, exp_name = self.output_dir.split('/')
        self.exp_name = f"{learn_obj}-{dataset_name}-{exp_name}"
    



        



class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  
            self.flush()
        except Exception:
            self.handleError(record)




def postprocess_pred(preds, with_sigmoid=True):
    
    # R -> [-1, 1]
    if with_sigmoid:
        preds = torch.sigmoid(preds)
    preds = 2 * (preds - 0.5)
    return torch.clamp(preds, min=0)


class RegionConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RegionConsistencyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred_mask):
        smoothness_loss = F.mse_loss(pred_mask[:, :, :-1, :], pred_mask[:, :, 1:, :]) + \
                          F.mse_loss(pred_mask[:, :, :, :-1], pred_mask[:, :, :, 1:])
        if self.reduction == 'mean':
            smoothness_loss = smoothness_loss.mean()
        return smoothness_loss