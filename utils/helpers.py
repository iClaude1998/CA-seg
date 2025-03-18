import os
import sys
sys.path.append('..')
import cv2
import torch
 

import matplotlib
import numpy as np
import importlib.util
import blobfile as bf


from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from datasets import build_dataset


INITIAL_LOG_LOSS_SCALE = 20.0


def import_or_skip(module_name):
    if importlib.util.find_spec(module_name) is None:
        print(f"Skipping import of {module_name}")
        return None
    return importlib.import_module(module_name)






def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)



def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))





def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

        


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None




def process_checkpoints(checkpoint):
    """convert the checkpoint from parallel to single gpu:

    Args:
        checkpoint (_type_): _description_
    """
    new_state_dict_model = {}
    for key in checkpoint['model'].keys():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict_model[new_key] = checkpoint['model'][key]
    checkpoint['model'] = new_state_dict_model
    
    new_state_dict_model = {}
    if checkpoint.get('model_ema', None) is not None:
        for key in checkpoint['model_ema'].keys():
            new_key = key.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict_model[new_key] = checkpoint['model_ema'][key]
        checkpoint['model_ema'] = new_state_dict_model
    
    return checkpoint


def mix_images_with_masks(images, masks, alpha_heatmap=0.5, colormap='jet'):
    """
    Mixes images with unsigned distance functions (USDFs) using a specified colormap and alpha blending.
    Args:
        images (torch.Tensor): A batch of images with shape (N, C, H, W).
        masks (torch.Tensor): A batch of masks with shape (N, 1, H, W).
        alpha_heatmap (float, optional): The blending factor for the heatmap overlay. Default is 0.5.
        colormap (str, optional): The colormap to use for the SDFs. Default is 'jet'.
    Returns:
        numpy.ndarray: The resulting images with the SDF heatmap overlay, with shape (N, H, W, C).
    """

    cmap = matplotlib.colormaps.get_cmap(colormap)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    
    # normalize masks
    masks = min_max_normalize(masks)
    
    rgb_heatmaps_np = cmap(masks)[..., :3]
    if len(rgb_heatmaps_np.shape) == 5:
        for i in range(rgb_heatmaps_np.shape[1]):
            rgb_heatmaps_np_i = rgb_heatmaps_np[:, i]
            images = (1 - alpha_heatmap) * images + alpha_heatmap * rgb_heatmaps_np_i
    else:
        images = (1 - alpha_heatmap) * images + alpha_heatmap * rgb_heatmaps_np
    return np.clip(images, a_min=0., a_max=1.)


def compute_metrics(preds, gts, mask_name, metric, thresh=126):
    # preds = preds.squeeze(1).cpu().numpy()
    # gts = gts.squeeze(1).cpu().numpy()
    outcomes = []
    for i in range(gts.shape[1]):
        pred = preds[:, i].cpu().numpy()
        gt = gts[:, i].cpu().numpy()
        if isinstance(thresh, int):
            pred = min_max_normalize(pred)
            pred = (255 * pred >= thresh)
        elif thresh == 'otsu':
            pred = min_max_normalize(pred)
            pred = otsu_thresholding(pred)
        else:
            pred = (pred.sigmoid() > 0.5)
        gt = (255 * gt > 0)
        # visualization_for_debug(preds, gts, mask_name)
        if metric == 'iou':
            intersection = np.logical_and(pred, gt)
            union = np.logical_or(pred, gt)
            intersection_batch = np.sum(intersection.astype('float32'), axis=(1, 2))
            union_batch = np.sum(union.astype('float32'), axis=(1, 2))
            
            mask = np.where(union_batch > 0, 1., 0.)
            union_batch = np.where(union_batch > 0, union_batch, 1e-3)
            outcome = intersection_batch * mask / union_batch
        elif metric == 'dice':
            intersection = np.logical_and(pred, gt)
            intersection_batch = np.sum(intersection.astype('float32'), axis=(1, 2))
            pred_batch = np.sum(pred.astype('float32'), axis=(1, 2))
            gt_batch = np.sum(gt.astype('float32'), axis=(1, 2))
            denorminztor = pred_batch + gt_batch
            
            mask = np.where(denorminztor > 0., 1., 0.)
            denorminztor = np.where(denorminztor > 0., denorminztor, 1e-3)
            outcome = (2 * intersection_batch * mask) / denorminztor
        outcomes.append(outcome)
    outcomes = np.stack(outcomes, axis=0)
   
    return np.mean(outcomes, axis=0)
    
    
    
def save_batch(mixed_img_predits_I, mixed_img_predits_II, mixed_img_gts, mask_names, vis_path):
    num_examples = mixed_img_gts.shape[0]
    if mixed_img_predits_I is None:
        
        for i in range(num_examples):

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))  
            ax[0].imshow(mixed_img_predits_II[i])
            ax[0].axis('off')
            ax[0].set_title('Predictions')   
            ax[1].imshow(mixed_img_gts[i])
            ax[1].axis('off')
            ax[1].set_title('Ground truths')   
            plt.savefig(os.path.join(vis_path, mask_names[i]))
            plt.close(fig)
    else:
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
        
        
        

def min_max_normalize(usdfs):
    
    mim_usdfs = np.min(usdfs, axis=(1, 2), keepdims=True)
    max_usdfs = np.max(usdfs, axis=(1, 2), keepdims=True)
    nonzero_max = (max_usdfs > 0).astype('float32')
    max_usdfs = np.where(max_usdfs > 0, max_usdfs, 1e-6)
    return nonzero_max * (usdfs - mim_usdfs) / (max_usdfs - mim_usdfs)



def otsu_thresholding(usdfs):
    usdfs = (255 * usdfs).astype('uint8')
    binary_images = np.zeros(usdfs.shape, dtype='uint8')
    
    B = usdfs.shape[0]
    for i in range(B):
        _, binary_images[i] = cv2.threshold(usdfs[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_images > 0
    


def visualization_for_debug(preds, gts, mask_name, save_dir='experiments/check_res'):
    
    B = preds.shape[0]
    for i in range(B):
        pred = 255 * preds[i].astype('float32')
        gt = gts[i].astype('float32')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pred)
        ax[0].axis('off')
        ax[0].set_title('Predictions')  
        ax[1].imshow(gt)
        ax[1].axis('off')
        ax[1].set_title('GroundTruth')  
        plt.savefig(os.path.join(save_dir, mask_name[i]))
        plt.close(fig)
     
        
def produce_out_dir(cfgs):
    if 'ISIC' in cfgs.datasets.basedir:
        clip_model = cfgs.model.clip.pretrain
        val_dir = train_outdir = f'/data/claude/datasets/medisegs/ISIC/Train/ISBI2016_ISIC_Part1_Training_LRP/{clip_model}'
        test_outdir = f'/data/claude/datasets/medisegs/ISIC/Test/ISBI2016_ISIC_Part1_Test_LRP/{clip_model}'
    else:
        raise ValueError(f"Unsupported dataset: {cfgs.datasets.basedir}, what do you wanna do ???")
    for adir in [train_outdir, val_dir, test_outdir]:
        os.makedirs(adir, exist_ok=True)
    return train_outdir, val_dir, test_outdir


def build_dataloaders(cfgs, preprocess, tokenizer, resolution, bz=None):
    if bz is not None:
        cfgs.datasets.batch_size = bz
    if hasattr(cfgs.datasets, 'train'):
        train_dataset = build_dataset(cfgs.datasets.train, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
        train_dl = DataLoader(train_dataset, batch_size=cfgs.datasets.batch_size, num_workers=cfgs.num_workers, shuffle=True)
        num_training_samples = len(train_dataset)
        
        # for i in tqdm(range(num_training_samples)):
        #     _ = train_dataset[i]
    else:
        train_dl = None
        num_training_samples = 0
        
    if hasattr(cfgs.datasets, 'val'):
        val_dataset = build_dataset(cfgs.datasets.val, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
        val_dl = DataLoader(val_dataset, batch_size=cfgs.datasets.batch_size, num_workers=cfgs.num_workers, shuffle=False)
        num_val_samples = len(val_dataset)
        
        # for i in tqdm(range(num_val_samples)):
        #     _ = val_dataset[i]
    else:
        val_dl = None
        num_val_samples = 0
        
    if hasattr(cfgs.datasets, 'test'):
        test_dataset = build_dataset(cfgs.datasets.test, [preprocess, tokenizer, resolution], cfgs.model.clip.inter_mode)
        num_test_samples = len(test_dataset)
        test_dl = DataLoader(test_dataset, batch_size=cfgs.datasets.batch_size, shuffle=False)
        
        # for i in tqdm(range(num_test_samples)):
        #     _ = test_dataset[i]
    else:
        test_dl = None
        num_test_samples = 0
    
    return train_dl, val_dl, test_dl, num_training_samples, num_val_samples, num_test_samples



    
    


