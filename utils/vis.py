import os 
import cv2 

from matplotlib import pyplot as plt 
from .img_process import gt2heatmap, interpolate_cam



def vis_batch(batch, save_dir, scores):
    bz = scores.size(0)
    for bid in range(bz):
        raw_height, raw_width =  batch['height'][bid], batch['width'][bid] 
        raw_img = cv2.imread(batch['img_path'][bid], cv2.IMREAD_COLOR)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        raw_mask = cv2.imread(batch['mask_path'][bid], cv2.IMREAD_COLOR)
        heatmap = gt2heatmap(raw_mask)
        
        vis_pred = interpolate_cam(scores[bid], raw_height, raw_width)
        vis_pred = (vis_pred.detach().cpu().numpy() * 255).astype('uint8')
        vis_pred = cv2.applyColorMap(vis_pred, cv2.COLORMAP_JET)
        vis_pred = raw_img * 0.6 + vis_pred * 0.4
        vis_pred = cv2.cvtColor(vis_pred.astype('uint8'), cv2.COLOR_BGR2RGB)
        
        vis_gt = raw_img * 0.6 + heatmap * 0.4
        vis_gt = cv2.cvtColor(vis_gt.astype('uint8'), cv2.COLOR_BGR2RGB)
        
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        axs[0].imshow(vis_pred)
        axs[0].axis('off')
        axs[1].imshow(vis_gt)
        axs[1].axis('off')
        fig.suptitle(batch['sentence'][bid], fontsize=8, y=0.83)
        save_name = os.path.join(save_dir, f"{batch['mask_name'][bid]}")
        plt.savefig(save_name)
        plt.close()