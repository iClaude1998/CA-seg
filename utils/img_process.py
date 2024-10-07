import cv2 
import torch 
from torch.nn import functional as F

def gt2heatmap(gt):
    if gt.ndim == 3:
        gt = gt[..., 0]

    distance_map = cv2.distanceTransform(gt, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_map = distance_map * gt
    normalized = cv2.normalize(distance_map, None, 0, 1, cv2.NORM_MINMAX) 
    return cv2.applyColorMap((255 * normalized[..., None]).astype('uint8'), cv2.COLORMAP_JET)

def inverse_binary(img):
    img = img * -1
    img[img == 0] = 255
    img[img < 0] = 0
    return img.astype('uint8')   

def interpolate_cam(cam, h, w):
    """interpolate cam

    Args:
        cam (torch.tensor): [h, w]
        h (int): target height
        w (int): target width

    Returns:
        torch.tensor: resized cam
    """
    
    # [1, 1, h, w]
    rcam = F.interpolate(cam[None, None, ...], [h, w], mode='bilinear')
    if rcam.max() > 0:
        rcam = (rcam - rcam.min()) / (rcam.max() - rcam.min())
    
    return rcam[0].permute(1, 2, 0)


def process_Relevant_score_batch(Rs, size):
    b, c = Rs.shape[:2]
    Rs = F.interpolate(Rs, size)
    mins = torch.min(Rs.view(b, c, -1), dim=-1, keepdim=True)[0]
    maxs = torch.max(Rs.view(b, c, -1), dim=-1, keepdim=True)[0]
    mask = (maxs > 0).to(torch.float32)
    maxs[maxs == 0] = -1
    Rs = (Rs - mins.unsqueeze(-1)) / (maxs - mins).unsqueeze(-1)
    return Rs * mask.unsqueeze(-1)
    
    