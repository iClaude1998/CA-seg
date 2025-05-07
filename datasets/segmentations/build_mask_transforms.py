import cv2
import torch 
import numpy as np
import torchvision.transforms as T

from torchvision.transforms import InterpolationMode


def refine_image_transforms(preprocess, image_resolution):
    
    new_preprocess = []
    for transfrom in preprocess.transforms:
        if isinstance(transfrom, T.Resize):
            new_preprocess.append(T.Resize(image_resolution, interpolation=InterpolationMode.BICUBIC, antialias=True))
        elif isinstance(transfrom, T.CenterCrop):
            new_preprocess.append(T.CenterCrop(image_resolution))
        else:
            new_preprocess.append(transfrom)

    return T.Compose(new_preprocess)


def build_usdf_transforms(image_resolution):
    return T.Compose([
                T.ToTensor(),
                T.Resize(image_resolution, interpolation=InterpolationMode.NEAREST, antialias=True),
                T.CenterCrop(image_resolution),
                minmx_normalization_usdf])
    
    
def build_mask_transforms(image_resolution):
    return T.Compose([
                T.Resize(image_resolution, interpolation=InterpolationMode.NEAREST, antialias=True),
                T.CenterCrop(image_resolution),
                T.ToTensor()])
    
    
def build_intermap_transforms(image_resolution, norm='minmax', resize=True, clamp=False):
    transforms = [T.ToTensor(),]
    if resize:
        transforms.append(Resize_Interpretability_Map(image_resolution))
    if clamp:
        transforms.append(T.Lambda(lambda x: torch.clamp(x, min=0)))
    if norm == 'minmax':
        transforms.append(minmx_normalization_usdf)
    elif norm == 'sigmoid':
        transforms.append(sigmoid_normalization_usdf)
    return T.Compose(transforms)
    


class Resize_Interpretability_Map(object):
    
    def __init__(self, image_resolution):
        if isinstance(image_resolution, int):
            image_resolution = (image_resolution, image_resolution)
        self.image_resolution = image_resolution
        self.resize_pipeline = T.Compose([T.Resize(image_resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
                                          T.CenterCrop(image_resolution)])
        
    def __call__(self, interpre_map):
        h, w = interpre_map.shape[-2:]
        if not (h == self.image_resolution[0] and w == self.image_resolution[1]):
            interpre_map = self.resize_pipeline(interpre_map)
        return interpre_map


def minmx_normalization_usdf(usdf):
    # normalize all to be in [0, 1] for guidance image
    if usdf.max() == 0:
        return torch.zeros_like(usdf)
    usdf = (usdf - usdf.min()) / (usdf.max() - usdf.min())
    return usdf


def sigmoid_normalization_usdf(usdf):
    # R -> [-1, 1]
    usdf = torch.sigmoid(usdf)
    usdf = 2 * (usdf - 0.5)
    return torch.clamp(usdf, min=0)


def usdf_function(mask):
    usdf_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    if usdf_map.max() > 0:
        usdf_map = (usdf_map - usdf_map.min()) / (usdf_map.max() - usdf_map.min()) 
    else:
        usdf_map = np.zeros_like(usdf_map).astype(np.float32)
    return usdf_map



