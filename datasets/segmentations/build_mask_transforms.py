import torch 
import torchvision.transforms as T

from torchvision.transforms import InterpolationMode



def build_usdf_transforms(image_resolution):
    return T.Compose([
                T.ToTensor(),
                T.Resize(image_resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_resolution),
                normalization_usdf])
    
def build_mask_transforms(image_resolution):
    return T.Compose([
                T.Resize(image_resolution, interpolation=InterpolationMode.NEAREST, antialias=True),
                T.CenterCrop(image_resolution),
                T.ToTensor()])


def normalization_usdf(usdf):
    # normalize all to be in [0, 1] for guidance image
    usdf = (usdf - usdf.min()) / (usdf.max() - usdf.min())
    return usdf