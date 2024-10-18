import torch 
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
    if usdf.max() == 0:
        return torch.zeros_like(usdf)
    usdf = (usdf - usdf.min()) / (usdf.max() - usdf.min())
    return usdf