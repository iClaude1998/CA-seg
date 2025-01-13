
import os 
import json
import torch
import numpy as np 

from PIL import Image
from typing import Optional
from torch.utils.data import Dataset
from .build_mask_transforms import build_usdf_transforms, build_mask_transforms, refine_image_transforms, build_intermap_transforms



class ISIC_image(Dataset):
    r"""
    Image-Text-Mask Dataset
    Args:
        tokenizer_type (TOKENIZER_TYPE): Type of tokenizer to use
        prompt_types (List[PROMPT_TYPE]): List of prompt types to use
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
        caps_file (Optional[str], optional): Path to captions file. Defaults to None.
        img_size (int,int): Size of image. Defaults to (224, 224).
        context_length (int, optional): Context length. Defaults to 77.
        img_transforms (Optional[A.Compose], optional): Transforms to apply to image. Defaults to None.
        mask_transforms (Optional[A.Compose], optional): Transforms to apply to mask. Defaults to None.
        override_prompt (Optional[str], optional): Text uesd for overriding prompt. Defaults to None.
        zero_prompt (bool, optional): Whether to send zero in the place of prompt. Defaults to False.
        data_num (Optional[int | float], optional): Number of data to use. For float Defaults to 1.0.

    Raises:
        TypeError: If tokenizer_type is not one of TOKENIZER_TYPE
        ValueError: If data_num is of type float and is not in range [0., 1.]
    """

    def __init__(
        self,
        preprocessors,
        images_dir: str,
        masks_dir: str,
        sdf_dir: str,
        layercam_dir: Optional[str] = None,
        caps_file: Optional[str] = None,
        image_size=None,
        featuremap_size=None
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sdf_dir = sdf_dir
        self.layercam_dir = layercam_dir
        
        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)
        self.mask_transforms = build_mask_transforms(featuremap_size)
        self.usdf_transforms = build_usdf_transforms(featuremap_size)

        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)
    
    
    def __len__(self):
        return len(self.imgs_captions)
    
    
    def __getitem__(self, idx):
        cap = self.imgs_captions[idx]
        mask_name = cap["mask_name"] 
        name = os.path.splitext(cap['img_name'])[0]

        # Ensure the image is read with RGB channels
        image = Image.open(f"{self.images_dir}/{cap['img_name']}").convert("RGB")
        mask = Image.open(f"{self.masks_dir}/{mask_name}")
        sdf_map = np.load(f"{self.sdf_dir}/{name}.npy")
        inter_map = np.load(f"{self.layercam_dir}/{name}_Segmentation_gcam.npy")
        
        h, w = mask.height, mask.width
        
        image = self.preprocess(image)
        mask = self.mask_transforms(mask)
        sdf_map = self.usdf_transforms(sdf_map)
        inter_map = torch.from_numpy(inter_map).float()
        
        return_dict = dict(
                        pixel_values=image,
                        mask=mask,
                        sdf_map=sdf_map,
                        inter_map=inter_map,
                        mask_name=cap["mask_name"],
                        height=h,
                        width=w,
                        img_path=f"{self.images_dir}/{cap['img_name']}",
                        mask_path=f"{self.masks_dir}/{mask_name}"
        )
        return return_dict


