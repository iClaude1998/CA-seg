import os 
import cv2
import zarr
import torch 
import pandas as pd
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset

from .build_mask_transforms import build_mask_transforms, refine_image_transforms, build_usdf_transforms, build_intermap_transforms





class Bioparse_navive(Dataset):
    
    
    def __init__(
        self,
        preprocessors,
        modality: str,
        organ: str,
        root_dir: str,
        split: str,
        view: str = None,
        train_rate: float = 0.8,
        image_size=None,  
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.modality = modality
        self.organ = organ
        self.view = view
        if split == 'train' or split == 'val':
            self.split = 'train'
        else:
            self.split = 'test'

        self.train_rate = train_rate

        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)

        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        self.img_root = os.path.join(self.root_dir, self.modality, self.split)
        self.mask_root = os.path.join(self.root_dir, self.modality, f"{self.split}_mask")
        
        pairs = []
        
        for iname in os.listdir(self.img_root):
            if self.view is not None and self.view not in iname:
                continue
            masks = []
            prefix, suffix = os.path.splitext(iname)
            for organ in self.organ:
                mask_name = f"{prefix}_{organ}{suffix}"
                if not os.path.exists(os.path.join(self.mask_root, mask_name)):
                    continue
                masks.append(mask_name)
            pair = [iname, masks]
            if len(masks) == len(self.organ):
                pairs.append(pair)
        self.pairs = pairs 
    
    def __len__(self):
        return len(self.pairs)
    
    

    def __getitem__(self, index) -> Dict[str, Any]:
        
        img_name, mask_names = self.pairs[index]

        image = Image.open(f"{self.img_root}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        intermap = np.zeros((7, 7))
        prompt = ''

        masks, sdf_maps = [], []
        for mask_name in mask_names:
    
            mask = Image.open(f"{self.mask_root}/{mask_name}").convert("L")
            sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            h, w = mask.height, mask.width
            mask = self.mask_transforms(mask)
            sdf_map = self.usdf_transforms(sdf_map)
            masks.append(mask)
            sdf_maps.append(sdf_map)
            
        intermap = torch.from_numpy(intermap).float()
        text_enc = self.tokenizer(prompt)
        masks = torch.cat(masks, dim=0)
        sdf_maps = torch.cat(sdf_maps, dim=0)
           
        
        return_dict= dict(
                    pixel_values=image,
                    img_name=os.path.split(img_name)[1],
                    mask_name=os.path.split(mask_name)[1],
                    height=h,
                    width=w,
                    img_path=f"{self.img_root}/{img_name}",
                    mask=masks,
                    mask_path=f"{self.mask_root}/{mask_name}",
                    sdf_map=sdf_maps,
                    inter_map=intermap,
                    )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict