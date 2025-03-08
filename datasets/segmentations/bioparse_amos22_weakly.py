import os 
import cv2
import torch 
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset
from collections import defaultdict as ddict
from .build_mask_transforms import build_mask_transforms, refine_image_transforms, build_usdf_transforms




class Bioparse_amos22_weakly(Dataset):
    
    
    def __init__(
        self,
        preprocessors,
        modality: str,
        organ: str,
        root_dir: str,
        split: str,
        train_rate: float = 0.8,
        image_size=None,
        featuremap_size=None,
        gcam_dir='gcam',
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.modality = modality
        self.organ = organ
        self.split = split
        if self.split == 'crop_train' or self.split == 'crop_val':
            split = 'crop_train'
        else:
            split = 'crop_test'
        self.train_rate = train_rate
        self.img_dir = os.path.join(root_dir, modality, f'{split}')
        self.mask_dir = os.path.join(root_dir, modality, f"{split}_mask")
        self.inter_dir = os.path.join(root_dir, modality, f"{split}_{gcam_dir}", self.organ)

        self.preprocess, _, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(featuremap_size)
        self.usdf_transforms = build_usdf_transforms(featuremap_size)
        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        self.intermap_name_list = []
        self.labels = []
        self.img_name_list = []
        hash_map = ddict(list)
        img_name_list = sorted([f for f in os.listdir(self.img_dir)])
        for name in img_name_list:
            prefix, suffix = os.path.splitext(name)
            organ = prefix.split('_')[-1]
            pf = prefix.split('_')[:-1]
            img_name = '_'.join(pf)
            hash_map[img_name].append(organ)
        
        for k, v in hash_map.items():
            if self.organ in v:
                self.labels.append(1)
                self.img_name_list.append(f"{k}_{self.organ}{suffix}")
                self.intermap_name_list.append(f"{k}_{self.organ}_gcam.npy")
            else:
                for o in v:
                    self.labels.append(0)
                    self.img_name_list.append(f"{k}_{o}{suffix}")
                    self.intermap_name_list.append(f"{k}_{o}_gcam.npy")
                    
        self.mask_name_list = self.img_name_list
            
        
            
    def __len__(self):
        return len(self.img_name_list)



    def __getitem__(self, index) -> Dict[str, Any]:
        
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.img_dir}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        intermap = np.load(f"{self.inter_dir}/{self.intermap_name_list[index]}")

        mask_name = self.mask_name_list[index]
        mask = Image.open(f"{self.mask_dir}/{mask_name}").convert("L")
        
        sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        h, w = mask.height, mask.width
        mask = self.mask_transforms(mask)
        sdf_map = self.usdf_transforms(sdf_map)
        
           
        return dict(
                    pixel_values=image,
                    img_name=img_name,
                    mask_name=mask_name,
                    height=h,
                    width=w,
                    img_path=f"{self.img_dir}/{img_name}",
                    mask=mask,
                    mask_path=f"{self.mask_dir}/{mask_name}",
                    sdf_map=sdf_map,
                    inter_map=torch.from_numpy(intermap).float(),
                    label=torch.tensor(self.labels[index]),
                    )