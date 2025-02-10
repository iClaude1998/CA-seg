import os 
import cv2
import zarr
import torch 
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset

from .build_mask_transforms import build_mask_transforms, refine_image_transforms, build_usdf_transforms, build_intermap_transforms



class Bioparse_segmentation(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing AMOS images and their corresponding masks.
    Attributes:
        root_dir (str): Root directory containing the dataset.
        modality (str): Modality of the images (e.g., CT, MRI).
        organ (str): Organ of interest.
        split (str): Dataset split (e.g., train, val, test).
        img_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        preprocess (callable): Preprocessing function for images.
        image_only (bool): Whether to load only images without masks.
        mask_transforms (callable): Preprocessing function for masks.
        img_name_list (list): List of image filenames.
        mask_name_list (list): List of mask filenames.
    Methods:
        __init__(preprocessors, modality, organ, root_dir, split, image_only=False, image_size=None):
            Initializes the dataset with the given parameters.
        produce_mask_names(img_name):
            Generates the corresponding mask filename for a given image filename.
        produce_sample_list():
            Produces the list of image and mask filenames, ensuring that masks exist for the images.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index):
            Retrieves the sample (image and mask) at the given index.
    """
    

    def __init__(
        self,
        preprocessors,
        modality: str,
        organ: str,
        root_dir: str,
        split: str,
        train_rate: float = 0.8,
        image_size=None,
        resize=False,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.modality = modality
        self.organ = organ
        self.split = split
        if self.split == 'train' or self.split == 'val':
            split = 'train'
        else:
            split = 'test'
        self.train_rate = train_rate
        self.img_dir = os.path.join(root_dir, modality, f'{split}')
        self.mask_dir = os.path.join(root_dir, modality, f"{split}_mask")
        self.inter_dir = os.path.join(root_dir, modality, f"{split}_cbm")
        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)
        self.intermap_transforms = build_intermap_transforms(image_resolution, None, resize)
        self.produce_sample_list()

    
    def produce_real_names(self, img_name):
        mask_names = []
        inter_names = []
        img_names = []
        prefix, suffix = os.path.splitext(img_name)
        for cls in self.organ:
            mask_candidate = f"{prefix}_{cls}{suffix}"
            intermap_candidate = f"{prefix}_{cls}.npy"
            if os.path.exists(f"{self.mask_dir}/{mask_candidate}") and os.path.exists(f"{self.inter_dir}/{intermap_candidate}"):
                mask_names.append(mask_candidate)
                inter_names.append(intermap_candidate)
                img_names.append(img_name)
            
        return mask_names, inter_names, img_names
    
    def produce_sample_list(self):
        img_name_list = sorted([f for f in os.listdir(self.img_dir)])
        self.mask_name_list = []
        self.intermap_name_list = []
        self.img_name_list = []
        for img_name in img_name_list:
            mask_names, inter_names, img_names = self.produce_real_names(img_name)
            self.mask_name_list.extend(mask_names)
            self.intermap_name_list.extend(inter_names)
            self.img_name_list.extend(img_names)
        
        if self.split == 'train':
            self.img_name_list = self.img_name_list[:int(len(self.img_name_list)*self.train_rate)]
            self.mask_name_list = self.mask_name_list[:int(len(self.mask_name_list)*self.train_rate)]
            self.intermap_name_list = self.intermap_name_list[:int(len(self.intermap_name_list)*self.train_rate)]
        elif self.split == 'val':
            self.img_name_list = self.img_name_list[int(len(self.img_name_list)*self.train_rate):]
            self.mask_name_list = self.mask_name_list[int(len(self.mask_name_list)*self.train_rate):]
            self.intermap_name_list = self.intermap_name_list[int(len(self.intermap_name_list)*self.train_rate):]

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.img_dir}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        intermap = np.load(f"{self.inter_dir}/{self.intermap_name_list[index]}")
        
        prompt = ''

        mask_name = self.mask_name_list[index]


        mask = Image.open(f"{self.mask_dir}/{mask_name}").convert("L")
        sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # sdf_map = self.sdf_dir[os.path.splitext(mask_name)[0]][:]
        h, w = mask.height, mask.width
        mask = self.mask_transforms(mask)
        sdf_map = self.usdf_transforms(sdf_map)
        intermap = self.intermap_transforms(intermap)
        text_enc = self.tokenizer(prompt)
           
        
        return_dict= dict(
                    pixel_values=image,
                    img_name=img_name,
                    mask_name=mask_name,
                    height=h,
                    width=w,
                    img_path=f"{self.img_dir}/{img_name}",
                    mask=mask,
                    mask_path=f"{self.mask_dir}/{mask_name}",
                    sdf_map=sdf_map,
                    inter_map=intermap,
                    )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict