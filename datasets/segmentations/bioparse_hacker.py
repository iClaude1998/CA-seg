import os 
import cv2
import torch 
import pandas as pd
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset

from .build_mask_transforms import build_mask_transforms, refine_image_transforms, build_usdf_transforms, build_intermap_transforms




class Bioparse_segmentation2_hacker(Dataset):
    """
    Bioparse_segmentation2_hacker is a custom PyTorch Dataset class designed for handling
    segmentation tasks in medical imaging. It supports various preprocessing and transformation
    techniques to prepare the data for training and evaluation.

    Attributes:
        preprocessors (tuple): A tuple containing the preprocessing function, tokenizer, and image resolution.
        modality (str): The imaging modality (e.g., CT, MRI) of the dataset.
        organ (str): The target organ for segmentation.
        root_dir (str): The root directory where the dataset is stored.
        split (str): The dataset split to use ('train', 'val', or 'test').
        train_rate (float): The proportion of data to use for training.
        image_size (int, optional): The desired image size for preprocessing.
        annotation_name (str): The name of the annotation file.
        cbm_dir (str): The directory for CBM-related data.

    Methods:
        __init__: Initializes the dataset with the given parameters.
        produce_sample_list: Generates a list of samples based on the annotations and split.
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
        annotation_name='annotation.csv',   
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
        self.annotation_path = os.path.join(root_dir, modality, annotation_name)
        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)
        self.intermap_transforms = build_intermap_transforms(7, None, True)
        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        anns = pd.read_csv(self.annotation_path)
        
        pattern = '|'.join(self.organ)
        anns = anns[anns['mask_path'].str.contains(pattern, na=False, regex=True)]
        train_split = int(len(anns) * 0.6)
        val_split = int(len(anns) * 0.8)
        
        if self.split == 'train':
            anns = anns[:train_split]
        elif self.split == 'val':
            anns = anns[train_split:val_split]
        else:
            anns = anns[val_split:]
        
        self.img_name_list = anns['img_path'].tolist()
        self.mask_name_list = anns['mask_path'].tolist()
        

    def __len__(self):
        return len(self.img_name_list)


    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.root_dir}/{self.modality}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        
        prompt = ''

        mask_name = self.mask_name_list[index]
        mask = Image.open(f"{self.root_dir}/{self.modality}/{mask_name}").convert("L")
        sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # sdf_map = self.sdf_dir[os.path.splitext(mask_name)[0]][:]
        h, w = mask.height, mask.width
        mask = self.mask_transforms(mask)
        intermap = self.intermap_transforms(sdf_map) / 10
        sdf_map = self.usdf_transforms(sdf_map)
        text_enc = self.tokenizer(prompt)
           
        return_dict= dict(
                    pixel_values=image,
                    img_name=os.path.split(img_name)[1],
                    mask_name=os.path.split(mask_name)[1],
                    height=h,
                    width=w,
                    img_path=f"{self.root_dir}/{self.modality}/{img_name}",
                    mask=mask,
                    mask_path=f"{self.root_dir}/{self.modality}/{mask_name}",
                    sdf_map=sdf_map,
                    inter_map=intermap,
                    )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict
    


class Bioparse_camus_hacker(Dataset):
    """
    Bioparse_camus_hacker is a custom PyTorch Dataset class designed for handling
    segmentation tasks in medical imaging, specifically for the CAMUS dataset. It supports various
    preprocessing and transformation techniques to prepare the data for training and evaluation.

    Attributes:
        preprocessors (tuple): A tuple containing the preprocessing function, tokenizer, and image resolution.
        view (str): The view type (e.g., 2CH, 4CH) of the dataset.
        organ (str): The target organ for segmentation.
        root_dir (str): The root directory where the dataset is stored.
        split (str): The dataset split to use ('train', 'val', or 'test').
        train_rate (float): The proportion of data to use for training.
        image_size (int, optional): The desired image size for preprocessing.
        annotation_name (str): The name of the annotation file.
        cbm_dir (str): The directory for CBM-related data.

    Methods:
        __init__: Initializes the dataset with the given parameters.
        produce_sample_list: Generates a list of samples based on the annotations and split.
    """
    

    def __init__(
        self,
        preprocessors,
        view: str,
        organ: str,
        root_dir: str,
        split: str,
        train_rate: float = 0.8,
        image_size=None,
        annotation_name='annotation.csv',   
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.view = view
        self.organ = organ
        self.split = split
        if self.split == 'train' or self.split == 'val':
            split = 'train'
        else:
            split = 'test'
        self.train_rate = train_rate
        self.annotation_path = os.path.join(root_dir, annotation_name)
        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)
        self.intermap_transforms = build_intermap_transforms(7, None, True)
        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        anns = pd.read_csv(self.annotation_path)
        
        pattern = '|'.join(self.organ)
        anns = anns[anns['mask_path'].str.contains(pattern, na=False, regex=True)]
        pattern = self.view
        anns = anns[anns['mask_path'].str.contains(pattern, na=False, regex=True)]
        train_split = int(len(anns) * 0.6)
        val_split = int(len(anns) * 0.8)
        
        if self.split == 'train':
            anns = anns[:train_split]
        elif self.split == 'val':
            anns = anns[train_split:val_split]
        else:
            anns = anns[val_split:]
        
        self.img_name_list = anns['img_path'].tolist()
        self.mask_name_list = anns['mask_path'].tolist()
        
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.root_dir}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        
        prompt = ''

        mask_name = self.mask_name_list[index]
        mask = Image.open(f"{self.root_dir}/{mask_name}").convert("L")
        sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        h, w = mask.height, mask.width
        mask = self.mask_transforms(mask)
        intermap = self.intermap_transforms(sdf_map) / 10
        sdf_map = self.usdf_transforms(sdf_map)
        text_enc = self.tokenizer(prompt)
           
        return_dict= dict(
                    pixel_values=image,
                    img_name=os.path.split(img_name)[1],
                    mask_name=os.path.split(mask_name)[1],
                    height=h,
                    width=w,
                    img_path=f"{self.root_dir}/{img_name}",
                    mask=mask,
                    mask_path=f"{self.root_dir}/{mask_name}",
                    sdf_map=sdf_map,
                    inter_map=intermap,
                    )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict
    
    

class Bioparse_segmentation_amos22_hacker(Dataset):
    """
    Bioparse_segmentation_amos22_hacker is a custom PyTorch Dataset class designed for handling
    segmentation tasks in medical imaging, specifically for the AMOS22 dataset. It supports various
    preprocessing and transformation techniques to prepare the data for training and evaluation.

    Attributes:
        preprocessors (tuple): A tuple containing the preprocessing function, tokenizer, and image resolution.
        modality (str): The imaging modality (e.g., CT, MRI) of the dataset.
        organ (str): The target organ for segmentation.
        root_dir (str): The root directory where the dataset is stored.
        split (str): The dataset split to use ('crop_train', 'crop_val', or 'crop_test').
        train_rate (float): The proportion of data to use for training.
        image_size (int, optional): The desired image size for preprocessing.
        annotation_name (str): The name of the annotation file.
        cbm_dir (str): The directory for CBM-related data.

    Methods:
        __init__: Initializes the dataset with the given parameters.
        produce_sample_list: Generates a list of samples based on the annotations and split.
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
        annotation_name='annotation.csv',  
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
        self.annotation_path = os.path.join(root_dir, modality, annotation_name)
        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)
        self.intermap_transforms = build_intermap_transforms(7, None, True)
        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        anns = pd.read_csv(self.annotation_path)
        
        pattern = '|'.join(self.organ)
        anns = anns[anns['mask_path'].str.contains(pattern, na=False, regex=True)]
        train_split = int(len(anns) * 0.6)
        val_split = int(len(anns) * 0.8)
        
        if self.split == 'crop_train':
            anns = anns[:train_split]
        elif self.split == 'crop_val':
            anns = anns[train_split:val_split]
        else:
            anns = anns[val_split:]
        
        self.img_name_list = anns['img_path'].tolist()
        self.mask_name_list = anns['mask_path'].tolist()
    

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.root_dir}/{self.modality}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)

        prompt = ''

        mask_name = self.mask_name_list[index]
        mask = Image.open(f"{self.root_dir}/{self.modality}/{mask_name}").convert("L")
        sdf_map = cv2.distanceTransform(np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # sdf_map = self.sdf_dir[os.path.splitext(mask_name)[0]][:]
        h, w = mask.height, mask.width
        mask = self.mask_transforms(mask)
        intermap = self.intermap_transforms(sdf_map) / 10
        sdf_map = self.usdf_transforms(sdf_map)
        text_enc = self.tokenizer(prompt)
           
    
        return_dict= dict(
                    pixel_values=image,
                    img_name=os.path.split(img_name)[1],
                    mask_name=os.path.split(mask_name)[1],
                    height=h,
                    width=w,
                    img_path=f"{self.root_dir}/{self.modality}/{img_name}",
                    mask=mask,
                    mask_path=f"{self.root_dir}/{self.modality}/{mask_name}",
                    sdf_map=sdf_map,
                    inter_map=intermap,
                    )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict
        