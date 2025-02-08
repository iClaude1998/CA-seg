import os 
import torch 
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset

from .build_mask_transforms import build_mask_transforms, refine_image_transforms, usdf_function, build_usdf_transforms


class Bioparse_image(Dataset):
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
        featuremap_size=None,
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
        self.inter_dir = os.path.join(root_dir, modality, f"{split}_gcam", "all")
        self.preprocess, _, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(featuremap_size)
        self.usdf_transforms = build_usdf_transforms(featuremap_size)
        self.produce_sample_list()

    
    def produce_mask_names(self, img_name):
        prefix, suffix = os.path.splitext(img_name)
        mask_names = [f"{prefix}_{organ}{suffix}" for organ in self.organ]
        return mask_names
    
    def produce_sample_list(self):
        img_name_list = [f for f in os.listdir(self.img_dir)]
        self.img_name_list = []
        self.mask_name_list = []
        for img_name in img_name_list:
            mask_names = self.produce_mask_names(img_name)
            mask_paths = [os.path.join(self.mask_dir, mask_name) for mask_name in mask_names]
            if all([os.path.exists(mask_path) for mask_path in mask_paths]):
                self.mask_name_list.append(mask_names)
                self.img_name_list.append(img_name)
        if self.split == 'train':
            self.img_name_list = self.img_name_list[:int(len(self.img_name_list)*self.train_rate)]
            self.mask_name_list = self.mask_name_list[:int(len(self.mask_name_list)*self.train_rate)]
        elif self.split == 'val':
            self.img_name_list = self.img_name_list[int(len(self.img_name_list)*self.train_rate):]
            self.mask_name_list = self.mask_name_list[int(len(self.mask_name_list)*self.train_rate):]
        self.intermap_name_list = [f"{os.path.splitext(img_name)[0]}_gcam.npy" for img_name in self.img_name_list]

   
        
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_name = self.img_name_list[index]
        image = Image.open(f"{self.img_dir}/{img_name}").convert("RGB")
        h, w = image.height, image.width
        image = self.preprocess(image)
        intermap = np.load(f"{self.inter_dir}/{self.intermap_name_list[index]}")

        mask_names = self.mask_name_list[index]
        masks = []
        sdf_maps = []
        for mask_name in mask_names:
            mask = Image.open(f"{self.mask_dir}/{mask_name}").convert("L")
            sdf_map = usdf_function(np.array(mask))
            h, w = mask.height, mask.width
            mask = self.mask_transforms(mask)
            sdf_map = self.usdf_transforms(sdf_map)
            masks.append(mask)
            sdf_maps.append(sdf_map)
        
        masks = torch.cat(masks)
        sdf_maps = torch.cat(sdf_maps)    
        
        return dict(
                    pixel_values=image,
                    img_name=img_name,
                    mask_name=[mask_name for mask_name in mask_names],
                    height=h,
                    width=w,
                    img_path=f"{self.img_dir}/{img_name}",
                    mask=masks,
                    mask_path=[f"{self.mask_dir}/{mask_name}" for mask_name in mask_names],
                    sdf_map=sdf_maps,
                    inter_map=torch.from_numpy(intermap).float(),
                    )