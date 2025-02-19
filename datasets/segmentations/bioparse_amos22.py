import os 
import cv2
import torch 
import numpy as np

from PIL import Image
from typing import Any, Dict
from torch.utils.data import Dataset

from .build_mask_transforms import build_mask_transforms, refine_image_transforms, build_usdf_transforms




class Bioparse_amos22(Dataset):
    """
    A PyTorch Dataset class for handling the Bioparse AMOS22 dataset.
    Args:
        preprocessors (tuple): A tuple containing preprocessing functions and image resolution.
        modality (str): The imaging modality (e.g., 'CT', 'MRI').
        organ (str): The target organ for segmentation.
        root_dir (str): The root directory of the dataset.
        split (str): The dataset split ('train', 'val', 'test').
        train_rate (float, optional): The proportion of training data. Defaults to 0.8.
        image_size (tuple, optional): The desired image size. Defaults to None.
        featuremap_size (tuple, optional): The desired feature map size. Defaults to None.
        gcam_dir (str, optional): The directory for Grad-CAM maps. Defaults to 'gcam'.
    Attributes:
        root_dir (str): The root directory of the dataset.
        modality (str): The imaging modality.
        organ (str): The target organ for segmentation.
        split (str): The dataset split.
        train_rate (float): The proportion of training data.
        img_dir (str): The directory containing images.
        mask_dir (str): The directory containing masks.
        inter_dir (str): The directory containing intermediate maps.
        preprocess (callable): The preprocessing function for images.
        mask_transforms (callable): The transformation function for masks.
        usdf_transforms (callable): The transformation function for SDF maps.
        img_name_list (list): The list of image filenames.
        mask_name_list (list): The list of mask filenames.
        intermap_name_list (list): The list of intermediate map filenames.
    Methods:
        produce_sample_list():
            Generates the list of sample filenames for images, masks, and intermediate maps.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index):
            Retrieves the sample at the specified index.
    Returns:
        dict: A dictionary containing the following keys:
            - pixel_values (Tensor): The preprocessed image tensor.
            - img_name (str): The image filename.
            - mask_name (str): The mask filename.
            - height (int): The height of the image.
            - width (int): The width of the image.
            - img_path (str): The path to the image file.
            - mask (Tensor): The transformed mask tensor.
            - mask_path (str): The path to the mask file.
            - sdf_map (Tensor): The transformed SDF map tensor.
            - inter_map (Tensor): The intermediate map tensor.
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
        # self.sdf_dir = zarr.open(os.path.join(root_dir, modality, f"{split}_usdf"), mode='r')
        self.preprocess, _, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)

        self.mask_transforms = build_mask_transforms(featuremap_size)
        self.usdf_transforms = build_usdf_transforms(featuremap_size)
        self.produce_sample_list()

    
    def produce_sample_list(self):
        
        intermap_list = []
        img_name_list = sorted([f for f in os.listdir(self.img_dir)])
        self.img_name_list = [f for f in img_name_list if os.path.splitext(f)[0].split("_")[-1] == self.organ]
        self.mask_name_list = self.img_name_list
        for name in self.img_name_list:
            prefix, _ = os.path.splitext(name)
            intermap_list.append(f"{prefix}_gcam.npy")
        self.intermap_name_list = intermap_list
            
        
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
        # sdf_map = self.sdf_dir[os.path.splitext(mask_name)[0]][:]
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
                    )