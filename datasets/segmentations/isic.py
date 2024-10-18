import os 
import json
import torch 
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, Optional

from .build_mask_transforms import build_usdf_transforms, build_mask_transforms, refine_image_transforms


class ISIC_seg(Dataset):
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
        prompt_type: str,
        images_dir: str,
        masks_dir: str,
        sdf_dir: str,
        caps_file: Optional[str] = None,
        override_prompt: Optional[str] = None,
        zero_prompt: bool = False,
        image_size=None
    ) -> None:
        super().__init__()

        self.prompt_type = prompt_type
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sdf_dir = sdf_dir

        self.preprocess, self.tokenizer, image_resolution = preprocessors

        self.zero_prompt = zero_prompt
        self.override_prompt = override_prompt
        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)
        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)

        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)
            # random.shuffle(self.imgs_captions)
       
    def __len__(self):
        return len(self.imgs_captions)

    def __getitem__(self, index) -> Dict[str, Any]:
        cap = self.imgs_captions[index]
        mask_name = cap["mask_name"] 
        name = os.path.splitext(cap['img_name'])[0]

        # Ensure the image is read with RGB channels
        image = Image.open(f"{self.images_dir}/{cap['img_name']}").convert("RGB")
        mask = Image.open(f"{self.masks_dir}/{mask_name}")
        sdf_map = np.load(f"{self.sdf_dir}/{name}.npy")

        h, w = mask.height, mask.width

        # Use overrided prompt if provided
        if self.override_prompt:
            prompt = self.override_prompt
        else:
            if self.prompt_type == "random":
                # Randomly select a prompt except the first one i.e., p0
                prompt = random.choice(list(cap["prompts"].values())[1:])
            else:
                prompt = cap["prompts"][self.prompt_type]

            if type(prompt) == list:
                prompt = random.choice(prompt)
        
        image = self.preprocess(image)
        mask = self.mask_transforms(mask)
        sdf_map = self.usdf_transforms(sdf_map)
        text_enc = self.tokenizer(prompt)
        
        return_dict = dict(
                        pixel_values=image,
                        mask=mask,
                        sdf_map=sdf_map,
                        mask_name=cap["mask_name"],
                        height=h,
                        width=w,
                        sentence=prompt,
                        img_path=f"{self.images_dir}/{cap['img_name']}",
                        mask_path=f"{self.masks_dir}/{mask_name}"
        )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
        return return_dict