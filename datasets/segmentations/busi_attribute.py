import os 
import cv2
import json
import torch 
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, Optional

from .build_mask_transforms import build_usdf_transforms, build_mask_transforms, refine_image_transforms, build_intermap_transforms


class busiattributes_seg(Dataset):
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
        inter_dir: Optional[str] = None,
        inter_layer: Optional[str] = None,
        caps_file: Optional[str] = None,
        image_size=None
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.prompt_type = prompt_type
        self.inter_dir = inter_dir
        self.inter_layer = inter_layer

        self.preprocess, self.tokenizer, image_resolution = preprocessors

        if image_size is not None and image_size != image_resolution:
            image_resolution = image_size
            self.preprocess = refine_image_transforms(self.preprocess, image_resolution)
        self.mask_transforms = build_mask_transforms(image_resolution)
        self.usdf_transforms = build_usdf_transforms(image_resolution)
        if self.inter_layer is not None:
            self.intermap_transforms = build_intermap_transforms(image_resolution)

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
        # mask = Image.open(f"{self.masks_dir}/{mask_name}")
        mask = cv2.imread(f"{self.masks_dir}/{mask_name}", cv2.IMREAD_GRAYSCALE)
        mask = 255 * (mask > 0).astype(np.uint8)
        mask = Image.fromarray(mask)
        sdf_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        if self.inter_dir is not None:
            assert self.inter_layer is not None, "Please provide the layer for the interpretability map"
            inter_map = np.load(f"{self.inter_dir}/{name}_{self.inter_layer}.npy")

        h, w = mask.height, mask.width
        # h, w = mask.shape[:2]

        # Use overrided prompt if provided

        prompt = cap["prompt_attr"]
        sentense_prompt = cap["prompts"][self.prompt_type]

        image = self.preprocess(image)
        mask = self.mask_transforms(mask)
        sdf_map = self.usdf_transforms(sdf_map)
        if self.inter_layer is not None:
            inter_map = self.intermap_transforms(inter_map)
        if prompt is None:
            prompt = {'number': 'no', 'shape': 'no', 'color': 'no', 'size': 'no', 'location': 'nowhere'}
            prompt['number'] = "A"
        for key in prompt:
            if prompt[key] is None:
                prompt[key] = 'no'
        # skin melanoma
        input_prompts = [f"{prompt['number']} tumor",
                         f"{prompt['shape']} tumor", 
                         f"{prompt['size']} tumor", 
                         f"{prompt['number']} tumor on the {prompt['location']}",
                         f"{prompt['number']} tumor",
                         f"{prompt['number']} {prompt['shape']}, tumor",
                         f"{prompt['number']} {prompt['size']}, tumor",
                         f"{prompt['number']} tumor locate on the {prompt['location']}",
                         "a picture of tumor",
                         f"tumor on the {prompt['location']}",
                         ]

        text_enc = self.tokenizer(input_prompts)
        
        return_dict = dict(
                        pixel_values=image,
                        mask=mask,
                        sdf_map=sdf_map,
                        mask_name=cap["mask_name"],
                        height=h,
                        width=w,
                        sentence=str(prompt),
                        img_path=f"{self.images_dir}/{cap['img_name']}",
                        mask_path=f"{self.masks_dir}/{mask_name}"
        )
        if isinstance(text_enc, dict):
            return_dict["input_ids"] = text_enc["input_ids"][0]
            return_dict["attention_mask"] = text_enc["attention_mask"][0]
        elif isinstance(text_enc, torch.Tensor):
            return_dict["input_ids"] = text_enc[0]
            
        if self.inter_layer is not None:
            return_dict["inter_map"] = inter_map
        return return_dict