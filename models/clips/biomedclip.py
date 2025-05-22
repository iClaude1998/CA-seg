import math
import torch

from torch import nn
from typing import List, Optional
from torch.nn import functional as F


from .utils import text_global_pool


class ImageEncoderWrapper(nn.Module):
    def __init__(self, visual_model, with_head=True, image_size=224):
        super().__init__()
        self.patch_size = visual_model.trunk.patch_embed.patch_size
        self.image_size = image_size
        self.visual_model = visual_model
        self.trunk = self.visual_model.trunk
        self.trunk.patch_embed.strict_img_size = False
        self.trunk.patch_embed.dynamic_img_pad = True
        if with_head:
            self.head = self.visual_model.head
        else:
            self.head = nn.Identity()
    
    def forward(self, x):
        W_in = x.shape[2]
        x = self.trunk.patch_embed(x)
        W = int(math.sqrt(x.shape[1]))
        if W_in != self.image_size:
            pos_embed = self.resample_abs_pos_embed(self.trunk.pos_embed, (W, W), 
                                                num_prefix_tokens=1)
        else:
            pos_embed = self.trunk.pos_embed
        to_cat = [self.trunk.cls_token.expand(x.shape[0], -1, -1)]
        x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
        x = self.trunk.pos_drop(x)
        x = self.trunk.patch_drop(x)
        x = self.trunk.norm_pre(x)
        x = self.trunk.blocks(x)
        x = self.trunk.norm(x)
        x = self.trunk.pool(x)
        x = self.trunk.fc_norm(x)
        x = self.trunk.head_drop(x)
        x = self.trunk.head(x)
        x = self.head(x)
        return x
    
    def pad(self, x):
        H, W = x.shape[2:]
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x
        
    
    def resample_abs_pos_embed(
        self,
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
        # sort out sizes, assume square if old size not provided
        num_pos_tokens = posemb.shape[1]
        num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
            return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=1)


        return posemb
        
    
    



class BiomedCLIPWrapper(nn.Module):
    
    def __init__(self, clip_model, device):
        super().__init__()
        self.inter_mode = False 
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()



class Biomedclip(nn.Module):
    """A wrapper for CLIP to support forward with a list of text inputs."""

    def __init__(self, clip_model):
        """Initialize the wrapper.

        Args:
            clip_model (CLIP): the CLIP model to be wrapped.
        """
        super().__init__()
        # copy all attributes from clip_model to self
        for k, v in vars(clip_model).items():
            setattr(self, k, v)
        
        _ = self.visual.trunk.blocks[-1].register_forward_hook(self.hook_fn)

        
    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype
    
    def hook_fn(self, module, input, output):
        self.intermediates = output

    @torch.no_grad()
    def encode_image(self, image, normalize=False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    @torch.no_grad()
    def encode_text(self, text, normalize=False):
        x = self.text(text)
        return F.normalize(x, dim=-1) if normalize else x
        
    @torch.no_grad()
    def forward(self, image, text_ids, normalize=True):
        image_features = self.encode_image(image, normalize)
        text_features = self.encode_text(text_ids, normalize)
        if not normalize:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        logits_per_text = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        intermediate = F.normalize(self.intermediates[:, 1:], dim=-1)
        
        return logits_per_image, logits_per_text, torch.stack([intermediate, intermediate, intermediate], dim=0)