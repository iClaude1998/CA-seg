import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CLIPImageEncoderWrapper(nn.Module):
    def __init__(self, visual_model, with_head=True, image_size=224):
        super().__init__()
        self.with_head = with_head
        self.patch_size = 16
        self.image_size = image_size
        self.visual_model = visual_model
        if with_head:
            self.proj = self.visual_model.proj
    
    def forward(self, x):
        x = self.pad(x)
        x = self.visual_model.conv1(x)
        W_in = x.shape[2]
        if W_in != (self.image_size // self.patch_size):
            pos_embed = self.resample_abs_pos_embed(self.visual_model.positional_embedding, (W_in, W_in), 
                                                num_prefix_tokens=1)
        else:
            pos_embed = self.visual_model.positional_embedding
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + pos_embed
        x = self.visual_model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.visual_model.ln_post(x[:, 0, :])
        if self.with_head:
            x = x @ self.proj
        
       
        return x
    
    def pad(self, x):
        H, W = x.shape[2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
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
        num_pos_tokens = posemb.shape[0]
        num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
            return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:num_prefix_tokens], posemb[num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype).squeeze(0)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=0)


        return posemb