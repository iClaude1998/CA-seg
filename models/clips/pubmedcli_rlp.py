import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Optional
from .auxilary import MultiheadAttention


class CustomVisionRLPBlock(nn.Module):
    
    def __init__(self, block):
        super().__init__()
        for k, v in vars(block).items():
            setattr(self, k, v)
        
        self.num_heads = self.attn.num_heads
        self.d_model = self.attn.embed_dim

        self.attn = MultiheadAttention(self.attn)
    
    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad
      
    def set_attn_prob(self, attn_prob):
      self.attn_prob = attn_prob

    def attention(self, x, attn_mask: Optional[torch.Tensor] = None):

        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, attention_probs_forward_hook=self.set_attn_prob,
                         attention_probs_backwards_hook=self.set_attn_grad)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomTransformer(nn.Module):
    def __init__(self, transformer,out_layers=[2, 5, 8, 11]):
        super().__init__()
        # for k, v in transformer.named_parameters():
        #   print(k)
        for k, v in vars(transformer).items():
            setattr(self, k, v)

        for module in ['cls_token', 'pos_embed', 'patch_embed', 'norm', 'projects']:
            if hasattr(self, module):
                delattr(self, module)

        self.resblocks = nn.Sequential(*[CustomVisionRLPBlock(block) for block in self.resblocks])
        self.layers = len(self.resblocks)
        self.out_layers = out_layers

    def forward(self, x: torch.Tensor):
        intermediate = []
        for i in range(self.layers):
            x = self.resblocks[i](x)
            if i in self.out_layers:
                intermediate.append(x)
        return x, intermediate


class CustomVisionTransformer(nn.Module):
    """A customized VisionTransformer to support CAM calculation."""

    def __init__(self, model, output_layers=[2, 5, 8, 11]):
        """Initialize the wrapper.

        Args:
            model (VisionTransformer): the VisionTransformer to be wrapped.
        """
        super().__init__()
        for k, v in vars(model).items():
            setattr(self, k, v)
        self.patch_size = self.conv1.weight[0]
        self.transformer = CustomTransformer(self.transformer, output_layers)

    def _patch_embed(self, x):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        return x

    def _pos_embed(self, x):
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        return x

    def forward_features(self, x):
        # x = self.trunk.patch_embed(x)
        x = self._patch_embed(x)
        x = self._pos_embed(x)
        # x = self.trunk.patch_drop(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, intermediate = self.transformer(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x, intermediate
    

    def forward_head(self, x):
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward(self, x):
        x, intermediate = self.forward_features(x)
        x = self.forward_head(x)
        intermediate = self.forward_patch(intermediate)
        
        return x, intermediate

    def forward_patch(self, x):
        x = torch.stack(x, dim=0).permute(2, 0, 1, 3)[..., 1:, :] # [num_layers, B, num_patch, d]
        x = self.ln_post(x)
        if self.proj is not None:
            x = torch.matmul(x, self.proj)
        return x
        
        


class PUBMEDCLIPWrapper(nn.Module):
    """A wrapper for CLIP to support forward with a list of text inputs."""

    def __init__(self, clip_model, outlayers=[2, 5, 8, 11]):
        """Initialize the wrapper.

        Args:
            clip_model (CLIP): the CLIP model to be wrapped.
        """
        super().__init__()
        # copy all attributes from clip_model to self
        for k, v in vars(clip_model).items():
            print(k)
            setattr(self, k, v)
        self.visual = CustomVisionTransformer(self.visual, outlayers)
        

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        # return CLS+patch embeddings and attention scores of all layers
        features, intermediate = self.visual(image.type(self.dtype))
        return features, intermediate

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text_ids):
        image_features, intermediate = self.encode_image(image)
        text_features = self.encode_text(text_ids)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        logits_per_text = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return logits_per_image, logits_per_text, intermediate
    
    
    
    
class PUBMEDCLIPLRP:
    def __init__(self, clip_model, device):
        self.device = device
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()
        self.image_attn_blocks = list(dict(self.clip_model.visual.transformer.resblocks.named_children()).values())
    
    def __call__(self, image, text_tokens, start_layer=-1):

        batch_size = image.size(0)
        if isinstance(text_tokens, str):
            text_tokens = [text_tokens]

        # forward pass
        logits_per_image, _, intermediate = self.clip_model(image, text_tokens)
        I = torch.eye(logits_per_image.size(0), requires_grad=True).to(self.device)
        one_hot = torch.sum(I * logits_per_image)
        self.clip_model.zero_grad() 
        
        image_attn_blocks = list(dict(self.clip_model.visual.transformer.resblocks.named_children()).values())
        if start_layer == -1: 
            # calculate index of last layer 
            start_layer = len(image_attn_blocks) - 1
        
        num_tokens = image_attn_blocks[0].attn_prob.size(-1)
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_prob.dtype).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_prob], retain_graph=True)[0].detach()
            cam = blk.attn_prob.detach() 
            cam = cam.reshape(-1, cam.size(-1), cam.size(-1))
            grad = grad.reshape(-1, grad.size(-1), grad.size(-1))
            cam = cam * grad 
            cam = cam.reshape(batch_size, -1, grad.size(-1), grad.size(-1))
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        return R[:, 0, 1:], intermediate
    
