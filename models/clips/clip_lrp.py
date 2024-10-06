import torch
import torch.nn.functional as F

from torch import nn
from typing import Optional

from .utils import text_global_pool
from .auxilary import MultiheadAttention

class CustomVisionRLPBlock(nn.Module):
  
    def __init__(self, block):
        super().__init__()
        for k, v in vars(block).items():
            setattr(self, k, v)
        self.num_heads = self.attn.num_heads
        self.d_model = self.attn.embed_dim
        self.attn = MultiheadAttention(self.attn)
        self.attn_grad = None
        self.attn_prob = None

    def attention(self, x, attn_mask: Optional[torch.Tensor] = None):

        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, attention_probs_forward_hook=self.set_attn_prob,
                         attention_probs_backwards_hook=self.set_attn_grad)[0]
    
    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad
      
    def set_attn_prob(self, attn_prob):
      self.attn_prob = attn_prob
        
    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):

        x = x + self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

  

class CustomTransformer(nn.Module):
    """A customized Transformer to support CAM calculation."""

    def __init__(self, transformer, outlayers):
        """Initialize the wrapper.

        Args:
            transformer (nn.Module): the Transformer to be wrapped.
        """
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
        self.outlayers = outlayers

    def forward(self, x):
        intermediates = []
        for i in range(self.layers):
            x = self.resblocks[i](x)
            if i in self.outlayers:
                intermediates.append(x)
        return x, intermediates


class CustomVisionTransformer(nn.Module):
    """A customized VisionTransformer to support CAM calculation."""

    def __init__(self, model, outlayers):
        """Initialize the wrapper.

        Args:
            model (VisionTransformer): the VisionTransformer to be wrapped.
        """
        super().__init__()
        for k, v in vars(model).items():
            setattr(self, k, v)
        self.patch_size = self.conv1.weight[0]
        self.outlayers = outlayers
        self.transformer = CustomTransformer(self.transformer, outlayers)

    def _patch_embed(self, x):

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        return x

    def _pos_embed(self, x):
        # self.pos_embed_new = upsample_position_embedding(
        #     self.trunk.pos_embed, (h // self.patch_size, w // self.patch_size))
        class_embedding = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([class_embedding, x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        return self.patch_dropout(x)

    def forward_features(self, x):
        # x = self.trunk.patch_embed(x)
        x = self._patch_embed(x)
        x = self._pos_embed(x)
        # x = self.trunk.patch_drop(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, intermediates = self.transformer(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x, intermediates
    
    def _global_pool(self, x: torch.Tensor):
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward_head(self, x):
        x = self.ln_post(x)
        pooled, _ = self._global_pool(x)
        if self.proj is not None:
            pooled = pooled @ self.proj
        return pooled

    def forward(self, x):
        x, intermediates = self.forward_features(x)
        x = self.forward_head(x)
        intermediates = self.forward_patch(intermediates)
        return x, intermediates
    
    def forward_patch(self, x):
        x = torch.stack(x, dim=0).permute(2, 0, 1, 3)[..., 1:, :] # [num_layers, B, num_patch, d]
        x = self.ln_post(x)
        if self.proj is not None:
            x = torch.matmul(x, self.proj)
        return x
    

class CLIPWrapper(nn.Module):
    """A wrapper for CLIP to support forward with a list of text inputs."""

    def __init__(self, clip_model, outlayers):
        """Initialize the wrapper.

        Args:
            clip_model (CLIP): the CLIP model to be wrapped.
        """
        super().__init__()
        # copy all attributes from clip_model to self
        for k, v in vars(clip_model).items():
            setattr(self, k, v)
        self.visual = CustomVisionTransformer(self.visual, outlayers)
        self.transformer.batch_first = True
        

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, normalize=False):
        # return CLS+patch embeddings and attention scores of all layers
        features, intermediates = self.visual(image.type(self.dtype))
        if normalize:
            features = F.normalize(features, dim=-1)
        return features, intermediates

    def encode_text(self, text, normalize=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x
        

    def forward(self, image, text_ids, normalize=False):
        image_features, intermediates = self.encode_image(image, normalize)
        text_features = self.encode_text(text_ids, normalize)
        if not normalize:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        logits_per_text = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return logits_per_image, logits_per_text, intermediates
    


class CLIPLRP:
    def __init__(self, clip_model, device):
        self.device = device
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()
    
    def to(self, device):
        self.clip_model = self.clip_model.to(device)
        self.device = device

    
    def __call__(self, image, text_tokens, start_layer=-1):

        batch_size = image.size(0)
        if isinstance(text_tokens, str):
            text_tokens = [text_tokens]

        # forward pass
        logits_per_image, _, intermediates = self.clip_model(image, text_tokens, normalize=True)
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
        return R[:, 0, 1:], intermediates