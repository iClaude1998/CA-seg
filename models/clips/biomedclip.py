import torch
from torch import nn
from torch.nn import functional as F

from .utils import text_global_pool



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