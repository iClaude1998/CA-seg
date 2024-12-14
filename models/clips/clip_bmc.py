import clip
import torch 
import open_clip

from torch import nn
from open_clip import create_model_from_pretrained


from .clip_pmc import ModifiedResNet



class ClipCBN(nn.Module):
    def __init__(self, backbone_name, num_cocncepts):
        super(ClipCBN, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "PubMedCLIP":
            clip_model = torch.load("pretrained/PubMedCLIP_ViT32.pth")
            model, _ = clip.load("ViT-B/32", jit=False, download_root="pretrained/clips")
            model.load_state_dict(clip_model['state_dict'])
            self.backbone = model.visual
            in_features = 512
        elif backbone_name == "MedICaT":
            model, _ , _ = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
            self.backbone = model.visual
            in_features = 768
        elif backbone_name == "PMC_CLIP":
            model = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
            model.load_state_dict(torch.load('pretrained/pmc_clip/image_encoder(resnet50).pth'))
            self.backbone = model
            in_features = 512
        elif backbone_name == "BiomedCLIP":
            model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir="pretrained/huggingface_hub/biomedclip")
            self.backbone = model.visual 
            in_features = 512

        self.num_cocncepts = num_cocncepts
        self.backbone = self.backbone.float()
        
        self.concept_head = nn.Sequential(*[nn.Linear(in_features, in_features),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features, num_cocncepts)])
        

    def forward(self, x):
        clip_features = self.backbone(x)
        bmc_features = self.concept_head(clip_features)
        return bmc_features

    def encode_image(self, x):
        return self.clip_model.encode_image(x)

    