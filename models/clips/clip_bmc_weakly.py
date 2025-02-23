
import copy
import torch
from torch import nn

from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from open_clip import create_model_from_pretrained

from . import ModifiedResNet

            
            
            

class BioMedCLIP_Weakly_Segmentor(nn.Module):
    
    def __init__(self, num_cocncepts, modality, organ):
        super(BioMedCLIP_Weakly_Segmentor, self).__init__()
        
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                                                cache_dir="pretrained/huggingface_hub/biomedclip")
        model.eval()
        self.vision_backbone = model.visual.trunk
        
        self.vision_backbone2 = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        self.vision_backbone2.load_state_dict(torch.load('pretrained/pmc_clip/image_encoder(resnet50).pth'))
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load('pretrained/pmc_clip/text_encoder.pth'))
        text_projection_layer = torch.load('pretrained/pmc_clip/text_projection_layer.pth')
        text_projection_layer = nn.Parameter(text_projection_layer)
        
        self.num_cocncepts = num_cocncepts
        
        txt_templeate = f"a {modality} of {organ}"
        tokens = tokenizer(txt_templeate, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        txt_features = text_encoder(tokens['input_ids'])
        txt_features = txt_features.pooler_output @ text_projection_layer.to(txt_features.pooler_output.device)
        
        self.anchor = F.normalize(txt_features, dim=-1)
        
        self.concept_head = nn.Linear(768, num_cocncepts)
        self.temperature = nn.Parameter(torch.tensor(100.0, requires_grad=True))
        
        self.freeze_backbone()
    

        
    def forward(self, x, cams):
        clip_features = self.vision_backbone(x)
        bmc_features = self.concept_head(clip_features)
        preds = torch.sum(self.temperature * bmc_features[..., None, None] * cams, dim=1, keepdim=True) # [B, 1, H, W]
        
        preds_ = F.interpolate(preds, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pos_x = x * preds_.sigmoid()
        neg_x = x * (1 - preds_.sigmoid())
        
        pos_clip_features = self.vision_backbone2(pos_x)
        neg_clip_features = self.vision_backbone2(neg_x)
        pos_clip_features = F.normalize(pos_clip_features, dim=-1)
        neg_clip_features = F.normalize(neg_clip_features, dim=-1)
        
        return pos_clip_features, neg_clip_features, preds
    
            
             
    def freeze_backbone(self):

        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        for param in self.vision_backbone2.parameters():
            param.requires_grad = False
            
         
            

class PMCCLIP_Weakly_Segmentor(nn.Module):
    
    def __init__(self, num_cocncepts, modality, organ):
        super(PMCCLIP_Weakly_Segmentor, self).__init__()
        
          
        self.vision_backbone = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        self.vision_backbone.load_state_dict(torch.load('pretrained/pmc_clip/image_encoder(resnet50).pth'))
        self.vision_backbone2 = copy.deepcopy(self.vision_backbone)
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load('pretrained/pmc_clip/text_encoder.pth'))
        text_projection_layer = torch.load('pretrained/pmc_clip/text_projection_layer.pth')
        text_projection_layer = nn.Parameter(text_projection_layer)
        
        self.num_cocncepts = num_cocncepts
        
        txt_templeate = f"a {modality} of {organ}"
        tokens = tokenizer(txt_templeate, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        txt_features = text_encoder(tokens['input_ids'])
        txt_features = txt_features.pooler_output @ text_projection_layer.to(txt_features.pooler_output.device)
        
        self.anchor = F.normalize(txt_features, dim=-1)
        
        self.concept_head = nn.Linear(768, num_cocncepts)
        self.temperature = nn.Parameter(torch.tensor(100.0, requires_grad=True))
        
        self.freeze_backbone()
    

        
    def forward(self, x, cams):
        clip_features = self.vision_backbone(x)
        bmc_features = self.concept_head(clip_features)
        preds = torch.sum(self.temperature * bmc_features[..., None, None] * cams, dim=1, keepdim=True) # [B, 1, H, W]
        
        preds_ = F.interpolate(preds, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pos_x = x * preds_.sigmoid()
        neg_x = x * (1 - preds_.sigmoid())
        
        pos_clip_features = self.vision_backbone2(pos_x)
        neg_clip_features = self.vision_backbone2(neg_x)
        pos_clip_features = F.normalize(pos_clip_features, dim=-1)
        neg_clip_features = F.normalize(neg_clip_features, dim=-1)
        
        return pos_clip_features, neg_clip_features, preds
    
            
             
    def freeze_backbone(self):

        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        for param in self.vision_backbone2.parameters():
            param.requires_grad = False