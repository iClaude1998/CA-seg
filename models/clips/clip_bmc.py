from torch import nn


class ClipCBN(nn.Module):
    def __init__(self, backbone, in_features, num_cocncepts, one_layer=False):
        super(ClipCBN, self).__init__()
        self.backbone = backbone.float()

        if one_layer == 1:
            self.concept_head = nn.Sequential(*[nn.Linear(in_features, num_cocncepts)])
        else:
            self.concept_head = nn.Sequential(*[nn.Linear(in_features, in_features),
                                                nn.LeakyReLU(),
                                                nn.Linear(in_features, num_cocncepts)])

        self.freeze_backbone()
        
        
    def forward(self, x):
        clip_features = self.backbone(x)
        bmc_features = self.concept_head(clip_features)
        return bmc_features


    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            


