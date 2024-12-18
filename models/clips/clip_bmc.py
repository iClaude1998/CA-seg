from torch import nn


<<<<<<< HEAD


=======
>>>>>>> 2cefacd321e42a9c9dc5f831ec0484f7c258df5a
class ClipCBN(nn.Module):
    def __init__(self, backbone, in_features, num_cocncepts):
        super(ClipCBN, self).__init__()
        self.backbone = backbone.float()
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
<<<<<<< HEAD
            param.requires_grad = False

    
=======
            param.requires_grad = False
>>>>>>> 2cefacd321e42a9c9dc5f831ec0484f7c258df5a
