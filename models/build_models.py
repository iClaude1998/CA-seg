import clip 
import torch
import open_clip

from .diffusion import UNetModel_v1preview
from .clips import CLIPWrapper, CLIPLRP, PUBMEDCLIPLRP, PUBMEDCLIPWrapper


def load_clip_and_tokenizer(cfgs, device):
    if cfgs.pretrain == 'ViT-B-32':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = CLIPLRP(CLIPWrapper(model, cfgs.outlayers), device)
    elif cfgs.pretrain == "MedICaT":
        model, _ , preprocess = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        tokenizer = open_clip.get_tokenizer('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        model = CLIPLRP(CLIPWrapper(model, cfgs.outlayers), device)
    elif cfgs.pretrain == "Pubmedclip":
        clip_model = torch.load("pretrain_weights/PubMedCLIP_ViT32.pth")
        model, preprocess = clip.load("ViT-B/32", jit=False)
        model.load_state_dict(clip_model['state_dict'])
        tokenizer = clip.tokenize
        model = PUBMEDCLIPLRP(PUBMEDCLIPWrapper(model, cfgs.outlayers), device)
    return model, tokenizer, preprocess


def create_diffusion(cfgs):
    if cfgs.channel_mult == "":
        if cfgs.image_size == 512 or cfgs.image_size == 256 or cfgs.image_size == 224:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfgs.image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif cfgs.image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {cfgs.image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in cfgs.channel_mult.split(","))

    attention_ds = []
    for res in cfgs.attention_resolutions.split(","):
        attention_ds.append(cfgs.image_size // int(res))

    return  UNetModel_v1preview(
                image_size=cfgs.image_size,
                in_channels=cfgs.in_channels,
                model_channels=cfgs.num_channels,
                out_channels=1,#(3 if not learn_sigma else 6),
                num_res_blocks=cfgs.num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=cfgs.dropout,
                channel_mult=channel_mult,
                num_classes=cfgs.num_classes,
                use_checkpoint=cfgs.use_checkpoint,
                use_fp16=cfgs.use_fp16,
                num_heads=cfgs.num_heads,
                num_head_channels=cfgs.num_head_channels,
                num_heads_upsample=cfgs.num_heads_upsample,
                use_scale_shift_norm=cfgs.use_scale_shift_norm,
                resblock_updown=cfgs.resblock_updown,
                use_new_attention_order=cfgs.use_new_attention_order,
                condition=cfgs.condition
    )