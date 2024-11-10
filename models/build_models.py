import os
import clip 
import torch
import open_clip

from transformers import PreTrainedTokenizerFast

from .diffusion import UNetModel_v1preview, UNetModel_v2preview, UNetModel_v1position, UNetModel_v2position
from .clips import CLIPWrapper, CLIPLRP, PUBMEDCLIPLRP, PUBMEDCLIPWrapper



def load_clip_and_tokenizer(cfgs, device):
    if cfgs.pretrain == 'ViT-B-32':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        resolution = model.visual.preprocess_cfg['size']
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        default_imgsize = 224
        model = CLIPLRP(CLIPWrapper(model, default_imgsize, cfgs.outlayers, cfgs.inter_mode, cfgs.proj_patch), device)
    elif cfgs.pretrain == "MedICaT":
        model, _ , preprocess = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        resolution = model.visual.preprocess_cfg['size']
        tokenizer = open_clip.get_tokenizer('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        default_imgsize = 448
        model = CLIPLRP(CLIPWrapper(model, default_imgsize, cfgs.outlayers), device)
    elif cfgs.pretrain == "Pubmedclip":
        clip_model = torch.load("pretrained/PubMedCLIP_ViT32.pth")
        model, preprocess = clip.load("ViT-B/32", jit=False, download_root="pretrained/clips")
        model.load_state_dict(clip_model['state_dict'])
        resolution = model.visual.input_resolution
        tokenizer = clip.tokenize
        default_imgsize = 224
        model = PUBMEDCLIPLRP(PUBMEDCLIPWrapper(model, default_imgsize, cfgs.outlayers, cfgs.inter_mode, cfgs.proj_patch), device)
    return model, tokenizer, preprocess, resolution



def create_diffusion(cfgs):
    if cfgs.channel_mult == "":
        if cfgs.image_size >= 224:
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
        attention_ds.append(int(res))
    
    if cfgs.version == 'v1':

        return  UNetModel_v1preview(
                    image_size=cfgs.image_size,
                    in_channels=cfgs.in_channels,
                    model_channels=cfgs.num_channels,
                    out_channels=cfgs.out_channels,#(3 if not learn_sigma else 6),
                    num_res_blocks=cfgs.num_res_blocks,
                    attention_resolutions=tuple(attention_ds),
                    dropout=cfgs.dropout,
                    condition_channels=cfgs.condition_channels,
                    channel_mult=channel_mult,
                    num_classes=cfgs.num_classes,
                    use_checkpoint=cfgs.use_checkpoint,
                    use_fp16=cfgs.use_fp16,
                    num_heads=cfgs.num_heads,
                    num_head_channels=cfgs.num_head_channels,
                    num_heads_upsample=cfgs.num_heads_upsample,
                    use_scale_shift_norm=cfgs.use_scale_shift_norm,
                    resblock_updown=cfgs.resblock_updown,
                    use_new_attention_order=cfgs.use_new_attention_order
        )
    elif cfgs.version == 'v2':
        return UNetModel_v2preview(
                        image_size=cfgs.image_size,
                        in_channels=cfgs.in_channels,
                        model_channels=cfgs.num_channels,
                        out_channels=cfgs.out_channels,
                        channels_clip=cfgs.channels_clip,
                        num_res_blocks=cfgs.num_res_blocks,
                        attention_resolutions=tuple(attention_ds),
                        dropout=cfgs.dropout,
                        channel_mult=channel_mult,
                        use_checkpoint=cfgs.use_checkpoint,
                        use_fp16=cfgs.use_fp16,
                        num_heads=cfgs.num_heads,
                        num_head_channels=cfgs.num_head_channels,
                        num_heads_upsample=cfgs.num_heads_upsample,
                        use_scale_shift_norm=cfgs.use_scale_shift_norm,
                        clip_allignment=cfgs.clip_allignment,
        )
    elif cfgs.version == 'v1p':
        return UNetModel_v1position(
                    image_size=cfgs.image_size,
                    in_channels=cfgs.in_channels,
                    pos_embed_dim=cfgs.pos_embed_dim,
                    model_channels=cfgs.num_channels,
                    out_channels=cfgs.out_channels,#(3 if not learn_sigma else 6),
                    combine=cfgs.combine,
                    fuse=cfgs.fuse,
                    num_res_blocks=cfgs.num_res_blocks,
                    attention_resolutions=tuple(attention_ds),
                    dropout=cfgs.dropout,
                    condition_channels=cfgs.condition_channels,
                    channel_mult=channel_mult,
                    num_classes=cfgs.num_classes,
                    use_checkpoint=cfgs.use_checkpoint,
                    use_fp16=cfgs.use_fp16,
                    num_heads=cfgs.num_heads,
                    num_head_channels=cfgs.num_head_channels,
                    num_heads_upsample=cfgs.num_heads_upsample,
                    use_scale_shift_norm=cfgs.use_scale_shift_norm,
                    resblock_updown=cfgs.resblock_updown,
                    use_new_attention_order=cfgs.use_new_attention_order
        )
    elif cfgs.version == 'v2p':
        return UNetModel_v2position(
                        image_size=cfgs.image_size,
                        in_channels=cfgs.in_channels,
                        pos_embed_dim=cfgs.pos_embed_dim,
                        model_channels=cfgs.num_channels,
                        out_channels=cfgs.out_channels,
                        combine=cfgs.combine,
                        fuse=cfgs.fuse,
                        channels_clip=cfgs.channels_clip,
                        num_res_blocks=cfgs.num_res_blocks,
                        attention_resolutions=tuple(attention_ds),
                        dropout=cfgs.dropout,
                        channel_mult=channel_mult,
                        use_checkpoint=cfgs.use_checkpoint,
                        use_fp16=cfgs.use_fp16,
                        num_heads=cfgs.num_heads,
                        num_head_channels=cfgs.num_head_channels,
                        num_heads_upsample=cfgs.num_heads_upsample,
                        use_scale_shift_norm=cfgs.use_scale_shift_norm,
                        clip_allignment=cfgs.clip_allignment,
        )