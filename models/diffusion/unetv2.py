import torch 

from torch import nn 
from .fp_convert import convert_module_to_f16, convert_module_to_f32
from .nn_block import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    layer_norm,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    Upsample,
    CLIPAttentionBlock,
)

NUM_CLASSES = 2


class UNetModel_v2preview(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        channels_clip,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        clip_allignment=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channels_clip = channels_clip
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.clip_allignment = clip_allignment

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.attn_encode_layers = []
        encode_count = 1
        for level, mult in enumerate(channel_mult):
            
            for _ in range(num_res_blocks):
                # 2*ResBlock or 2*[ResBlock, AttentionBlock]
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        CLIPAttentionBlock(ch,
                                           channels_clip,
                                           image_size//ds,
                                           num_heads=1,
                                           num_head_channels=-1,
                                           linear_allignment=clip_allignment,
                                           use_checkpoint=False)
                    )
                    self.attn_encode_layers.append(encode_count)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                encode_count += 1
                self._feature_size += ch
                input_block_chans.append(ch)


            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                encode_count += 1

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            CLIPAttentionBlock(
                ch,
                channels_clip,
                image_size//ds,
                num_heads=1,
                num_head_channels=-1,
                linear_allignment=clip_allignment,
                use_checkpoint=False),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.attn_decode_layers = []
        decode_count = 0
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        CLIPAttentionBlock(ch,
                                           channels_clip,
                                           image_size//ds,
                                           num_heads=1,
                                           num_head_channels=-1,
                                           linear_allignment=clip_allignment,
                                           use_checkpoint=False)
                    )
                    self.attn_decode_layers.append(decode_count)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                decode_count += 1

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels , out_channels, 3, padding=1)),
        )
        features = 32

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    def enhance(self, c, h):
        cu = layer_norm(c.size()[1:])(c)
        hu = layer_norm(h.size()[1:])(h)
        return cu * hu * h
    

    def forward(self, x, timesteps, clip_emb):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        if self.in_channels == 1:
            h = h[:, -1:, ...]
        elif self.in_channels == 2:
            h = h[:, -2:, ...]
        
        clip_emb = clip_emb.type(self.dtype)
        for ind, module in enumerate(self.input_blocks):
            if len(emb.size()) > 2:
                emb = emb.squeeze()
            if ind == self.attn_encode_layers[0]:
                cem = clip_emb[0]
            elif ind == self.attn_encode_layers[1]:
                cem = clip_emb[1]
            else:
                cem = None
            h = module(h, emb, cem)
            hs.append(h)
        # hs[0] -> out of input layer
        # hs[3] -> out of 1st resblock 1
        # hs[6] -> out of 2nd resblock 1
        # hs[9] -> out of 3rd resblock 2
        # hs[12] -> out of 4th resblock 2
        # hs[15] -> out of 5th resblock 4
        # hs[18] -> out of 6th resblock 4
        
        # clip 
        h = self.middle_block(h, emb, clip_emb[-1])
        for ind, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            if ind == self.attn_decode_layers[0]:
                cem = clip_emb[1]
            elif ind == self.attn_decode_layers[1]:
                cem = clip_emb[0]
            elif ind == self.attn_decode_layers[2]:
                cem = clip_emb[0]
            else:
                cem = None
            h = module(h, emb, cem)
        # h = h.type(x.dtype)
        out = self.out(h)
        return out
    


    


