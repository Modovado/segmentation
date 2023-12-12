from __future__ import annotations
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import List, Tuple
# from modules.new_baseline_modules import LK_MBConv_strip, \
#     BFP_Head, Head, \
#     CH_Block_global_local, SP_Block_global_local, \
#     CH_SP_Block_global_local

# from modules.refined_modules import \
#     BFP_Head, Head, \
#     CH_SP_Block_global_local, \
#     CH_SP_Block_global_local_ViT_22B, \
#     CH_SP_PX_Block_global_local_ViT_22B, \
#     Yolo_Neck

# encoder reduction
from modules.modules import PFR, PFR_SVD, CAFR, CFR

from modules.modules import \
    BFP_Head, Head, \
    CH_SP_PX_Block_global_local_ViT_22B, \
    Yolo_Neck


class Encoder(nn.Module):
    """
        Encoder
        - backbone
        - channel reduction
        """

    def __init__(self,
                 encoder_model: str = "resnet34",
                 in_channels: int = 4,
                 pretrained=False):
        super().__init__()

        self.encoder = timm.create_model(encoder_model, features_only=True, pretrained=pretrained, in_chans=in_channels,
                                         out_indices=(1, 2, 3, 4))

        # info
        self.features_info_channels = self.encoder.feature_info.channels()  # [64, 64, 128, 256, 512]
        self.features_info_reduction = self.encoder.feature_info.reduction()  # [2, 4, 8, 16, 32] (256, 128, 64, 32, 16)

        # Encoder_channels_list
        encoder_channels = self.encoder.feature_info.channels()  # [64, 64, 128, 256, 512]
        encoder_channels = encoder_channels[::-1]  # reverse [64, 64, 128, 256, 512] --> [512, 256, 128, 64, 64]

    def forward(self, x):
        features = self.encoder(x)[::-1]  # reverse channels order

        return features


class Decoder(nn.Module):
    """
    Decoder
    - in_channels, skip_channels, out_channels
    - scale(up/down), depth
    - block
        [BLOCK PARAMS]
        - kernel_size
        - dilation
        - expand_ratio
        - act_layer
        - norm_layer
        - split_size:
        - num_heads
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,

                 scale: int | float,
                 depth: int,

                 block,
                 block_kernel_size: int = 5,
                 kernel_size: int = 51,
                 dilation: int = 5,
                 expand_ratio: int | float = 1.,
                 drop_path: float = 0.1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 ffn_act_layer: Type[nn.Module] = nn.GELU,
                 reduction: int = 4,

                 wave: str = 'db1',

                 split_size: int = 1,
                 num_heads: int = 1,
                 attn_drop: float = 0.1,
                 qk_scale: float = None) -> None:
        super().__init__()

        self.scale = scale

        self.blocks = nn.ModuleList(
            [block(in_channels=in_channels,
                   kernel_size=block_kernel_size,
                   dilation=dilation,
                   expand_ratio=expand_ratio,
                   drop_path=drop_path,
                   norm_layer=norm_layer,
                   act_layer=act_layer,
                   ffn_act_layer=ffn_act_layer,
                   reduction=reduction,
                   wave=wave,
                   split_size=split_size,
                   num_heads=num_heads,
                   attn_drop=attn_drop,
                   qk_scale=qk_scale)
             for _ in range(depth)]
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.conv1 = LK_MBConv_strip(in_channels + skip_channels,
        #                              out_channels,
        #                              kernel_size=kernel_size,
        #                              dilation=dilation,
        #                              expand_ratio=expand_ratio,
        #                              norm_layer=norm_layer,
        #                              act_layer=act_layer)
        #
        # self.conv2 = LK_MBConv_strip(out_channels,
        #                              out_channels,
        #                              kernel_size=kernel_size,
        #                              dilation=dilation,
        #                              expand_ratio=expand_ratio,
        #                              norm_layer=norm_layer,
        #                              act_layer=act_layer)

    def forward(self, x, skip=None):

        loss = 0

        # Block
        # for layer in self.blocks:
        #     x, aux_loss = layer(x)
        #     loss += aux_loss

        # up/downsample (pool)
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        if skip is not None:
            if isinstance(skip, list):  # list

                skip_cat = torch.cat(skip[:], dim=1)
                x = torch.cat([x, skip_cat], dim=1)

            else:  # not list
                x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x, loss


class Model(nn.Module):

    def __init__(self,
                 encoder_model: str = "resnet34",
                 in_channels: int = 4,
                 pretrained=False,
                 reduction_channels: Tuple[int] = (256, 128, 64, 32),
                 # [INPUT/SKIP] original channels: (512, 256, 128, 64, 16)<ENCODER / NECK>
                 decoder_channels: Tuple[int] = (128, 64, 32, 16),
                 # [OUTPUT] (128, 64, 32, 16)<DECODER> (256, 128, 64, 32)
                 decoder_sizes: Tuple[int] = (16, 32, 64, 128),
                 # [OUTPUT] (16, 32, 64, 128, 256)<DECODER> (256, 128, 64, 32)
                 depth: Tuple[int] = (1, 1, 1),
                 classes: int = 5,
                 sizes: int = 512,
                 split_size: Tuple[int] = (8, 8, 8),
                 num_heads: Tuple[int] = (8, 8, 8)):
        # (512, 256, 128, 64, 32)
        super().__init__()

        # [ENCODER]
        self.encoder = Encoder(encoder_model=encoder_model,
                               in_channels=in_channels,
                               pretrained=pretrained)

        # [CHANNEL_LIST]
        encoder_channels = self.encoder.features_info_channels[::-1]
        len_feats = len(self.encoder.features_info_channels)

        print(self.encoder.features_info_channels)  # [64, 64, 128, 256, 512]
        print(self.encoder.features_info_reduction)  # [2, 4, 8, 16, 32]

        # [REDUCTION] : PFR CAFR CFR PFR_SVD
        reduction_blocks = [
            PFR_SVD(in_channels,
                reduced_channels)
            for in_channels, reduced_channels
            in zip(encoder_channels, reduction_channels[:len_feats])]

        print(reduction_blocks)

        decoder_up_in_channels = list(reduction_channels[:len_feats])
        decoder_up_skip_channels = list(reduction_channels[1:len_feats]) + [0]
        decoder_up_out_channels = list(decoder_channels)

        print(f'{decoder_up_in_channels=}')
        print(f'{decoder_up_skip_channels=}')
        print(f'{decoder_up_out_channels=}')

        depth_up, depth_down = depth[::-1], \
                               depth

        split_size_up, split_size_down = split_size[::-1], \
                                         split_size

        num_heads_up, num_heads_down = num_heads[::-1], \
                                       num_heads
        scale_up, scale_down = 2, \
                               0.5

        # up_blocks = [CH_Block_global_local, CH_Block_global_local, CH_Block_global_local, CH_Block_global_local]
        # up_blocks = [SP_Block_global_local, SP_Block_global_local, SP_Block_global_local, SP_Block_global_local]
        # up_blocks = [CH_SP_Block_global_local, CH_SP_Block_global_local, CH_SP_Block_global_local,
        #              CH_SP_Block_global_local]
        up_blocks = [CH_SP_PX_Block_global_local_ViT_22B, CH_SP_PX_Block_global_local_ViT_22B,
                     CH_SP_PX_Block_global_local_ViT_22B, CH_SP_PX_Block_global_local_ViT_22B]
        # down_blocks = [SP_Block_global_local, SP_Block_global_local, SP_Block_global_local, SP_Block_global_local]

        # [Decoder_UP] : Spatial Attention SP_Block
        decoder_up_blocks = [
            Decoder(in_ch, skip_ch, out_ch,
                    scale=scale_up,
                    depth=depth,
                    block=block,
                    block_kernel_size=3,
                    drop_path=0.1,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU,
                    ffn_act_layer=nn.GELU,
                    split_size=split_size,
                    num_heads=num_heads,
                    attn_drop=0.1,
                    qk_scale=None
                    ) for in_ch, skip_ch, out_ch,
                          depth,
                          block,
                          split_size,
                          num_heads in
            zip(decoder_up_in_channels, decoder_up_skip_channels, decoder_up_out_channels,
                depth_up,
                up_blocks,
                split_size_up,
                num_heads_up)]

        self.reduction_blocks = nn.ModuleList(reduction_blocks)
        self.decoder_up_blocks = nn.ModuleList(decoder_up_blocks)
        # self.decoder_down_blocks = nn.ModuleList(decoder_down_blocks)

        self.bfp_head = BFP_Head(channel_list=reduction_channels,
                                 size_list=decoder_sizes,
                                 kernel_size=5)

        self.yolo_neck = Yolo_Neck(channel_list=reduction_channels[:len_feats])

        self.upsample_head = Head(channel_list=decoder_channels[-1],
                                  size_list=decoder_sizes[-1],
                                  out_channels=classes,
                                  out_sizes=sizes,
                                  kernel_size=5)
        # aux output
        self.concat_head = Head(channel_list=decoder_channels[1:-1],
                                size_list=decoder_sizes[1:-1],
                                out_channels=classes,
                                out_sizes=sizes,
                                kernel_size=5)

    def _channel_cat(self, input):

        # concat channel-wise
        return torch.cat(input, dim=1)

    def forward(self, x):

        B, C, H, W = x.shape
        # print(B, C, H, W)

        # [INIT]
        aux_loss_total = torch.zeros(1).to(x.device)  # aux_loss

        # [ENCODER] features : [512, 256, 128, 64, 64]
        features = self.encoder(x)

        # [REDUCTION]
        for reduction_index, reduction_block in enumerate(self.reduction_blocks):
            # no_loss
            x = reduction_block(features[reduction_index])
            features[reduction_index] = x

            # loss
            # x, aux_loss = reduction_block(features[reduction_index])
            # features[reduction_index] = x
            # aux_loss_total += aux_loss

        # for feat in features:
        #     print(feat.shape)

        # [try bfp first]
        # bfp = self.bfp_head(features)
        # x = bfp[0]
        # skips = bfp[1:]  # not include FIRST feature

        # [try yolo neck]
        # yolo = self.yolo_neck(features)

        # x = yolo[0]
        # skips = yolo[1:]  # not include FIRST feature

        # [yolo neck with bfp]
        # bfp = self.bfp_head(yolo)
        # x = bfp[0]
        # skips = bfp[1:]  # not include FIRST feature

        # x = features[0]
        # skips = features[1:]  # not include FIRST feature

        # [bfp neck with yolo]
        # bfp = self.bfp_head(features)
        # yolo = self.yolo_neck(bfp)

        # x = yolo[0]
        # skips = yolo[1:]  # not include FIRST feature

        outputs = []  # list for output features

        # [DECODER_UP]  # [256, 128, 64, 32, 16]
        x = features[0]
        skips = features[1:]  # not include FIRST feature

        # print(f'{len(self.decoder_up_blocks)=}')
        for up_index, decoder_up_block in enumerate(self.decoder_up_blocks):
            skip = skips[up_index] if up_index < len(skips) else None  # skip channel
            x, aux_loss = decoder_up_block(x, skip)
            # features[up_index + 1] = x
            outputs.append(x)
            aux_loss_total += aux_loss

        # print(len(outputs))
        # bfp = self.bfp_head(outputs)
        # output = self.upsample_head(bfp[-1])
        # output = self.upsample_head(features[-1])
        #
        #
        # # [OUTPUT] output and aux_output
        # output = features[-1]
        # output = self.upsample_head(bfp[-1])
        # output = self.upsample_head(features[-1])
        #
        if self.training:  # train
            output = self.upsample_head(outputs[-1])

            # for i in outputs:
            #     print(i.shape)
            aux_output = self.concat_head(outputs[:-1])  # concat_output
            # aux_output = self.concat_head(features[:-1])  # concat_output
            return output, aux_output, aux_loss_total

        else:  # val or test
            output = self.upsample_head(outputs[-1])
            return output, aux_loss_total

        # # no more aux
        # return output, aux_loss_total


if __name__ == '__main__':
    from thop import profile

    # B, C, H, W = x.shape
    input = torch.ones(1, 3, 1024, 1024, dtype=torch.float).cuda()

    model = Model(encoder_model='swsl_resnet18', in_channels=3, pretrained=True).cuda()

    # model.eval()
    model.train()

    # print(model)
    # model.train()
    # model.eval()
    # output, output_loss = model(input)
    output, aux_output, output_loss = model(input)
    print(output.shape)
    print(aux_output.shape)
    # print(output_loss)

    macs, parameters = profile(model, inputs=(input,))
    print(f'macs:{macs / 1e9} G, parameter:{parameters / 1e6} M')
