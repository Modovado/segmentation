from __future__ import annotations
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import List, Tuple

# encoder reduction
from modules.modules import PFR, PFR_SVD, CAFR, CFR

# decoder block
from modules.modules import \
    Head, \
    Block

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
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 scale: int | float,
                 depth: int,
                 # block
                 block,
                 block_kernel_size: int = 5,
                 block_reduction: int = 4,
                 block_norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 block_act_layer: Type[nn.Module] = nn.ReLU,
                 block_split_size: int = 1,
                 block_num_heads: int = 1,
                 block_attn_drop: float = 0.,
                 block_qk_scale: float = None,
                 # gating network
                 block_k: int = 3,
                 block_hid_channels: int = 512,
                 block_pool_channels: int = 16,
                 block_pool_sizes: int = 16,
                 # experts
                 block_num_experts: int = 4,
                 block_loss_coef: float = 0.1,
                 # ca experts
                 block_waves: Tuple[str, ...] = ('db1', 'sym2', 'coif1', 'bior1.3'),
                 # pa experts
                 block_kernel_sizes: tuple = (1, 3, 7, 11),
                 # ffn
                 block_ffn_act_layer: Type[nn.Module] = nn.GELU,
                 block_drop_path: float = 0.1) -> None:
        super().__init__()

        dim = in_channels  # AKA dim

        self.scale = scale

        self.blocks = nn.ModuleList(
            [block(dim=dim,
                   kernel_size=block_kernel_size,
                   reduction=block_reduction,
                   norm_layer=block_norm_layer,
                   act_layer=block_act_layer,
                   split_size=block_split_size,
                   num_heads=block_num_heads,
                   attn_drop=block_attn_drop,
                   qk_scale=block_qk_scale,
                   # gating network
                   k=block_k,
                   hid_channels=block_hid_channels,
                   pool_channels=block_pool_channels,
                   pool_sizes=block_pool_sizes,
                   # experts
                   num_experts=block_num_experts,
                   loss_coef=block_loss_coef,
                   # ca experts
                   waves=block_waves,
                   # pa experts
                   kernel_sizes=block_kernel_sizes,
                   # ffn
                   ffn_act_layer=block_ffn_act_layer,
                   drop_path=block_drop_path)
             for _ in range(depth)]
        )

        self.conv = nn.Sequential(
            #
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,
                x: torch.Tensor,
                skip: torch.Tensor = None) -> torch.Tensor:

        loss = 0

        # [Attention] Block
        for layer in self.blocks:
            x, aux_loss = layer(x)
            loss += aux_loss

        # [SAMPLE]
        # up/downsample (pool)
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        if skip is not None:
            if isinstance(skip, list):  # list
                skip_cat = torch.cat(skip[:], dim=1)
                x = torch.cat([x, skip_cat], dim=1)
            else:  # not list
                x = torch.cat([x, skip], dim=1)

        # [CONV]
        x = self.conv(x)

        return x, loss


class Model(nn.Module):

    def __init__(self,
                 # encoder
                 encoder_model: str = "resnet34",
                 in_channels: int = 4,
                 pretrained: bool = False,

                 reduction_channels_scale: float = 0.5,

                 depth: Tuple[int] = (1, 1, 1),
                 classes: int = 5,

                 split_size: Tuple[int] = (8, 8, 8),  #
                 num_heads: Tuple[int] = (8, 8, 8),  #

                 # block
                 block_kernel_size: int = 5,
                 block_reduction: int = 4,
                 block_norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 block_act_layer: Type[nn.Module] = nn.ReLU,
                 block_attn_drop: float = 0.,
                 block_qk_scale: float = None,
                 # gating network
                 block_k: int = 3,
                 block_hid_channels: int = 512,
                 block_pool_channels: int = 16,
                 block_pool_sizes: int = 16,
                 # experts
                 block_num_experts: int = 4,
                 block_loss_coef: float = 0.1,
                 # ca experts
                 block_waves: Tuple[str, ...] = ('db1', 'sym2', 'coif1', 'bior1.3'),
                 # pa experts
                 block_kernel_sizes: tuple = (1, 3, 7, 11),
                 # ffn
                 block_ffn_act_layer: Type[nn.Module] = nn.GELU,
                 block_drop_path: float = 0.1,
                 # head
                 head_kernel_size: int = 5) -> None:

        super().__init__()

        # [ENCODER]
        self.encoder = Encoder(encoder_model=encoder_model,
                               in_channels=in_channels,
                               pretrained=pretrained)
        # encoder_channels
        encoder_channels = self.encoder.features_info_channels[::-1]
        len_feats = len(self.encoder.features_info_channels)
        # print(f'{encoder_channels=}')
        # print(self.encoder.features_info_channels)  # [64, 64, 128, 256, 512]
        # print(self.encoder.features_info_reduction)  # [2, 4, 8, 16, 32]

        # [REDUCTION]
        # reduction channels
        reduction_channels = [int(channel * reduction_channels_scale) for channel in encoder_channels]
        print(f'{reduction_channels=}')

        # reduction blocks : PFR(!)/CAFR/CFR/PFR_SVD(!)
        reduction_blocks = [
            CFR(in_channels,
                reduced_channels)
            for in_channels, reduced_channels
            in zip(encoder_channels, reduction_channels[:len_feats])]
        # print(f'{reduction_blocks=}')

        # [DECODER]
        # decoder_channels
        decoder_up_in_channels = list(reduction_channels[:len_feats])  # reduction
        decoder_up_skip_channels = list(reduction_channels[1:len_feats]) + [0]
        decoder_up_out_channels = list(reduction_channels[1:len_feats]) + [int(reduction_channels[-1] * 0.5)]

        # decoder_sizes
        decoder_sizes = self.encoder.features_info_reduction[::-1]

        print(f'{decoder_up_in_channels=}')
        print(f'{decoder_up_skip_channels=}')
        print(f'{decoder_up_out_channels=}')
        print(f'{decoder_sizes=}')

        # up/down args
        depth_up, depth_down = depth[::-1], depth
        split_size_up, split_size_down = split_size[::-1], split_size
        num_heads_up, num_heads_down = num_heads[::-1], num_heads
        scale_up, scale_down = 2, 0.5
        up_blocks = [Block, Block, Block]
        # down_blocks = [SP_Block_global_local, SP_Block_global_local, SP_Block_global_local, SP_Block_global_local]

        # [DECODER_UP]
        decoder_up_blocks = [
            Decoder(in_ch, skip_ch, out_ch,
                    scale=scale_up,
                    depth=depth,
                    # block
                    block=block,
                    block_kernel_size=block_kernel_size,
                    block_reduction=block_reduction,
                    block_norm_layer=block_norm_layer,
                    block_act_layer=block_act_layer,
                    block_split_size=split_size,  #
                    block_num_heads=num_heads,  #
                    block_attn_drop=block_attn_drop,
                    block_qk_scale=block_qk_scale,
                    block_k=block_k,
                    block_hid_channels=block_hid_channels,
                    block_pool_channels=block_pool_channels,
                    block_pool_sizes=block_pool_sizes,
                    # experts
                    block_num_experts=block_num_experts,
                    block_loss_coef=block_loss_coef,
                    # ca experts
                    block_waves=block_waves,
                    # pa experts
                    block_kernel_sizes=block_kernel_sizes,
                    # ffn
                    block_ffn_act_layer=block_ffn_act_layer,
                    block_drop_path=block_drop_path,
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

        # [HEAD]
        # output
        self.head = Head(channel_list=decoder_up_out_channels[-1],
                         reduction_list=decoder_sizes[-1],
                         out_channels=classes,
                         kernel_size=head_kernel_size)
        # aux output
        self.aux_head = Head(channel_list=decoder_up_out_channels[1:-1],
                             reduction_list=decoder_sizes[1:-1],
                             out_channels=classes,
                             kernel_size=head_kernel_size)

    def forward(self, x):

        B, C, H, W = x.shape

        # [INIT]
        aux_loss_total = torch.zeros(1).to(x.device)  # aux_loss

        # [ENCODER] # [512, 256, 128, 64]
        features = self.encoder(x)

        # [REDUCTION] # [256, 128, 64, 32]
        for reduction_index, reduction_block in enumerate(self.reduction_blocks):

            x = reduction_block(features[reduction_index])
            features[reduction_index] = x

        # [NECK]

        outputs = []  # list for output features

        # [DECODER_UP]  # [256, 128, 64, 32]
        x = features[0]
        skips = features[1:]  # not include FIRST feature

        for up_index, decoder_up_block in enumerate(self.decoder_up_blocks):

            skip = skips[up_index] if up_index < len(skips) else None  # skip channel
            x, aux_loss = decoder_up_block(x, skip)
            outputs.append(x)
            aux_loss_total += aux_loss

        # [OUTPUT] output and aux_output
        if self.training:  # train
            output = self.head(outputs[-1])
            aux_output = self.aux_head(outputs[:-1])  # concat_output

            return output, aux_output, aux_loss_total

        else:  # val or test
            output = self.head(outputs[-1])
            return output, aux_loss_total


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
