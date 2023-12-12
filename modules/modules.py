from __future__ import annotations
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.models.layers import DropPath

"""
Basic Function
"""
# DWT
class DWT(nn.Module):
    def __init__(self,
                 wave: str = 'db1') -> None:
        super().__init__()

        self.requires_grad = True

        # [mode] db1 -> zero, other -> periodization
        mode = 'zero' if wave == 'db1' else 'periodization'

        self.xfm = DWTForward(J=1, wave=wave, mode=mode)

    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, h, w = x.shape
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Yl, Yh : ll, (lh, hl, hh)
        ll, Yh = self.xfm(x)
        # * -> de-list
        # Yh : torch.Size([b, c, 3, h, w])
        lh, hl, hh = torch.tensor_split(*Yh, 3, dim=2)
        # lh, hl, hh : torch.Size([b, c, 1, h, w]) squeeze -> torch.Size([b, c, h, w])
        lh, hl, hh = self._squeeze(lh), self._squeeze(hl), self._squeeze(hh)

        # return torch.cat((ll, lh, hl, hh), 1), ll, torch.cat((lh, hl), 1), hh
        return [ll, lh, hl, hh]
# IDWT
class IDWT(nn.Module):
    def __init__(self,
                 wave: str = 'db1') -> None:
        super().__init__()
        self.requires_grad = True

        # [mode] db1 -> zero, other -> periodization
        mode = 'zero' if wave == 'db1' else 'periodization'

        self.ifm = DWTInverse(wave=wave, mode=mode)

    def _unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, c, 1, h, w)

    def _cat(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y, z), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (ll, lh, hl, hh) / Yl, Yh : ll, (lh, hl, hh)
        ll, lh, hl, hh = torch.tensor_split(x, 4, dim=1)
        lh, hl, hh = self._unsqueeze(lh), self._unsqueeze(hl), self._unsqueeze(hh)
        # REMEMBER the brackets -->　[]
        Yh = [self._cat(lh, hl, hh)]
        h = self.ifm((ll, Yh))

        return h
# Convolution + BatchNorm + ReLU
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # conv
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2)),
            # norm_layer
            norm_layer(out_channels),
            # act_layer
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x
# Channel Attention from CBAM
class CA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int = 4,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        assert in_channels >= reduction, f'in_channels`{in_channels} should >= reduction`{reduction}`'

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP -- although we use Conv here
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1),
            act_layer(),
            nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1),
        )

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maxPool = self.max_pool(x)
        avgPool = self.avg_pool(x)

        maxOut = self.mlp(maxPool)
        avgOut = self.mlp(avgPool)

        y = self.sigmoid(maxOut + avgOut)

        return x * y
# Spatial Pool(max mean ...) for `Spatial Attention`
class SpatialPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MaxPool, AvgPool
        return torch.cat(
            (torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
# Spatial Attention
class SA(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.sa = nn.Sequential(
            SpatialPool(),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sa(x)
        return attn * x
# Pixel Attention
class PA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 act_layer: Type[nn.Module] = nn.Sigmoid,
                 ) -> None:
        super().__init__()

        dim = in_channels
        self.pa = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
            act_layer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pa(x)
        return attn * x
# FFN
class FFN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 act_layer: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)

        return x
# [Gating Network]
class GN_gate(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int = 512,
                 pool_channels: int = 16,
                 pool_sizes: int = 16,
                 k: int = 2,
                 num_experts: int = 4,
                 eps: float = 1e-10) -> None:
        super().__init__()

        self.k = k  # k of topk
        self.num_experts = num_experts

        # gating network - pooling and w_gate
        # {Conv1x1 - ReLU - Conv1x1 - AvgPool - Conv1x1 - Flatten}
        self.pooling = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(pool_sizes),
            nn.Conv2d(pool_channels, pool_channels, kernel_size=1),
            nn.Flatten(),
        )
        self.w_gate = nn.Parameter(torch.zeros(pool_channels * pool_sizes ** 2, num_experts), requires_grad=True)
        self.eps = eps

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + self.eps)

    def topk(self, t, k=1):
        values, index = t.topk(k=k, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))

        return values, index

    def forward(self, x):
        # [logits]
        logits = self.pooling(x) @ self.w_gate

        # find top k of value and its index
        expert_logits, expert_indices = self.topk(logits, k=self.k)

        # [expert_gates]
        expert_gates = F.softmax(expert_logits, dim=1)

        # <for calculating loss>
        # [gates] : scatter [expert_indices, expert_gates]
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, expert_indices, expert_gates)

        # [importances] : sum(gate) = sum(softmax(logits))
        importance = gates.sum(0)

        # [load] : sum(ceiling(gate)) (gates > 0).sum(0)
        load = (gates > 0).sum(0)

        # [loss] : cv_square(importance) & cv_square(load)
        gating_loss = self.cv_squared(importance) + self.cv_squared(load)

        return expert_gates, expert_indices, gating_loss
# Interpolate
class Interpolate(nn.Module):
    def __init__(self,
                 scale_factor: int | float = 1.,
                 mode: str = 'bicubic',
                 align_corners: bool = False) -> None:
        super().__init__()

        self.intp = F.interpolate

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x
# Lazy ConvBNReLU
class LazyConvBNReLU(nn.Module):
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # conv
            nn.LazyConv2d(out_channels, kernel_size=kernel_size, padding=(kernel_size // 2)),
            norm_layer(out_channels),
            act_layer(),
            # conv
            nn.LazyConv2d(out_channels, kernel_size=kernel_size, padding=(kernel_size // 2)),
            norm_layer(out_channels),
            act_layer(),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x

class UpsampleConvBNReLU(nn.Module):
    """
    up sample -> (channel * 0.5, size * 2) * depth
    using LazyConv
    """

    def __init__(self,
                 in_channels: int,
                 depth: int = 5,
                 kernel_size: int = 5,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 mode: str = 'bicubic',
                 align_corners: bool = False) -> None:
        super().__init__()

        channel_list = [int(in_channels * 0.5 ** n) for n in range(1, depth + 1)]
        # print(channel_list)
        self.upsample = nn.Sequential(*[nn.Sequential(
            Interpolate(scale_factor=2,
                        mode=mode,
                        align_corners=align_corners),
            LazyConvBNReLU(out_channels,
                           kernel_size=kernel_size,
                           norm_layer=norm_layer,
                           act_layer=act_layer),
            )
            for out_channels in channel_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        return x

"""
Feature Reduction
"""
# PFR (PCA Feature Reduction)
class PFR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduced_channels: int,
                 center: bool = False,
                 niter: int = 3,
                 eps: float = 1e-4) -> None:
        super().__init__()

        assert in_channels > reduced_channels, \
            f'in_channels`{in_channels} should > reduced_channels`{reduced_channels}`'

        self.in_channels = in_channels
        self.q = reduced_channels  # n_components
        self.center = center
        self.niter = niter
        self.eps = eps

    def _flatten_dispatch(self,
                          x: torch.Tensor) -> List[torch.Tensor]:
        """
        Flatten and Dispatch 4D input to 2D input, because PCA only accepts 2D input

        Flatten 4D input to 3D input and Dispatch 3D input to 2D input
        Flatten: [B, C, H, W] -> [B, (H * W), C]
        Dispatch: [B, C, H, W] -> B * [C, H, W]
        """

        # Flatten: [B, C, H, W] -> [B, (H * W), C]
        flatten_x = rearrange(x, 'b c h w  -> b (h w) c ')

        # Dispatch: [B, C, H, W] -> B * [C, H, W]
        batchsize = flatten_x.size(0)
        output = torch.tensor_split(flatten_x, batchsize, dim=0)

        return output

    def _combine_unflatten(self,
                           x: List[torch.Tensor],
                           h: int,
                           w: int) -> torch.Tensor:
        """
        Combine and Unflatten 2D input to original 4D input

        Combine 2D dispatch input to 3D input and Unflatten 3D input to 4D
        Combine: B * [(H * W), C] -> [B, (H * W), C]
        Unflatten: [B, (H * W), C] -> [B, C, H, W]
        """
        # Combine: B * [(H * W), C] -> [B, (H * W), C]
        combine_x = torch.cat(x, dim=0)

        # Unflatten: [B, (H * W), C] -> [B, C, H, W]
        output = rearrange(combine_x, 'b (h w) c -> b c h w', h=h, w=w)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W] """
        B, C, H, W = x.shape

        assert C == self.in_channels, f'C `{C}`should == in_channels`{self.in_channels}`'

        # flatten_dispatch
        flatten_dispatch_x = self._flatten_dispatch(x)

        # for storing patches
        patches_output = []

        for index, input in enumerate(flatten_dispatch_x):

            try:
                U, S, V = torch.pca_lowrank(input, q=self.q, center=self.center, niter=self.niter)

            except:  # add some noise
                U, S, V = torch.pca_lowrank(input + self.eps * input.mean() * torch.rand_like(input),
                                            q=self.q, center=self.center, niter=self.niter)

            # projection
            pca_output = input @ V

            # patches
            patches_output.append(pca_output)

        # combine_unflatten
        output = self._combine_unflatten(patches_output, h=H, w=W)

        return output
# PFR_svd (PCA Feature Reduction using SVD)
class PFR_SVD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduced_channels: int) -> None:
        super().__init__()

        assert in_channels > reduced_channels, \
            f'in_channels`{in_channels} should > reduced_channels`{reduced_channels}`'

        self.in_channels = in_channels
        self.reduced_channels = reduced_channels

    def _flatten_dispatch(self,
                          x: torch.Tensor) -> List[torch.Tensor]:
        """
        Flatten and Dispatch 4D input to 2D input, because PCA only accepts 2D input

        Flatten 4D input to 3D input and Dispatch 3D input to 2D input
        Flatten: [B, C, H, W] -> [B, (H * W), C]
        Dispatch: [B, C, H, W] -> B * [C, H, W]
        """

        # Flatten: [B, C, H, W] -> [B, (H * W), C]
        flatten_x = rearrange(x, 'b c h w  -> b (h w) c ')

        # Dispatch: [B, C, H, W] -> B * [C, H, W]
        batchsize = flatten_x.size(0)
        output = torch.tensor_split(flatten_x, batchsize, dim=0)

        return output

    def _combine_unflatten(self,
                           x: List[torch.Tensor],
                           h: int,
                           w: int) -> torch.Tensor:
        """
        Combine and Unflatten 2D input to original 4D input

        Combine 2D dispatch input to 3D input and Unflatten 3D input to 4D
        Combine: B * [(H * W), C] -> [B, (H * W), C]
        Unflatten: [B, (H * W), C] -> [B, C, H, W]
        """
        # Combine: B * [(H * W), C] -> [B, (H * W), C]
        combine_x = torch.cat(x, dim=0)

        # Unflatten: [B, (H * W), C] -> [B, C, H, W]
        output = rearrange(combine_x, 'b (h w) c -> b c h w', h=h, w=w)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W] """
        B, C, H, W = x.shape

        assert C == self.in_channels, f'C `{C}`should == in_channels`{self.in_channels}`'

        # flatten_dispatch
        flatten_dispatch_x = self._flatten_dispatch(x)

        # for storing patches
        patches_output = []

        for index, input in enumerate(flatten_dispatch_x):
            # squeeze
            input = input.squeeze(0)

            print(f'{input.shape=}')

            U, S, V = torch.linalg.svd(input)

            # projection
            pca_output = input @ V[:, :self.reduced_channels]

            # unsqueeze
            pca_output = pca_output.unsqueeze(0)

            # patches
            patches_output.append(pca_output)

        # combine_unflatten
        output = self._combine_unflatten(patches_output, h=H, w=W)

        return output
# Channel Attention Map from CBAM
class CA_Value(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int = 4,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 ) -> None:
        super().__init__()

        assert in_channels >= reduction, f'in_channels`{in_channels} should >= reduction`{reduction}`'

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP -- although we use Conv here
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1),
            act_layer(),
            nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1),
        )

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maxPool = self.max_pool(x)
        avgPool = self.avg_pool(x)

        maxOut = self.mlp(maxPool)
        avgOut = self.mlp(avgPool)

        y = self.sigmoid(maxOut + avgOut)

        return y
# CAFR (CA Feature Reduction)
class CAFR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduced_channels: int) -> None:
        super().__init__()

        assert in_channels > reduced_channels, \
            f'in_channels`{in_channels} should > reduced_channels`{reduced_channels}`'

        self.in_channels = in_channels
        self.reduced_channels = reduced_channels
        self.ca_value = CA_Value(in_channels=in_channels)

    def _dispatch(self,
                  x: torch.Tensor) -> List[torch.Tensor]:
        """
        Dispatch 4D input to multiple 3D input
        Dispatch: [B, C, H, W] -> B * [C, H, W]
        """
        # Dispatch: [B, C, H, W] -> B * [C, H, W]
        batchsize = x.size(0)
        output = torch.tensor_split(x, batchsize, dim=0)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W] """
        B, C, H, W = x.shape

        ca_map = self.ca_value(x)

        # find top k value, index
        _, indices = ca_map.topk(k=self.reduced_channels, dim=1)  # channel-axis(dim=1)

        # Flatten(dim=1): [B, reduced_C, 1, 1] -> [B, reduced_C]
        indices = torch.flatten(indices, start_dim=1)

        # batched index_select
        # per_batch_x :[C, H, W], index : [reduced_C]
        output = torch.cat([torch.index_select(per_batch_x, 0, index).unsqueeze(0)
                            for per_batch_x, index in zip(x, indices)])

        return output
# CFR (Conv Feature Reduction)
class CFR(nn.Module):
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
                 in_channels: int,
                 reduced_channels: int,
                 kernel_size: int = 1,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()

        assert in_channels >= reduced_channels, \
            f'in_channels`{in_channels} should >= reduced_channels`{reduced_channels}`'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      reduced_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            norm_layer(reduced_channels),
            act_layer()
        ) if in_channels > reduced_channels else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels,
                      reduced_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            norm_layer(reduced_channels),
            act_layer()
        )if in_channels > reduced_channels else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

"""
Channel Attention; CA
"""
# [CA_local] WFCA_lmh
class CA_local(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 wave: str = 'db1') -> None:
        super().__init__()

        # dwt / idwt
        self.dwt = DWT(wave=wave)
        self.idwt = IDWT(wave=wave)
        # Body
        self.body_lh = ConvBNReLU(in_channels, in_channels,
                                  kernel_size=kernel_size,
                                  norm_layer=norm_layer,
                                  act_layer=act_layer)

        self.body_m = ConvBNReLU(in_channels * 2, in_channels * 2,
                                 kernel_size=kernel_size,
                                 norm_layer=norm_layer,
                                 act_layer=act_layer)
        # Channel Attention
        self.ca_lh = CA(in_channels=in_channels,
                        reduction=reduction,
                        act_layer=act_layer)

        self.ca_m = CA(in_channels=in_channels * 2,
                       reduction=reduction,
                       act_layer=act_layer)

        self.cat_conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1)
        self.idwt_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # conv after idwt
        # dwt2lmh ([ll, lh, hl, hh] to [low, mid, high])

    def _dwt2lmh(self, x: list):
        ll, lh, hl, hh = x

        low = ll
        mid = torch.cat([lh, hl], dim=1)
        high = hh

        # low, mid, high
        return [low, mid, high]

        # lmh2dwt ([low, mid, high] to [ll, lh, hl, hh])

    def _lmh2dwt(self, x: list):
        low, mid, high = x

        ll = low
        lh, hl = torch.tensor_split(mid, 2, dim=1)
        hh = high

        # ll, lh, hl, hh
        return [ll, lh, hl, hh]

    def _channel_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # concat channel-wise
        return torch.cat(inputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dwt
        # [ll, lh, hl, hh] to [low, mid, high]
        low, mid, high = self._dwt2lmh(self.dwt(x))

        # l = low, m = mid, h = high
        # Body_conv
        low, mid, high = self.body_lh(low), self.body_m(mid), self.body_lh(high)

        # CA
        low = self.ca_lh(self.body_lh(low))
        mid = self.ca_m(self.body_m(mid))
        high = self.ca_lh(self.body_lh(high))

        # channel concat, cat_conv, idwt-> [output]
        x_out = self.idwt(self._channel_cat([low, mid, high]))

        return x_out
# [CA_global]  WFCA_lmh_SWin
class CA_global(nn.Module):
    def __init__(self,
                 dim: int,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.,
                 qk_scale: float = None,
                 wave: str = 'db1') -> None:
        super().__init__()

        in_channels = dim  # AKA in_channels

        # dwt / idwt
        self.dwt = DWT(wave=wave)
        self.idwt = IDWT(wave=wave)

        # qkv conv is kind of working like body conv
        self.to_qkv_conv_lh = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.to_qkv_conv_m = nn.Conv2d(dim * 2, dim * 2 * 3, kernel_size=1, bias=False)

        lh_dim = dim
        m_dim = dim * 2

        lh_head_dim = lh_dim // num_heads
        m_head_dim = m_dim // num_heads

        self.lh_scale = qk_scale or lh_head_dim ** -0.5
        self.m_scale = qk_scale or m_head_dim ** -0.5

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

    def _dwt2lmh(self, x: list):
        ll, lh, hl, hh = x

        low = ll
        mid = torch.cat([lh, hl], dim=1)
        high = hh

        # low, mid, high
        return [low, mid, high]

        # lmh2dwt ([low, mid, high] to [ll, lh, hl, hh])

    def _lmh2dwt(self, x: list):
        low, mid, high = x

        ll = low
        lh, hl = torch.tensor_split(mid, 2, dim=1)
        hh = high

        # ll, lh, hl, hh
        return [ll, lh, hl, hh]

    def _channel_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # concat channel-wise
        return torch.cat(inputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""

        low, mid, high = self._dwt2lmh(self.dwt(x))

        _, _, H, W = low.shape  # [B, 4C, H/2, W/2]

        assert H % self.sp == 0 and W % self.sp == 0, \
            f'{H=} or {W=} cannot be divided by split_size={self.sp} '

        h = H // self.sp
        w = W // self.sp
        hsp = wsp = self.sp

        # qkv(Body)
        low_q, low_k, low_v = self.to_qkv_conv_lh(low).chunk(3, dim=1)
        mid_q, mid_k, mid_v = self.to_qkv_conv_m(mid).chunk(3, dim=1)
        high_q, high_k, high_v = self.to_qkv_conv_lh(high).chunk(3, dim=1)

        # from [B, 4C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), 4C/head]

        low_q, low_k, low_v = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                                                      head=self.num_heads,
                                                      h=h, w=w, hsp=hsp, wsp=wsp), (low_q, low_k, low_v))

        mid_q, mid_k, mid_v = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                                                      head=self.num_heads,
                                                      h=h, w=w, hsp=hsp, wsp=wsp), (mid_q, mid_k, mid_v))

        high_q, high_k, high_v = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                                                         head=self.num_heads,
                                                         h=h, w=w, hsp=hsp, wsp=wsp), (high_q, high_k, high_v))

        # low attn
        low_q = low_q * self.lh_scale
        low_attn = (low_q @ low_k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C
        low_attn = F.softmax(low_attn, dim=-1, dtype=low_attn.dtype)
        low_attn = self.attn_drop(low_attn)
        x_low = low_attn @ low_v

        x_low = rearrange(x_low, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
        # mid attn
        mid_q = mid_q * self.m_scale
        mid_attn = (mid_q @ mid_k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C
        mid_attn = F.softmax(mid_attn, dim=-1, dtype=mid_attn.dtype)
        mid_attn = self.attn_drop(mid_attn)
        x_mid = mid_attn @ mid_v

        x_mid = rearrange(x_mid, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

        # high attn
        high_q = high_q * self.lh_scale
        high_attn = (high_q @ high_k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C
        high_attn = F.softmax(high_attn, dim=-1, dtype=high_attn.dtype)
        high_attn = self.attn_drop(high_attn)
        x_high = high_attn @ high_v

        x_high = rearrange(x_high, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                           head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

        x_out = self.idwt(self._channel_cat([x_low, x_mid, x_high]))
        return x_out
# [CA_moe]
# [CA_global_local]
class CA_local_global(nn.Module):
    def __init__(self,
                 dim: int,
                 # local
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 # global
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.,
                 qk_scale: float = None,
                 wave: str = 'db1') -> None:
        super().__init__()

        in_channels = dim  # AKA in_channels

        # [CA] - WFCA
        self.ca_local_block = CA_local(in_channels=in_channels,
                                       kernel_size=kernel_size,
                                       reduction=reduction,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer,
                                       wave=wave)

        self.ca_global_block = CA_global(dim=dim,
                                         split_size=split_size,
                                         num_heads=num_heads,
                                         attn_drop=attn_drop,
                                         qk_scale=qk_scale,
                                         wave=wave)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [CA]
        ca_out = self.ca_local_block(x) + self.ca_global_block(x)

        return ca_out
# [CA_moe]
class CA_local_global_moe(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.,
                 qk_scale: float = None,
                 # gating network
                 k: int = 3,
                 hid_channels: int = 512,
                 pool_channels: int = 16,
                 pool_sizes: int = 16,
                 # experts
                 num_experts: int = 4,
                 loss_coef: float = 0.1,
                 waves: Tuple[str, ...] = ('db1', 'sym2', 'coif1', 'bior1.3')) -> None:

        super().__init__()

        in_channels = dim  # AKA in_channels

        self.ca_experts = nn.ModuleList([CA_local_global(dim=dim,
                                                         kernel_size=kernel_size,
                                                         reduction=reduction,
                                                         norm_layer=norm_layer,
                                                         act_layer=act_layer,
                                                         split_size=split_size,
                                                         num_heads=num_heads,
                                                         attn_drop=attn_drop,
                                                         qk_scale=qk_scale,
                                                         wave=wave)
                                         for wave in waves])
        # gating network
        self.gating = GN_gate(in_channels=in_channels,
                              hid_channels=hid_channels,
                              pool_channels=pool_channels,
                              pool_sizes=pool_sizes,
                              k=k,
                              num_experts=num_experts)

        self.loss_coef = loss_coef

    def _dispatch(self, input):
        # separate batches
        # [b, c, h, w] to  b * [1, c, h, w]
        batchsize = input.size(0)
        return torch.tensor_split(input, batchsize, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""

        # loss
        loss = 0

        # [GATING NETWORK]
        expert_gates, expert_indices, gating_loss = self.gating(x)  # gating

        # loss
        loss += gating_loss

        # for storing patches(batches)
        patches_output = []

        # dispatch
        dispatch_x = self._dispatch(x)  # B * [C, H, W]

        for i, (input, gates, indices) in enumerate(zip(dispatch_x, expert_gates, expert_indices)):

            # one patch
            patch_output = 0

            # gate(value) * pa_experts[index]
            for gate, index in zip(gates, indices):
                if gate == 0:  # if gate is 0, we don't need to compute experts
                    continue
                patch_output += gate * self.ca_experts[index](input)

            # patches
            patches_output.append(patch_output)

        # concat the dispatches
        output = torch.cat(patches_output, dim=0)

        loss *= self.loss_coef

        return output, loss
"""
Spatial Attention; SA
"""
# [SA_local] LK
class SA_local(nn.Module):
    def __init__(self,
                 in_channels: int) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        self.conv0_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.conv0_2 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)

        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)

        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)

        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        attn = self.sigmoid(attn)

        return attn * u
# [SA_global] CSWin
class SA_global(nn.Module):
    """
    Split Dim w/ Two Attention Operations - Vertical & Horizontal
    """

    def __init__(self,
                 dim: int,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.1,
                 qk_scale: float = None) -> None:
        super().__init__()

        # dim split into 2 parts
        dim = dim // 2

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.lepe_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""
        B, C, H, W = x.shape

        # create a list for later concat
        attened_x = []

        # split channel [:C // 2] , [C // 2:]
        x1, x2 = rearrange(x, 'b (split c) h w -> split b c h w', split=2)
        inputs = [x1, x2]

        # two attn operations
        assert H % self.sp == 0 and W % self.sp == 0, \
            f'{H=} or {W=} cannot be divided by split_size={self.sp} '

        v_h, v_hsp, v_w, v_wsp = 1, H, W // self.sp, self.sp  # vertical
        h_h, h_hsp, h_w, h_wsp = H // self.sp, self.sp, 1, W  # horizontal
        params = [(v_h, v_hsp, v_w, v_wsp), (h_h, h_hsp, h_w, h_wsp)]

        for index, (x, (h, hsp, w, wsp)) in enumerate(zip(inputs, params)):
            # from [B, C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            q, k, v = self.to_qkv_conv(x).chunk(3, dim=1)

            # lepe from v
            lepe = self.lepe_conv(v)

            q, k, v, lepe = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp)  -> (b h w) head (hsp wsp) c',
                                                    head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp), (q, k, v, lepe))

            # attention
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v) + lepe

            # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
            x = rearrange(x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
            attened_x.append(x)

        x = torch.cat(attened_x, dim=1)

        return x
# [SA_global] CSWin + Swin
class SA_global_Swin(nn.Module):
    """
    Split Dim w/ Two Attention Operations - Vertical & Horizontal
    """

    def __init__(self,
                 dim: int,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.1,
                 qk_scale: float = None):
        super().__init__()

        # dim split into 2 parts
        dim = dim // 2

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.lepe_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        #
        self.window_size = (split_size, split_size)  # Wh, Ww

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # print(f'{self.relative_position_bias_table=}')

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        # print(relative_position_index)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""
        B, C, H, W = x.shape

        # create a list for later concat
        attened_x = []

        # split channel [:C // 2] , [C // 2:]
        x1, x2 = rearrange(x, 'b (split c) h w -> split b c h w', split=2)
        inputs = [x1, x2]
        # [CSWIN]
        # two attn operations
        assert H % self.sp == 0 and W % self.sp == 0, \
            f'{H=} or {W=} cannot be divided by split_size={self.sp} '

        # v_h, v_hsp, v_w, v_wsp = 1, H, W // self.sp, self.sp  # vertical
        # h_h, h_hsp, h_w, h_wsp = H // self.sp, self.sp, 1, W  # horizontal
        # params = [(v_h, v_hsp, v_w, v_wsp), (h_h, h_hsp, h_w, h_wsp)]

        # [SWIN]
        h, hsp, w, wsp = H // self.sp, self.sp, W // self.sp, self.sp
        params = [(h, hsp, w, wsp), (h, hsp, w, wsp)]

        for index, (x, (h, hsp, w, wsp)) in enumerate(zip(inputs, params)):
            # from [B, C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            q, k, v = self.to_qkv_conv(x).chunk(3, dim=1)

            # lepe from v
            lepe = self.lepe_conv(v)

            q, k, v, lepe = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head (hsp wsp) c',
                                                    head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp), (q, k, v, lepe))

            # attention
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

            # [SWIN]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            attn = attn + relative_position_bias.unsqueeze(0)
            attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            # [CSWIN]
            x = (attn @ v) + lepe

            # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
            x = rearrange(x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
            attened_x.append(x)

        x = torch.cat(attened_x, dim=1)

        return x
"""
Pixel Attention; PA
"""
class PA_MOE_gate(nn.Module):
    def __init__(self,
                 in_channels: int,
                 # gating network
                 k: int = 2,
                 hid_channels: int = 512,
                 pool_channels: int = 16,
                 pool_sizes: int = 16,
                 # experts
                 num_experts: int = 4,
                 loss_coef: float = 0.1,
                 kernel_sizes: tuple = (1, 3, 7, 11)) -> None:

        super().__init__()
        # Pixel Attention Experts
        self.pa_experts = nn.ModuleList([PA(in_channels=in_channels, kernel_size=kernel_size)
                                         for kernel_size in kernel_sizes])

        # gating network
        self.gating = GN_gate(in_channels=in_channels,
                              hid_channels=hid_channels,
                              pool_channels=pool_channels,
                              pool_sizes=pool_sizes,
                              k=k,
                              num_experts=num_experts)

        self.loss_coef = loss_coef

    def _dispatch(self, input):
        # separate batches
        # [b, c, h, w] to  b * [1, c, h, w]
        batchsize = input.size(0)
        return torch.tensor_split(input, batchsize, dim=0)

    def forward(self, x):

        # loss
        loss = 0

        # [GATING NETWORK]
        expert_gates, expert_indices, gating_loss = self.gating(x)  # gating

        # loss
        loss += gating_loss

        # for storing patches(batches)
        patches_output = []

        # dispatch
        dispatch_x = self._dispatch(x)  # B * [C, H, W]

        for i, (input, gates, indices) in enumerate(zip(dispatch_x, expert_gates, expert_indices)):

            # one patch
            patch_output = 0

            # gate(value) * pa_experts[index]
            for gate, index in zip(gates, indices):
                if gate == 0:  # if gate is 0, we don't need to compute experts
                    continue
                patch_output += gate * self.pa_experts[index](input)

            # patches
            patches_output.append(patch_output)

        # concat the dispatches
        output = torch.cat(patches_output, dim=0)

        loss *= self.loss_coef

        return output, loss

"""
Attention/Block
"""
# consisted of CA SA PA
class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.,
                 qk_scale: float = None,
                 # gating network
                 k: int = 3,
                 hid_channels: int = 512,
                 pool_channels: int = 16,
                 pool_sizes: int = 16,
                 # experts
                 num_experts: int = 4,
                 loss_coef: float = 0.1,
                 # ca experts
                 waves: Tuple[str, ...] = ('db1', 'sym2', 'coif1', 'bior1.3'),
                 # pa experts
                 kernel_sizes: tuple = (1, 3, 7, 11)) -> None:
        super().__init__()

        in_channels = dim  # AKA in_channels

        # CA_moe
        self.ca_moe_block = CA_local_global_moe(dim=dim,
                                                kernel_size=kernel_size,
                                                reduction=reduction,
                                                norm_layer=norm_layer,
                                                act_layer=act_layer,
                                                split_size=split_size,
                                                num_heads=num_heads,
                                                attn_drop=attn_drop,
                                                qk_scale=qk_scale,
                                                # gating network
                                                k=k,
                                                hid_channels=hid_channels,
                                                pool_channels=pool_channels,
                                                pool_sizes=pool_sizes,
                                                num_experts=num_experts,
                                                # ca experts
                                                waves=waves,
                                                loss_coef=loss_coef)
        # SA_local
        self.sa_local_block = SA_local(in_channels=in_channels)
        # SA_global
        self.sa_global_block = SA_global_Swin(dim=dim,
                                              split_size=split_size,
                                              num_heads=num_heads,
                                              attn_drop=attn_drop,
                                              qk_scale=qk_scale)
        # PA_moe
        self.pa_moe_block = PA_MOE_gate(in_channels=in_channels,
                                        k=k,
                                        hid_channels=hid_channels,
                                        pool_channels=pool_channels,
                                        pool_sizes=pool_sizes,
                                        num_experts=num_experts,
                                        kernel_sizes=kernel_sizes,
                                        loss_coef=loss_coef,
                                        )

        self.concat_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        # self.aggre_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = 0

        # [CA_MOE]
        ca_out, ca_loss = self.ca_moe_block(x)

        # [SA]
        sa_out = self.sa_local_block(x) + self.sa_global_block(x)  # SA

        # [PA_MOE]
        pa_out, pa_loss = self.pa_moe_block(x)

        ca_sa_out = self.concat_conv(torch.cat((ca_out, sa_out), dim=1))
        x_out = self.concat_conv(torch.cat((ca_sa_out, pa_out), dim=1))

        loss += (ca_loss + pa_loss)  # moe aux_loss

        return x_out, loss
# ViT-22B
class Block(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 split_size: int = 2,
                 num_heads: int = 4,
                 attn_drop: float = 0.,
                 qk_scale: float = None,
                 # gating network
                 k: int = 3,
                 hid_channels: int = 512,
                 pool_channels: int = 16,
                 pool_sizes: int = 16,
                 # experts
                 num_experts: int = 4,
                 loss_coef: float = 0.1,
                 # ca experts
                 waves: Tuple[str, ...] = ('db1', 'sym2', 'coif1', 'bior1.3'),
                 # pa experts
                 kernel_sizes: tuple = (1, 3, 7, 11),
                 # ffn
                 ffn_act_layer: Type[nn.Module] = nn.GELU,
                 drop_path: float = 0.1) -> None:
        super().__init__()

        in_channels = dim  # AKA in_channels

        self.norm = nn.BatchNorm2d(in_channels)

        self.attn = Attention(dim=dim,
                              kernel_size=kernel_size,
                              reduction=reduction,
                              norm_layer=norm_layer,
                              act_layer=act_layer,
                              split_size=split_size,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              qk_scale=qk_scale,
                              # gating network
                              k=k,
                              hid_channels=hid_channels,
                              pool_channels=pool_channels,
                              pool_sizes=pool_sizes,
                              # experts
                              num_experts=num_experts,
                              loss_coef=loss_coef,
                              # ca experts
                              waves=waves,
                              # pa experts
                              kernel_sizes=kernel_sizes
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = FFN(in_channels,
                       act_layer=ffn_act_layer)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        loss = 0

        # x + DROP_PATH(FFN(norm_x) + ATTN(norm_x)) DROP_PATH is optional
        # norm_x = self.norm(x)
        # x = self.residual(x) + self.drop_path(self.attn(norm_x) + self.ffn(norm_x))
        # x = x + self.attn(norm_x) + self.ffn(norm_x)

        # norm_first
        norm_x = self.norm(x)

        # attn & ffn
        attn_x, aux_loss = self.attn(self.norm(x))
        loss += aux_loss
        ffn_x = self.ffn(norm_x)

        #  x = x + self.attn(norm_x) + self.ffn(norm_x)
        x = x + attn_x + ffn_x

        # y, aux_loss = self.attn(self.norm(x))
        # x = x + y
        # loss += aux_loss
        # x = x + self.ffn(self.norm2(x))

        # norm_last
        # y, aux_loss = self.attn(x)
        # x = x + self.norm1(y)
        # loss += aux_loss
        # x = x + self.norm2(self.ffn(x))
        return x, loss

"""
Head
"""
# [Final_Head]
class Head(nn.Module):

    def __init__(self,
                 channel_list: List[int] | Tuple[int] | int = (256, 128, 64, 32, 16),
                 reduction_list: Tuple[int] = (16, 32, 64, 128, 256),
                 out_channels: int = 4,
                 kernel_size: int = 5,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        if isinstance(channel_list, int):
            channel_list = [channel_list]

        if isinstance(reduction_list, int):
            reduction_list = [reduction_list]

        assert len(channel_list) == len(reduction_list)
        num_levels = len(channel_list)

        self.num_levels = num_levels

        upsample_blocks = []

        for i, (in_channel, in_reduction) in enumerate(zip(channel_list, reduction_list)):

            # round & to int
            num_upsample = self._log_scale(in_reduction)

            upsample_blocks.append(
                UpsampleConvBNReLU(in_channel,
                                   depth=num_upsample,
                                   kernel_size=kernel_size,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))

        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        self.out_conv = nn.LazyConv2d(out_channels, kernel_size=1)

    def _log_scale(self, x, base=2):
        import math
        return int(math.log(x, base))

    def _channel_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        # concat channel-wise
        return torch.cat(inputs, dim=1)

    def forward(self, x: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:

        if not isinstance(x, list):
            x = [x]

        assert len(x) == self.num_levels

        feats = []

        for i, (feat, upsample_block) in enumerate(zip(x, self.upsample_blocks)):
            feats.append(upsample_block(feat))

        x_out = self.out_conv(self._channel_cat(feats))

        return x_out
