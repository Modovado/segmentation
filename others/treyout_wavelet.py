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


# DWT / IDWT
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


# Channel Attention from CBAM
class CA(nn.Module):
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


########################################################################################################################

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


########################################################################################################################

# [CA_local] non-WFCA
class CA_local_non_WFCA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 wave: str = 'db1') -> None:
        super().__init__()

        # Body
        self.body = ConvBNReLU(in_channels, in_channels,
                               kernel_size=kernel_size,
                               norm_layer=norm_layer,
                               act_layer=act_layer)

        # Channel Attention
        self.ca = CA(in_channels=in_channels,
                     reduction=reduction,
                     act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x_out = self.ca(x)

        return x_out


# [CA_global] non-WFCA
class CA_global_non_WFCA(nn.Module):
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
                 wave: str = 'db1'):
        super().__init__()

        in_channels = dim  # AKA in_channels

        self.to_qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""

        _, _, H, W = x.shape  # [B, 4C, H/2, W/2]

        assert H % self.sp == 0 and W % self.sp == 0, \
            f'{H=} or {W=} cannot be divided by split_size={self.sp} '

        h = H // self.sp
        w = W // self.sp
        hsp = wsp = self.sp

        # qkv
        q, k, v = self.to_qkv_conv(x).chunk(3, dim=1)

        # from [B, 4C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), 4C/head]

        q, k, v = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head c (hsp wsp)',
                                          head=self.num_heads,
                                          h=h, w=w, hsp=hsp, wsp=wsp), (q, k, v))

        # attn
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = attn @ v

        x_out = rearrange(x, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

        return x_out


# [CA_local] <adjust> WFCA_lmh
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

        # channel concat, cat_conv, idwt and, idwt_conv-> [output]
        x_out = self.idwt(self._channel_cat([low, mid, high]))

        return x_out


# [CA_global] <adjust> WFCA_lmh_CSWin
class CA_global(nn.Module):
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

                 wave: str = 'db1'):
        super().__init__()

        in_channels = dim  # AKA in_channels

        # dwt / idwt
        self.dwt = DWT(wave=wave)
        self.idwt = IDWT(wave=wave)

        # i think qkv conv is working like body conv
        # Body
        # self.body_lh = ConvBNReLU(in_channels, in_channels,
        #                           kernel_size=kernel_size,
        #                           norm_layer=norm_layer,
        #                           act_layer=act_layer)
        #
        # self.body_m = ConvBNReLU(in_channels * 2, in_channels * 2,
        #                          kernel_size=kernel_size,
        #                          norm_layer=norm_layer,
        #                          act_layer=act_layer)

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

        self.cat_conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1)
        self.idwt_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # conv after idwt

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

        # Body
        # low, mid, high = self.body_lh(low), self.body_m(mid), self.body_lh(high)

        # qkv
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

        # low = low + ca_low
        # mid = mid + ca_mid
        # high = high + ca_high

        x_out = self.idwt(self._channel_cat([x_low, x_mid, x_high]))
        # x_out = self.idwt_conv(self.idwt(self.cat_conv(self._channel_cat([low, mid, high]))))
        return x_out


# [SA_local] LK
class SA_local(nn.Module):
    def __init__(self,
                 in_channels: int):
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

            q, k, v, lepe = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp)   -> (b h w) head (hsp wsp) c',
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


# [SA_global] shifted_CSWin
class SA_global_shifted(nn.Module):
    """
    [Shift] features - default (0, 2) - no shift, shift two
    [Split] Dim w/ Two Attention Operations - Vertical & Horizontal then concat
    """

    def __init__(self,
                 dim: int,
                 split_size: int = 4,
                 num_heads: int = 8,
                 attn_drop: float = 0.1,
                 qk_scale: float = None,
                 shift_sizes: Tuple[int, ...] = (0, 2)):
        super().__init__()

        # shift_sizes
        self.shift_sizes = shift_sizes

        assert dim // 2 >= num_heads, \
            f'after dim splitting into 2, `{dim // 2}` should be >= num_heads `{num_heads}`'

        # dim split into 2 parts
        split_dim = dim // 2

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

        head_dim = split_dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv_conv = nn.Conv2d(split_dim, split_dim * 3, kernel_size=1, bias=False)
        self.lepe_conv = nn.Conv2d(split_dim, split_dim, kernel_size=3, stride=1, padding=1, groups=split_dim)

        self.shift_concat = nn.Conv2d(dim * len(shift_sizes), dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""
        B, C, H, W = x.shape

        # create a list for later concat
        attend_shifted_x = []

        for shift_size in self.shift_sizes:

            # create a list for later concat
            attened_splitted_x = []

            """shift"""
            if shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            else:
                shifted_x = x

            """split"""
            # split channel [:C // 2] , [C // 2:]
            x1, x2 = rearrange(shifted_x, 'b (split c) h w -> split b c h w', split=2)
            inputs = [x1, x2]

            # two attn operations
            assert H % self.sp == 0 and W % self.sp == 0, \
                f'{H=} or {W=} cannot be divided by split_size={self.sp} '

            v_h, v_hsp, v_w, v_wsp = 1, H, W // self.sp, self.sp  # vertical
            h_h, h_hsp, h_w, h_wsp = H // self.sp, self.sp, 1, W  # horizontal
            params = [(v_h, v_hsp, v_w, v_wsp), (h_h, h_hsp, h_w, h_wsp)]

            for index, (x_split, (h, hsp, w, wsp)) in enumerate(zip(inputs, params)):

                # from [B, C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
                q, k, v = self.to_qkv_conv(x_split).chunk(3, dim=1)

                # lepe from v
                lepe = self.lepe_conv(v)

                q, k, v, lepe = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp)   -> (b h w) head (hsp wsp) c',
                                                        head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp),
                                    (q, k, v, lepe))

                # attention
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
                attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
                attn = self.attn_drop(attn)

                splitted_x = (attn @ v) + lepe

                # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
                splitted_x = rearrange(splitted_x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                                       head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

                # reverse cyclic shift
                if shift_size > 0:
                    splitted_x = torch.roll(splitted_x, shifts=(shift_size, shift_size), dims=(1, 2))

                attened_splitted_x.append(splitted_x)

            # concat
            concat_splitted_x = torch.cat(attened_splitted_x, dim=1)
            attend_shifted_x.append(concat_splitted_x)

        # concat
        # concat_shifted_x = torch.cat(attend_shifted_x, dim=1)
        # output = self.shift_concat(concat_shifted_x)

        # aggregate
        output = torch.add(*attend_shifted_x)
        # output = self.shift_concat(concat_shifted_x)

        return output


class PA_global(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 1,
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


class PA_local(nn.Module):
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


# ADD　PA
class CA_SA_PA_global_local(nn.Module):
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

                 wave: str = 'db2'):
        super().__init__()

        in_channels = dim  # AKA in_channels
        # WFCA
        self.ca_local_block = CA_local(in_channels=in_channels,
                                       kernel_size=kernel_size,
                                       reduction=reduction,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer,
                                       wave=wave)

        self.ca_global_block = CA_global(dim=dim,
                                         kernel_size=kernel_size,
                                         reduction=reduction,
                                         norm_layer=norm_layer,
                                         act_layer=act_layer,
                                         split_size=split_size,
                                         num_heads=num_heads,
                                         attn_drop=attn_drop,
                                         qk_scale=qk_scale,
                                         wave=wave)

        # WFCA
        # self.ca_local_block = CA_local_(in_channels=in_channels,
        #                                 kernel_size=kernel_size,
        #                                 reduction=reduction,
        #                                 norm_layer=norm_layer,
        #                                 act_layer=act_layer,
        #                                 wave=wave)
        #
        # self.ca_global_block = CA_global_(dim=dim,
        #                                   kernel_size=kernel_size,
        #                                   reduction=reduction,
        #                                   norm_layer=norm_layer,
        #                                   act_layer=act_layer,
        #                                   split_size=split_size,
        #                                   num_heads=num_heads,
        #                                   attn_drop=attn_drop,
        #                                   qk_scale=qk_scale,
        #                                   wave=wave)

        self.sa_local_block = SA_local(in_channels=in_channels)
        # self.sa_local_block = SA()
        # self.sa_global_block = SA_global(dim=dim,
        #                                  split_size=split_size,
        #                                  num_heads=num_heads,
        #                                  attn_drop=attn_drop,
        #                                  qk_scale=qk_scale)

        self.sa_global_block = SA_global_Swin(dim=dim,
                                              split_size=split_size,
                                              num_heads=num_heads,
                                              attn_drop=attn_drop,
                                              qk_scale=qk_scale)

        self.pa_local_block = PA_local(in_channels=dim)

        self.pa_global_block = PA_global(in_channels=dim)

        self.concat_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.aggre_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SA + CA
        # x_out = self.sa_local_block(x) + self.sa_global_block(x)
        # x_out = self.ca_local_block(x_out) + self.ca_global_block(x_out)

        # CA + SA
        # x_out = self.ca_local_block(x) + self.ca_global_block(x)
        # x_out = self.sa_local_block(x_out) + self.sa_global_block(x_out)

        # CA & SA parallel
        # concat
        # ca_out = self.concat_conv(torch.cat((self.ca_local_block(x), self.ca_global_block(x)), dim=1))
        # sa_out = self.concat_conv(torch.cat((self.sa_local_block(x), self.sa_global_block(x)), dim=1))
        # pa_out = self.concat_conv(torch.cat((self.pa_local_block(x), self.pa_global_block(x)), dim=1))

        # ca_out = self.aggre_conv(self.ca_local_block(x) + self.ca_global_block(x))
        # sa_out = self.aggre_conv(self.sa_local_block(x) + self.sa_global_block(x))
        # pa_out = self.aggre_conv(self.pa_local_block(x) + self.pa_global_block(x))

        ca_out = self.ca_local_block(x) + self.ca_global_block(x)
        sa_out = self.sa_local_block(x) + self.sa_global_block(x)
        pa_out = self.pa_local_block(x) + self.pa_global_block(x)
        # pa_out = self.concat_conv(torch.cat((self.pa_local_block(x), self.pa_global_block(x)), dim=1))

        # CA & PA concat & conv
        # ca_pa_out = self.concat_conv(torch.cat((ca_out, pa_out), dim=1))
        # # CAPA & SA concat & conv
        # ca_pa_sa_out = self.concat_conv(torch.cat((ca_pa_out, sa_out), dim=1))
        # x_out = ca_pa_sa_out

        # SA & PA concat & conv
        # sa_pa_out = self.concat_conv(torch.cat((sa_out, pa_out), dim=1))
        # SAPA & CA concat & conv
        # sa_pa_ca_out = self.concat_conv(torch.cat((sa_pa_out, ca_out), dim=1))
        # x_out = sa_pa_ca_out

        # CA & SA concat & conv
        # ca_sa_out = self.concat_conv(torch.cat((ca_out, sa_out), dim=1))
        # CASA & PA concat & conv
        # ca_sa_pa_out = self.concat_conv(torch.cat((ca_sa_out, pa_out), dim=1))
        # x_out = ca_sa_pa_out

        # CA & SA aggre & conv
        ca_sa_out = self.aggre_conv(ca_out + sa_out)

        # CASA & PA aggre & conv
        ca_sa_pa_out = self.aggre_conv(ca_sa_out + pa_out)
        x_out = ca_sa_pa_out

        # ca_out = self.ca_local_block(x) + self.ca_global_block(x)
        # ca_out = self.ca_local_block(x) + self.ca_global_block(x)
        # sa_out = self.sa_local_block(x) + self.sa_global_block(x)
        # pa_out = self.pa_local_block(x) + self.pa_global_block(x)

        # aggregation
        # x_out = ca_out + sa_out + pa_out
        # concat
        # x_concat = torch.concat((ca_out, sa_out), dim=1)
        # x_out = self.concat_conv(x_concat)

        return x_out


# test_residual_x
class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x + self.conv1x1(x) + self.conv3x3(x)

        return x


# [Channel & Spatial & Pixel Transformer_Block] ViT-22B
class CH_SP_PX_Block_global_local_ViT_22B(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 13,
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

        self.norm = nn.BatchNorm2d(in_channels)

        self.attn = CA_SA_PA_global_local(dim=in_channels,
                                          kernel_size=kernel_size,
                                          reduction=reduction,
                                          norm_layer=norm_layer,
                                          act_layer=act_layer,
                                          split_size=split_size,
                                          num_heads=num_heads,
                                          attn_drop=attn_drop,
                                          qk_scale=qk_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = FFN(in_channels,
                       act_layer=ffn_act_layer)

        self.residual = Residual(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        loss = 0

        # x + DROP_PATH(FFN(norm_x) + ATTN(norm_x)) DROP_PATH is optional
        norm_x = self.norm(x)
        # x = self.residual(x) + self.drop_path(self.attn(norm_x) + self.ffn(norm_x))
        x = x + self.attn(norm_x) + self.ffn(norm_x)

        # norm_first
        # y, aux_loss = self.attn(self.norm1(x))
        # x = x + y
        # loss += aux_loss
        # x = x + self.ffn(self.norm2(x))

        # norm_last
        # y, aux_loss = self.attn(x)
        # x = x + self.norm1(y)
        # loss += aux_loss
        # x = x + self.norm2(self.ffn(x))
        return x, loss


########################################################################################################################

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
            # norm_layer
            norm_layer(out_channels),
            # act_layer
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        return x


class DownsampleConvBNReLU(nn.Module):
    """
    down sample -> (channel * 2, size * 0.5) * depth
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

        channel_list = [int(in_channels * 2 ** n) for n in range(1, depth + 1)]

        self.downsample = nn.Sequential(*[nn.Sequential(
            LazyConvBNReLU(out_channels,
                           kernel_size=kernel_size,
                           norm_layer=norm_layer,
                           act_layer=act_layer),
            Interpolate(scale_factor=0.5,
                        mode=mode,
                        align_corners=align_corners))
            for out_channels in channel_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

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
            LazyConvBNReLU(out_channels,
                           kernel_size=kernel_size,
                           norm_layer=norm_layer,
                           act_layer=act_layer),
            Interpolate(scale_factor=2,
                        mode=mode,
                        align_corners=align_corners))
            for out_channels in channel_list])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        return x


# [BFP_Head]
class BFP_Head(nn.Module):
    """
    channel_list should be decremental
    size_list should be incremental

    BFP (Balanced Feature Pyrmamids)
    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.
    """

    def __init__(self,
                 channel_list: Tuple[int] = (256, 128, 64, 32, 16),
                 size_list: Tuple[int] = (16, 32, 64, 128, 256),
                 kernel_size: int = 5,
                 refine_level: int = 2,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:

        super().__init__()

        assert len(channel_list) == len(size_list)
        num_levels = len(channel_list)

        assert 0 <= refine_level < num_levels
        self.num_levels = num_levels

        resize_blocks = []
        resize_inverse_blocks = []

        for i in range(num_levels):
            if i < refine_level:  # resize - upsample / resize_inverse - downsample
                resize_blocks.append(
                    UpsampleConvBNReLU(channel_list[i],
                                       depth=abs(i - refine_level),
                                       kernel_size=kernel_size,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer))

                resize_inverse_blocks.append(
                    DownsampleConvBNReLU(channel_list[refine_level],
                                         depth=abs(i - refine_level),
                                         kernel_size=kernel_size,
                                         norm_layer=norm_layer,
                                         act_layer=act_layer))

            elif i > refine_level:  # resize - downsample / resize_inverse - upsample
                resize_blocks.append(
                    DownsampleConvBNReLU(channel_list[i],
                                         depth=abs(i - refine_level),
                                         kernel_size=kernel_size,
                                         norm_layer=norm_layer,
                                         act_layer=act_layer))
                resize_inverse_blocks.append(
                    UpsampleConvBNReLU(channel_list[refine_level],
                                       depth=abs(i - refine_level),
                                       kernel_size=kernel_size,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer))

            else:  # Identity
                resize_blocks.append(nn.Identity())
                resize_inverse_blocks.append(nn.Identity())

        self.resize_blocks = nn.ModuleList(resize_blocks)
        self.resize_inverse_blocks = nn.ModuleList(resize_inverse_blocks)

        self.refine = LazyConvBNReLU(
            channel_list[refine_level],
            kernel_size=3,
            norm_layer=norm_layer,
            act_layer=act_layer)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(x) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []

        for input_feat, resize_block in zip(x, self.resize_blocks):
            feats.append(resize_block(input_feat))

        # feats
        bsf = sum(feats) / len(feats)

        # AFA
        # afa_feats_concat = []
        # attn_values = self.attn_value(torch.cat(feats, dim=1))
        # for feat in feats:
        #     afa_feats_concat.append(attn_values * feat)

        # step 2: refine gathered features
        bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        out = []

        for input_feat, resize_inverse_block in zip(x, self.resize_inverse_blocks):
            # orininal features + refined features
            res = input_feat + resize_inverse_block(bsf)
            out.append(res)

        return out


# [Final_Head]
class Head(nn.Module):
    """
    channel_list should be decremental
    size_list should be incremental
    """

    def __init__(self,
                 channel_list: Tuple[int] = (256, 128, 64, 32, 16),
                 size_list: Tuple[int] = (16, 32, 64, 128, 256),
                 out_channels: int = 4,
                 out_sizes: int = 512,
                 kernel_size: int = 5,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__()

        if isinstance(channel_list, int):
            channel_list = [channel_list]

        if isinstance(size_list, int):
            size_list = [size_list]

        assert len(channel_list) == len(size_list)
        num_levels = len(channel_list)

        self.num_levels = num_levels

        upsample_blocks = []

        for i, (in_channel, in_sizes) in enumerate(zip(channel_list, size_list)):
            # round & to int
            num_upsample = self._log_scale(in_sizes, out_sizes)

            upsample_blocks.append(
                UpsampleConvBNReLU(in_channel,
                                   depth=num_upsample,
                                   kernel_size=kernel_size,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))

        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        self.out_conv = nn.LazyConv2d(out_channels, kernel_size=1)

    def _log_scale(self, x, y, base=2):
        import math
        if x >= y:
            return int(math.log(x // y, base))
        elif x < y:
            return int(math.log(y // x, base))

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


########################################################################################################################

# moe?
class FFN_moe(nn.Module):
    def __init__(self,
                 in_channels: int,
                 act_layer: Type[nn.Module] = nn.GELU,
                 expert: int = 4) -> None:
        super().__init__()
        self.gate = nn.Conv2d(1, 1, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)

        return x


# [YOLO_Neck]
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):

    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class neck_decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 scale: int | float):
        super().__init__()

        self.scale = scale
        self.down_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.c2f = C2f(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):

        # UP / DOWN
        if self.scale > 1:  # UP
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        else:  # DOWN
            x = self.down_conv(x)

        # CONCAT
        x = torch.cat([x, skip], dim=1)

        # C2F
        x = self.c2f(x)

        return x


class Yolo_Neck(nn.Module):
    def __init__(self,
                 channel_list: Tuple[int] = (256, 128, 64, 32)
                 ):
        super().__init__()

        up_in_channels, down_in_channels = channel_list[:-1], channel_list[::-1][:-1]
        up_skip_channels, down_skip_channels = channel_list[1:], channel_list[::-1][1:]
        up_out_channels, down_out_channels = channel_list[1:], channel_list[::-1][1:]
        scale_up, scale_down = 2, 0.5

        up_blocks = [neck_decoder(in_channels=in_ch,
                                  skip_channels=skip_ch,
                                  out_channels=out_ch,
                                  scale=scale_up)
                     for in_ch, skip_ch, out_ch in zip(up_in_channels, up_skip_channels, up_out_channels)]
        down_blocks = [neck_decoder(in_channels=in_ch,
                                    skip_channels=skip_ch,
                                    out_channels=out_ch,
                                    scale=scale_down)
                       for in_ch, skip_ch, out_ch in zip(down_in_channels, down_skip_channels, down_out_channels)]

        self.up_blocks = nn.ModuleList(up_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        x = features[0]
        skips = features[1:]

        for up_index, block in enumerate(self.up_blocks):
            skip = skips[up_index] if up_index < len(skips) else None  # skip channel
            x = block(x, skip)
            features[up_index + 1] = x
            # print(f'{up_index=} {x.shape=}')

        # down
        features = features[::-1]

        x = features[0]
        skips = features[1:]

        for down_index, block in enumerate(self.down_blocks):
            skip = skips[down_index] if down_index < len(skips) else None  # skip channel
            x = block(x, skip)
            features[down_index + 1] = x
            # print(f'{up_index=} {x.shape=}')

        # reverse back
        features = features[::-1]

        return features


# [SWIN]

# [SA_global] CSWin
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


# [SA_global] CSWin
class SA_global_shifted_Swin(nn.Module):
    """
    [Shift] features - default (0, 2) - no shift, shift two
    [Split] Dim w/ Two Attention Operations - Vertical & Horizontal then concat
    """

    def __init__(self,
                 dim: int,
                 split_size: int = 4,
                 num_heads: int = 8,
                 attn_drop: float = 0.1,
                 qk_scale: float = None,
                 shift_sizes: Tuple[int, ...] = (0, 2)):
        super().__init__()

        # shift_sizes
        self.shift_sizes = shift_sizes

        assert dim // 2 >= num_heads, \
            f'after dim splitting into 2, `{dim // 2}` should be >= num_heads `{num_heads}`'

        # dim split into 2 parts
        split_dim = dim // 2

        # split_size
        self.sp = split_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

        head_dim = split_dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv_conv = nn.Conv2d(split_dim, split_dim * 3, kernel_size=1, bias=False)
        self.lepe_conv = nn.Conv2d(split_dim, split_dim, kernel_size=3, stride=1, padding=1, groups=split_dim)

        self.shift_concat = nn.Conv2d(dim * len(shift_sizes), dim, kernel_size=1, bias=False)

        # [SWIN]
        self.window_size = (split_size, split_size)  # Wh, Ww

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size : [B, C, H, W]"""
        B, C, H, W = x.shape

        # create a list for later concat
        attend_shifted_x = []

        for shift_size in self.shift_sizes:

            # create a list for later concat
            attened_splitted_x = []

            """shift"""
            if shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            else:
                shifted_x = x

            """split"""
            # split channel [:C // 2] , [C // 2:]
            x1, x2 = rearrange(shifted_x, 'b (split c) h w -> split b c h w', split=2)
            inputs = [x1, x2]

            # two attn operations
            assert H % self.sp == 0 and W % self.sp == 0, \
                f'{H=} or {W=} cannot be divided by split_size={self.sp} '

            # [CSWIN]
            # v_h, v_hsp, v_w, v_wsp = 1, H, W // self.sp, self.sp  # vertical
            # h_h, h_hsp, h_w, h_wsp = H // self.sp, self.sp, 1, W  # horizontal
            # params = [(v_h, v_hsp, v_w, v_wsp), (h_h, h_hsp, h_w, h_wsp)]

            # [SWIN]
            h, hsp, w, wsp = H // self.sp, self.sp, W // self.sp, self.sp
            params = [(h, hsp, w, wsp), (h, hsp, w, wsp)]

            for index, (x_split, (h, hsp, w, wsp)) in enumerate(zip(inputs, params)):

                # from [B, C, H, W] to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
                q, k, v = self.to_qkv_conv(x_split).chunk(3, dim=1)

                # lepe from v
                lepe = self.lepe_conv(v)

                q, k, v, lepe = map(lambda t: rearrange(t, 'b (c head) (h hsp) (w wsp) -> (b h w) head (hsp wsp) c',
                                                        head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp),
                                    (q, k, v, lepe))

                # attention
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

                # [SWIN]
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

                attn = attn + relative_position_bias.unsqueeze(0)
                attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
                attn = self.attn_drop(attn)

                splitted_x = (attn @ v) + lepe

                # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
                splitted_x = rearrange(splitted_x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                                       head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

                # reverse cyclic shift
                if shift_size > 0:
                    splitted_x = torch.roll(splitted_x, shifts=(shift_size, shift_size), dims=(1, 2))

                attened_splitted_x.append(splitted_x)

            # concat
            concat_splitted_x = torch.cat(attened_splitted_x, dim=1)
            attend_shifted_x.append(concat_splitted_x)

        # concat
        # concat_shifted_x = torch.cat(attend_shifted_x, dim=1)
        # output = self.shift_concat(concat_shifted_x)

        # aggregate
        output = torch.add(*attend_shifted_x)
        # output = self.shift_concat(concat_shifted_x)

        return output


# [CA_global] <adjust> WFCA_lmh_SWin !!!NOT WORKING!!!
class CA_global_Swin(nn.Module):
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

                 wave: str = 'db1'):
        super().__init__()

        in_channels = dim  # AKA in_channels

        # dwt / idwt
        self.dwt = DWT(wave=wave)
        self.idwt = IDWT(wave=wave)

        # i think qkv conv is working like body conv
        # Body
        # self.body_lh = ConvBNReLU(in_channels, in_channels,
        #                           kernel_size=kernel_size,
        #                           norm_layer=norm_layer,
        #                           act_layer=act_layer)
        #
        # self.body_m = ConvBNReLU(in_channels * 2, in_channels * 2,
        #                          kernel_size=kernel_size,
        #                          norm_layer=norm_layer,
        #                          act_layer=act_layer)

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

        self.cat_conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1)
        self.idwt_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # conv after idwt

        # [SWIN]
        self.window_size = (split_size, split_size)  # Wh, Ww

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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
        self.register_buffer("relative_position_index", relative_position_index)

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

        # Body
        # low, mid, high = self.body_lh(low), self.body_m(mid), self.body_lh(high)

        # qkv
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

        # [SWIN]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        low_attn = low_attn + relative_position_bias.unsqueeze(0)
        low_attn = F.softmax(low_attn, dim=-1, dtype=low_attn.dtype)
        low_attn = self.attn_drop(low_attn)
        x_low = low_attn @ low_v

        x_low = rearrange(x_low, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
        # mid attn
        mid_q = mid_q * self.m_scale
        mid_attn = (mid_q @ mid_k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C

        # [SWIN]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        mid_attn = mid_attn + relative_position_bias.unsqueeze(0)
        mid_attn = F.softmax(mid_attn, dim=-1, dtype=mid_attn.dtype)
        mid_attn = self.attn_drop(mid_attn)
        x_mid = mid_attn @ mid_v

        x_mid = rearrange(x_mid, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

        # high attn
        high_q = high_q * self.lh_scale
        high_attn = (high_q @ high_k.transpose(-2, -1))  # B head C N @ B head N C --> B head C C

        # [SWIN]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        high_attn = high_attn + relative_position_bias.unsqueeze(0)
        high_attn = F.softmax(high_attn, dim=-1, dtype=high_attn.dtype)
        high_attn = self.attn_drop(high_attn)
        x_high = high_attn @ high_v

        x_high = rearrange(x_high, '(b h w) head c (hsp wsp) -> b (c head) (h hsp) (w wsp)',
                           head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

        # low = low + ca_low
        # mid = mid + ca_mid
        # high = high + ca_high

        x_out = self.idwt(self._channel_cat([x_low, x_mid, x_high]))
        # x_out = self.idwt_conv(self.idwt(self.cat_conv(self._channel_cat([low, mid, high]))))
        return x_out

# attention gate
