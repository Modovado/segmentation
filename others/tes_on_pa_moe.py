
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




###################################################################

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


# Gating Network
class GN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int = 512,
                 k: int = 2,
                 num_experts: int = 4,
                 act_layer: Type[nn.Module] = nn.Softmax,
                 eps: float = 1e-10) -> None:
        super().__init__()

        self.k = k  # k of topk
        self.num_experts = num_experts

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, num_experts, kernel_size=1),  # [b, 4]
            nn.Flatten(),
            act_layer(dim=1),
        )
        self.eps = eps

    def topk(self, t, k=1):
        values, index = t.topk(k=k, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def forward(self, x):
        logits = self.gating(x)

        # find top k value
        expert_value, expert_index = self.topk(logits, k=self.k)

        # Coefficient of Variation # need fix
        loss = logits.float().std() / (logits.float().mean() + self.eps)
        # print(f'loss_std : {loss}')

        # other version?
        # loss = logits.float().var() / (logits.float().mean() ** 2 + eps)
        # print(f'loss_var : {loss}')

        # if k < num_experts, softmax again
        if self.k < self.num_experts:
            expert_value = F.softmax(expert_value)

        return expert_value, expert_index, loss


class PA_MOE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_sizes: tuple = (1, 3, 7, 11),
                 k: int = 2,
                 num_experts: int = 4,
                 loss_coef: float = 0.1) -> None:

        super().__init__()

        # Pixel Attention Experts
        self.pa_experts = nn.ModuleList([PA(in_channels=in_channels, kernel_size=kernel_size)
                                         for kernel_size in kernel_sizes])

        # gating network
        self.gating = GN(in_channels=in_channels, k=k, num_experts=num_experts)

        self.loss_coef = loss_coef

    def forward(self, x):

        # loss
        loss = 0

        # GN first, then PA index for PA
        expert_value, expert_index, expert_loss = self.gating(x)  # gating

        # loss
        loss += expert_loss

        # for storing
        output = []

        # flatten tensor gn_value, gn_index
        gn_value = expert_value.view(-1)
        gn_index = expert_index.view(-1)

        for i, (value, index) in enumerate(zip(gn_value, gn_index)):

            # value * pa_experts[index]
            output.append(value * self.pa_experts[index](x))

        # stack then sum
        output = torch.sum(torch.stack(output), dim=0)

        loss *= self.loss_coef

        return output, loss



#################################################

if __name__ == '__main__':
    from thop import profile
    # B, C, H, W = x.shape
    input = torch.ones(1, 4, 512, 512, dtype=torch.float)

    model = PA_MOE(in_channels=4)

    # model.eval()
    # model.train()

    # print(model)
    outpu, l = model(input)
    # model.train()
    # model.eval()
    # output, output_loss = model(input)
    # output, aux_output, output_loss = model(input)
    print(outpu.shape)
    # print(aux_output.shape)
    # print(output_loss)

    macs, parameters = profile(model, inputs=(input, ))
    print(f'macs:{macs / 1e9} G, parameter:{parameters / 1e6} M')
