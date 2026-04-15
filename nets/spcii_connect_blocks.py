# -*- coding: utf-8 -*-
"""
Minimal SPCII + Connect blocks used by the full GC-MambaWater decoder.

This file intentionally keeps only the components that are actually referenced by
the released complete model:
    - SELayer
    - Connect
    - ChannelInteraction
    - SPCII_Attention

All other historical experiment code has been removed to keep the release clean.
"""

import math

import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Connect(nn.Module):
    """
    Continuity-aware auxiliary head.

    Outputs:
        aux_seg : auxiliary segmentation logits
        con0    : 3x3 local connectivity response
        con1    : dilated connectivity response
    """

    def __init__(self, num_classes: int, num_neighbor: int, embedding_dim: int = 768, dropout_ratio: float = 0.1):
        super().__init__()
        self.seg_branch = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
        )
        self.connect_branch = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_neighbor, 3, padding=1, dilation=1),
        )
        self.se = SELayer(num_neighbor)
        self.connect_branch_d1 = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3),
        )
        self.se_d1 = SELayer(num_neighbor)
        self._init_weight()

    def forward(self, x: torch.Tensor):
        aux_seg = self.seg_branch(x)
        con0 = self.se(self.connect_branch(x))
        con1 = self.se_d1(self.connect_branch_d1(x))
        return aux_seg, con0, con1

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ChannelInteraction(nn.Module):
    def __init__(self, k_size: int = 5):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.unsqueeze(1)
        y = self.conv1d(y)
        return y.squeeze(1)


class SPCII_Attention(nn.Module):
    """
    Spatial-Position and Channel Interaction Integration attention.
    """

    def __init__(self, channels: int, num_groups_1d_conv: int = 1):
        super().__init__()
        self.channels = channels
        self.num_groups = max(1, min(num_groups_1d_conv, channels))

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))

        k_size = int(abs((math.log(channels, 2) + 1) / 2))
        k_size = k_size if k_size % 2 else k_size + 1

        if self.num_groups > 1 and channels % self.num_groups == 0:
            self.interaction_h = nn.ModuleList([ChannelInteraction(k_size) for _ in range(self.num_groups)])
            self.interaction_w = nn.ModuleList([ChannelInteraction(k_size) for _ in range(self.num_groups)])
        else:
            self.interaction_h = ChannelInteraction(k_size)
            self.interaction_w = ChannelInteraction(k_size)
            self.num_groups = 1

        # Kept for strict checkpoint compatibility with the original training code.
        self.conv_cat = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        f_h = self.avg_pool_h(x).squeeze(-1) + self.max_pool_h(x).squeeze(-1)
        f_w = self.avg_pool_w(x).squeeze(-2) + self.max_pool_w(x).squeeze(-2)

        f_h = f_h.permute(0, 2, 1).contiguous().view(b * h, c)
        f_w = f_w.permute(0, 2, 1).contiguous().view(b * w, c)

        if self.num_groups > 1:
            group_c = c // self.num_groups
            f_h = f_h.view(b * h, self.num_groups, group_c)
            f_w = f_w.view(b * w, self.num_groups, group_c)

            h_out = [self.interaction_h[i](f_h[:, i, :]) for i in range(self.num_groups)]
            w_out = [self.interaction_w[i](f_w[:, i, :]) for i in range(self.num_groups)]
            g_h = torch.cat(h_out, dim=1)
            g_w = torch.cat(w_out, dim=1)
        else:
            g_h = self.interaction_h(f_h)
            g_w = self.interaction_w(f_w)

        g_h = self.sigmoid(g_h).view(b, h, c).permute(0, 2, 1).unsqueeze(-1)
        g_w = self.sigmoid(g_w).view(b, w, c).permute(0, 2, 1).unsqueeze(-2)
        return x * g_h * g_w
