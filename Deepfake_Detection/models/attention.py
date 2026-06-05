import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        mid = max(in_ch // reduction, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(mid, in_ch, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(self.avg(x)) + self.fc(self.max(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg, mx], 1))) * x


class DualAttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ch   = ChannelAttention(c)
        self.sp   = SpatialAttention()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True))

    def forward(self, x):
        return self.sp(self.ch(self.conv(x))) + x


class RecursiveResidualGroup(nn.Module):
    def __init__(self, c, n=2):
        super().__init__()
        self.blocks = nn.ModuleList([DualAttentionBlock(c) for _ in range(n)])

    def forward(self, x):
        r = x
        for b in self.blocks:
            x = b(x)
        return x + r