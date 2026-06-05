import torch
import torch.nn as nn
import torch.utils.checkpoint
from torchvision import models
from config import Config
from models.attention import RecursiveResidualGroup


class SimplifiedHRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),          nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            RecursiveResidualGroup(256, Config.NUM_RRG_BLOCKS),
        )
        self.t1 = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(nn.Conv2d(256, 128, 3, stride=2, padding=1),
                          nn.BatchNorm2d(128), nn.ReLU(True)),
        ])
        self.s2h = RecursiveResidualGroup(256, 2)
        self.s2l = RecursiveResidualGroup(128, 2)
        self.h2l = nn.Sequential(nn.Conv2d(256, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128))
        self.l2h = nn.Sequential(nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256))

    def forward(self, x):
        x = self.stage1(self.stem(x))
        xh, xl = self.s2h(self.t1[0](x)), self.s2l(self.t1[1](x))
        return xh + self.l2h(xl), xl + self.h2l(xh)


class EfficientNetBackboneWithMap(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        eff = models.efficientnet_b0(pretrained=pretrained)
        feats = list(eff.features.children())
        self.features_early = nn.Sequential(*feats[:7])
        self.features_late  = nn.Sequential(*feats[7:])
        self.avgpool        = nn.AdaptiveAvgPool2d(1)
        self.reduce         = nn.Sequential(
            nn.Linear(1280, Config.SPATIAL_FEATURE_DIM),
            nn.ReLU(True), nn.Dropout(0.3),
        )
        self.early_ch = 192
        self._apply_grad_ckpt = Config.USE_GRAD_CKPT

    def forward(self, x):
        if self._apply_grad_ckpt and self.training:
            feat_map = torch.utils.checkpoint.checkpoint(
                self.features_early, x, use_reentrant=False)
        else:
            feat_map = self.features_early(x)
        out = self.features_late(feat_map)
        feat_vec = self.reduce(self.avgpool(out).flatten(1))
        return feat_vec, feat_map