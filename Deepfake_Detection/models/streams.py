import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.backbones import SimplifiedHRNet, EfficientNetBackboneWithMap


class SpatialStreamHRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hrnet      = SimplifiedHRNet()
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(256, Config.SPATIAL_FEATURE_DIM), nn.ReLU(True), nn.Dropout(0.3))
        self.feature_dim = Config.SPATIAL_FEATURE_DIM

    def forward(self, x):
        xh, _ = self.hrnet(x)
        return self.projection(self.gap(xh).flatten(1)), xh


class SpatialStreamEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone    = EfficientNetBackboneWithMap(pretrained)
        self.feature_dim = Config.SPATIAL_FEATURE_DIM

    def forward(self, x):
        feat_vec, feat_map = self.backbone(x)
        return feat_vec, feat_map


class FrequencyStreamDCT(nn.Module):
    def __init__(self, in_ch=9):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_ch,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.AdaptiveAvgPool2d(1),
        )
        self.projection  = nn.Sequential(
            nn.Linear(512, Config.FREQ_FEATURE_DIM), nn.ReLU(True), nn.Dropout(0.3))
        self.feature_dim = Config.FREQ_FEATURE_DIM

    def forward(self, x):
        return self.projection(self.conv_layers(x).flatten(1))


class LNPStream(nn.Module):
    def __init__(self, in_ch=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_ch, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d(1),
        )
        self.projection  = nn.Sequential(
            nn.Linear(256, Config.LNP_FEATURE_DIM), nn.ReLU(True), nn.Dropout(0.3))
        self.feature_dim = Config.LNP_FEATURE_DIM

    def forward(self, x):
        return self.projection(self.conv_layers(x).flatten(1))


class BetaStatisticsEncoder(nn.Module):
    def __init__(self, in_dim=63):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(128, Config.BETA_FEATURE_DIM), nn.ReLU(True), nn.Dropout(0.3),
        )
        self.output_dim = Config.BETA_FEATURE_DIM

    def forward(self, x):
        return self.encoder(x)


class MaskSupervisionModule(nn.Module):
    # Sigmoid removed — BCEWithLogitsLoss applies it internally (autocast-safe)
    def __init__(self, in_ch):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.2),
            nn.Conv2d(128,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, feat_map):
        m = self.predictor(feat_map)
        return F.interpolate(m, (Config.IMG_SIZE, Config.IMG_SIZE),
                             mode='bilinear', align_corners=False)