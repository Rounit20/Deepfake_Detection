import torch.nn as nn
from config import Config
from models.streams import (
    SpatialStreamEfficientNet, SpatialStreamHRNet,
    FrequencyStreamDCT, LNPStream, BetaStatisticsEncoder, MaskSupervisionModule,
)
from models.fusion import GatedAttentionFusion


class DeepfakeDetectorV54(nn.Module):
    def __init__(self, num_classes=2, use_mask=True,
                 backbone='efficientnet', use_beta=True):
        super().__init__()
        self.use_mask   = use_mask
        self.backbone_t = backbone
        self.use_beta   = use_beta

        if backbone == 'efficientnet':
            self.spatial_stream = SpatialStreamEfficientNet(Config.USE_PRETRAINED)
            mask_in_ch = self.spatial_stream.backbone.early_ch
        else:
            self.spatial_stream = SpatialStreamHRNet()
            mask_in_ch = 256

        self.frequency_stream = FrequencyStreamDCT(Config.DCT_CHANNELS)
        self.lnp_stream       = LNPStream(Config.LNP_CHANNELS)
        self.beta_encoder     = BetaStatisticsEncoder(Config.NUM_AC_COEFFICIENTS) if use_beta else None

        if use_mask:
            self.mask_adapter = nn.Sequential(
                nn.Conv2d(mask_in_ch, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
            self.mask_module  = MaskSupervisionModule(256)
        else:
            self.mask_adapter = None
            self.mask_module  = None

        dims = [self.spatial_stream.feature_dim,
                self.frequency_stream.feature_dim,
                self.lnp_stream.feature_dim]
        if use_beta:
            dims.append(self.beta_encoder.output_dim)

        self.fusion     = GatedAttentionFusion(dims)
        self.classifier = nn.Linear(self.fusion.output_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, spatial, dct_freq, beta, lnp):
        spatial_feat, feat_map = self.spatial_stream(spatial)

        pred_mask = None
        if self.use_mask and self.mask_module is not None:
            adapted   = self.mask_adapter(feat_map)
            pred_mask = self.mask_module(adapted)

        freq_feat = self.frequency_stream(dct_freq)
        lnp_feat  = self.lnp_stream(lnp)

        feats = [spatial_feat, freq_feat, lnp_feat]
        if self.use_beta and self.beta_encoder is not None:
            feats.append(self.beta_encoder(beta))

        fused  = self.fusion(feats)
        logits = self.classifier(fused)
        return logits, pred_mask