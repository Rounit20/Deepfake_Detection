import torch
import torch.nn as nn


class GatedAttentionFusion(nn.Module):
    def __init__(self, stream_dims: list):
        super().__init__()
        total = sum(stream_dims)
        n     = len(stream_dims)
        self.gate_fc     = nn.Sequential(
            nn.Linear(total, 64), nn.ReLU(True),
            nn.Linear(64, n),    nn.Sigmoid(),
        )
        self.norms       = nn.ModuleList([nn.LayerNorm(d) for d in stream_dims])
        self.fusion_proj = nn.Sequential(
            nn.Linear(total, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512,  256),  nn.ReLU(True), nn.Dropout(0.4),
        )
        self.output_dim  = 256
        self.stream_dims = stream_dims

    def forward(self, feats: list):
        normed = [n(f) for n, f in zip(self.norms, feats)]
        cat    = torch.cat(normed, dim=1)
        gates  = self.gate_fc(cat)
        gated  = [normed[i] * gates[:, i:i+1] for i in range(len(normed))]
        return self.fusion_proj(torch.cat(gated, dim=1))