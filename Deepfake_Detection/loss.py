import torch
import torch.nn as nn
from config import Config


class CombinedLoss(nn.Module):
    # BCEWithLogitsLoss replaces BCELoss — autocast-safe, Sigmoid fused internally
    def __init__(self, mask_weight=0.3, label_smoothing=0.1):
        super().__init__()
        self.ce  = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing if Config.USE_LABEL_SMOOTHING else 0.0)
        self.bce = nn.BCEWithLogitsLoss()
        self.w   = mask_weight

    def forward(self, logits, pred_mask, labels, target_mask):
        cls = self.ce(logits, labels)
        if pred_mask is not None:
            mk  = self.bce(pred_mask, target_mask)
            tot = cls + self.w * mk
        else:
            mk  = torch.tensor(0.0, device=logits.device)
            tot = cls
        return tot, cls, mk