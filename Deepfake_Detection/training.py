import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)
from config import Config


AMP_DTYPE = torch.bfloat16 if (
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
) else torch.float16


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.val = self.avg = self.sum = self.count = 0
    def update(self, v, n=1):
        self.val = v; self.sum += v * n; self.count += n; self.avg = self.sum / self.count


def get_param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay,    'weight_decay': Config.WEIGHT_DECAY},
            {'params': no_decay, 'weight_decay': 0.0}]


def train_epoch(model, loader, criterion, optimizer, dev, epoch,
                scaler=None, accum_steps=1):
    model.train()
    loss_m, cls_m, mask_m, acc_m = (AverageMeter() for _ in range(4))
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Train')
    for i, (sp, dct_f, beta, lnp, mask, labels) in enumerate(pbar):
        sp, dct_f, beta, lnp, mask, labels = (
            t.to(dev, non_blocking=True)
            for t in (sp, dct_f, beta, lnp, mask, labels))

        if Config.USE_MIXED_PRECISION and scaler:
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                logits, pm = model(sp, dct_f, beta, lnp)
                loss, cls, ml = criterion(logits, pm, labels, mask)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        else:
            logits, pm = model(sp, dct_f, beta, lnp)
            loss, cls, ml = criterion(logits, pm, labels, mask)
            loss = loss / accum_steps
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if Config.USE_GRADIENT_CLIPPING:
                if scaler: scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        acc = (logits.argmax(1) == labels).float().mean()
        loss_m.update(loss.item() * accum_steps, sp.size(0))
        cls_m.update(cls.item(), sp.size(0))
        if ml.item() > 0: mask_m.update(ml.item(), sp.size(0))
        acc_m.update(acc.item(), sp.size(0))

        pbar.set_postfix(loss=f'{loss_m.avg:.4f}', cls=f'{cls_m.avg:.4f}',
                         mask=f'{mask_m.avg:.4f}', acc=f'{acc_m.avg:.4f}')

        if (i + 1) % Config.EMPTY_CACHE_FREQ == 0:
            torch.cuda.empty_cache()

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, dev):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, desc='Validation')
    for sp, dct_f, beta, lnp, mask, labels in pbar:
        sp, dct_f, beta, lnp, mask, labels = (
            t.to(dev, non_blocking=True)
            for t in (sp, dct_f, beta, lnp, mask, labels))
        logits, pm = model(sp, dct_f, beta, lnp)
        loss, _, _ = criterion(logits, pm, labels, mask)
        probs = F.softmax(logits, 1)[:, 1]
        preds = logits.argmax(1)

        loss_m.update(loss.item(), sp.size(0))
        acc_m.update((preds == labels).float().mean().item(), sp.size(0))
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        pbar.set_postfix(loss=f'{loss_m.avg:.4f}', acc=f'{acc_m.avg:.4f}')

    metrics = {
        'loss':      loss_m.avg,
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall':    recall_score(all_labels, all_preds, zero_division=0),
        'f1':        f1_score(all_labels, all_preds, zero_division=0),
        'auc':       roc_auc_score(all_labels, all_probs)
                     if len(set(all_labels)) > 1 else 0.0,
    }
    return metrics, all_labels, all_preds, all_probs


def find_threshold(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    idx = np.argmax(tpr - fpr)
    t   = thr[idx]
    print(f"\n🎯 Optimal threshold: {t:.4f}  (TPR={tpr[idx]:.4f}, FPR={fpr[idx]:.4f})")
    return t