import math
import numpy as np
import torch
from torch.nn import functional as F


def boundary(mask, width):
    # Explicit zero padding includes boundaries touching image edges.
    padded = F.pad(mask.float()[:, None], (width,)*4, value=0)
    eroded = -F.max_pool2d(-padded, 2*width+1, stride=1)
    return mask & ~(eroded[:, 0] > .5)


class Metrics:
    def __init__(self, classes=1, boundary_ratio=.02):
        self.binary = classes == 1
        self.classes = 2 if self.binary else classes
        self.cm = torch.zeros(self.classes, self.classes, dtype=torch.float64)
        self.bi = torch.zeros(self.classes, dtype=torch.float64)
        self.bu = torch.zeros(self.classes, dtype=torch.float64)
        self.ratio = boundary_ratio

    @torch.no_grad()
    def update(self, logits, target):
        pred = (logits[:, 0] > 0).long() if self.binary else logits.argmax(1)
        valid = target != -100
        encoded = target[valid]*self.classes + pred[valid]
        self.cm += torch.bincount(encoded, minlength=self.classes**2).reshape(self.classes, self.classes).cpu()
        width = max(1, round(self.ratio*math.hypot(*target.shape[-2:])))
        # Do not count artificial boundaries next to ignored pixels.
        near_ignore = F.max_pool2d((~valid).float()[:, None], 2*width+1, 1, width)[:, 0] > 0
        safe = valid & ~near_ignore
        for c in range(self.classes):
            bp = boundary((pred == c) & valid, width) & safe
            bt = boundary((target == c) & valid, width) & safe
            self.bi[c] += (bp & bt).sum().cpu()
            self.bu[c] += (bp | bt).sum().cpu()

    def compute(self):
        tp = self.cm.diag()
        union = self.cm.sum(0)+self.cm.sum(1)-tp
        denom = self.cm.sum(0)+self.cm.sum(1)
        iou = torch.where(union > 0, tp/union, torch.nan)
        dice = torch.where(denom > 0, 2*tp/denom, torch.nan)
        biou = torch.where(self.bu > 0, self.bi/self.bu, torch.nan)
        select = slice(1, 2) if self.binary else slice(None)
        return dict(accuracy=float(tp.sum()/self.cm.sum()) if self.cm.sum() else float('nan'),
                    iou=float(iou[select].nanmean()), miou=float(iou.nanmean()),
                    dice=float(dice[select].nanmean()), biou=float(biou[select].nanmean()),
                    per_class_iou=iou.tolist(), confusion_matrix=self.cm.tolist())


def forgetting(matrix):
    a = np.asarray(matrix, float)
    n = len(a)
    delta = np.diag(a)-a[-1]
    return dict(last_iou=float(np.mean(a[-1])), avg_iou=float(np.mean(np.diag(a))),
                ff_iou=float(np.mean(delta[:-1])) if n > 1 else 0., per_task_forgetting=delta.tolist())
