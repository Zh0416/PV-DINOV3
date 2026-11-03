"""A task-specific segmentation head trained only from one labeled support."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SupportOnlySegmenter(nn.Module):
    """Linear patch classifier whose weights are created from support features."""

    def __init__(self, channels):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        self.channels = int(channels)
        self.network = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.network.weight)
        nn.init.zeros_(self.network.bias)

    def forward(self, features):
        if (features.ndim != 4 or features.shape[1] != self.channels or
                not torch.isfinite(features).all()):
            raise ValueError('Expected finite BCHW features with the trained channel count')
        return self.network(features.float())


def _grid_target(mask, grid):
    if (mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-2] % 16 or
            mask.shape[-1] % 16 or not torch.isfinite(mask).all() or
            (mask < 0).any() or (mask > 1).any()):
        raise ValueError('Support mask must be a finite B1HW tensor in [0, 1]')
    target = F.interpolate(mask.float(), grid, mode='area')
    if not (target > 0).any() or not (target < 1).any():
        raise ValueError('Support mask needs both foreground and background')
    return target


def _loss(logits, target, available):
    positive = (target * available).sum().clamp_min(1.)
    negative = ((1 - target) * available).sum().clamp_min(1.)
    bce = 0.5 * ((F.softplus(-logits) * target * available).sum() / positive +
                 (F.softplus(logits) * (1 - target) * available).sum() / negative)
    probability = logits.sigmoid()
    intersection = (probability * target * available).sum()
    dice = 1 - (2 * intersection + 1) / ((probability * available).sum() + positive + 1)
    return bce + 0.5 * dice


def _support_threshold(probability, target, bins=4096):
    values = probability.detach().cpu().numpy().reshape(-1)
    labels = target.detach().cpu().numpy().astype(bool).reshape(-1)
    if not labels.any() or labels.all() or not np.isfinite(values).all():
        raise ValueError('Support threshold requires both classes and finite scores')
    edges = np.linspace(0., 1., bins + 1, dtype=np.float64)
    counts = np.stack([
        np.histogram(values[labels], edges)[0],
        np.histogram(values[~labels], edges)[0],
    ])
    tp = counts[:, ::-1].cumsum(1)[0]
    fp = counts[:, ::-1].cumsum(1)[1]
    positive = counts[0].sum()
    iou = tp / np.maximum(positive + fp, 1)
    best = np.flatnonzero(iou == iou.max())[0]
    return float(edges[bins - 1 - best]), float(iou[best]), int(bins)


def fit_from_support(features, mask, *, steps=80, learning_rate=.05, seed=20261103):
    """Fit four fresh heads and calibrate from held-out support patches."""
    if not isinstance(features, torch.Tensor) or features.ndim != 4 or features.shape[0] != 1:
        raise ValueError('Support features must have shape 1,C,H,W')
    if not isinstance(mask, torch.Tensor) or mask.shape[0] != 1:
        raise ValueError('Support mask must have batch size one')
    if not isinstance(steps, int) or steps <= 0 or learning_rate <= 0:
        raise ValueError('steps and learning_rate must be positive')
    torch.manual_seed(seed)
    features = F.normalize(features.detach().float(), dim=1)
    target = _grid_target(mask.to(features.device), features.shape[-2:])
    rows, columns = torch.meshgrid(
        torch.arange(features.shape[-2], device=features.device),
        torch.arange(features.shape[-1], device=features.device), indexing='ij')
    partitions = (rows.remainder(2) * 2 + columns.remainder(2))[None, None]
    out_of_fold = torch.empty_like(target)
    states = []
    history = []
    for partition in range(4):
        held = partitions == partition
        available = (~held).float()
        if (target * held).sum() <= 0 or (target * available).sum() <= 0:
            raise ValueError('Each support partition needs photovoltaic pixels')
        model = SupportOnlySegmenter(features.shape[1]).to(features.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = _loss(logits, target, available)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite support-only loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step == steps:
                history.append(dict(partition=partition, step=step, loss=float(loss.detach())))
        model.eval()
        with torch.no_grad():
            out_of_fold[held] = model(features).sigmoid()[held]
        states.append({key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()})
    support_probability = F.interpolate(out_of_fold, mask.shape[-2:],
                                        mode='bilinear', align_corners=False)
    threshold, support_iou, bins = _support_threshold(support_probability, mask)
    return dict(models=states, channels=features.shape[1], threshold=threshold,
                held_out_support_iou=support_iou, threshold_bins=bins, steps=steps,
                learning_rate=learning_rate, seed=seed, history=history,
                target_grid=list(features.shape[-2:]))


def predict(task, features, output_size):
    if not isinstance(features, torch.Tensor) or features.ndim != 4 or features.shape[0] != 1:
        raise ValueError('Query features must have shape 1,C,H,W')
    if task['channels'] != features.shape[1]:
        raise ValueError('Query channel count differs from support task')
    features = F.normalize(features.detach().float(), dim=1)
    models = []
    for state in task['models']:
        model = SupportOnlySegmenter(task['channels'])
        model.load_state_dict(state)
        models.append(model.to(features.device).eval())
    with torch.no_grad():
        patch_probability = torch.stack([model(features).sigmoid() for model in models]).mean(0)
        probability = F.interpolate(patch_probability, output_size,
                                    mode='bilinear', align_corners=False)[0, 0]
    threshold = float(task['threshold'])
    return probability.cpu().numpy(), probability.cpu().numpy() >= threshold
