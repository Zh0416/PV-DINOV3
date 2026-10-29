"""Frozen response construction and calibration for A0/A1/A2 diagnostics."""

import numpy as np
import torch
from torch.nn import functional as F


def foreground_prototype(features, mask):
    """Area weights preserve small foregrounds without inventing positive pixels."""
    weights = F.interpolate(mask.float(), features.shape[-2:], mode='area')
    if not (weights.sum((-2, -1)) > 0).all():
        raise ValueError('Support has no foreground')
    pooled = (features.float() * weights).sum((-2, -1)) / weights.sum((-2, -1))
    return F.normalize(pooled, dim=1)


def prototype_response(features, prototype, size):
    similarity = (F.normalize(features.float(), dim=1) * prototype[:, :, None, None]).sum(1, keepdim=True)
    return F.interpolate(similarity, size=size, mode='bilinear', align_corners=False)


def foreground_background_prototypes(features, mask):
    """Pool both classes from the same support; mixed patches keep area weights."""
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('Support mask must contain finite weights in [0, 1]')
    if not ((1 - mask).sum((-2, -1)) > 0).all():
        raise ValueError('Support has no background')
    return foreground_prototype(features, mask), foreground_prototype(features, 1 - mask)


def foreground_background_response(features, foreground, background, size):
    """Half the cosine margin is in [-1, 1]; positive means foreground wins.

    The factor 1/2 only matches the calibration histogram's score range. No
    temperature, learned weight, query normalization or softmax is introduced.
    """
    normalized = F.normalize(features.float(), dim=1)
    direction = (foreground - background)[:, :, None, None]
    margin = (normalized * direction).sum(1, keepdim=True) * .5
    return F.interpolate(margin, size=size, mode='bilinear', align_corners=False)


def overlap_response(scores, boxes, size):
    """Each crop contributes its native CLIP cosine to every covered pixel."""
    total, count = np.zeros(size, np.float32), np.zeros(size, np.float32)
    if len(scores) != len(boxes):
        raise ValueError('Crop scores and boxes differ')
    for score, (left, top, right, bottom) in zip(scores, boxes):
        total[top:bottom, left:right] += float(score)
        count[top:bottom, left:right] += 1
    if not (count > 0).all():
        raise ValueError('Crop grid leaves uncovered pixels')
    return total / count


def score_histogram(score, target, bins=4096):
    score, target = np.asarray(score), np.asarray(target, dtype=bool)
    if score.shape != target.shape or not np.isfinite(score).all():
        raise ValueError('Invalid response/target')
    values = np.clip(score, -1, 1)
    return np.stack([np.histogram(values[target], bins, (-1, 1))[0],
                     np.histogram(values[~target], bins, (-1, 1))[0]])


def calibrate_threshold(hist):
    """One threshold per method/run; labels belong to the calibration group only."""
    tp, fp = hist[:, ::-1].cumsum(1)[:, ::-1]
    positive = hist[0].sum()
    if positive == 0:
        raise ValueError('Calibration requires foreground')
    iou = tp / np.maximum(positive + fp, 1)
    # Prefer fewer positive predictions when calibration IoUs tie.
    best = np.flatnonzero(iou == iou.max())[-1]
    return {'threshold': float(-1 + 2 * best / hist.shape[1]),
            'calibration_micro_iou': float(iou[best]), 'bins': int(hist.shape[1])}


def confusion(score, target, threshold):
    pred, target = np.asarray(score) >= threshold, np.asarray(target, dtype=bool)
    return dict(tp=int((pred & target).sum()), fp=int((pred & ~target).sum()),
                fn=int((~pred & target).sum()), tn=int((~pred & ~target).sum()))


def metrics(counts):
    tp, fp, fn, tn = [counts[k] for k in ('tp', 'fp', 'fn', 'tn')]
    ratio = lambda a, b: float(a / b) if b else None
    return dict(**counts, iou=ratio(tp, tp + fp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                false_positive_rate=ratio(fp, fp + tn),
                predicted_fraction=ratio(tp + fp, tp + fp + fn + tn))


def ranking_metrics(hist):
    tp, fp = hist[:, ::-1].cumsum(1)
    p, n = hist.sum(1)
    if p == 0 or n == 0:
        return {'pixel_ap_histogram': None, 'pixel_auroc_histogram': None}
    recall, precision = tp / p, tp / np.maximum(tp + fp, 1)
    ap = (np.diff(np.r_[0, recall]) * precision).sum()
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = integrate(np.r_[0, recall], np.r_[0, fp / n])
    return {'pixel_ap_histogram': float(ap), 'pixel_auroc_histogram': float(auc)}
