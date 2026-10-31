import torch
from torch.nn import functional as F


def binary_loss(logits, target):
    if logits.shape != target.shape:
        raise ValueError(f'Logits/target mismatch: {logits.shape}/{target.shape}')
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return .5 * bce + .5 * dice


def balanced_bce(logits, target):
    """Fractional targets with equal foreground/background mass per image."""
    loss_pos = F.softplus(-logits) * target
    loss_neg = F.softplus(logits) * (1 - target)
    positives, negatives = target.sum(), (1 - target).sum()
    if positives > 0 and negatives > 0:
        return .5 * (loss_pos.sum() / positives + loss_neg.sum() / negatives)
    return (loss_pos.sum() + loss_neg.sum()) / target.numel()


def segmentation_loss(logits, mask):
    target = F.interpolate(mask.float(), logits.shape[-2:], mode='area')
    prob = logits.sigmoid()
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return balanced_bce(logits, target) + .5 * dice
