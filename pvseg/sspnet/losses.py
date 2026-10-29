"""Paper Eq. 8–13: CE + meta Dice + .2 pseudo BCE + .04 spatial Dice."""

import torch
from torch.nn import functional as F

from .modules import resize


def dice(probability, target):
    probability, target = probability.float().flatten(1), target.float().flatten(1)
    return (1 - (2 * (probability * target).sum(1) + 1) /
            (probability.sum(1) + target.sum(1) + 1)).mean()


def balanced_meta_ce(logits, mask):
    """PV adaptation: equal mean CE per present class, then per episode.

    Apply CE directly to raw meta logits so a confidently wrong foreground
    pixel retains a corrective gradient. Empty queries supervise background
    alone; their missing foreground class contributes no artificial term.
    """
    target = mask[:, 0].long()
    pixel = F.cross_entropy(logits.float(), target, reduction='none')
    counts, means = [], []
    for label in (0, 1):
        selected = target == label
        count = selected.flatten(1).sum(1)
        counts.append(count)
        means.append((pixel * selected).flatten(1).sum(1) / count.clamp_min(1))
    present = torch.stack(counts, 1) > 0
    return ((torch.stack(means, 1) * present).sum(1) / present.sum(1)).mean()


def objective(outputs, mask, foreground_weight=1., meta_ce_weight=0.):
    if meta_ce_weight < 0:
        raise ValueError('meta_ce_weight must be nonnegative')
    target = mask[:, 0].long()
    weight = outputs['logits'].new_tensor([1., foreground_weight]).float()
    main = F.cross_entropy(outputs['logits'].float(), target, weight=weight)
    meta_probability = outputs['meta_logits'].float().softmax(1)
    # Weighted two-class Dice; PV defaults to equal class weights, not LoveDA's heuristic.
    meta = (dice(meta_probability[:, 0], 1 - target) +
            foreground_weight * dice(meta_probability[:, 1], target)) / (1 + foreground_weight)
    pseudo_target = resize(mask.float(), outputs['pseudo_logits'].shape[-2:], 'nearest')
    cam_target = resize(mask.float(), outputs['spatial_cam'].shape[-2:], 'nearest')
    pseudo = F.binary_cross_entropy_with_logits(outputs['pseudo_logits'].float(), pseudo_target)
    cam = dice(outputs['spatial_cam'], cam_target)
    total = main + meta + .2 * pseudo + .04 * cam
    terms = {'main_ce': main, 'meta_dice': meta, 'pseudo_bce': pseudo, 'spatial_dice': cam}
    # Default remains the reconstructed paper objective. This optional term
    # is an explicit diagnostic adaptation, never silently enabled.
    if meta_ce_weight:
        terms['balanced_meta_ce'] = balanced_meta_ce(outputs['meta_logits'], mask)
        total = total + meta_ce_weight * terms['balanced_meta_ce']
    return total, {'total': total, **terms}
