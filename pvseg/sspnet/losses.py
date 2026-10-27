"""Paper Eq. 8–13: CE + meta Dice + .2 pseudo BCE + .04 spatial Dice."""

import torch
from torch.nn import functional as F

from .modules import resize


def dice(probability, target):
    probability, target = probability.float().flatten(1), target.float().flatten(1)
    return (1 - (2 * (probability * target).sum(1) + 1) /
            (probability.sum(1) + target.sum(1) + 1)).mean()


def objective(outputs, mask, foreground_weight=1.):
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
    return total, {'total': total, 'main_ce': main, 'meta_dice': meta,
                   'pseudo_bce': pseudo, 'spatial_dice': cam}
