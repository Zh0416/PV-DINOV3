"""Train dynamic fusion and Haar correction from one labeled support image."""

import numpy as np
import torch
from torch.nn import functional as F

from .frequency_gate import FrequencyCorrection
from .frozen_baselines import foreground_background_response
from .hard_negative_gate import ResidualFusionGate, residual_fusion
from .support_only_segmenter import _support_threshold


def haar_features(rgb):
    rgb = np.asarray(rgb)
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or
            min(rgb.shape[:2]) < 32 or any(size % 16 for size in rgb.shape[:2])):
        raise ValueError('Expected uint8 RGB with dimensions divisible by 16')
    image = rgb.astype(np.float32) / 255
    luma = .2126 * image[..., 0] + .7152 * image[..., 1] + .0722 * image[..., 2]
    height, width = luma.shape
    patches = luma.reshape(height // 16, 16, width // 16, 16).transpose(0, 2, 1, 3)
    maps = []
    for _ in range(2):
        top_left = patches[..., ::2, ::2]
        top_right = patches[..., ::2, 1::2]
        bottom_left = patches[..., 1::2, ::2]
        bottom_right = patches[..., 1::2, 1::2]
        maps.extend(np.log1p(1000 * np.square(band).mean(axis=(-2, -1)))
                    for band in ((top_left - top_right + bottom_left - bottom_right) / 2,
                                 (top_left + top_right - bottom_left - bottom_right) / 2,
                                 (top_left - top_right - bottom_left + bottom_right) / 2))
        patches = (top_left + top_right + bottom_left + bottom_right) / 2
    features = np.stack(maps).astype(np.float32)
    features -= features.mean(axis=(1, 2), keepdims=True)
    features /= features.std(axis=(1, 2), keepdims=True) + 1e-5
    return torch.from_numpy(features.copy())[None]


def _prototypes(features, target, available):
    labels = (target * available).detach()
    background = ((1 - target) * available).detach()
    if labels.sum() <= 0 or background.sum() <= 0:
        raise ValueError('Support partition needs both classes')
    normalized = F.normalize(features.detach().float(), dim=1)
    foreground = F.normalize((normalized * labels).sum((-2, -1)) /
                             labels.sum((-2, -1)), dim=1)
    negative = F.normalize((normalized * background).sum((-2, -1)) /
                           background.sum((-2, -1)), dim=1)
    return foreground, negative


def _margin(features, prototypes):
    return foreground_background_response(features, prototypes[0], prototypes[1],
                                          features.shape[-2:])


def _evidence(dino, geo, haar, dino_prototypes, geo_prototypes):
    spatial = _margin(dino, dino_prototypes)
    semantic = _margin(geo, geo_prototypes)
    base = torch.cat((spatial, semantic, spatial - semantic,
                      haar.square().mean(1, keepdim=True).sqrt()), 1)
    return spatial, semantic, base


def _score(fusion, correction, spatial, semantic, base, haar):
    alpha, residual = fusion(base)
    delta = correction(base, haar)
    return residual_fusion(spatial, semantic, alpha, residual) + delta


def _loss(score, target, available):
    logits = 8 * score
    positive = (target * available).sum().clamp_min(1.)
    negative = ((1 - target) * available).sum().clamp_min(1.)
    bce = .5 * ((F.softplus(-logits) * target * available).sum() / positive +
                (F.softplus(logits) * (1 - target) * available).sum() / negative)
    probability = logits.sigmoid()
    intersection = (probability * target * available).sum()
    dice = 1 - (2 * intersection + 1) / ((probability * available).sum() + positive + 1)
    return bce + .5 * dice


def _new_modules(device):
    fusion = ResidualFusionGate(4, hidden=16, initial_alpha=.5,
                                max_residual=.3).to(device)
    correction = FrequencyCorrection(base_channels=4, frequency_channels=6,
                                     hidden=16, max_correction=.3).to(device)
    return fusion, correction


def fit_from_support(rgb, dino, geo, mask, *, steps=80, learning_rate=.002,
                     seed=20261103):
    if (not isinstance(dino, torch.Tensor) or not isinstance(geo, torch.Tensor) or
            not isinstance(mask, torch.Tensor) or dino.ndim != 4 or geo.ndim != 4 or
            dino.shape[0] != 1 or geo.shape[0] != 1 or
            dino.shape[-2:] != geo.shape[-2:] or dino.device != geo.device or
            mask.shape != (1, 1, *np.asarray(rgb).shape[:2]) or
            not isinstance(steps, int) or steps <= 0 or learning_rate <= 0 or
            not torch.isfinite(dino).all() or not torch.isfinite(geo).all() or
            not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any()):
        raise ValueError('Invalid aligned support features, RGB, or mask')
    torch.manual_seed(seed)
    grid = dino.shape[-2:]
    target = F.interpolate(mask.to(dino.device).float(), grid, mode='area')
    haar = haar_features(rgb).to(dino.device)
    if haar.shape[-2:] != grid:
        raise ValueError('Haar and encoder grids differ')
    rows, columns = torch.meshgrid(torch.arange(grid[0], device=dino.device),
                                   torch.arange(grid[1], device=dino.device),
                                   indexing='ij')
    partitions = (rows.remainder(2) * 2 + columns.remainder(2))[None, None]
    out_of_fold = torch.empty_like(target)
    states = []
    history = []
    for partition in range(4):
        held = partitions == partition
        available = (~held).float()
        if (target * held).sum() <= 0 or ((1 - target) * held).sum() <= 0:
            raise ValueError('Every held-out support partition needs both classes')
        dino_prototypes = _prototypes(dino, target, available)
        geo_prototypes = _prototypes(geo, target, available)
        spatial, semantic, base = _evidence(dino, geo, haar,
                                             dino_prototypes, geo_prototypes)
        fusion, correction = _new_modules(dino.device)
        fusion_initial = [parameter.detach().clone() for parameter in fusion.parameters()]
        correction_initial = [parameter.detach().clone() for parameter in correction.parameters()]
        parameters = list(fusion.parameters()) + list(correction.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                                      weight_decay=1e-4)
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            score = _score(fusion, correction, spatial, semantic, base, haar)
            loss = _loss(score, target, available)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite support fusion loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step == steps:
                history.append(dict(partition=partition, step=step,
                                    loss=float(loss.detach())))
        fusion.eval()
        correction.eval()
        with torch.no_grad():
            probability = (8 * _score(fusion, correction, spatial,
                                      semantic, base, haar)).sigmoid()
            out_of_fold[held] = probability[held]
        states.append(dict(fusion={key: value.detach().cpu().clone()
                                   for key, value in fusion.state_dict().items()},
                           correction={key: value.detach().cpu().clone()
                                       for key, value in correction.state_dict().items()},
                           updated_fusion_tensors=sum(not torch.equal(parameter, initial)
                                                      for parameter, initial in
                                                      zip(fusion.parameters(), fusion_initial)),
                           updated_haar_tensors=sum(not torch.equal(parameter, initial)
                                                    for parameter, initial in
                                                    zip(correction.parameters(), correction_initial)),
                           dino_prototypes=[value.detach().cpu().clone()
                                            for value in dino_prototypes],
                           geo_prototypes=[value.detach().cpu().clone()
                                           for value in geo_prototypes]))
    support_probability = F.interpolate(out_of_fold, mask.shape[-2:],
                                        mode='bilinear', align_corners=False)
    threshold, held_out_iou, bins = _support_threshold(support_probability, mask)
    return dict(models=states, threshold=threshold,
                held_out_support_iou=held_out_iou, threshold_bins=bins,
                steps=steps, learning_rate=learning_rate, seed=seed,
                grid=list(grid), dino_channels=dino.shape[1],
                geo_channels=geo.shape[1], history=history)


def predict(task, rgb, dino, geo):
    if (dino.ndim != 4 or geo.ndim != 4 or dino.shape[0] != 1 or
            geo.shape[0] != 1 or dino.shape[1] != task['dino_channels'] or
            geo.shape[1] != task['geo_channels'] or
            list(dino.shape[-2:]) != task['grid'] or
            dino.shape[-2:] != geo.shape[-2:] or dino.device != geo.device):
        raise ValueError('Query features differ from support task')
    haar = haar_features(rgb).to(dino.device)
    if haar.shape[-2:] != dino.shape[-2:]:
        raise ValueError('Query Haar grid differs from encoder grid')
    probabilities = []
    with torch.no_grad():
        for state in task['models']:
            fusion, correction = _new_modules(dino.device)
            fusion.load_state_dict(state['fusion'])
            correction.load_state_dict(state['correction'])
            fusion.eval()
            correction.eval()
            spatial, semantic, base = _evidence(dino, geo, haar,
                [value.to(dino.device) for value in state['dino_prototypes']],
                [value.to(dino.device) for value in state['geo_prototypes']])
            probabilities.append((8 * _score(fusion, correction, spatial,
                                             semantic, base, haar)).sigmoid())
        probability = F.interpolate(torch.stack(probabilities).mean(0),
                                    np.asarray(rgb).shape[:2], mode='bilinear',
                                    align_corners=False)[0, 0]
    score = probability.cpu().numpy()
    return score, score >= task['threshold']
