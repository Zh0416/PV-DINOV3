"""Support-conditioned dual-prior feature fusion with optional Haar modulation."""

import torch
from torch import nn
from torch.nn import functional as F

from .frequency_gate import FrequencyCorrection
from .support_haar_fusion import _margin, _prototypes


class FeatureFusionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context = nn.Sequential(nn.Conv2d(2 * channels + 2, 32, 3, padding=1),
                                     nn.GELU())
        self.alpha = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def forward(self, dino, geo, spatial, semantic):
        inputs = torch.cat((dino, geo, spatial, semantic), dim=1)
        return torch.sigmoid(self.alpha(self.context(inputs)))


class HaarFeatureModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(6, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, haar):
        return features * (1 + .2 * torch.tanh(self.projection(haar)))


class CAMGuidedDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.guidance = nn.Conv2d(2, 1, 3, padding=1)
        nn.init.zeros_(self.guidance.weight)
        nn.init.zeros_(self.guidance.bias)
        self.decoder = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(64, 32, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, features, spatial, semantic, guided=True):
        if guided:
            gain = 1 + .5 * torch.tanh(self.guidance(torch.cat((spatial, semantic), 1)))
            features = features * gain
        return self.decoder(features)


class DualPriorSegmenter(nn.Module):
    """Train only the projections, feature gate, optional frequency arm and decoder."""

    def __init__(self, channels=128, branch='both', fusion='dynamic',
                 frequency='none', cam_guidance=True):
        super().__init__()
        if channels <= 0 or branch not in ('dino', 'geo', 'both') or \
                fusion not in ('fixed', 'dynamic') or frequency not in ('none', 'score', 'feature'):
            raise ValueError('Invalid dual-prior configuration')
        self.branch, self.fusion = branch, fusion
        self.frequency, self.cam_guidance = frequency, cam_guidance
        self.dino_projection = nn.Conv2d(1024, channels, 1)
        self.geo_projection = nn.Conv2d(512, channels, 1)
        self.feature_gate = FeatureFusionGate(channels)
        self.feature_modulation = HaarFeatureModulation(channels)
        self.score_correction = FrequencyCorrection(base_channels=4,
                                                      frequency_channels=6,
                                                      hidden=16, max_correction=.3)
        self.decoder = CAMGuidedDecoder(channels)
        if branch == 'dino':
            self.geo_projection.requires_grad_(False)
        elif branch == 'geo':
            self.dino_projection.requires_grad_(False)
        if branch != 'both' or fusion == 'fixed':
            self.feature_gate.requires_grad_(False)
        if frequency != 'feature':
            self.feature_modulation.requires_grad_(False)
        if frequency != 'score':
            self.score_correction.requires_grad_(False)
        if not cam_guidance:
            self.decoder.guidance.requires_grad_(False)

    def forward(self, support_dino, support_geo, support_mask,
                query_dino, query_geo, output_size, query_haar=None):
        if (support_dino.ndim != 4 or support_geo.ndim != 4 or
                query_dino.ndim != 4 or query_geo.ndim != 4 or
                support_dino.shape[1] != 1024 or query_dino.shape[1] != 1024 or
                support_geo.shape[1] != 512 or query_geo.shape[1] != 512 or
                support_dino.shape[0] != query_dino.shape[0] or
                support_geo.shape[0] != query_geo.shape[0] or
                support_dino.shape[-2:] != support_geo.shape[-2:] or
                query_dino.shape[-2:] != query_geo.shape[-2:] or
                support_mask.shape != (support_dino.shape[0], 1,
                                       16 * support_dino.shape[-2],
                                       16 * support_dino.shape[-1]) or
                len(output_size) != 2 or any(size <= 0 for size in output_size)):
            raise ValueError('Expected aligned DINO, Geo and support-mask grids')
        if self.frequency != 'none' and (query_haar is None or
                query_haar.shape != (query_dino.shape[0], 6, *query_dino.shape[-2:])):
            raise ValueError('Frequency branch requires aligned six-channel Haar')
        target = F.interpolate(support_mask.float(), support_dino.shape[-2:], mode='area')
        available = torch.ones_like(target)
        spatial = _margin(query_dino, _prototypes(support_dino, target, available))
        semantic = _margin(query_geo, _prototypes(support_geo, target, available))
        dino = F.normalize(self.dino_projection(query_dino.float()), dim=1)
        geo = F.normalize(self.geo_projection(query_geo.float()), dim=1)
        if self.branch == 'dino':
            fused = dino
            alpha = torch.zeros_like(spatial)
        elif self.branch == 'geo':
            fused = geo
            alpha = torch.ones_like(spatial)
        else:
            alpha = (self.feature_gate(dino, geo, spatial, semantic)
                     if self.fusion == 'dynamic' else torch.full_like(spatial, .5))
            fused = (1 - alpha) * dino + alpha * geo
        if self.frequency == 'feature':
            fused = self.feature_modulation(fused, query_haar.float())
        low_logits = self.decoder(fused, spatial, semantic, self.cam_guidance)
        if self.frequency == 'score':
            base = torch.cat((spatial, semantic, spatial - semantic,
                              query_haar.float().square().mean(1, keepdim=True).sqrt()), 1)
            low_logits = low_logits + 8 * self.score_correction(base, query_haar.float())
        logits = F.interpolate(low_logits, size=output_size, mode='bilinear',
                               align_corners=False)
        return dict(low_logits=low_logits, logits=logits, spatial_cam=spatial,
                    semantic_cam=semantic, alpha=alpha)


def segmentation_loss(logits, target):
    if logits.shape != target.shape:
        raise ValueError('Logits and target must have the same pixel grid')
    positive = target.sum()
    negative = (1 - target).sum()
    positive_loss = (F.softplus(-logits) * target).sum() / positive.clamp_min(1)
    negative_loss = (F.softplus(logits) * (1 - target)).sum() / negative.clamp_min(1)
    bce = (.5 * (positive_loss + negative_loss) if positive > 0 and negative > 0
           else positive_loss + negative_loss)
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / \
        (probability.sum() + positive + 1)
    return bce + .5 * dice
