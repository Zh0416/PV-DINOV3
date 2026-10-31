"""Frozen PV/greenhouse contrast in GeoRSCLIP space, beside the A2 prior.

This module consumes projected GeoRSCLIP *image* embeddings, not DINO tokens.
It requires no greenhouse masks or gradient updates. Its discrimination still
depends on the pretrained encoder and must be evaluated on real imagery.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .sspnet.encoders import PV_PROMPT


# Keep the original A3 positive prompt so the first ablation changes only the
# competing concepts/correction. Prompt ensembles are a separate experiment.
DEFAULT_CONCEPT_PROMPTS = {
    'pv': (PV_PROMPT,),
    'greenhouse': (
        'a satellite aerial image of agricultural greenhouses used for growing crops',
        'a satellite aerial image of agricultural greenhouses covered with plastic film',
        'a satellite aerial image of glass agricultural greenhouses',
        'a satellite aerial image of agricultural polytunnels',
    ),
    'roof': ('a satellite aerial image of buildings with metal rooftops',),
    'water': ('a satellite aerial image of water in lakes and ponds',),
    'soil': ('a satellite aerial image of bare soil and open ground',),
    'vegetation': ('a satellite aerial image of vegetation and agricultural fields',),
}


@torch.no_grad()
def encode_concepts(model, tokenizer, prompts=None, device='cpu'):
    """Encode a prompt bank with the caller's frozen, pinned GeoRSCLIP model.

    Return per-template normalized embeddings for GreenhouseContrastPrior.
    Use the same RS5M ViT-B/32 QuickGELU weights as the image branch. This helper
    does not load weights or change the caller's train/eval state.
    """
    prompts = DEFAULT_CONCEPT_PROMPTS if prompts is None else prompts
    if model.training:
        raise ValueError('Encode concepts with the frozen GeoRSCLIP model in eval mode')
    if (not prompts or any(not templates or isinstance(templates, str)
                           or any(not isinstance(t, str) or not t.strip() for t in templates)
                           for templates in prompts.values())):
        raise ValueError('Each concept needs a nonempty sequence of prompt strings')
    flat = [text for templates in prompts.values() for text in templates]
    encoded = F.normalize(model.encode_text(tokenizer(flat).to(device)).float(), dim=-1)
    result, offset = {}, 0
    for name, templates in prompts.items():
        result[name] = encoded[offset:offset + len(templates)].detach()
        offset += len(templates)
    return result


class GreenhouseContrastPrior(nn.Module):
    """Subtract bounded greenhouse evidence from A3's semantic contribution.

    Inputs:
      geo: B,C,H,W overlap means of unit projected GeoRSCLIP crop embeddings;
           do NOT normalize again after overlap averaging.
      spatial: B,1,H,W A2 foreground/background cosine margin divided by two.

    Evidence is positive only if greenhouse beats PV AND the strongest other
    background concept. Equal/ambiguous scores do not trigger a correction.
    The A2 contribution is preserved. All scalar settings are fixed per run;
    choose them using train/calibration data, never individual query masks.
    """

    def __init__(self, concept_embeddings, semantic_weight=.7, strength=1.,
                 max_penalty=.2, pv_margin=0., other_margin=0.,
                 cam_scale=1., cam_bias=0.):
        super().__init__()
        names = tuple(concept_embeddings)
        if 'pv' not in names or 'greenhouse' not in names or len(names) < 3:
            raise ValueError('Need pv, greenhouse and at least one other background concept')
        settings = (semantic_weight, strength, max_penalty, pv_margin, other_margin,
                    cam_scale, cam_bias)
        if (not all(math.isfinite(v) for v in settings)
                or not 0 <= semantic_weight <= 1 or strength < 0
                or not 0 <= max_penalty <= 2 or pv_margin < 0 or other_margin < 0
                or cam_scale <= 0):
            raise ValueError('Invalid fixed fusion/correction/calibration settings')
        vectors, dim = [], None
        for bank in concept_embeddings.values():
            bank = torch.as_tensor(bank).detach().float()
            if bank.ndim != 2 or not bank.shape[0] or not bank.shape[1]:
                raise ValueError('Each concept bank must have shape templates, embedding_dim')
            if not torch.isfinite(bank).all() or (bank.norm(dim=1) < 1e-7).any():
                raise ValueError('Concept embeddings must be finite and nonzero')
            if dim is not None and bank.shape[1] != dim:
                raise ValueError('Concept embedding dimensions differ')
            dim = bank.shape[1]
            # Mean cosine over templates; do not renormalize the template mean.
            vectors.append(F.normalize(bank, dim=1).mean(0))
        self.register_buffer('concept_vectors', torch.stack(vectors))
        self.concept_names = names
        self.pv_index, self.gh_index = names.index('pv'), names.index('greenhouse')
        self.other_indices = [i for i, name in enumerate(names) if name not in ('pv', 'greenhouse')]
        self.semantic_weight, self.strength = semantic_weight, strength
        self.max_penalty = max_penalty
        self.pv_margin, self.other_margin = pv_margin, other_margin
        self.cam_scale, self.cam_bias = cam_scale, cam_bias

    def forward(self, geo, spatial, output_size=None):
        if geo.ndim != 4 or spatial.ndim != 4 or spatial.shape[1] != 1:
            raise ValueError('Expected geo B,C,H,W and spatial B,1,H,W')
        if (geo.shape[1] != self.concept_vectors.shape[1]
                or geo.shape[0] != spatial.shape[0] or geo.shape[2:] != spatial.shape[2:]):
            raise ValueError('Geo/text dimensions or spatial grid do not match')
        geo, spatial = geo.float(), spatial.float()
        if not torch.isfinite(geo).all() or not torch.isfinite(spatial).all():
            raise ValueError('Prior inputs must be finite')
        if (geo.norm(dim=1) > 1.001).any() or (spatial.abs() > 1.001).any():
            raise ValueError('Use unit crop embeddings before overlap means and A2 scores in [-1,1]')
        # Geo/text remain in their pretrained common space; DINO features never
        # enter this dot product. Mean crop cosines are preserved exactly.
        scores = torch.einsum('bchw,kc->bkhw', geo, self.concept_vectors.float())
        pv = scores[:, self.pv_index:self.pv_index + 1]
        gh = scores[:, self.gh_index:self.gh_index + 1]
        other = scores[:, self.other_indices].amax(1, keepdim=True)
        evidence = torch.minimum(gh - pv - self.pv_margin,
                                 gh - other - self.other_margin).relu()
        penalty = (self.strength * evidence).clamp(max=self.max_penalty)
        corrected = (pv - penalty).clamp(-1, 1)
        alpha = self.semantic_weight
        fused = (1 - alpha) * spatial + alpha * corrected
        maps = {'pv_semantic': pv, 'greenhouse_semantic': gh,
                'other_semantic': other, 'pv_greenhouse_margin': pv - gh,
                'greenhouse_evidence': evidence, 'semantic_penalty': penalty,
                'semantic_corrected': corrected, 'spatial': spatial,
                'baseline_score': (1 - alpha) * spatial + alpha * pv,
                'fused_score': fused}
        if output_size is not None:
            maps = {name: F.interpolate(value, size=output_size, mode='bilinear',
                                         align_corners=False) for name, value in maps.items()}
        # Monotone interface for downstream CAM modules. Default scale/bias are
        # uncalibrated; this map must not be described as a probability estimate.
        maps['fused_cam'] = torch.sigmoid(self.cam_scale * maps['fused_score'] + self.cam_bias)
        return maps

    def get_extra_state(self):
        # State dicts retain concept order AND fixed settings, not just vectors.
        return {name: getattr(self, name) for name in (
            'concept_names', 'semantic_weight', 'strength', 'max_penalty',
            'pv_margin', 'other_margin', 'cam_scale', 'cam_bias')}

    def set_extra_state(self, state):
        if tuple(state['concept_names']) != self.concept_names:
            raise ValueError('Concept order differs from saved prior')
        for name, value in state.items():
            if name != 'concept_names':
                setattr(self, name, value)
