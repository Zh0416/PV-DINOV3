"""SSPNet reconstruction. Inference accepts RGB query + labeled supports only."""

import torch
from torch import nn

from .modules import (ASPP, DCMA, DMSF, ShotWeighting, hierarchical_prior,
                      masked_pool, resize)


class SSPNet(nn.Module):
    def __init__(self, cnn, dino, text_embedding, text_metadata, shots=1, hidden=256,
                 heads=8, query_chunk=256, routing='literal'):
        super().__init__()
        self.cnn, self.dino = cnn.requires_grad_(False), dino.requires_grad_(False)
        # Freezing does not erase gradients left by base-stage pretraining.
        # Remove them at the stage boundary so later audits detect new grads.
        self.cnn.zero_grad(set_to_none=True)
        self.dino.zero_grad(set_to_none=True)
        self.cnn.eval()
        self.dino.eval()
        self.register_buffer('raw_text', text_embedding.detach().clone())
        self.shots, self.routing = shots, routing
        self.dmsf = DMSF(dino.dim, text_embedding.shape[-1], hidden)
        self.down_q, self.down_s = [nn.Sequential(nn.Conv2d(cnn.mid_dim, hidden, 1, bias=False),
            nn.ReLU(), nn.Dropout2d(.5)) for _ in range(2)]
        self.query_merge = nn.Sequential(nn.Conv2d(hidden * 2 + 4, hidden, 1, bias=False), nn.ReLU())
        self.support_merge = nn.Sequential(nn.Conv2d(hidden * 2 + 1, hidden, 1, bias=False), nn.ReLU())
        self.shot_weighting = ShotWeighting(shots)
        self.dcma = DCMA(hidden, heads, query_chunk)
        self.decoder = ASPP(hidden)
        self.meta_head = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                       nn.Dropout2d(.1), nn.Conv2d(hidden, 2, 1))
        # Gram adjustment follows BAM; in this single known-class PV adaptation,
        # combine corresponding base/meta classes instead of treating PV as an
        # "other base class" that must suppress the meta foreground.
        self.gram_merge = nn.Conv2d(2, 1, 1, bias=False)
        self.class_merges = nn.ModuleList(nn.Conv2d(2, 1, 1, bias=False) for _ in range(2))
        for layer in [self.gram_merge, *self.class_merges]:
            nn.init.zeros_(layer.weight)
            layer.weight.data[0, 0, 0, 0] = 1
        self.metadata = {'format': 'sspnet_pv_dinov3_v1', 'shots': shots, 'hidden': hidden,
            'heads': heads, 'routing': routing, 'dino': getattr(dino, 'metadata', {}),
            'cnn': getattr(cnn, 'provenance', {}), 'text': text_metadata,
            'base_merge': 'per_class_known_PV', 'input': 'RGB_float_0_1'}

    def train(self, mode=True):
        super().train(mode)
        self.cnn.eval()
        self.dino.eval()
        return self

    def forward(self, query, support_images, support_masks):
        if support_images.ndim != 5 or support_masks.ndim != 5:
            raise ValueError('Supports must be [B,K,3,H,W], masks [B,K,1,H,W]')
        if support_images.shape[1] != self.shots or support_masks.shape[:2] != support_images.shape[:2]:
            raise ValueError('Support count differs from configured shot count')
        images = list(support_images.unbind(1))
        masks = list(support_masks.unbind(1))
        for mask in masks:
            if not (resize(mask, (query.shape[-2] // 16, query.shape[-1] // 16), 'nearest').flatten(1).sum(1) > 0).all():
                raise ValueError('Support foreground disappeared on the DINO patch grid')
        # Encode one image at a time so 5-shot does not multiply encoder activations.
        with torch.no_grad():
            qcnn, qdino = self.cnn.features(query), self.dino(query)
            scnn = [self.cnn.features(image) for image in images]
            sdino = [self.dino(image) for image in images]
        return self.forward_from_features(qcnn, qdino, scnn, sdino, masks, query.shape[-2:])

    def forward_from_features(self, qcnn, qdino, scnn, sdino, masks, output_size):
        """Train the same network from frozen encoder outputs, without query labels.

        Cache only the encoder outputs. DMSF projections, CAMs, assembly and
        DCMA must run with autograd on every episode, including support pooling.
        """
        if not (len(scnn) == len(sdino) == len(masks) == self.shots):
            raise ValueError('Cached support count differs from configured shot count')
        if any(not (resize(m.float(), s.shape[-2:], 'nearest').flatten(1).sum(1) > 0).all()
               for s, m in zip(sdino, masks)):
            raise ValueError('Support foreground disappeared on the DINO patch grid')
        cam = self.dmsf(qdino, sdino, masks, self.raw_text)
        q = self.down_q(qcnn['mid'])
        s = [self.down_s(x['mid']) for x in scnn]
        size = q.shape[-2:]
        weights, discrepancy = self.shot_weighting(qcnn['low'], [x['low'] for x in scnn])
        prototypes = [masked_pool(x, m) for x, m in zip(s, masks)]
        prototype = sum(weights[:, i, None, None, None] * p for i, p in enumerate(prototypes))
        fg, dominant, routing = hierarchical_prior(qcnn['levels'], [x['levels'] for x in scnn], masks, size, self.routing)
        fused = resize(cam['fused'], size)
        q = self.query_merge(torch.cat([q, prototype.expand_as(q), fg, dominant, fused, 1 - fused], 1))
        s = [self.support_merge(torch.cat([x, p.expand_as(x), resize(m, x.shape[-2:], 'nearest')], 1))
             for x, p, m in zip(s, prototypes, masks)]
        refined, pseudo = self.dcma(q, s, masks, fused)
        meta = self.meta_head(self.decoder(refined))
        base_prob, meta_prob = qcnn['base_logits'].float().softmax(1), meta.float().softmax(1)
        est = discrepancy.expand_as(meta_prob[:, :1])
        adjusted = [self.gram_merge(torch.cat([meta_prob[:, i:i+1], est], 1)) for i in range(2)]
        final = torch.cat([merge(torch.cat([adjusted[i], base_prob[:, i:i+1]], 1))
                           for i, merge in enumerate(self.class_merges)], 1)
        return {'logits': resize(final, output_size), 'meta_logits': resize(meta, output_size),
                'base_logits': resize(qcnn['base_logits'], output_size),
                'pseudo_logits': pseudo, 'spatial_cam': cam['spatial'],
                'semantic_cam': cam['semantic'], 'prior_cam': cam['fused'],
                'fused_cam': fused, 'gamma': cam['gamma'],
                'shot_weights': weights, 'routing': routing}
