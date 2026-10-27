"""DMSF, hierarchical priors and DCMA; no pretrained dependency at import time."""

import torch
from torch import nn
from torch.nn import functional as F


def resize(x, size, mode='bilinear'):
    return F.interpolate(x, size=size, mode=mode,
                         **({'align_corners': True} if mode == 'bilinear' else {}))


def minmax(x, eps=1e-7):
    # Compute reductions in FP32, including under AMP.
    x = x.float()
    low = x.amin(dim=(-2, -1), keepdim=True)
    return (x - low) / (x.amax(dim=(-2, -1), keepdim=True) - low + eps)


def masked_pool(features, mask):
    mask = resize(mask.float(), features.shape[-2:], 'nearest')
    return (features * mask).sum((-2, -1), keepdim=True) / mask.sum((-2, -1), keepdim=True).clamp_min(1)


def cosine_map(features, prototype):
    return (F.normalize(features.float(), dim=1, eps=1e-7) *
            F.normalize(prototype.float(), dim=1, eps=1e-7)).sum(1, keepdim=True)


class Projection(nn.Module):
    def __init__(self, source, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(source, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, x):
        return self.net(x)

    def map(self, x):
        return self(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CAMRefine(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return (x + .1 * self.net(x).tanh()).clamp(0, 1)


class DMSF(nn.Module):
    def __init__(self, dino_dim=1024, text_dim=512, hidden=256):
        super().__init__()
        self.visual = Projection(dino_dim, hidden)
        self.text_to_dino = nn.Sequential(
            nn.Linear(text_dim, dino_dim), nn.LayerNorm(dino_dim), nn.GELU(),
            nn.Linear(dino_dim, dino_dim), nn.LayerNorm(dino_dim), nn.GELU(),
            nn.Linear(dino_dim, dino_dim))
        self.text_hidden = Projection(dino_dim, hidden)
        self.spatial_refine, self.semantic_refine = CAMRefine(), CAMRefine()
        self.beta = nn.Parameter(torch.tensor(.7))

    def forward(self, query, supports, masks, text):
        q = self.visual.map(query)
        prototypes = [masked_pool(self.visual.map(s), m) for s, m in zip(supports, masks)]
        # Equal shot averaging follows the hierarchical-prior convention; Gram
        # weights are used for the CNN support prototype and learner merge.
        spatial = self.spatial_refine(minmax(torch.stack([cosine_map(q, p) for p in prototypes]).mean(0)))
        t = self.text_hidden(self.text_to_dino(text)).unsqueeze(-1).unsqueeze(-1)
        semantic = self.semantic_refine(minmax(cosine_map(q, t)))
        gamma = self.beta.sigmoid()
        return {'spatial': spatial, 'semantic': semantic,
                'fused': gamma * spatial + (1 - gamma) * semantic, 'gamma': gamma}


def hierarchical_prior(query_levels, support_levels, masks, target_size, routing='literal'):
    """Eq. 10–12. Expose the normalized-max degeneracy rather than hide it."""
    if routing not in ('literal', 'raw_max'):
        raise ValueError('routing must be literal or explicitly experimental raw_max')
    fg_maps, dominant_maps, chosen, ties = [], [], [], []
    for levels, mask in zip(support_levels, masks):
        pairs, scores = [], []
        for q, s in zip(query_levels, levels):
            raw_fg = cosine_map(q, masked_pool(s, mask))
            raw_bg = cosine_map(q, masked_pool(s, 1 - mask))
            fg, bg = resize(minmax(raw_fg), target_size), resize(minmax(raw_bg), target_size)
            pairs.append((fg, (fg - bg).relu()))
            scores.append((fg if routing == 'literal' else raw_fg).flatten(1).amax(1))
        pick_base = scores[1] > scores[0]
        selector = pick_base[:, None, None, None]
        fg_maps.append(torch.where(selector, pairs[1][0], pairs[0][0]))
        dominant_maps.append(torch.where(selector, pairs[1][1], pairs[0][1]))
        chosen.append(pick_base)
        ties.append(torch.isclose(scores[0], scores[1], atol=1e-6, rtol=0))
    return (torch.stack(fg_maps).mean(0), torch.stack(dominant_maps).mean(0),
            {'base_selected': torch.stack(chosen, 1), 'near_tie': torch.stack(ties, 1)})


def gram(x):
    x = F.normalize(x.float().flatten(2), dim=2, eps=1e-7)
    return x @ x.transpose(1, 2)


class ShotWeighting(nn.Module):
    """Sorted Gram discrepancy reweighting, adapted from BAM (MIT)."""
    def __init__(self, shots):
        super().__init__()
        self.shots = shots
        if shots > 1:
            self.reweight = nn.Sequential(nn.Linear(shots, shots), nn.ReLU(), nn.Linear(shots, shots))

    def forward(self, query, supports):
        if len(supports) != self.shots:
            raise ValueError(f'Checkpoint configured for {self.shots} shots, got {len(supports)}')
        gq = gram(query)
        distances = torch.stack([(gq - gram(s)).norm(dim=(1, 2)) / gq.shape[1] for s in supports], 1)
        if self.shots == 1:
            weights = torch.ones_like(distances)
        else:
            values, indices = distances.sort(1)
            weights = self.reweight(values).gather(1, indices.argsort(1)).softmax(1)
        return weights, (weights * distances).sum(1)[:, None, None, None]


class DCMA(nn.Module):
    def __init__(self, hidden=256, heads=8, query_chunk=256):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.hidden, self.heads, self.query_chunk = hidden, heads, query_chunk
        self.modulate = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.q, self.k, self.v = [nn.Linear(hidden, hidden, bias=False) for _ in range(3)]
        self.out = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.pseudo = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.pseudo[-1].weight)
        nn.init.zeros_(self.pseudo[-1].bias)

    def attend(self, query, support, mask):
        b, nq, c = query.shape
        ns, h = support.shape[1], self.heads
        q = self.q(query).reshape(b, nq, h, c // h).transpose(1, 2)
        k = self.k(support).reshape(b, ns, h, c // h).transpose(1, 2)
        v = self.v(support).reshape(b, ns, h, c // h).transpose(1, 2)
        # Eq. 7: if no foreground exists, eta_fg=0 and masking is disabled.
        # Actual PV episodes reject empty supports earlier, but the module stays finite.
        eta = mask.any(1, keepdim=True)
        bias = (-1000 * ((~mask) & eta).to(q.dtype))[:, None, None, :]
        chunk = self.query_chunk or nq
        attended = torch.cat([F.scaled_dot_product_attention(
            q[:, :, start:start + chunk], k, v, attn_mask=bias, dropout_p=0.)
            for start in range(0, nq, chunk)], dim=2)
        x = self.norm1(query + self.out(attended.transpose(1, 2).reshape(b, nq, c)))
        return x + self.ffn(self.norm2(x))

    def forward(self, query, supports, masks, cam):
        b, c, height, width = query.shape
        w = self.modulate(torch.cat([cam, 1 - cam], 1))
        q = (query * (1 + .1 * w)).flatten(2).transpose(1, 2)
        s = torch.cat([x.flatten(2).transpose(1, 2) for x in supports], 1)
        mask = torch.cat([resize(m, x.shape[-2:], 'nearest').flatten(1) > .5
                          for m, x in zip(masks, supports)], 1)
        x = self.attend(q, s, mask).transpose(1, 2).reshape(b, c, height, width)
        return x, self.pseudo(x).clamp(-5, 5)


class ASPP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(hidden, hidden, 1)] + [
            nn.Conv2d(hidden, hidden, 3, padding=r, dilation=r) for r in (6, 12, 18)])
        self.global_branch = nn.Conv2d(hidden, hidden, 1)
        self.reduce = nn.Sequential(nn.Conv2d(5 * hidden, hidden, 1), nn.ReLU())
        self.residual = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())

    def forward(self, x):
        maps = [F.relu(branch(x)) for branch in self.branches]
        maps.append(resize(F.relu(self.global_branch(F.adaptive_avg_pool2d(x, 1))), x.shape[-2:]))
        x = self.reduce(torch.cat(maps, 1))
        return x + self.residual(x)
