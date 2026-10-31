"""Learn only convex CAM weights after the unchanged brightness correction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def gate_inputs(spatial, semantic, brightness_z, brightness_gate, effective_penalty):
    """Eight frozen scalar maps; no masks or encoder embeddings enter this API."""
    values = [v.detach().float() for v in
              (spatial, semantic, brightness_z, brightness_gate, effective_penalty)]
    if (any(v.ndim != 4 or v.shape[1] != 1 or v.shape != values[0].shape for v in values)
            or any(not torch.isfinite(v).all() for v in values)):
        raise ValueError('Expected five finite matching B,1,H,W maps')
    d, s, z, g, penalty = values
    def local_std(x):
        mean = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        second = F.avg_pool2d(x.square(), 3, 1, 1, count_include_pad=False)
        return (second - mean.square()).clamp_min(0).add(1e-8).sqrt()
    return torch.cat((d, s, z.clamp(-1, 2), g, penalty, s - d,
                      local_std(d), local_std(s)), 1)


def convex_fusion(spatial, semantic, alpha, output_size=None):
    """Interpolate each operand FIRST, then fuse in output coordinates.

    This differs from interpolating alpha*S+(1-alpha)*D when alpha varies.
    Detaching the two frozen CAMs also prevents accidental encoder training.
    """
    d, s = spatial.detach().float(), semantic.detach().float()
    if d.shape != s.shape or alpha.shape != d.shape:
        raise ValueError('CAMs and alpha must have matching B,1,H,W shapes')
    if not all(torch.isfinite(x).all() for x in (d, s, alpha)) or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError('Alpha must be finite and within [0,1]')
    if output_size is not None:
        d, s = [F.interpolate(x, output_size, mode='bilinear', align_corners=False) for x in (d, s)]
        # Centering is algebraically the same interpolation, while constant .7
        # remains bit-exact instead of drifting by one FP32 interpolation ULP.
        alpha = F.interpolate(alpha - .7, output_size, mode='bilinear', align_corners=False) + .7
    return (1 - alpha) * d + alpha * s


class CAMWeight(nn.Module):
    """A fixed, learned global, or learned local semantic weight. No logit head."""
    def __init__(self, mode='local', initial_alpha=.7, hidden=8):
        super().__init__()
        if mode not in ('fixed', 'global', 'local') or not 0 < initial_alpha < 1:
            raise ValueError('Invalid mode or initial weight')
        self.mode = mode
        self.register_buffer('initial_logit', torch.tensor(math.log(initial_alpha / (1 - initial_alpha))))
        if mode == 'global':
            self.delta = nn.Parameter(torch.zeros(()))
        elif mode == 'local':
            self.net = nn.Sequential(nn.Conv2d(8, hidden, 3, padding=1, padding_mode='replicate'),
                                     nn.GELU(), nn.Conv2d(hidden, 1, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        inputs = inputs.detach().float()
        if inputs.ndim != 4 or inputs.shape[1] != 8 or not torch.isfinite(inputs).all():
            raise ValueError('Expected finite B,8,H,W gate inputs')
        delta = self.net(inputs) if self.mode == 'local' else (self.delta if self.mode == 'global' else 0.)
        return torch.sigmoid(self.initial_logit + delta).expand_as(inputs[:, :1])


def weight_regularization(alpha, initial=.7):
    anchor = (alpha - initial).square().mean()
    tv = ((alpha[..., 1:, :] - alpha[..., :-1, :]).abs().mean()
          + (alpha[..., :, 1:] - alpha[..., :, :-1]).abs().mean())
    return anchor, tv
