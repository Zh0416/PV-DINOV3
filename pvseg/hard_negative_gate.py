"""A small residual gate for reducing supervised hard-background false positives.

The existing dynamic alpha module can only interpolate two frozen CAMs.  This
module keeps that interpolation and adds a bounded, zero-initialized residual
so a hard background can be lowered even when both CAMs are high.
"""

import torch
import math
from torch import nn
from torch.nn import functional as F


class ResidualFusionGate(nn.Module):
    """Predict a local alpha and a bounded score correction.

    ``forward`` returns ``alpha`` in [0, 1] and a signed residual in
    ``[-max_residual, max_residual]``.  The last layer is zero initialized, so
    the module starts exactly at the supplied initial alpha and the original
    CAM score.
    """

    def __init__(self, channels, hidden=16, initial_alpha=0.7, max_residual=0.15):
        super().__init__()
        if channels <= 0 or hidden <= 0 or not 0 < initial_alpha < 1:
            raise ValueError("Invalid gate dimensions or initial alpha")
        if not 0 < max_residual <= 1:
            raise ValueError("max_residual must be in (0, 1]")
        self.channels = int(channels)
        self.max_residual = float(max_residual)
        self.register_buffer(
            "initial_logit",
            torch.tensor(math.log(initial_alpha / (1 - initial_alpha)))
        )
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
        )
        self.alpha_head = nn.Conv2d(hidden, 1, 1)
        self.residual_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, inputs):
        inputs = inputs.detach().float()
        if (inputs.ndim != 4 or inputs.shape[1] != self.channels
                or not torch.isfinite(inputs).all()):
            raise ValueError(f"Expected finite B,{self.channels},H,W inputs")
        hidden = self.body(inputs)
        alpha = torch.sigmoid(self.initial_logit + self.alpha_head(hidden))
        residual = self.max_residual * torch.tanh(self.residual_head(hidden))
        return alpha, residual


def residual_fusion(spatial, semantic, alpha, residual, output_size=None):
    """Interpolate frozen CAMs, alpha, and residual before combining them."""
    values = (spatial, semantic, alpha, residual)
    if any(v.ndim != 4 for v in values) or len({tuple(v.shape) for v in values}) != 1:
        raise ValueError("CAM, alpha, and residual shapes must match")
    if (not all(torch.isfinite(v).all() for v in values)
            or (alpha < 0).any() or (alpha > 1).any()):
        raise ValueError("Fusion inputs must be finite and alpha must be in [0, 1]")
    d, s = spatial.detach().float(), semantic.detach().float()
    if output_size is not None:
        d = F.interpolate(d, output_size, mode="bilinear", align_corners=False)
        s = F.interpolate(s, output_size, mode="bilinear", align_corners=False)
        alpha = F.interpolate(alpha - 0.7, output_size, mode="bilinear", align_corners=False) + 0.7
        residual = F.interpolate(residual, output_size, mode="bilinear", align_corners=False)
    return (1 - alpha) * d + alpha * s + residual


def gate_regularization(alpha, residual, initial_alpha=0.7, residual_weight=1.0,
                        tv_weight=1.0):
    """Return compact anchor, residual, and total-variation regularizers."""
    anchor = (alpha - initial_alpha).square().mean()
    residual_penalty = residual.square().mean()
    tv = ((alpha[..., 1:, :] - alpha[..., :-1, :]).abs().mean()
          + (alpha[..., :, 1:] - alpha[..., :, :-1]).abs().mean())
    tv += ((residual[..., 1:, :] - residual[..., :-1, :]).abs().mean()
           + (residual[..., :, 1:] - residual[..., :, :-1]).abs().mean())
    return anchor, residual_weight * residual_penalty + tv_weight * tv
