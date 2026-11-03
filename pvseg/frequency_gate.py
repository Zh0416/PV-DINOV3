"""Frequency-conditioned extension of the bounded residual fusion gate."""

import math

import torch
from torch import nn


class FrequencyResidualGate(nn.Module):
    def __init__(self, base_channels=15, frequency_channels=9, hidden=16,
                 initial_alpha=.7, max_residual=.15):
        super().__init__()
        if min(base_channels, frequency_channels, hidden) <= 0:
            raise ValueError('Channel counts must be positive')
        self.base_channels = base_channels
        self.frequency_channels = frequency_channels
        self.max_residual = max_residual
        self.register_buffer('initial_logit',
                             torch.tensor(math.log(initial_alpha / (1 - initial_alpha))))
        self.body = nn.Sequential(nn.Conv2d(base_channels, hidden, 3, padding=1,
                                            padding_mode='replicate'), nn.GELU())
        self.frequency_projection = nn.Conv2d(frequency_channels, hidden, 1, bias=False)
        self.frequency_gate = nn.Conv2d(hidden, hidden, 1)
        self.alpha_head = nn.Conv2d(hidden, 1, 1)
        self.residual_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.frequency_projection.weight)
        for head in (self.alpha_head, self.residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, base, frequency):
        base, frequency = base.detach().float(), frequency.detach().float()
        if (base.ndim != 4 or frequency.ndim != 4 or
                base.shape[1] != self.base_channels or
                frequency.shape[1] != self.frequency_channels or
                base.shape[0] != frequency.shape[0] or
                base.shape[2:] != frequency.shape[2:] or
                not torch.isfinite(base).all() or not torch.isfinite(frequency).all()):
            raise ValueError('Expected finite, aligned base and frequency maps')
        hidden = self.body(base)
        update = torch.tanh(self.frequency_projection(frequency))
        hidden = hidden + torch.sigmoid(self.frequency_gate(hidden)) * update
        alpha = torch.sigmoid(self.initial_logit + self.alpha_head(hidden))
        residual = self.max_residual * torch.tanh(self.residual_head(hidden))
        return alpha, residual


class FrequencyCorrection(nn.Module):
    """A bounded correction that starts at exactly zero on a frozen baseline."""

    def __init__(self, base_channels=15, frequency_channels=9, hidden=16,
                 max_correction=.06):
        super().__init__()
        self.base_channels = base_channels
        self.frequency_channels = frequency_channels
        self.max_correction = max_correction
        self.context = nn.Sequential(nn.Conv2d(base_channels, hidden, 3, padding=1,
                                               padding_mode='replicate'), nn.GELU())
        self.frequency_projection = nn.Conv2d(frequency_channels, hidden, 1, bias=False)
        self.frequency_gate = nn.Conv2d(hidden, hidden, 1)
        self.head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.frequency_projection.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base, frequency):
        base, frequency = base.detach().float(), frequency.detach().float()
        if (base.ndim != 4 or frequency.ndim != 4 or
                base.shape[1] != self.base_channels or
                frequency.shape[1] != self.frequency_channels or
                base.shape[0] != frequency.shape[0] or
                base.shape[2:] != frequency.shape[2:] or
                not torch.isfinite(base).all() or not torch.isfinite(frequency).all()):
            raise ValueError('Expected finite, aligned base and frequency maps')
        hidden = self.context(base)
        update = torch.tanh(self.frequency_projection(frequency))
        hidden = hidden + torch.sigmoid(self.frequency_gate(hidden)) * update
        return self.max_correction * torch.tanh(self.head(hidden))
