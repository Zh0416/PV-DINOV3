"""Frozen generic DINOv3-SAT feature extractor for support-only tasks."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


MODEL = 'vit_large_patch16_dinov3.sat493m'
MODEL_SHA256 = '8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286'
MEAN = (.430, .411, .296)
STD = (.213, .156, .143)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class FrozenDINOv3SAT(nn.Module):
    def __init__(self, weights, image_size=896):
        super().__init__()
        import timm
        if timm.__version__ != '1.0.29':
            raise RuntimeError('Expected timm==1.0.29')
        weights = Path(weights)
        config = json.loads(weights.with_name('config.json').read_text(encoding='utf-8'))
        cfg = config.get('pretrained_cfg', {})
        if (config.get('architecture') != 'vit_large_patch16_dinov3' or
                cfg.get('tag') != 'sat493m' or sha256(weights) != MODEL_SHA256 or
                tuple(cfg['mean']) != MEAN or tuple(cfg['std']) != STD):
            raise ValueError('Unexpected generic DINOv3-SAT checkpoint')
        self.encoder = timm.create_model(MODEL, pretrained=True, num_classes=0,
                                         img_size=image_size,
                                         pretrained_cfg_overlay=dict(file=str(weights),
                                                                     hf_hub_id=None, url=''))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb):
        mean = rgb.new_tensor(MEAN)[None, :, None, None]
        std = rgb.new_tensor(STD)[None, :, None, None]
        maps = self.encoder.forward_intermediates((rgb - mean) / std,
            indices=[23], norm=True, output_fmt='NCHW',
            return_prefix_tokens=False, intermediates_only=True)
        if len(maps) != 1 or maps[0].shape[1:] != (1024, rgb.shape[-2] // 16,
                                                    rgb.shape[-1] // 16):
            raise RuntimeError('Unexpected DINOv3 patch feature shape')
        return maps[0]
