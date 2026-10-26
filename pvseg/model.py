"""SegDINO root decoder adapted to the timm SAT encoder (not an exact v1 reproduction).

Decoder topology adapted from script-Yang/SegDINO dpt.py and blocks.py,
commit c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41; see third_party/SegDINO-LICENSE.
"""

import hashlib
import inspect
import json
from pathlib import Path

import timm
from timm.data import resolve_model_data_config
import torch
from torch import nn
from torch.nn import functional as F


MODEL_ID = 'vit_large_patch16_dinov3.sat493m'
HF_REPOSITORY = 'timm/' + MODEL_ID
HF_REVISION = 'ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c'
INDICES = (4, 11, 17, 23)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_weights(weights=None, download=False, cache_dir=None):
    if weights:
        path = Path(weights).resolve()
        config_path = path.with_name('config.json')
        if not path.is_file() or path.suffix != '.safetensors' or not config_path.is_file():
            raise ValueError('Expected timm-converted model.safetensors and adjacent config.json')
        origin = {'kind': 'mounted_timm_weights', 'path': str(path)}
    elif download:
        if not cache_dir or not Path(cache_dir).resolve().is_relative_to(Path('/kaggle/working').resolve()):
            raise ValueError('Downloads must stay in an explicit /kaggle/working cache')
        if not Path('/kaggle/working').is_dir():
            raise RuntimeError('Official weight download is allowed only inside Kaggle')
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(HF_REPOSITORY, 'model.safetensors', revision=HF_REVISION,
                                   cache_dir=cache_dir))
        config_path = Path(hf_hub_download(HF_REPOSITORY, 'config.json', revision=HF_REVISION,
                                          cache_dir=cache_dir))
        origin = {'kind': 'huggingface', 'repository': HF_REPOSITORY, 'revision': HF_REVISION,
                  'filename': 'model.safetensors', 'path': str(path)}
    else:
        raise ValueError('Provide --weights or explicitly allow --download_weights in Kaggle')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('architecture') != 'vit_large_patch16_dinov3' or config.get('pretrained_cfg', {}).get('tag') != 'sat493m':
        raise ValueError('Weight config is not the requested timm SAT-493M model')
    origin.update(sha256=sha256_file(path), config_path=str(config_path))
    return path, config, origin


class SegDINODecoder(nn.Module):
    def __init__(self, in_channels, features=128, out_channels=(96, 192, 384, 768)):
        super().__init__()
        self.config = {'features': features, 'out_channels': list(out_channels), 'nclass': 1,
                       'topology': 'SegDINO root projections/scratch/transpose/concat/1x1'}
        self.projects = nn.ModuleList(nn.Conv2d(in_channels, c, 1) for c in out_channels)
        self.scratch = nn.Module()
        for i, c in enumerate(out_channels, 1):
            setattr(self.scratch, f'layer{i}_rn', nn.Conv2d(c, features, 3, padding=1, bias=False))
        self.scratch.output_conv = nn.Conv2d(features * 4, 1, 1)
        self.proj = nn.ConvTranspose2d(features, features, 4, stride=4, bias=False)

    def forward(self, maps, output_size):
        if len(maps) != 4:
            raise ValueError(f'Expected four actual maps, found {len(maps)}')
        aligned = [getattr(self.scratch, f'layer{i + 1}_rn')(project(x))
                   for i, (project, x) in enumerate(zip(self.projects, maps))]
        first = self.proj(aligned[0])
        resized = [first] + [F.interpolate(x, size=first.shape[-2:], mode='bilinear', align_corners=True)
                             for x in aligned[1:]]
        logits = self.scratch.output_conv(torch.cat(resized, dim=1))
        return F.interpolate(logits, size=output_size, mode='bilinear', align_corners=True)


class FrozenSATSegDINO(nn.Module):
    def __init__(self, weight_path, weight_config, weight_source, img_size=896):
        super().__init__()
        if MODEL_ID not in timm.list_models('*dinov3*', pretrained=True, include_tags=True):
            raise RuntimeError(f'{MODEL_ID} not registered in timm {timm.__version__}')
        self.backbone = timm.create_model(
            MODEL_ID, pretrained=True, num_classes=0, img_size=img_size,
            pretrained_cfg_overlay={'file': str(weight_path), 'hf_hub_id': None, 'url': ''})
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.indices = INDICES
        signature = inspect.signature(self.backbone.forward_intermediates)
        required = {'indices', 'norm', 'output_fmt', 'intermediates_only', 'return_prefix_tokens'}
        if not required.issubset(signature.parameters):
            raise RuntimeError(f'Unsupported intermediate API: {signature}')
        self.data_config = resolve_model_data_config(self.backbone)
        for key in ['mean', 'std']:
            if tuple(self.data_config[key]) != tuple(weight_config['pretrained_cfg'][key]):
                raise ValueError(f'Loaded model {key} disagrees with pinned weight config')
        patch_size = tuple(self.backbone.patch_embed.patch_size)
        if patch_size != (16, 16) or len(self.backbone.blocks) <= max(self.indices):
            raise ValueError('Unexpected SAT patch size or depth')
        self.decoder = SegDINODecoder(self.backbone.num_features)
        self.metadata = {
            'backbone_id': MODEL_ID, 'weight_source': weight_source,
            'normalization': {k: list(self.data_config[k]) for k in ['mean', 'std']},
            'indices_zero_based': list(self.indices), 'intermediate_norm': True,
            'patch_size': list(patch_size), 'decoder': self.decoder.config,
            'api': str(signature), 'backbone_class': type(self.backbone).__module__ + '.' + type(self.backbone).__name__,
            'constructor': {'num_classes': 0, 'img_size': img_size},
            'encoder_signature': str(inspect.signature(type(self.backbone).__init__)),
            'pretrained_cfg': self.backbone.pretrained_cfg,
        }

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, images):
        self.backbone.eval()
        with torch.no_grad():
            maps = self.backbone.forward_intermediates(
                images, indices=list(self.indices), norm=True, output_fmt='NCHW',
                return_prefix_tokens=False, intermediates_only=True)
        expected = (images.shape[0], self.backbone.num_features, images.shape[-2] // 16, images.shape[-1] // 16)
        if len(maps) != 4 or any(tuple(x.shape) != expected for x in maps):
            raise RuntimeError(f'Actual feature shapes {[list(x.shape) for x in maps]} differ from {expected}')
        return maps

    def forward(self, images):
        return self.decoder(self.extract_features(images), images.shape[-2:])
