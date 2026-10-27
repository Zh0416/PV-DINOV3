"""Frozen foundation encoders and a separately pretrained PSPNet base learner."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .modules import resize

SAT_MODEL = 'vit_large_patch16_dinov3.sat493m'
SAT_SHA = '8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286'
SAT_MEAN, SAT_STD = (.430, .411, .296), (.213, .156, .143)
CNN_MEAN, CNN_STD = (.485, .456, .406), (.229, .224, .225)
PV_PROMPT = ('a satellite aerial image of photovoltaic solar panels and solar arrays, '
             'dark rectangular modules arranged in regular rows on rooftops or on the ground')
GEO_REVISION = '4920188e6eba4e711ef9848cfd7cb77e874ee33f'
GEO_SHA = '129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_rgb(rgb, mean, std):
    return (rgb - rgb.new_tensor(mean)[None, :, None, None]) / rgb.new_tensor(std)[None, :, None, None]


def freeze_bn(module):
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class FrozenSAT(nn.Module):
    dim = 1024

    def __init__(self, weights, img_size=512):
        super().__init__()
        import timm
        if timm.__version__ != '1.0.29':
            raise RuntimeError('This adapter requires the verified timm==1.0.29 API')
        weights = Path(weights)
        config = json.loads(weights.with_name('config.json').read_text())
        cfg = config.get('pretrained_cfg', {})
        digest = sha256(weights)
        if (config.get('architecture') != 'vit_large_patch16_dinov3' or cfg.get('tag') != 'sat493m'
                or digest != SAT_SHA or tuple(cfg['mean']) != SAT_MEAN or tuple(cfg['std']) != SAT_STD):
            raise ValueError('SAT weights/config differ from the verified existing Kaggle asset')
        self.encoder = timm.create_model(SAT_MODEL, pretrained=True, num_classes=0, img_size=img_size,
            pretrained_cfg_overlay={'file': str(weights), 'hf_hub_id': None, 'url': ''})
        self.requires_grad_(False)
        self.eval()
        self.metadata = {'model': SAT_MODEL, 'sha256': digest, 'block_zero_based': 23,
                         'patch_size': 16, 'mean': list(SAT_MEAN), 'std': list(SAT_STD)}

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb):
        x = normalize_rgb(rgb, SAT_MEAN, SAT_STD)
        maps = self.encoder.forward_intermediates(x, indices=[23], norm=True, output_fmt='NCHW',
            return_prefix_tokens=False, intermediates_only=True)
        if len(maps) != 1 or maps[0].shape[1:] != (1024, rgb.shape[-2] // 16, rgb.shape[-1] // 16):
            raise RuntimeError(f'Unexpected DINO patch feature shapes: {[x.shape for x in maps]}')
        return maps[0]


def cache_georsclip_text(checkpoint, output, prompt=PV_PROMPT, device='cpu'):
    """Cache only the frozen raw embedding; all learned projections stay trainable."""
    import open_clip
    digest = sha256(checkpoint)
    if digest != GEO_SHA:
        raise ValueError('Expected the pinned RS5M ViT-B/32 base checkpoint, not a retrieval fine-tune')
    clip = open_clip.create_model('ViT-B-32', pretrained=None)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    missing, unexpected = clip.load_state_dict(state, strict=False)
    text_keys = [k for k in clip.state_dict() if not k.startswith('visual.')]
    missing_text = sorted(set(text_keys) & set(missing))
    if missing_text or unexpected:
        raise ValueError(f'GeoRSCLIP checkpoint mismatch: text_missing={missing_text}, unexpected={unexpected}')
    clip.requires_grad_(False).eval().to(device)
    tokens = open_clip.get_tokenizer('ViT-B-32')([prompt]).to(device)
    with torch.no_grad():
        text = F.normalize(clip.encode_text(tokens).float(), dim=-1).cpu()
    if text.shape != (1, 512) or not torch.isfinite(text).all():
        raise ValueError('Invalid GeoRSCLIP text embedding')
    meta = {'architecture': 'ViT-B-32', 'repository': 'Zilun/GeoRSCLIP',
            'filename': 'ckpt/RS5M_ViT-B-32.pt', 'sha256': digest, 'revision': GEO_REVISION,
            'prompt': prompt, 'missing_visual_keys': [k for k in missing if k.startswith('visual.')]}
    torch.save({'format': 'sspnet_georsclip_raw_text_v1', 'embedding': text, 'metadata': meta}, output)
    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, meta


def load_text_cache(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['format'] != 'sspnet_georsclip_raw_text_v1' or state['embedding'].shape != (1, 512)
            or state['metadata']['sha256'] != GEO_SHA or not torch.isfinite(state['embedding']).all()):
        raise ValueError('Not a validated GeoRSCLIP raw text cache')
    return state['embedding'], state['metadata']


class PyramidPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU()) for size in (1, 2, 3, 6))

    def forward(self, x):
        return torch.cat([x] + [resize(branch(x), x.shape[-2:]) for branch in self.branches], 1)


class PSPNet(nn.Module):
    """ResNet-50, output stride 8; torchvision stem is an explicit reconstruction choice."""
    mid_dim = 1536

    def __init__(self, imagenet=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if imagenet else None,
                       replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = cnn.layer1, cnn.layer2, cnn.layer3, cnn.layer4
        self.ppm = PyramidPool()
        self.head = nn.Sequential(nn.Conv2d(4096, 512, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout2d(.1))
        self.classifier = nn.Conv2d(512, 2, 1)
        self.provenance = {'architecture': 'torchvision_resnet50_os8_psp_1_2_3_6',
                           'initialization': 'IMAGENET1K_V1' if imagenet else 'random',
                           'segmentation_pretrained': False}

    def features(self, rgb):
        x = self.stem(normalize_rgb(rgb, CNN_MEAN, CNN_STD))
        low = self.layer1(x)
        block2 = self.layer2(low)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        base = self.head(self.ppm(block4))
        return {'low': low, 'mid': torch.cat([block2, block3], 1),
                'levels': (block4, base), 'base_logits': self.classifier(base)}

    def forward(self, rgb):
        return resize(self.features(rgb)['base_logits'], rgb.shape[-2:])
