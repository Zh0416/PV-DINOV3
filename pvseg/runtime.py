"""Reproducibility, AMP selection, decoder-only checkpoints, and PNG outputs."""

from contextlib import nullcontext
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment():
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'cuda_version': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
              'versions': {}, 'gpus': []}
    for name in ['torchvision', 'timm', 'numpy', 'Pillow', 'safetensors', 'huggingface-hub']:
        result['versions'][name] = metadata.version(name)
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        result['gpus'].append({'index': index, 'name': properties.name,
                               'total_memory_bytes': properties.total_memory,
                               'capability': list(torch.cuda.get_device_capability(index))})
    try:
        import psutil
        result['ram'] = psutil.virtual_memory()._asdict()
    except ImportError:
        pass
    try:
        result['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result['git_commit'] = None
    return result


class Precision:
    def __init__(self, device):
        self.device = torch.device(device)
        self.dtype = None
        if self.device.type == 'cuda':
            capability = torch.cuda.get_device_capability(self.device)
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif capability[0] >= 7:
                self.dtype = torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.dtype == torch.float16)

    def context(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.dtype else nullcontext()

    def describe(self):
        return {'amp_enabled': self.dtype is not None, 'dtype': str(self.dtype),
                'grad_scaler_enabled': self.scaler.is_enabled()}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory():
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'free_device_bytes': free, 'total_device_bytes': total}


def timed(call):
    sync()
    start = time.perf_counter()
    result = call()
    sync()
    return result, time.perf_counter() - start


def decoder_optimizer(model, lr=1e-4):
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr, weight_decay=1e-4)
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected or any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError('Optimizer must contain exactly the trainable decoder parameters')
    return optimizer


def save_checkpoint(path, model, optimizer, config, software, **extra):
    torch.save({'format': 'pvseg_decoder_v1', 'decoder_state': model.decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'model': model.metadata,
                'config': config, 'software': software, 'rng_cpu': torch.get_rng_state(),
                **extra}, path)


def load_checkpoint(path, model, optimizer=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'pvseg_decoder_v1' or state['model']['backbone_id'] != model.metadata['backbone_id']:
        raise ValueError('Checkpoint format or backbone mismatch')
    if state['model']['weight_source']['sha256'] != model.metadata['weight_source']['sha256']:
        raise ValueError('Checkpoint requires a different pretrained weight file')
    for key in ['normalization', 'decoder', 'indices_zero_based', 'intermediate_norm']:
        if state['model'][key] != model.metadata[key]:
            raise ValueError(f'Checkpoint/model {key} mismatch')
    model.decoder.load_state_dict(state['decoder_state'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state'])
    return state


def save_prediction(output, batch, logits, normalization, prefix='', smoke=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(normalization['mean']).view(3, 1, 1)
    std = torch.tensor(normalization['std']).view(3, 1, 1)
    for index, sid in enumerate(batch['sample_id']):
        rgb = ((batch['image'][index].cpu() * std + mean).clamp(0, 1) * 255).round().byte()
        rgb = rgb.permute(1, 2, 0).numpy()
        truth = batch['mask'][index, 0].cpu().numpy() >= .5
        pred = logits[index, 0].detach().float().sigmoid().cpu().numpy() >= .5
        Image.fromarray(pred.astype('uint8') * 255).save(output / f'{prefix}{sid}_prediction.png')
        overlay = rgb.copy()
        overlay[pred] = (overlay[pred] * .55 + np.array([255, 30, 30]) * .45).astype('uint8')
        size = 256
        canvas = Image.new('RGB', (size * 4, size + 40), 'white')
        draw = ImageDraw.Draw(canvas)
        caption = 'random-decoder engineering check only' if smoke else 'pilot prediction; not final validation'
        draw.text((4, 3), f'{sid} | {caption}', fill='black')
        for col, (array, title) in enumerate(zip([rgb, truth.astype('uint8') * 255,
                                                 pred.astype('uint8') * 255, overlay],
                                                ['RGB', 'Ground truth', 'Prediction >=0.5', 'Prediction overlay'])):
            panel = Image.fromarray(array).convert('RGB')
            method = Image.Resampling.NEAREST if col in [1, 2] else Image.Resampling.LANCZOS
            canvas.paste(panel.resize((size, size), method), (col * size, 40))
            draw.text((col * size + 4, 23), title, fill='black')
        canvas.save(output / f'{prefix}{sid}_overlay.png')
