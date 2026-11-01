"""Apply a saved residual/alpha candidate without reading the query mask.

This consumes the frozen DINO/Geo/brightness caches already used by the
experiment. For new imagery generate the same upstream CAMs first; this is
not a standalone RGB foundation-model extractor or the full SSPNet decoder.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pvseg.dynamic_brightness_fusion import CAMWeight, gate_inputs
from pvseg.hard_negative_gate import ResidualFusionGate, residual_fusion
from pvseg.local_texture import texture_features


def predict(checkpoint, rgb, spatial, semantic, penalty, brightness_z, brightness_gate, support):
    cfg = checkpoint['config']
    model = (CAMWeight('local', hidden=16) if cfg.get('alpha_only') else
             ResidualFusionGate(cfg['channels'], hidden=16, max_residual=.15))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    tensor = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]
    effective = penalty * brightness_gate
    d, s = tensor(spatial), tensor(np.clip(semantic - effective, -1, 1))
    x = gate_inputs(d, s, tensor(brightness_z), tensor(brightness_gate), tensor(effective))
    if cfg['channels'] > 8:
        x = torch.cat((x, torch.from_numpy(texture_features(rgb))[None]), 1)[:, :cfg['channels']]
    if cfg.get('zero_fft'):
        x[:, 8:10] = 0
    with torch.no_grad():
        if cfg.get('alpha_only'):
            a = model(x)
            r = torch.zeros_like(a)
        else:
            a, r = model(x)
        score = residual_fusion(d, s, a, r, (896, 896))[0, 0].numpy()
    threshold = checkpoint['calibration'][str(support)]['threshold']
    return score, score >= threshold, a[0, 0].numpy(), r[0, 0].numpy()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--sample-id', required=True)
    p.add_argument('--support', type=int, choices=(42,43,44), default=42)
    p.add_argument('--score-cache', type=Path, required=True)
    p.add_argument('--brightness-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    scores, brightness = np.load(args.score_cache), np.load(args.brightness_cache)
    prefix = f'{args.sample_id}_{args.support}_'
    with Image.open(args.image) as im:
        rgb = np.asarray(im.convert('RGB'))
    score, mask, alpha, residual = predict(ckpt, rgb, scores[prefix+'spatial'], scores[prefix+'pv_semantic'],
        scores[prefix+'semantic_penalty'], brightness[args.sample_id+'_z'], brightness[args.sample_id+'_gate'], args.support)
    args.output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(mask.astype(np.uint8)*255).save(args.output/'mask.png')
    np.savez_compressed(args.output/'prediction.npz', score=score, alpha=alpha, residual=residual)
    (args.output/'metadata.json').write_text(json.dumps(dict(sample_id=args.sample_id, support=args.support,
        checkpoint=str(args.checkpoint), threshold=ckpt['calibration'][str(args.support)]['threshold'],
        query_mask_read=False), indent=2), encoding='utf-8')
