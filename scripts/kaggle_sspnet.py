"""Bounded GPU verification using actual SAT, GeoRSCLIP and ImageNet weights.

Downloads stay on Kaggle. This is an execution check, not a trained result.
"""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GEO_REVISION = '4920188e6eba4e711ef9848cfd7cb77e874ee33f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/kaggle/working/sspnet_smoke')
    p.add_argument('--weights', default='/kaggle/input/notebook5f7aef8a4f/verified_sat/model.safetensors')
    args = p.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Use Kaggle; do not download pretrained assets to the local source checkout')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completion.json').exists():
        raise RuntimeError('This smoke output already exists')
    # Keep the installed CUDA torch/torchvision. pip has no reason to replace them.
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import torch
    from huggingface_hub import hf_hub_download
    from pvseg.data import discover_root, read_manifest
    from pvseg.runtime import Precision, environment, memory, seed_everything, write_json
    from pvseg.sspnet import SSPNet, checkpoint
    from pvseg.sspnet.data import Episodes, ResizedPV, make_split
    from pvseg.sspnet.encoders import (FrozenSAT, PSPNet, SAT_SHA,
        cache_georsclip_text, load_text_cache, sha256)
    from pvseg.sspnet.losses import objective
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the real encoder check')
    torch.set_num_threads(2)
    seed_everything(42)
    write_json(output/'environment.json', environment())
    with (output/'tests.txt').open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    print((output/'tests.txt').read_text(), flush=True)
    result.check_returncode()
    assets = Path('/kaggle/working/sspnet_assets')
    assets.mkdir(exist_ok=True)
    weights = Path(args.weights)
    if not weights.is_file():
        # Kaggle may nest notebook sources under a different mount name.
        matches = [p for p in Path('/kaggle/input').rglob('model.safetensors') if p.parent.name == 'verified_sat']
        if len(matches) > 1:
            raise ValueError(f'Ambiguous mounted SAT checkpoints: {matches}')
        if matches:
            weights = matches[0]
        else:
            # Some Kaggle API source attachments resolve to the latest saved
            # output (the inference archive omits SAT), even with /4 supplied.
            # Recover the identical pinned bytes INSIDE Kaggle, never locally.
            from pvseg.model import resolve_weights
            source, _, origin = resolve_weights(download=True, cache_dir=str(assets/'hf_sat'))
            stable = assets/'verified_sat'
            stable.mkdir(exist_ok=True)
            shutil.copy2(source, stable/'model.safetensors')
            shutil.copy2(origin['config_path'], stable/'config.json')
            weights = stable/'model.safetensors'
            write_json(output/'sat_recovery.json', origin)
    if sha256(weights) != SAT_SHA:
        raise ValueError('Mounted SAT hash mismatch')
    geo = hf_hub_download('Zilun/GeoRSCLIP', 'ckpt/RS5M_ViT-B-32.pt',
                         revision=GEO_REVISION, cache_dir=str(assets/'hf_cache'))
    text_cache = assets/'georsclip_pv_text.pt'
    _, text_meta = cache_georsclip_text(geo, text_cache, device='cpu')
    text_meta['revision'] = GEO_REVISION
    # Persist actual bytes so future Kaggle runs need no GeoRSCLIP re-download.
    shutil.copy2(geo, assets/'RS5M_ViT-B-32.pt')
    write_json(output/'asset_manifest.json', {'sat_path': str(weights), 'sat_sha256': SAT_SHA,
        'georsclip': text_meta, 'text_cache': str(text_cache),
        'normalization_policy': 'raw RGB normalized independently inside each frozen encoder'})
    torch.cuda.empty_cache()
    common = ['--data-root', '/kaggle/input', '--pilot', '--epochs', '1', '--size', '512',
              '--batch-size', '1', '--accumulation', '1', '--max-steps', '2', '--seed', '42']
    # Separate processes release stage-1 gradients and optimizer memory before SAT loading.
    subprocess.run([sys.executable, str(REPO/'scripts/sspnet.py'), 'base',
        '--output', str(output/'base'), *common], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(REPO/'scripts/sspnet.py'), 'episodic',
        '--output', str(output/'episodic'), '--weights', str(weights), '--text-cache', str(text_cache),
        '--base-checkpoint', str(output/'base/last.pt'), '--allow-untrained-base', *common],
        cwd=REPO, check=True)
    # Inference on one real query per category, each with five distinct labeled supports.
    # Reuse learned 1-shot modules; the separate 5-shot reweight MLP remains untrained.
    root = discover_root('/kaggle/input')
    train_rows, val_rows, split = make_split(read_manifest(root), pilot=True)
    selected = [next(r for r in val_rows if r['category'] == category) for category in sorted({r['category'] for r in val_rows})]
    dataset = Episodes(ResizedPV(root, selected, 512), ResizedPV(root, train_rows, 512),
                       split['spatial_groups'], shots=5)
    cnn = PSPNet(False)
    checkpoint.load(output/'base/last.pt', cnn)
    cnn.provenance = dict(cnn.provenance, checkpoint_sha256=sha256(output/'base/last.pt'))
    text, meta = load_text_cache(text_cache)
    model = SSPNet(cnn, FrozenSAT(weights, 512), text, meta, shots=5).cuda().eval()
    learned = torch.load(output/'episodic/last.pt', map_location='cpu', weights_only=True)
    # Exact matching for shared modules; 5-shot MLP has no 1-shot counterpart.
    own = model.state_dict()
    shared = {k: v for k, v in learned['state_dict'].items() if k in own and v.shape == own[k].shape}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    if unexpected or any(not (k.startswith('dino.') or k.startswith('shot_weighting.reweight.')) for k in missing):
        raise ValueError(f'Unexpected 1-to-5-shot transfer mismatch: {missing}, {unexpected}')
    precision, five_shot = Precision('cuda'), []
    for index in range(len(dataset)):
        item = dataset[index]
        with torch.no_grad(), precision.context():
            result = model(item['query'][None].cuda(), item['supports'][None].cuda(), item['support_masks'][None].cuda())
            loss, terms = objective(result, item['target'][None].cuda())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite 5-shot loss')
        five_shot.append({'query': item['query_id'], 'category': item['category'],
            'supports': item['support_ids'].split('|'), 'shape': list(result['logits'].shape),
            'loss': float(loss), 'weights': result['shot_weights'].float().cpu().tolist(),
            'gamma': float(result['gamma']), 'memory': memory()})
        print(json.dumps(five_shot[-1]), flush=True)
    write_json(output/'five_shot.json', five_shot)
    completion = {'status': 'passed', 'engineering_only': True,
        'sat_sha256': SAT_SHA, 'georsclip': text_meta,
        'base': json.loads((output/'base/completion.json').read_text()),
        'episodic': json.loads((output/'episodic/completion.json').read_text()),
        'five_shot': five_shot,
        'limitations': ['two updates per stage are NOT pretrained accuracy',
            '5-shot reweight MLP is untrained; separate formal 5-shot training required',
            '512 input verified; 896 not yet measured', 'development split only; supervised known class']}
    write_json(output/'completion.json', completion)
    print('SSPNET REAL-WEIGHT SMOKE PASSED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = Path('/kaggle/working/sspnet_smoke')
        failure.mkdir(exist_ok=True)
        (failure/'failure.txt').write_text(traceback.format_exc())
        raise
