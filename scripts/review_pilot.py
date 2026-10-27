"""Read-only pilot review and Kaggle-side archive. No training or model downloads."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PILOT_COMMIT = '4cf2aef6fe490a04be36feca0c84efff8594d06f'
WEIGHT_SHA = '8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286'
ERROR_IDS = ['pv003283', 'pv003289', 'pv003520', 'pv003519', 'pv003313']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def info(path, hash_file=True):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size,
            'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            'sha256': digest(path) if hash_file else None}


def stats(records):
    return {'pairs': len(records), 'categories': dict(Counter(r['category'] for r in records)),
            'scenes': sorted({r['source_scene'] for r in records}),
            'scene_count': len({r['source_scene'] for r in records}),
            'uncertainty': dict(Counter(str(r['source_label_uncertain_filename']).lower() for r in records)),
            'by_category': {c: {'pairs': sum(r['category'] == c for r in records),
                'scenes': sorted({r['source_scene'] for r in records if r['category'] == c}),
                'uncertainty': dict(Counter(str(r['source_label_uncertain_filename']).lower()
                    for r in records if r['category'] == c))} for c in sorted({r['category'] for r in records})}}


def spatial_review(train, validation, all_rows, saved):
    from pvseg.spatial import bounds, overlap, scene_groups
    groups, edges = scene_groups(all_rows)

    def basename(row):
        return row['source_image'].replace('\\', '/').rsplit('/', 1)[-1].casefold()

    def pairs(first, second):
        footprints = [(r, bounds(r)) for r in second]
        overlaps, touches, nearby = [], [], []
        nearest = None
        for a in first:
            ab = bounds(a)
            for b, bb in footprints:
                ids = [a['sample_id'], b['sample_id']]
                if overlap(ab, bb):
                    overlaps.append(ids)
                dx = max(0., ab[0] - bb[2], bb[0] - ab[2])
                dy = max(0., ab[1] - bb[3], bb[1] - ab[3])
                if dx <= 1e-10 and dy <= 1e-10 and not overlap(ab, bb):
                    touches.append(ids)
                # Geographic bounding-box separation estimate, not a surveyed distance.
                lat = math.radians((ab[1] + ab[3] + bb[1] + bb[3]) / 4)
                distance = 6371008.8 * math.hypot(math.radians(dx) * math.cos(lat), math.radians(dy))
                record = {'ids': ids, 'bbox_gap_m_approx': distance}
                if nearest is None or distance < nearest['bbox_gap_m_approx']:
                    nearest = record
                pixel = max(ab[2]-ab[0], ab[3]-ab[1], bb[2]-bb[0], bb[3]-bb[1]) / 896
                if dx <= pixel and dy <= pixel:
                    nearby.append(ids)
        return {'intersections': overlaps, 'touching_or_corner_pairs': touches,
                'within_one_nominal_pixel_pairs': nearby, 'nearest': nearest}

    assigned = saved['scene_assignment_before_sampling']
    full_train = [r for r in all_rows if assigned[r['source_scene']] == 'train']
    full_val = [r for r in all_rows if assigned[r['source_scene']] == 'validation']
    result = {'status': '已验证', 'train': stats(train), 'validation': stats(validation),
        'same_source_scene': sorted({r['source_scene'] for r in train} & {r['source_scene'] for r in validation}),
        'same_source_image_basename': sorted({basename(r) for r in train} & {basename(r) for r in validation}),
        'same_manifest_image_sha256': sorted({r['image_sha256'] for r in train} & {r['image_sha256'] for r in validation}),
        'same_manifest_mask_sha256': sorted({r['label_sha256'] for r in train} & {r['label_sha256'] for r in validation}),
        'selected_pair_geometry': pairs(train, validation),
        'all_assigned_pair_geometry': pairs(full_train, full_val),
        'recomputed_groups_match': groups == saved['spatial_groups'],
        'recomputed_overlap_edges_match': edges == saved['cross_scene_overlap_edges'],
        'coordinate_fields': ['crs', 'transform', 'row_off', 'col_off', 'width', 'height', 'source_image'],
        'limitations': ['Uses supplied EPSG:4326 affine metadata; not independently surveyed.',
            'Approximate bounding-box distance is not an exact geodesic polygon distance.',
            'No additional geographic buffer or proof of statistical independence.',
            'Identical empty-mask hashes alone do not imply duplicate image leakage.',
            'Validation was used for checkpoint selection and cannot serve as an untouched test.']}
    return result


def make_panels(folder, sid, rgb, truth, prediction):
    import numpy as np
    from PIL import Image, ImageDraw
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(folder/'rgb.png')
    Image.fromarray(truth.astype('uint8')*255).save(folder/'ground_truth.png')
    Image.fromarray(prediction.astype('uint8')*255).save(folder/'prediction.png')
    overlay = rgb.copy()
    overlay[prediction] = (overlay[prediction]*.55 + np.array([255, 30, 30])*.45).astype('uint8')
    Image.fromarray(overlay).save(folder/'prediction_overlay.png')
    errors = rgb.copy()
    for selector, color in [(prediction & truth, [30, 220, 30]),
                             (prediction & ~truth, [255, 30, 30]), (~prediction & truth, [30, 120, 255])]:
        errors[selector] = (errors[selector]*.4 + np.array(color)*.6).astype('uint8')
    Image.fromarray(errors).save(folder/'error_overlay.png')
    canvas = Image.new('RGB', (5*384, 426), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), f'{sid} | fixed best checkpoint, threshold=0.5; TP green / FP red / FN blue', fill='black')
    for i, (array, title) in enumerate(zip([rgb, truth.astype('uint8')*255, prediction.astype('uint8')*255, overlay, errors],
            ['RGB', 'Ground truth', 'Prediction >=0.5', 'Prediction overlay', 'Error overlay'])):
        resampling = Image.Resampling.NEAREST if i in [1, 2] else Image.Resampling.LANCZOS
        canvas.paste(Image.fromarray(array).convert('RGB').resize((384, 384), resampling), (384*i, 42))
        draw.text((384*i+5, 24), title, fill='black')
    canvas.save(folder/'review_panel.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', type=Path)
    parser.add_argument('--output_dir', type=Path, default=Path('/kaggle/working/PV-DINOV3/outputs/archive/pilot_baseline'))
    args = parser.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Kaggle-side review only; no local model downloads')
    if args.output_dir.exists():
        raise RuntimeError('Archive already exists; refusing overwrite')
    args.output_dir.mkdir(parents=True)
    output = args.output_dir
    started = time.perf_counter()
    (output/'README.md').write_text('''# Pilot baseline archive: inference and review only

Original pilot commit: `4cf2aef6fe490a04be36feca0c84efff8594d06f`.
See `review_completion.json` for actual completion; no file name implies a passed check.
`source_file_inventory.json` records readable files, byte sizes, mounted mtimes and hashes.
Mounted mtimes may reflect platform export/staging rather than original training write time.

`best_decoder.pt` is the unchanged original mixed training checkpoint, including
decoder tensors, optimizer, scaler, RNG, config and provenance. It does not contain
backbone tensors. `checkpoint_metadata.json` lists its actual keys and tensor shapes.
`inference_decoder.pt` is an additional export containing identical decoder tensors
and matching metadata, without optimizer/scaler/RNG or backbone tensors.

Required backbone: timm/vit_large_patch16_dinov3.sat493m, revision
ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c, SHA256
8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286.
No backbone download is performed. Supply the exact already saved model.safetensors
and its adjacent config.json as a read-only Kaggle input; normalization is
mean=[0.430,0.411,0.296], std=[0.213,0.156,0.143]. Use timm 1.0.29 and the original
CUDA PyTorch environment recorded in checkpoint_metadata.json.

From the fixed review checkout, load either checkpoint for one original 896x896 RGB image:

```bash
python scripts/infer_pilot.py --image /ACTUAL/IMAGE.bmp \\
  --checkpoint /ACTUAL/ARCHIVE/inference_decoder.pt \\
  --weights /ACTUAL/MOUNT/verified_sat/model.safetensors \\
  --output /kaggle/working/single_prediction.png
```

Paths must be taken from actual mount inventory. There is no automatic resize,
threshold search, augmentation or training. Threshold is fixed at 0.5. RGB/GT,
binary prediction, prediction overlay and TP/FP/FN overlay are under errors/ID/.
Green=TP, red=FP, blue=FN. Detailed per-sample metrics are separate from historical
aggregate metrics; suspected causes require human annotation review. Masks are unchanged.

This working directory is temporary until a Save Version run completes. After
completion select that version in Kaggle's Version history, inspect Output and
confirm best_decoder.pt plus manifests/checksums. Mount that exact output version
in a later session to verify persistence. A permanent Dataset/Model is not implied.
Do not rerun pilot training, change labels or treat these validation results as
independent cross-scene generalization measurements.
''', encoding='utf-8')
    live = Path('/kaggle/working/PV-DINOV3/outputs/pilot')
    write(output/'initial_environment.json', {'live_pilot_exists': live.exists(),
        'live_checkpoint_exists': (live/'best_decoder.pt').is_file(),
        'working_entries': [str(p) for p in Path('/kaggle/working').iterdir()],
        'input_entries': [str(p) for p in Path('/kaggle/input').iterdir()],
        'saved_output_source': 'zh0416/notebook5f7aef8a4f/4',
        'downloads_permitted': False, 'training_permitted': False,
        'note': 'Existence describes this fresh session, not an inaccessible prior interactive session.'})
    candidates = [args.source_root] if args.source_root else ([live] if (live/'completion.json').is_file() else [])
    if not candidates:
        candidates = [p.parent for p in Path('/kaggle/input').rglob('completion.json')
            if read(p).get('software', {}).get('git_commit') == PILOT_COMMIT]
    if len(candidates) != 1:
        raise RuntimeError(f'Need unique saved pilot source, found {candidates}')
    source = candidates[0]
    print('VERIFIED actual saved pilot directory:', source, flush=True)
    from PIL import Image
    required = ['best_decoder.pt', 'config.json', 'best_checkpoint_metadata.json', 'history.json',
        'epoch_metrics.csv', 'train_manifest.csv', 'validation_manifest.csv', 'selected_sample_ids.json',
        'spatial_split.json', 'selected_pixel_audit.json', 'label_source_report.json', 'environment.json', 'completion.json']
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    inventory = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        record = info(path)
        if path.suffix == '.json':
            read(path)
        elif path.suffix == '.csv':
            rows(path)
        elif path.suffix == '.png':
            with Image.open(path) as image:
                image.verify()
        elif path.suffix == '.txt':
            path.read_text(encoding='utf-8')
        record['readability'] = '已验证' if path.suffix != '.pt' else 'pending torch.load'
        inventory.append(record)
    write(output/'source_file_inventory.json', {'files': inventory,
        'mtime_scope': 'Actual mounted file stat; Kaggle export/mount may replace original modification times.'})
    completion = read(source/'completion.json')
    if completion['software']['git_commit'] != PILOT_COMMIT or completion['epochs'] != 3:
        raise ValueError('Not the fixed pilot result')
    import torch
    state = torch.load(source/'best_decoder.pt', map_location='cpu', weights_only=True)
    if state['epoch'] != 2 or state['format'] != 'pvseg_decoder_v1':
        raise ValueError('Wrong checkpoint epoch or format')
    model_meta = state['model']
    if model_meta['weight_source']['sha256'] != WEIGHT_SHA or state['software']['git_commit'] != PILOT_COMMIT:
        raise ValueError('Checkpoint provenance mismatch')
    # torch preserves tuples in timm pretrained_cfg; JSON serializes them as arrays.
    normalized_model_meta = json.loads(json.dumps(model_meta, default=str))
    sidecar_model_meta = read(source/'best_checkpoint_metadata.json')['model']
    write(output/'metadata_comparison.json', {'json_normalized_equal': normalized_model_meta == sidecar_model_meta,
        'policy': 'Normalize JSON-compatible tuples to arrays before comparing saved sidecar to torch metadata; no model values changed.'})
    if normalized_model_meta != sidecar_model_meta:
        raise ValueError('Actual checkpoint metadata disagrees with sidecar')
    if json.loads(json.dumps(state['config'], default=str)) != read(source/'config.json') or state['validation'] != read(source/'best_checkpoint_metadata.json')['validation']:
        raise ValueError('Actual checkpoint configuration/metrics differ from sidecars')
    if not all(torch.isfinite(t).all().item() for t in state['decoder_state'].values()):
        raise ValueError('Nonfinite decoder tensors')
    for record in inventory:
        if record['path'].endswith('/best_decoder.pt'):
            record['readability'] = '已验证: torch.load(weights_only=True)'
    write(output/'source_file_inventory.json', {'files': inventory,
        'mtime_scope': 'Actual mounted file stat; Kaggle export/mount may replace original modification times.'})
    # Preserve the original mixed checkpoint and all existing reports/predictions.
    for path in sorted(source.rglob('*')):
        if path.is_file():
            target = output/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if digest(source/'best_decoder.pt') != digest(output/'best_decoder.pt'):
        raise AssertionError('Archive copy hash mismatch')
    tensor_meta = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'numel': v.numel()}
                   for k, v in state['decoder_state'].items()}
    checkpoint_meta = {'status': '已验证', 'source': info(source/'best_decoder.pt'),
        'archive': info(output/'best_decoder.pt'), 'keys': sorted(state), 'format': state['format'],
        'epoch': state['epoch'], 'contains_optimizer_state': 'optimizer_state' in state,
        'optimizer_state_entries': len(state.get('optimizer_state', {}).get('state', {})),
        'contains_scaler_state': 'scaler_state' in state, 'contains_rng_cpu': 'rng_cpu' in state,
        'contains_backbone_tensors': False, 'decoder_tensors': tensor_meta,
        'model': model_meta, 'original_software': state['software'],
        'description': 'Original mixed training checkpoint: decoder tensors, optimizer, scaler, RNG, config and metadata; backbone stored separately.'}
    write(output/'checkpoint_metadata.json', checkpoint_meta)
    inference = {'format': 'pvseg_inference_decoder_v1', 'decoder_state': state['decoder_state'],
        'model': model_meta, 'training_git_commit': PILOT_COMMIT, 'epoch': 2,
        'source_checkpoint_sha256': digest(source/'best_decoder.pt'), 'software': state['software']}
    torch.save(inference, output/'inference_decoder.pt')
    exported = torch.load(output/'inference_decoder.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, exported['decoder_state'][k]) for k, v in state['decoder_state'].items()):
        raise AssertionError('Inference export differs from original decoder')
    write(output/'inference_export.json', {'status': '已验证', **info(output/'inference_decoder.pt'),
        'keys': sorted(inference), 'optimizer_included': False, 'backbone_included': False,
        'requires_backbone_sha256': WEIGHT_SHA, 'tensor_equality_to_original': True})
    print('VERIFIED checkpoint load/copy/inference export; no training.', flush=True)
    from pvseg.data import discover_root, read_manifest, CATEGORIES
    root = discover_root('/kaggle/input')
    all_rows = read_manifest(root)
    indexed = {r['sample_id']: r for r in all_rows}
    train, validation = rows(source/'train_manifest.csv'), rows(source/'validation_manifest.csv')
    for row in train + validation:
        if indexed.get(row['sample_id']) != {k: v for k, v in row.items() if k != 'split'}:
            raise ValueError(f'Mounted dataset manifest changed: {row["sample_id"]}')
    spatial = spatial_review(train, validation, all_rows, read(source/'spatial_split.json'))
    write(output/'spatial_review.json', spatial)
    print('VERIFIED spatial review:', spatial['selected_pair_geometry'], flush=True)
    # No remote resolution: require already mounted bytes with the exact original SHA.
    weight_candidates = [p for p in Path('/kaggle/input').rglob('model.safetensors') if p.with_name('config.json').is_file()]
    weight = next((p for p in weight_candidates if digest(p) == WEIGHT_SHA), None)
    result = {'checkpoint': '已验证', 'archive_copy': '已验证', 'spatial_review': '已验证',
        'new_inference': '未验证', 'errors': [], 'threshold': .5, 'training_steps': 0,
        'source_pilot_commit': PILOT_COMMIT,
        'review_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}
    if weight is None:
        result['errors'].append('No mounted exact-hash SAT weights. No download or retraining attempted.')
        write(output/'review_completion.json', result)
        return
    from pvseg.model import FrozenSATSegDINO, resolve_weights
    from pvseg.runtime import load_checkpoint, seed_everything, Precision, environment
    from pvseg.data import PV896Dataset
    from pvseg.metrics import BinaryCounts
    from torch.utils.data import DataLoader
    if metadata.version('timm') != '1.0.29' or str(torch.__version__) != state['software']['torch']:
        raise RuntimeError('Software differs from verified API/precision environment')
    for path in ['pvseg/model.py', 'pvseg/data.py', 'pvseg/metrics.py', 'pvseg/runtime.py']:
        old = subprocess.check_output(['git', 'show', f'{PILOT_COMMIT}:{path}'], cwd=REPO)
        if hashlib.sha256(old).hexdigest() != digest(REPO/path):
            raise RuntimeError(f'Core implementation changed since pilot: {path}')
    seed_everything(42)
    weight_path, weight_config, weight_source = resolve_weights(weight, download=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FrozenSATSegDINO(weight_path, weight_config, weight_source, 896).to(device).eval()
    load_checkpoint(source/'best_decoder.pt', model)
    model.requires_grad_(False)
    before = {k: v._version for k, v in model.named_parameters()}
    precision = Precision(device)
    dataset = PV896Dataset(root, validation, **model_meta['normalization'], augment=False)
    aggregate, categories, samples = BinaryCounts(), {}, []
    original = {s['sample_id']: s for s in completion['best_validation']['samples']}
    from PIL import Image
    import numpy as np
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            x, target = batch['image'].to(device), batch['mask'].to(device)
            with precision.context():
                logits = model(x)
            if logits.shape != target.shape or not torch.isfinite(logits).all().item():
                raise AssertionError('Invalid inference output')
            sid, category = batch['sample_id'][0], batch['category'][0]
            counts = BinaryCounts()
            counts.update(logits, target)
            aggregate.update(logits, target)
            categories.setdefault(category, BinaryCounts()).update(logits, target)
            metric = counts.report()
            truth = target[0, 0].cpu().numpy() >= .5
            pred = logits[0, 0].float().sigmoid().cpu().numpy() >= .5
            old_pred = np.array(Image.open(source/'best_predictions'/f'{sid}_prediction.png')) == 255
            row = indexed[sid]
            image_hash, mask_hash = digest(root/row['image']), digest(root/row['label'])
            if image_hash != row['image_sha256'] or mask_hash != row['label_sha256']:
                raise ValueError(f'Actual image/mask bytes differ from original manifest: {sid}')
            record = {'sample_id': sid, 'category': category, 'source_scene': row['source_scene'],
                'source_label_uncertain_filename': row['source_label_uncertain_filename'],
                'metadata': row, 'image_path': str(root/row['image']), 'mask_path': str(root/row['label']),
                'image_sha256': image_hash, 'mask_sha256': mask_hash,
                **metric, 'prediction_diff_pixels_vs_original': int((pred != old_pred).sum()),
                'count_metrics_match_original': all(metric[k] == original[sid][k] for k in ['tp', 'fp', 'fn']),
                'status': '已验证', 'cause': '待人工确认'}
            samples.append(record)
            if sid in ERROR_IDS:
                with Image.open(root/row['image']) as image:
                    rgb = np.array(image)
                make_panels(output/'errors'/sid, sid, rgb, truth, pred)
                write(output/'errors'/sid/'sample.json', record)
    result.update(new_inference='已验证', overall=aggregate.report(),
        per_category={c: v.report() for c, v in categories.items()},
        matches_original_counts=all(s['count_metrics_match_original'] for s in samples),
        matches_original_predictions=all(s['prediction_diff_pixels_vs_original'] == 0 for s in samples),
        parameters_unchanged=all(v._version == before[k] for k, v in model.named_parameters()),
        weights={**info(weight), 'copied_to_archive': False, 'downloaded': False},
        software=environment(), precision=precision.describe(), seconds=time.perf_counter()-started,
        reviewed_error_ids=[s['sample_id'] for s in samples if s['sample_id'] in ERROR_IDS],
        missing_requested_error_ids=sorted(set(ERROR_IDS)-{s['sample_id'] for s in samples}),
        note='Per-sample review is separate from historical aggregate; validation is not an independent test.')
    write(output/'sample_metrics.json', samples)
    fields = ['sample_id', 'category', 'source_scene', 'source_label_uncertain_filename', 'tp', 'fp', 'fn',
              'iou', 'f1', 'precision', 'recall', 'prediction_diff_pixels_vs_original', 'status', 'cause']
    with (output/'sample_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    write(output/'review_completion.json', result)
    print('REVIEW COMPLETE', json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
