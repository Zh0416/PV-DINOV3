"""Frozen native-896 A0 GeoRSCLIP crops and A1 DINOv3 foreground prototypes.

Only calibration labels set thresholds. Named query labels are used after response
construction for auditing/evaluation. Assets and original data stay on Kaggle.
"""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def choose_nonoverlap(rows, count, seed):
    from pvseg.spatial import bounds, overlap
    pool, selected = sorted(rows, key=lambda r: r['sample_id']), []
    random.Random(seed).shuffle(pool)
    for row in pool:
        if any(overlap(bounds(row), bounds(other)) or row['image_sha256'] == other['image_sha256'] for other in selected):
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError('Not enough distinct nonoverlapping samples')


def plan_samples(rows, config):
    from pvseg.data import CATEGORIES
    from pvseg.spatial import scene_groups
    groups, edges = scene_groups(rows)
    group_for = {s: key for key, scenes in groups.items() for s in scenes}
    support_key, calibration_key = config['support_group'], config['calibration_group']
    if support_key == calibration_key or any(k not in groups for k in (support_key, calibration_key)):
        raise ValueError('Invalid role groups')
    by_id = {r['sample_id']: r for r in rows}
    named = [by_id[x['sample_id']] for x in config['records']]
    if len(named) != len({r['sample_id'] for r in named}):
        raise ValueError('Duplicate requested IDs')
    query = [r for r in named if group_for[r['source_scene']] not in (support_key, calibration_key)]
    preview = [r for r in named if r not in query]
    named_ids = {r['sample_id'] for r in named}
    support_pool = [r for r in rows if group_for[r['source_scene']] == support_key
                    and CATEGORIES[r['category']] == 'positive' and r['sample_id'] not in named_ids]
    supports = []
    from pvseg.spatial import bounds, overlap
    for seed in config['support_seeds']:
        pool = [r for r in support_pool if not any(overlap(bounds(r), bounds(s)) or
                r['image_sha256'] == s['image_sha256'] for s in supports)]
        supports.extend(choose_nonoverlap(pool, 1, seed))
    calibration = []
    for index, category in enumerate(CATEGORIES):
        pool = [r for r in rows if group_for[r['source_scene']] == calibration_key
                and r['category'] == category and r['sample_id'] not in named_ids]
        # Also exclude overlaps with calibration records already selected.
        from pvseg.spatial import bounds, overlap
        pool = [r for r in pool if not any(overlap(bounds(r), bounds(c)) for c in calibration)]
        calibration.extend(choose_nonoverlap(pool, config['calibration_per_category'], 100 + index))
    roles = {'support': supports, 'calibration': calibration, 'query': query}
    for first, second in [('support', 'calibration'), ('support', 'query'), ('calibration', 'query')]:
        if {group_for[r['source_scene']] for r in roles[first]} & {group_for[r['source_scene']] for r in roles[second]}:
            raise AssertionError('Spatial group leakage')
        if {r['image_sha256'] for r in roles[first]} & {r['image_sha256'] for r in roles[second]}:
            raise AssertionError('Hash leakage')
    audit = {'groups': groups, 'cross_scene_overlap_edges': edges,
             'roles': {key: [{'sample_id': r['sample_id'], 'source_scene': r['source_scene'],
                             'group': group_for[r['source_scene']], 'category': CATEGORIES[r['category']],
                             'image_sha256': r['image_sha256'], 'label_sha256': r['label_sha256'],
                             'source_label_uncertain_filename': r.get('source_label_uncertain_filename')}
                            for r in records] for key, records in roles.items()},
             'preview_only_ids': [r['sample_id'] for r in preview],
             'support_runs': [{'seed_label': seed, 'support_id': r['sample_id']}
                              for seed, r in zip(config['support_seeds'], supports)],
             'policy': 'Separate complete spatial groups for support, threshold calibration and named queries. Preview-only same-calibration-scene samples excluded from all aggregate metrics.',
             'scope': 'Exploratory user-curated development set; not a new independent benchmark.'}
    return supports, calibration, query, preview, audit


def load_pil(root, row):
    from PIL import Image
    with Image.open(root / row['image']) as image:
        return image.convert('RGB')


def make_panel(path, rgb, target, maps, thresholds, ranges, footer):
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    width, height = 224, 224
    labels = ['RGB', 'Ground truth', 'A0 response', 'A0 mask', 'A1 response (run 42)', 'A1 mask (run 42)']
    mask_image = lambda a: Image.fromarray((np.asarray(a) * 255).astype('uint8')).convert('RGB')
    pictures = [rgb, mask_image(target)]
    for name in ('A0', 'A1_42'):
        lo, hi = ranges[name]
        color = colormaps['turbo'](np.clip((maps[name] - lo) / max(hi - lo, 1e-8), 0, 1))[..., :3]
        pictures.extend([Image.fromarray((color * 255).astype('uint8')),
                         mask_image(maps[name] >= thresholds[name]['threshold'])])
    canvas = Image.new('RGB', (6 * width, height + 72), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (picture, label) in enumerate(zip(pictures, labels)):
        canvas.paste(picture.resize((width, height), Image.Resampling.BILINEAR if i not in (1, 3, 5) else Image.Resampling.NEAREST), (i * width, 26))
        draw.text((i * width + 8, 7), label, fill='black')
    draw.text((8, height + 36), footer, fill='black')
    draw.text((8, height + 52), 'Heat scales fixed from calibration 1%-99% cosine; BLUE=low, RED=high. Masks use calibrated global thresholds.', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(REPO / 'configs/a0a1_selected_20261029.json'))
    p.add_argument('--output', default='/kaggle/working/a0a1_diagnostic')
    args = p.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run on Kaggle; original data and weights must stay there')
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Preserve existing run output; start a fresh run')
    output.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pvseg.data import CATEGORIES, PV896Dataset, discover_root, read_manifest
    from pvseg.frozen_baselines import (calibrate_threshold, confusion, foreground_prototype,
        metrics, overlap_response, prototype_response, ranking_metrics, score_histogram)
    from pvseg.sspnet.encoders import FrozenSAT, GEO_SHA, PV_PROMPT, SAT_SHA, sha256
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    write_json(output / 'protocol.json', dict(config, a0='GeoRSCLIP native image/text cosine, 224px windows, stride112, mean overlap reconstruction',
        a1='Frozen DINOv3 last-block native896 feature cosine to one labeled-support area-weighted foreground prototype',
        thresholds='4096-bin maximum foreground micro-IoU on separate 12-image calibration group; highest threshold breaks ties',
        training='No gradient updates; 3 separate 1-shot support draws, not 3-shot or an ensemble',
        normalization='No per-query min-max; cosine responses kept in native units',
        supervision='Threshold calibration uses 12 masks plus 1 support mask per A1 run; not a total-one-label experiment'))
    with (output / 'tests.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_frozen_baselines.py', '-v'],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    root = discover_root('/kaggle/input')
    rows = read_manifest(root)
    supports, calibration, queries, previews, audit = plan_samples(rows, config)
    write_json(output / 'selection_audit.json', audit)
    write_json(output / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'precision': 'FP32, no AMP',
        'packages': {k: importlib.metadata.version(k) for k in ['timm', 'open_clip_torch', 'numpy']},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
    active = supports + calibration + queries + previews
    dataset = PV896Dataset(root, active, [0, 0, 0], [1, 1, 1])
    items = {}
    supplied = {r['sample_id']: r for r in config['records']}
    observed = []
    for index, row in enumerate(active):
        item = dataset[index]
        sid = row['sample_id']
        # Verify exact mounted bytes against manifest, not only matching filenames.
        if sha256(root / row['image']) != row['image_sha256'] or sha256(root / row['label']) != row['label_sha256']:
            raise ValueError('Image/label hash mismatch: ' + sid)
        if sid in supplied and bool(item['mask'].any()) != supplied[sid]['pv_present']:
            raise ValueError('Workbook presence disagrees with label: ' + sid)
        items[sid] = item
        observed.append({'sample_id': sid, 'foreground_pixels': int(item['mask'].sum()),
                         'foreground_fraction': float(item['mask'].mean()), 'shape': list(item['image'].shape)})
    write_json(output / 'data_observed.json', observed)
    print(json.dumps({'phase': 'audit', 'queries': len(queries), 'calibration': len(calibration),
                      'supports': [r['sample_id'] for r in supports], 'previews': len(previews)}), flush=True)
    # Reuse verified mounted assets; missing or ambiguous mounts fail explicitly.
    sat_matches = [x for x in Path('/kaggle/input').rglob('model.safetensors') if x.parent.name == 'verified_sat']
    geo_matches = list(Path('/kaggle/input').rglob('RS5M_ViT-B-32.pt'))
    if len(sat_matches) != 1 or len(geo_matches) != 1:
        raise ValueError(f'Expected one mounted SAT and GeoRSCLIP asset, got {len(sat_matches)}/{len(geo_matches)}')
    sat_path, geo_path = sat_matches[0], geo_matches[0]
    if sha256(sat_path) != SAT_SHA or sha256(geo_path) != GEO_SHA:
        raise ValueError('Pretrained weight hash mismatch')
    all_rows = calibration + queries + previews
    responses = {r['sample_id']: {} for r in all_rows}
    # A0 uses both original CLIP branches; no DINO/CLIP embedding-space mixing.
    import open_clip
    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, force_quick_gelu=True)
    state = torch.load(geo_path, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    clip.load_state_dict(state, strict=True)
    del state
    clip.requires_grad_(False).cuda().eval()
    with torch.no_grad():
        text = F.normalize(clip.encode_text(open_clip.get_tokenizer('ViT-B-32')([PV_PROMPT]).cuda()).float(), dim=-1)
        for index, row in enumerate(all_rows):
            image = load_pil(root, row)
            boxes = [(x, y, x + 224, y + 224) for y in range(0, 673, 112) for x in range(0, 673, 112)]
            crops = torch.stack([preprocess(image.crop(box)) for box in boxes])
            scores = []
            for batch in crops.split(32):
                visual = F.normalize(clip.encode_image(batch.cuda()).float(), dim=-1)
                scores.extend((visual @ text.T).squeeze(1).cpu().tolist())
            responses[row['sample_id']]['A0'] = overlap_response(scores, boxes, (896, 896))
            print(json.dumps({'phase': 'A0', 'done': index + 1, 'total': len(all_rows), 'id': row['sample_id']}), flush=True)
    write_json(output / 'encoders.json', {'a0_checkpoint_sha256': GEO_SHA, 'a0_prompt': PV_PROMPT,
        'a0_preprocess': str(preprocess), 'a0_window': 224, 'a0_stride': 112, 'a0_crops_per_image': 49,
        'a1_checkpoint_sha256': SAT_SHA, 'a1_block_zero_based': 23, 'a1_patch_size': 16,
        'a1_feature_shape': [1, 1024, 56, 56], 'a1_mask_downsampling': 'area',
        'a0_limitation': 'Window-level semantic response, not a native dense segmentation head; response resolution and boundary quality are limited.'})
    del clip, text, crops, batch, visual
    gc.collect()
    torch.cuda.empty_cache()
    sat = FrozenSAT(sat_path, img_size=896).cuda().eval()
    prototypes = []
    with torch.no_grad():
        for row in supports:
            item = items[row['sample_id']]
            features = sat(item['image'][None].cuda())
            prototypes.append(foreground_prototype(features, item['mask'][None].cuda()))
        for index, row in enumerate(all_rows):
            features = sat(items[row['sample_id']]['image'][None].cuda())
            for seed, prototype in zip(config['support_seeds'], prototypes):
                responses[row['sample_id']]['A1_' + str(seed)] = prototype_response(features, prototype, (896, 896))[0, 0].cpu().numpy()
            print(json.dumps({'phase': 'A1', 'done': index + 1, 'total': len(all_rows), 'id': row['sample_id']}), flush=True)
    names = list(responses[all_rows[0]['sample_id']])
    thresholds, display_ranges = {}, {}
    for name in names:
        hist = sum(score_histogram(responses[r['sample_id']][name], items[r['sample_id']]['mask'][0].numpy()) for r in calibration)
        thresholds[name] = calibrate_threshold(hist)
        cumulative = hist.sum(0).cumsum()
        indices = np.searchsorted(cumulative, [.01 * cumulative[-1], .99 * cumulative[-1]])
        display_ranges[name] = [-1 + 2 * int(i) / hist.shape[1] for i in indices]
    write_json(output / 'thresholds.json', {'methods': thresholds, 'display_cosine_ranges': display_ranges,
        'calibration_ids': [r['sample_id'] for r in calibration], 'named_query_labels_used': False})
    totals = {name: dict(tp=0, fp=0, fn=0, tn=0) for name in names}
    histograms = {name: np.zeros((2, 4096), np.int64) for name in names}
    per_image = []
    for row in queries + previews:
        sid, maps = row['sample_id'], responses[row['sample_id']]
        target = items[sid]['mask'][0].numpy().astype(bool)
        is_query = row in queries
        record = {'sample_id': sid, 'included_in_summary': is_query, 'category': CATEGORIES[row['category']],
                  'source_scene': row['source_scene'], 'annotation': supplied[sid], 'foreground_pixels': int(target.sum()), 'methods': {}}
        for name in names:
            count = confusion(maps[name], target, thresholds[name]['threshold'])
            hist = score_histogram(maps[name], target)
            record['methods'][name] = dict(metrics(count), **ranking_metrics(hist),
                score_min=float(maps[name].min()), score_max=float(maps[name].max()),
                score_mean_foreground=float(maps[name][target].mean()) if target.any() else None,
                score_mean_background=float(maps[name][~target].mean()) if (~target).any() else None)
            if is_query:
                for key in count:
                    totals[name][key] += count[key]
                histograms[name] += hist
        per_image.append(record)
        fmt = lambda v: f'{v:.3f}' if v is not None else 'n/a'
        footer = f"{sid} | {'QUERY' if is_query else 'PREVIEW ONLY: calibration scene'} | GT fraction {target.mean():.4f} | A0 IoU {fmt(record['methods']['A0']['iou'])} | A1 IoU {fmt(record['methods']['A1_42']['iou'])}"
        make_panel(output / 'predictions' / (sid + '.png'), load_pil(root, row), target, maps, thresholds, display_ranges, footer)
    write_json(output / 'per_image.json', per_image)
    summary = {}
    for name in names:
        entries = [r for r in per_image if r['included_in_summary']]
        negatives = [r['methods'][name] for r in entries if r['foreground_pixels'] == 0]
        positive = [r['methods'][name] for r in entries if r['foreground_pixels'] > 0]
        summary[name] = dict(metrics(totals[name]), **ranking_metrics(histograms[name]),
            positive_image_mean_iou=float(np.mean([r['iou'] for r in positive])),
            positive_images_with_tp=sum(r['tp'] > 0 for r in positive),
            negative_pixel_false_positive_rate=sum(r['fp'] for r in negatives) / sum(r['fp'] + r['tn'] for r in negatives) if negatives else None,
            negative_images_with_any_fp=sum(r['fp'] > 0 for r in negatives),
            threshold=thresholds[name]['threshold'])
    a1_iou = [summary['A1_' + str(seed)]['iou'] for seed in config['support_seeds']]
    a1_ap = [summary['A1_' + str(seed)]['pixel_ap_histogram'] for seed in config['support_seeds']]
    result = {'query_count': len(queries), 'positive_queries': sum(r['foreground_pixels'] > 0 for r in per_image if r['included_in_summary']),
        'negative_queries': sum(r['foreground_pixels'] == 0 for r in per_image if r['included_in_summary']),
        'calibration_count': len(calibration), 'methods': summary,
        'A1_three_distinct_supports': {'micro_iou_mean': float(np.mean(a1_iou)), 'micro_iou_population_std': float(np.std(a1_iou)),
                                      'pixel_ap_mean': float(np.mean(a1_ap)), 'support_ids': [r['sample_id'] for r in supports]},
        'scope': 'Frozen exploratory localization/segmentation on user-curated development queries; threshold tuning uses separate labels; no model training and no new independent test.'}
    write_json(output / 'summary.json', result)
    # Keep only small diagnostic reports and derived thumbnails; no feature maps or checkpoints.
    import csv
    with (output / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'included_in_summary', 'method', 'foreground_pixels', 'iou', 'f1', 'precision', 'recall', 'false_positive_rate', 'pixel_ap_histogram', 'tp', 'fp', 'fn', 'tn'])
        writer.writeheader()
        for record in per_image:
            for name, m in record['methods'].items():
                writer.writerow({key: (record.get(key) if key in record else name if key == 'method' else m.get(key)) for key in writer.fieldnames})
    from PIL import Image, ImageDraw
    support_canvas = Image.new('RGB', (3 * 448, 258), 'white')
    draw = ImageDraw.Draw(support_canvas)
    for i, (row, seed) in enumerate(zip(supports, config['support_seeds'])):
        support_canvas.paste(load_pil(root, row).resize((224, 224)), (i * 448, 30))
        mask = items[row['sample_id']]['mask'][0].numpy()
        support_canvas.paste(Image.fromarray((mask * 255).astype('uint8')).convert('RGB').resize((224, 224), Image.Resampling.NEAREST), (i * 448 + 224, 30))
        draw.text((i * 448 + 8, 8), f"Run {seed}: {row['sample_id']} | RGB + support label", fill='black')
    support_canvas.save(output / 'supports.png')
    write_json(output / 'completion.json', {'status': 'complete', 'query_count': len(queries),
        'named_count': len(config['records']), 'preview_only_ids': [r['sample_id'] for r in previews],
        'all_encoders_frozen': True, 'gradient_updates': 0, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('A0 A1 COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        output = Path('/kaggle/working/a0a1_diagnostic')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
