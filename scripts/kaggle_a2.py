"""Paired frozen A1/A2 diagnostic; exact roles and A1 counts must reproduce v2."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kaggle_a0a1 import load_pil, plan_samples, write_json


def make_panel(path, rgb, target, maps, thresholds, ranges, footer):
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    labels = ['RGB', 'Ground truth', 'A1 response (run42)', 'A1 mask (run42)',
              'A2 response (run42)', 'A2 mask (run42)']
    mask_image = lambda a: Image.fromarray((np.asarray(a) * 255).astype('uint8')).convert('RGB')
    pictures = [rgb, mask_image(target)]
    for name in ('A1_42', 'A2_42'):
        lo, hi = ranges[name]
        heat = colormaps['turbo'](np.clip((maps[name] - lo) / max(hi - lo, 1e-8), 0, 1))[..., :3]
        pictures.extend([Image.fromarray((heat * 255).astype('uint8')),
                         mask_image(maps[name] >= thresholds[name]['threshold'])])
    canvas = Image.new('RGB', (1344, 296), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (picture, label) in enumerate(zip(pictures, labels)):
        mode = Image.Resampling.NEAREST if i in (1, 3, 5) else Image.Resampling.BILINEAR
        canvas.paste(picture.resize((224, 224), mode), (i * 224, 26))
        draw.text((i * 224 + 8, 7), label, fill='black')
    draw.text((8, 260), footer, fill='black')
    draw.text((8, 276), 'Separate fixed calibration 1%-99% scales; colors are not probabilities. Global calibrated thresholds.', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main(output, config_path):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run on Kaggle; original data and weights must stay there')
    if output.exists():
        raise RuntimeError('Preserve existing run output; start a fresh session')
    output.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    import torch
    from pvseg.data import CATEGORIES, PV896Dataset, discover_root, read_manifest
    from pvseg.frozen_baselines import (calibrate_threshold, confusion, foreground_background_prototypes,
        foreground_background_response, metrics, prototype_response, ranking_metrics, score_histogram)
    from pvseg.sspnet.encoders import FrozenSAT, SAT_SHA, sha256
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    with (output / 'tests.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                        '-p', 'test_frozen_baselines.py', '-v'], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    reference_paths = [p.parent for p in Path('/kaggle/input').rglob('summary.json')
                       if p.parent.name == 'a0a1_diagnostic']
    if len(reference_paths) != 1:
        raise RuntimeError('Mount exactly one completed A0/A1 v2 output')
    reference = reference_paths[0]
    read = lambda name: json.loads((reference / name).read_text(encoding='utf-8'))
    if read('completion.json')['status'] != 'complete' or read('environment.json')['source_commit'] != 'b2e7eb2cdaf302fbaa7948da63c491b10c55dc3e':
        raise RuntimeError('Reference run is not the expected completed A0/A1 v2')
    reference_records = {r['sample_id']: r for r in read('per_image.json')}
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = discover_root('/kaggle/input')
    supports, calibration, queries, previews, audit = plan_samples(read_manifest(root), config)
    if audit != read('selection_audit.json'):
        raise RuntimeError('Role/spatial audit differs from A0/A1 reference')
    write_json(output / 'selection_audit.json', audit)
    write_json(output / 'protocol.json', dict(config,
        a1='Unchanged foreground-only cosine, recomputed as paired control',
        a2='(cos(query,foreground)-cos(query,background))/2, then bilinear upsample; both prototypes from same support',
        background='Area-downsample complement of support foreground mask, one normalized weighted mean',
        thresholds='4096 bins [-1,1], maximum micro-IoU on unchanged 12-image calibration set; highest breaks ties',
        supervision='1 support mask plus 12 calibration masks per independent run',
        training='No gradient updates, no trainable module, 3 separate 1-shot experiments, no ensemble',
        normalization='Native signed cosine margin, no query min-max or learned temperature',
        reference_kernel='zh0416/pv-a0-a1-selected-check/2'))
    active = supports + calibration + queries + previews
    dataset = PV896Dataset(root, active, [0, 0, 0], [1, 1, 1])
    items, observed = {}, []
    supplied = {r['sample_id']: r for r in config['records']}
    conflict_ids = {r['sample_id'] for r in audit['annotation_conflicts']}
    for index, row in enumerate(active):
        sid = row['sample_id']
        item = dataset[index]
        if sha256(root / row['image']) != row['image_sha256'] or sha256(root / row['label']) != row['label_sha256']:
            raise ValueError('Image/label hash mismatch: ' + sid)
        disagrees = sid in supplied and bool(item['mask'].any()) != supplied[sid]['pv_present']
        if disagrees != (sid in conflict_ids):
            raise ValueError('Actual label disagrees with conflict audit: ' + sid)
        items[sid] = item
        observed.append({'sample_id': sid, 'foreground_pixels': int(item['mask'].sum()),
                         'foreground_fraction': float(item['mask'].mean()), 'shape': list(item['image'].shape),
                         'workbook_presence_disagrees_with_label': disagrees})
    if observed != read('data_observed.json'):
        raise RuntimeError('Observed data differs from A0/A1 reference')
    write_json(output / 'data_observed.json', observed)
    print(json.dumps({'phase': 'audit', 'queries': len(queries), 'calibration': len(calibration),
                      'supports': [r['sample_id'] for r in supports], 'previews': len(previews)}), flush=True)
    sat_matches = [p for p in Path('/kaggle/input').rglob('model.safetensors') if p.parent.name == 'verified_sat']
    if len(sat_matches) != 1 or sha256(sat_matches[0]) != SAT_SHA:
        raise ValueError('Expected exactly one verified mounted SAT asset')
    sat = FrozenSAT(sat_matches[0], img_size=896).cuda().eval()
    if any(p.requires_grad for p in sat.parameters()):
        raise RuntimeError('Encoder unexpectedly trainable')
    write_json(output / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'precision': 'FP32, no AMP',
        'packages': {k: importlib.metadata.version(k) for k in ('timm', 'numpy')},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'reference_source_commit': read('environment.json')['source_commit'],
        'encoder': sat.metadata, 'all_encoder_parameters_frozen': True})
    prototypes, prototype_info = [], []
    with torch.no_grad():
        for row, seed in zip(supports, config['support_seeds']):
            item = items[row['sample_id']]
            features = sat(item['image'][None].cuda())
            foreground, background = foreground_background_prototypes(features, item['mask'][None].cuda())
            prototypes.append((foreground, background))
            prototype_info.append({'seed_label': seed, 'support_id': row['sample_id'],
                'foreground_fraction': float(item['mask'].mean()),
                'foreground_background_cosine': float((foreground * background).sum()),
                'foreground_norm': float(foreground.norm()), 'background_norm': float(background.norm())})
        write_json(output / 'prototypes.json', prototype_info)
        all_rows = calibration + queries + previews
        responses = {}
        for index, row in enumerate(all_rows):
            features = sat(items[row['sample_id']]['image'][None].cuda())
            maps = {}
            for seed, (foreground, background) in zip(config['support_seeds'], prototypes):
                maps['A1_' + str(seed)] = prototype_response(features, foreground, (896, 896))[0, 0].cpu().numpy()
                maps['A2_' + str(seed)] = foreground_background_response(features, foreground, background, (896, 896))[0, 0].cpu().numpy()
            responses[row['sample_id']] = maps
            print(json.dumps({'phase': 'A1_A2', 'done': index + 1, 'total': len(all_rows), 'id': row['sample_id']}), flush=True)
    names = list(responses[all_rows[0]['sample_id']])
    thresholds, ranges = {}, {}
    for name in names:
        hist = sum(score_histogram(responses[r['sample_id']][name], items[r['sample_id']]['mask'][0].numpy()) for r in calibration)
        thresholds[name] = calibrate_threshold(hist)
        cumulative = hist.sum(0).cumsum()
        indices = np.searchsorted(cumulative, [.01 * cumulative[-1], .99 * cumulative[-1]])
        ranges[name] = [-1 + 2 * int(i) / hist.shape[1] for i in indices]
        if name.startswith('A1_') and thresholds[name] != read('thresholds.json')['methods'][name]:
            raise RuntimeError('A1 calibration failed exact reproduction: ' + name)
    write_json(output / 'thresholds.json', {'methods': thresholds, 'display_response_ranges': ranges,
        'calibration_ids': [r['sample_id'] for r in calibration], 'named_query_labels_used': False})
    totals = {name: dict(tp=0, fp=0, fn=0, tn=0) for name in names}
    histograms = {name: np.zeros((2, 4096), np.int64) for name in names}
    per_image = []
    query_ids = {r['sample_id'] for r in queries}
    for row in queries + previews:
        sid = row['sample_id']
        maps = responses[sid]
        target = items[sid]['mask'][0].numpy().astype(bool)
        record = {'sample_id': sid, 'included_in_summary': sid in query_ids, 'category': CATEGORIES[row['category']],
            'preview_reason': audit['preview_reasons'].get(sid), 'source_scene': row['source_scene'],
            'annotation': supplied[sid], 'foreground_pixels': int(target.sum()), 'methods': {}}
        for name in names:
            count = confusion(maps[name], target, thresholds[name]['threshold'])
            hist = score_histogram(maps[name], target)
            record['methods'][name] = dict(metrics(count), **ranking_metrics(hist),
                score_min=float(maps[name].min()), score_max=float(maps[name].max()),
                score_mean_foreground=float(maps[name][target].mean()) if target.any() else None,
                score_mean_background=float(maps[name][~target].mean()) if (~target).any() else None)
            if name.startswith('A1_') and any(count[k] != reference_records[sid]['methods'][name][k] for k in count):
                raise RuntimeError('A1 confusion failed exact reproduction: ' + sid + '/' + name)
            if record['included_in_summary']:
                for key in count:
                    totals[name][key] += count[key]
                histograms[name] += hist
        per_image.append(record)
        fmt = lambda v: f'{v:.3f}' if v is not None else 'n/a'
        status = 'QUERY' if sid in query_ids else ('PREVIEW: label conflict' if sid in conflict_ids else 'PREVIEW: calibration group')
        footer = (f"{sid} | {status} | GT {target.mean():.4f} | "
                  f"A1 IoU {fmt(record['methods']['A1_42']['iou'])} | A2 IoU {fmt(record['methods']['A2_42']['iou'])}")
        make_panel(output / 'predictions' / (sid + '.png'), load_pil(root, row), target, maps, thresholds, ranges, footer)
    write_json(output / 'per_image.json', per_image)
    entries = [r for r in per_image if r['included_in_summary']]
    summary = {}
    for name in names:
        negatives = [r['methods'][name] for r in entries if not r['foreground_pixels']]
        positives = [r['methods'][name] for r in entries if r['foreground_pixels']]
        summary[name] = dict(metrics(totals[name]), **ranking_metrics(histograms[name]),
            positive_image_mean_iou=float(np.mean([r['iou'] for r in positives])),
            positive_images_with_tp=sum(r['tp'] > 0 for r in positives),
            negative_pixel_false_positive_rate=sum(r['fp'] for r in negatives) / sum(r['fp'] + r['tn'] for r in negatives),
            negative_images_with_any_fp=sum(r['fp'] > 0 for r in negatives), threshold=thresholds[name]['threshold'])
        if name.startswith('A1_') and summary[name] != read('summary.json')['methods'][name]:
            raise RuntimeError('A1 aggregate failed exact reproduction: ' + name)
    support_results = {}
    for method in ('A1', 'A2'):
        values = [summary[method + '_' + str(seed)] for seed in config['support_seeds']]
        support_results[method] = {'support_ids': [r['sample_id'] for r in supports],
            'means': {k: float(np.mean([v[k] for v in values])) for k in
                      ('iou', 'f1', 'precision', 'recall', 'pixel_ap_histogram', 'negative_pixel_false_positive_rate')},
            'micro_iou_population_std': float(np.std([v['iou'] for v in values]))}
    result = {'query_count': len(queries), 'positive_queries': sum(r['foreground_pixels'] > 0 for r in entries),
        'negative_queries': sum(r['foreground_pixels'] == 0 for r in entries), 'calibration_count': len(calibration),
        'methods': summary, 'three_distinct_supports': support_results,
        'paired_delta_A2_minus_A1': {str(seed): {k: summary['A2_' + str(seed)][k] - summary['A1_' + str(seed)][k]
            for k in ('iou', 'precision', 'recall', 'pixel_ap_histogram', 'negative_pixel_false_positive_rate')}
            for seed in config['support_seeds']},
        'scope': 'Exploratory curated development set, no training, no independent test, no full SSPNet claim'}
    write_json(output / 'summary.json', result)
    with (output / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'included_in_summary', 'method', 'foreground_pixels',
            'iou', 'f1', 'precision', 'recall', 'false_positive_rate', 'pixel_ap_histogram', 'tp', 'fp', 'fn', 'tn'])
        writer.writeheader()
        for record in per_image:
            for name, values in record['methods'].items():
                writer.writerow({k: record[k] if k in record else name if k == 'method' else values.get(k) for k in writer.fieldnames})
    shutil.copyfile(reference / 'supports.png', output / 'supports.png')
    write_json(output / 'reference_reproduction.json', {'status': 'exact', 'roles_identical': True,
        'input_hashes_verified': True, 'observed_data_identical': True, 'calibration_identical': True,
        'all_60_images_3_A1_confusions_identical': True, 'all_A1_aggregate_metrics_identical': True,
        'reference_kernel': 'zh0416/pv-a0-a1-selected-check/2'})
    write_json(output / 'completion.json', {'status': 'complete', 'query_count': len(queries),
        'named_count': len(config['records']), 'preview_only_ids': [r['sample_id'] for r in previews],
        'all_encoders_frozen': True, 'gradient_updates': 0, 'a1_exact_reproduction': True,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('A2 COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/a0a1_selected_20261029.json')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/a2_diagnostic'))
    args = parser.parse_args()
    try:
        main(args.output, args.config)
    except Exception:
        # Do not modify a preexisting completed run if invoked accidentally.
        if not (args.output / 'completion.json').exists():
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
