"""Frozen spatially aligned A2/GeoRSCLIP fusion; reproduce all prior controls."""

import argparse
import csv
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
from scripts.kaggle_a0a1 import load_pil, plan_samples, write_json


def make_panel(path, rgb, target, maps, thresholds, ranges, footer):
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    labels = ['RGB', 'Ground truth', 'A0 response', 'A1 mask (run42)',
              'A2 response (run42)', 'A2 mask (run42)', 'A3 response (run42)', 'A3 mask (run42)']
    mask_image = lambda a: Image.fromarray((np.asarray(a) * 255).astype('uint8')).convert('RGB')
    def heat(name):
        lo, hi = ranges[name]
        color = colormaps['turbo'](np.clip((maps[name] - lo) / max(hi - lo, 1e-8), 0, 1))[..., :3]
        return Image.fromarray((color * 255).astype('uint8'))
    pictures = [rgb, mask_image(target), heat('A0'),
                mask_image(maps['A1_42'] >= thresholds['A1_42']['threshold'])]
    for name in ('A2_42', 'A3_42'):
        pictures.extend([heat(name), mask_image(maps[name] >= thresholds[name]['threshold'])])
    canvas = Image.new('RGB', (1792, 296), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (picture, label) in enumerate(zip(pictures, labels)):
        mode = Image.Resampling.NEAREST if i in (1, 3, 5, 7) else Image.Resampling.BILINEAR
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
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pvseg.data import CATEGORIES, PV896Dataset, discover_root, read_manifest
    from pvseg.frozen_baselines import (aligned_crop_features, calibrate_fusion, calibrate_threshold,
        confusion, foreground_background_prototypes, foreground_background_response, fuse_responses,
        metrics, overlap_response, prototype_response, ranking_metrics, score_histogram)
    from pvseg.sspnet.encoders import FrozenSAT, GEO_SHA, PV_PROMPT, SAT_SHA, sha256
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    with (output / 'tests.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                        '-p', 'test_frozen_baselines.py', '-v'], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    references = {}
    for label, folder, commit in [('A01', 'a0a1_diagnostic', 'b2e7eb2cdaf302fbaa7948da63c491b10c55dc3e'),
                                  ('A2', 'a2_diagnostic', '59e7ed6d502444fbb7d38e97cbf75408080f7ae4')]:
        matches = [p.parent for p in Path('/kaggle/input').rglob('summary.json') if p.parent.name == folder]
        if len(matches) != 1:
            raise RuntimeError('Mount exactly one completed reference: ' + folder)
        directory = matches[0]
        read_file = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
        if read_file('completion.json')['status'] != 'complete' or read_file('environment.json')['source_commit'] != commit:
            raise RuntimeError('Unexpected reference source/status: ' + label)
        references[label] = {n: read_file(n + '.json') for n in
                            ('summary', 'thresholds', 'selection_audit', 'data_observed', 'per_image')}
        references[label]['path'] = directory
        references[label]['records'] = {r['sample_id']: r for r in references[label]['per_image']}
    baseline = references['A01']
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = discover_root('/kaggle/input')
    supports, calibration, queries, previews, audit = plan_samples(read_manifest(root), config)
    if any(audit != ref['selection_audit'] for ref in references.values()):
        raise RuntimeError('Role/spatial audit differs from reference')
    write_json(output / 'selection_audit.json', audit)
    write_json(output / 'protocol.json', dict(config,
        a3='(1-alpha)*(DINO foreground-background cosine)/2 + alpha*aligned GeoRSCLIP cosine',
        alignment='L2-normalized 512d Geo crop vectors averaged on corresponding 14x14 cells of 56x56 DINO grid; no renormalization after overlap',
        fusion='Equivalent to direct-sum feature readout; no learned shared embedding projection',
        alpha_candidates=[i / 10 for i in range(11)],
        selection='One shared alpha maximizes mean of three calibration IoUs; tie prefers smaller alpha; one threshold per run',
        thresholds='4096 bins [-1,1], max calibration micro-IoU; tie prefers higher threshold',
        supervision='1 support mask plus unchanged 12 calibration masks per independent run',
        training='No gradient updates; no ensemble; both encoders frozen',
        controls='Recompute exact A0/A1/A2; GeoAligned isolates spatial grid smoothing',
        query_labels_used_for_selection=False))
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
    if any(observed != ref['data_observed'] for ref in references.values()):
        raise RuntimeError('Observed data differs from reference')
    write_json(output / 'data_observed.json', observed)
    print(json.dumps({'phase': 'audit', 'queries': len(queries), 'calibration': len(calibration),
                      'supports': [r['sample_id'] for r in supports], 'previews': len(previews)}), flush=True)
    sat_matches = [p for p in Path('/kaggle/input').rglob('model.safetensors') if p.parent.name == 'verified_sat']
    geo_matches = list(Path('/kaggle/input').rglob('RS5M_ViT-B-32.pt'))
    if len(sat_matches) != 1 or len(geo_matches) != 1:
        raise ValueError('Expected exactly one mounted SAT and GeoRSCLIP asset')
    sat_path, geo_path = sat_matches[0], geo_matches[0]
    if sha256(sat_path) != SAT_SHA or sha256(geo_path) != GEO_SHA:
        raise ValueError('Pretrained weight hash mismatch')
    all_rows = calibration + queries + previews
    responses = {r['sample_id']: {} for r in all_rows}
    import open_clip
    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, force_quick_gelu=True)
    state = torch.load(geo_path, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    clip.load_state_dict(state, strict=True)
    del state
    clip.requires_grad_(False).cuda().eval()
    if any(p.requires_grad for p in clip.parameters()):
        raise RuntimeError('GeoRSCLIP unexpectedly trainable')
    alignment_max_error = 0.
    with torch.no_grad():
        text = F.normalize(clip.encode_text(open_clip.get_tokenizer('ViT-B-32')([PV_PROMPT]).cuda()).float(), dim=-1)
        for index, row in enumerate(all_rows):
            image = load_pil(root, row)
            boxes = [(x, y, x + 224, y + 224) for y in range(0, 673, 112) for x in range(0, 673, 112)]
            crops = torch.stack([preprocess(image.crop(box)) for box in boxes])
            scores, vectors = [], []
            for batch in crops.split(32):
                visual = F.normalize(clip.encode_image(batch.cuda()).float(), dim=-1)
                scores.extend((visual @ text.T).squeeze(1).cpu().tolist())
                vectors.append(visual)
            native = overlap_response(scores, boxes, (896, 896))
            grid = aligned_crop_features(torch.cat(vectors), boxes, (896, 896), (56, 56))
            semantic_grid = (grid * text[:, :, None, None]).sum(1, keepdim=True)
            # Before interpolation, exact coordinate correspondence to native A0.
            expected_grid = native.reshape(56, 16, 56, 16).mean((1, 3))
            error = float(np.abs(semantic_grid[0, 0].cpu().numpy() - expected_grid).max())
            if error > 2e-6:
                raise RuntimeError('Semantic feature alignment failed numeric equivalence')
            alignment_max_error = max(alignment_max_error, error)
            maps = responses[row['sample_id']]
            maps['A0'] = native
            maps['GeoAligned'] = F.interpolate(semantic_grid, size=(896, 896), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            print(json.dumps({'phase': 'Geo_align', 'done': index + 1, 'total': len(all_rows), 'id': row['sample_id']}), flush=True)
    write_json(output / 'encoders.json', {'geo_checkpoint_sha256': GEO_SHA, 'geo_prompt': PV_PROMPT,
        'geo_model': 'ViT-B-32, QuickGELU, native image and text branches', 'geo_preprocess': str(preprocess),
        'geo_window': 224, 'geo_stride': 112, 'geo_crops_per_image': 49,
        'geo_aligned_feature_shape': [1, 512, 56, 56], 'overlap_mean_renormalized': False,
        'alignment_max_abs_error_vs_native_grid': alignment_max_error,
        'sat_checkpoint_sha256': SAT_SHA, 'sat_block_zero_based': 23, 'sat_patch_size': 16,
        'sat_feature_shape': [1, 1024, 56, 56], 'mask_downsampling': 'area',
        'output_size': [896, 896], 'learned_alignment': False})
    del clip, text, crops, batch, visual, vectors, grid, semantic_grid
    gc.collect()
    torch.cuda.empty_cache()
    sat = FrozenSAT(sat_path, img_size=896).cuda().eval()
    if any(p.requires_grad for p in sat.parameters()):
        raise RuntimeError('SAT encoder unexpectedly trainable')
    write_json(output / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'precision': 'FP32, no AMP',
        'packages': {k: importlib.metadata.version(k) for k in ('timm', 'open_clip_torch', 'numpy')},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
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
        for index, row in enumerate(all_rows):
            features = sat(items[row['sample_id']]['image'][None].cuda())
            if features.shape != (1, 1024, 56, 56):
                raise RuntimeError('Unexpected SAT grid')
            maps = responses[row['sample_id']]
            for seed, (foreground, background) in zip(config['support_seeds'], prototypes):
                maps['A1_' + str(seed)] = prototype_response(features, foreground, (896, 896))[0, 0].cpu().numpy()
                maps['A2_' + str(seed)] = foreground_background_response(features, foreground, background, (896, 896))[0, 0].cpu().numpy()
            print(json.dumps({'phase': 'DINO_A1_A2', 'done': index + 1, 'total': len(all_rows), 'id': row['sample_id']}), flush=True)
    # Select fusion using calibration images ONLY; query masks are not passed.
    fusion = calibrate_fusion(
        {seed: [responses[r['sample_id']]['A2_' + str(seed)] for r in calibration] for seed in config['support_seeds']},
        [responses[r['sample_id']]['GeoAligned'] for r in calibration],
        [items[r['sample_id']]['mask'][0].numpy() for r in calibration], [i / 10 for i in range(11)])
    alpha = fusion['selected_semantic_weight']
    write_json(output / 'fusion_selection.json', dict(fusion,
        calibration_ids=[r['sample_id'] for r in calibration], query_labels_used=False))
    for maps in responses.values():
        for seed in config['support_seeds']:
            maps['A3_' + str(seed)] = fuse_responses(maps['A2_' + str(seed)], maps['GeoAligned'], alpha)
    names = list(responses[all_rows[0]['sample_id']])
    thresholds, ranges = {}, {}
    def previous_for(name):
        return baseline if name == 'A0' or name.startswith('A1_') else references['A2'] if name.startswith('A2_') else None
    for name in names:
        hist = sum(score_histogram(responses[r['sample_id']][name], items[r['sample_id']]['mask'][0].numpy()) for r in calibration)
        thresholds[name] = calibrate_threshold(hist)
        cumulative = hist.sum(0).cumsum()
        indices = np.searchsorted(cumulative, [.01 * cumulative[-1], .99 * cumulative[-1]])
        ranges[name] = [-1 + 2 * int(i) / hist.shape[1] for i in indices]
        previous = previous_for(name)
        if previous and thresholds[name] != previous['thresholds']['methods'][name]:
            raise RuntimeError('Calibration failed exact reproduction: ' + name)
        if name.startswith('A3_') and thresholds[name] != fusion['selected_runs'][name.split('_')[1]]:
            raise RuntimeError('Fusion selection threshold changed')
    write_json(output / 'thresholds.json', {'methods': thresholds, 'display_response_ranges': ranges,
        'calibration_ids': [r['sample_id'] for r in calibration], 'named_query_labels_used': False,
        'shared_semantic_weight': alpha})
    print(json.dumps({'phase': 'calibration_complete', 'semantic_weight': alpha}), flush=True)
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
            previous = previous_for(name)
            if previous and any(count[k] != previous['records'][sid]['methods'][name][k] for k in count):
                raise RuntimeError('Confusion failed exact reproduction: ' + sid + '/' + name)
            if record['included_in_summary']:
                for key in count:
                    totals[name][key] += count[key]
                histograms[name] += hist
        per_image.append(record)
        fmt = lambda v: f'{v:.3f}' if v is not None else 'n/a'
        status = 'QUERY' if sid in query_ids else ('PREVIEW: label conflict' if sid in conflict_ids else 'PREVIEW: calibration group')
        footer = (f"{sid} | {status} | GT {target.mean():.4f} | alpha {alpha:.1f} | "
                  f"A1 {fmt(record['methods']['A1_42']['iou'])} | A2 {fmt(record['methods']['A2_42']['iou'])} | A3 {fmt(record['methods']['A3_42']['iou'])}")
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
        previous = previous_for(name)
        if previous and summary[name] != previous['summary']['methods'][name]:
            raise RuntimeError('Aggregate failed exact reproduction: ' + name)
    support_results = {}
    for method in ('A1', 'A2', 'A3'):
        values = [summary[method + '_' + str(seed)] for seed in config['support_seeds']]
        support_results[method] = {'support_ids': [r['sample_id'] for r in supports],
            'means': {k: float(np.mean([v[k] for v in values])) for k in
                      ('iou', 'f1', 'precision', 'recall', 'pixel_ap_histogram', 'negative_pixel_false_positive_rate')},
            'micro_iou_population_std': float(np.std([v['iou'] for v in values]))}
    result = {'query_count': len(queries), 'positive_queries': sum(r['foreground_pixels'] > 0 for r in entries),
        'negative_queries': sum(r['foreground_pixels'] == 0 for r in entries), 'calibration_count': len(calibration),
        'methods': summary, 'three_distinct_supports': support_results, 'selected_semantic_weight': alpha,
        'paired_delta_A3_minus_A2': {str(seed): {k: summary['A3_' + str(seed)][k] - summary['A2_' + str(seed)][k]
            for k in ('iou', 'precision', 'recall', 'pixel_ap_histogram', 'negative_pixel_false_positive_rate')}
            for seed in config['support_seeds']},
        'scope': 'Exploratory curated development set, frozen fixed fusion, no learned common embedding alignment, no independent test or full SSPNet claim'}
    write_json(output / 'summary.json', result)
    with (output / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'included_in_summary', 'method', 'foreground_pixels',
            'iou', 'f1', 'precision', 'recall', 'false_positive_rate', 'pixel_ap_histogram', 'tp', 'fp', 'fn', 'tn'])
        writer.writeheader()
        for record in per_image:
            for name, values in record['methods'].items():
                writer.writerow({k: record[k] if k in record else name if k == 'method' else values.get(k) for k in writer.fieldnames})
    shutil.copyfile(baseline['path'] / 'supports.png', output / 'supports.png')
    write_json(output / 'reference_reproduction.json', {'status': 'exact', 'roles_identical': True,
        'input_hashes_verified': True, 'observed_data_identical': True, 'calibration_identical': True,
        'all_60_images_A0_3_A1_3_A2_confusions_identical': True, 'all_control_aggregate_metrics_identical': True,
        'reference_kernels': ['zh0416/pv-a0-a1-selected-check/2', 'zh0416/pv-a2-selected-check/1']})
    write_json(output / 'completion.json', {'status': 'complete', 'query_count': len(queries),
        'named_count': len(config['records']), 'preview_only_ids': [r['sample_id'] for r in previews],
        'all_encoders_frozen': True, 'gradient_updates': 0, 'a0_a1_a2_exact_reproduction': True,
        'selected_semantic_weight': alpha, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('A3 COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/a0a1_selected_20261029.json')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/a3_diagnostic'))
    args = parser.parse_args()
    try:
        main(args.output, args.config)
    except Exception:
        if not (args.output / 'completion.json').exists():
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
