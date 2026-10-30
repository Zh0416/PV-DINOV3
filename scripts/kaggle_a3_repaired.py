"""Train SSPNet DMSF priors through the downstream network on native896 PV.

Bounded known-class development adaptation, not the paper's novel-class FSS
benchmark. All training data, feature caches and weights stay on Kaggle.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kaggle_a0a1 import load_pil, plan_samples, write_json


def main(output):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run on Kaggle; original data and weights stay there')
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from pvseg.data import CATEGORIES, PV896Dataset, discover_root, read_manifest
    from pvseg.frozen_baselines import (calibrate_threshold, confusion, metrics, score_histogram,
        ranking_metrics, foreground_background_prototypes, foreground_background_response)
    from pvseg.runtime import Precision, environment, memory, seed_everything
    from pvseg.sspnet import SSPNet, checkpoint
    from pvseg.sspnet.encoders import FrozenSAT, PSPNet, freeze_bn, load_text_cache, sha256
    from pvseg.sspnet.learned_protocol import training_plan, episode_schedule
    from pvseg.sspnet.losses import objective
    from pvseg.sspnet.modules import resize

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    seed_everything(42)
    precision = SimpleNamespace(context=nullcontext, scaler=torch.amp.GradScaler('cuda', enabled=False), describe=lambda: {'enabled': False, 'dtype': 'float32'})
    config = json.loads((REPO / 'configs/a3_repaired_20261030.json').read_text())
    selection = json.loads((REPO / 'configs/a0a1_selected_20261029.json').read_text(encoding='utf-8'))
    write_json(output / 'protocol.json', dict(config, roles=selection,
        method='Trainable DMSF shared visual projection + text-to-DINO MLP + semantic projection; learned sigmoid(beta) CAM fusion; prior assembly/DCMA/ASPP',
        cache='Only frozen encoders, FP32. All projections/support pooling/CAMs rerun with autograd each episode.',
        losses='CE_final + Dice_meta + 0.2 BCE_pseudo + 0.04 Dice_spatial + 1.0 balanced raw-meta CE',
        checkpoint_tie='earlier update', checkpoint_criterion='mean calibration full-final micro-IoU at probability >= 0.5; ties foreground', query_labels_used_for_selection=False,
        supervision='22 additional training masks (4 nonempty,18 empty), 12 calibration masks, 1 held-out support per inference; known PV adaptation',
        deviations='DINOv3 replaces DINOv2; binary known PV base/meta merge; torchvision PSP stem; AdamW episode optimizer; one training scene; bounded steps; no augmentation'))
    with (output / 'tests.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print((output / 'tests.txt').read_text(), flush=True)
    references = {}
    for name, folder, commit in [('A01', 'a0a1_diagnostic', 'b2e7eb2cdaf302fbaa7948da63c491b10c55dc3e'),
                                  ('A2', 'a2_diagnostic', '59e7ed6d502444fbb7d38e97cbf75408080f7ae4')]:
        matches = [x.parent for x in Path('/kaggle/input').rglob('summary.json') if x.parent.name == folder]
        if len(matches) != 1:
            raise ValueError('Mount exactly one reference ' + folder)
        read = lambda n: json.loads((matches[0] / (n + '.json')).read_text(encoding='utf-8'))
        if read('completion')['status'] != 'complete' or read('environment')['source_commit'] != commit:
            raise ValueError('Unexpected reference version: ' + name)
        references[name] = {n: read(n) for n in ('summary', 'thresholds', 'per_image', 'selection_audit', 'data_observed')}
    root = discover_root('/kaggle/input')
    rows = read_manifest(root)
    supports, calibration, queries, previews, audit = plan_samples(rows, selection)
    if any(audit != v['selection_audit'] for v in references.values()):
        raise ValueError('Historical allocation changed')
    train, allocation = training_plan(rows, selection, audit)
    if len(train) != 22 or len(allocation['positive_query_ids']) != 4:
        raise ValueError('Unexpected prespecified training allocation')
    write_json(output / 'selection_audit.json', audit)
    write_json(output / 'training_allocation.json', allocation)
    active = train + supports + calibration
    if len(active) != len({r['sample_id'] for r in active}):
        raise ValueError('Duplicate active IDs')
    dataset = PV896Dataset(root, active, [0, 0, 0], [1, 1, 1])
    items, observed = {}, []
    for i, row in enumerate(active):
        sid = row['sample_id']
        if any(sha256(root / row[k]) != row[k + '_sha256'] for k in ('image', 'label')):
            raise ValueError('Input hash mismatch: ' + sid)
        items[sid] = dataset[i]
        observed.append(dict(sample_id=sid, foreground_pixels=int(items[sid]['mask'].sum())))
    expected = {r['sample_id']: r['foreground_pixels'] for r in references['A01']['data_observed']}
    if any(expected[r['sample_id']] != r['foreground_pixels'] for r in observed if r['sample_id'] in expected):
        raise ValueError('Historical label pixels changed')
    write_json(output / 'data_observed.json', observed)
    env = environment()
    env.update(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
               train_precision=precision.describe(), encoder_cache_precision='FP32 no AMP')
    write_json(output / 'environment.json', env)
    schedule = episode_schedule(allocation, config['episode_steps'], config['seed'])
    write_json(output / 'episode_schedule.json', [dict(query=q, support=s) for q, s in schedule])
    base_matches = [p.parent for p in Path('/kaggle/input').rglob('base.pt') if p.parent.name == 'a3_learned']
    if len(base_matches) != 1:
        raise ValueError('Mount exactly the previous learned A3 base checkpoint')
    previous = base_matches[0]
    old_env = json.loads((previous / 'environment.json').read_text())
    old_allocation = json.loads((previous / 'training_allocation.json').read_text())
    if old_env['source_commit'] != 'a6733794eb2fc2cf43c762ecf62952c5b1e4b15c' or old_allocation != allocation:
        raise ValueError('Previous base provenance/allocation mismatch')
    base = PSPNet(imagenet=False).cuda()
    state = checkpoint.load(previous / 'base.pt', base)
    if state['step'] != 160 or base.provenance['training_split_sha256'] != allocation['split_sha256']:
        raise ValueError('Unexpected base pretraining')
    old_schedule = json.loads((previous / 'episode_schedule.json').read_text())
    if old_schedule != [dict(query=q, support=s) for q, s in schedule]:
        raise ValueError('Episode schedule differs from old learned A3')
    write_json(output / 'base_reuse.json', dict(source_commit=old_env['source_commit'],
        step=state['step'], checkpoint_sha256=sha256(previous / 'base.pt'), allocation_identical=True,
        schedule_identical=True, initialization='fresh stage2 seed42; not diagnostic checkpoint'))
    del state
    base.requires_grad_(False).eval()
    gc.collect()
    torch.cuda.empty_cache()

    def find(name, parent=None):
        found = [x for x in Path('/kaggle/input').rglob(name) if parent is None or x.parent.name == parent]
        if len(found) != 1:
            raise ValueError(f'Expected one asset {name}; got {len(found)}')
        return found[0]

    dino = FrozenSAT(find('model.safetensors', 'verified_sat'), img_size=896).cuda().eval()
    text, text_meta = load_text_cache(find('georsclip_pv_text.pt'))
    seed_everything(config['seed'])
    net = SSPNet(base, dino, text, text_meta, hidden=config['hidden'], heads=config['heads'],
                 query_chunk=config['query_chunk'], routing=config['routing']).cuda()
    cache = output / 'feature_cache'
    cache.mkdir()
    cpu = lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else {k: cpu(v) for k, v in x.items()} if isinstance(x, dict) else tuple(cpu(v) for v in x)
    gpu = lambda x: x.cuda() if isinstance(x, torch.Tensor) else {k: gpu(v) for k, v in x.items()} if isinstance(x, dict) else tuple(gpu(v) for v in x)

    @torch.no_grad()
    def encode(sid):
        rgb = items[sid]['image'][None].cuda()
        return dict(cnn=net.cnn.features(rgb), dino=net.dino(rgb))

    cache_rows = train + supports + calibration
    for i, row in enumerate(cache_rows):
        if shutil.disk_usage(output).free < 1024 ** 3:
            raise RuntimeError('Insufficient feature-cache disk space')
        features = encode(row['sample_id'])
        torch.save(cpu(features), cache / (row['sample_id'] + '.pt'))
        del features
        print(json.dumps(dict(phase='frozen_FP32_cache', done=i + 1, total=len(cache_rows))), flush=True)
    write_json(output / 'cache.json', dict(ids=[r['sample_id'] for r in cache_rows], dtype='float32',
        bytes=sum(p.stat().st_size for p in cache.iterdir()), trainable_features_cached=False,
        dino=dino.metadata, text=text_meta))

    def read_features(sid):
        return gpu(torch.load(cache / (sid + '.pt'), map_location='cpu', weights_only=True))

    def forward(q, s, sid):
        return net.forward_from_features(q['cnn'], q['dino'], [s['cnn']], [s['dino']],
                                         [items[sid]['mask'][None].cuda()], (896, 896))

    @torch.no_grad()
    def prior_scores(q, s, sid):
        cams = net.dmsf(q['dino'], [s['dino']], [items[sid]['mask'][None].cuda()], net.raw_text)
        return {k: resize(cams[k].float(), (896, 896))[0, 0].cpu().numpy()
                for k in ('spatial', 'semantic', 'fused')}

    @torch.no_grad()
    def calibrate(step):
        net.eval()
        hist = {f'{name}_{seed}': np.zeros((2, 4096), np.int64)
                for name in ('spatial', 'semantic', 'fused', 'final', 'meta') for seed in selection['support_seeds']}
        fixed_counts = {str(seed): dict(tp=0, fp=0, fn=0, tn=0) for seed in selection['support_seeds']}
        for row in calibration:
            sid = row['sample_id']
            q = read_features(sid)
            for seed, support in zip(selection['support_seeds'], supports):
                ssid = support['sample_id']
                s = read_features(ssid)
                with precision.context():
                    out = forward(q, s, ssid)
                maps = {name: resize(out[key].float(), (896, 896))[0, 0].cpu().numpy()
                        for name, key in [('spatial', 'spatial_cam'), ('semantic', 'semantic_cam'), ('fused', 'prior_cam')]}
                maps['final'] = out['logits'].float().softmax(1)[0, 1].cpu().numpy()
                maps['meta'] = out['meta_logits'].float().softmax(1)[0, 1].cpu().numpy()
                c = confusion(maps['final'], items[sid]['mask'][0].numpy(), 0.5)
                for k in c:
                    fixed_counts[str(seed)][k] += c[k]
                for name, score in maps.items():
                    hist[f'{name}_{seed}'] += score_histogram(score, items[sid]['mask'][0].numpy())
                del s, out
            del q
        thresholds = {name: calibrate_threshold(h) for name, h in hist.items()}
        return dict(step=step, thresholds=thresholds, fixed_final_counts=fixed_counts,
                    mean_calibration_iou=float(np.mean([metrics(fixed_counts[str(seed)])['iou']
                                                       for seed in selection['support_seeds']])),
                    gamma=float(net.dmsf.beta.sigmoid()), query_labels_used=False)

    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                 lr=config['episode_lr'], weight_decay=config['weight_decay'])
    initial = {k: p.detach().cpu().clone() for k, p in net.named_parameters() if p.requires_grad}
    history, gradient_max, selections = [], {}, []
    best = -1.
    successful_updates = 0
    started = time.monotonic()
    for step, (qid, sid) in enumerate(schedule, 1):
        q, s = read_features(qid), read_features(sid)
        net.train()
        optimizer.zero_grad(set_to_none=True)
        with precision.context():
            out = forward(q, s, sid)
            loss, terms = objective(out, items[qid]['mask'][None].cuda(), meta_ce_weight=config['meta_ce_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite episode loss')
        precision.scaler.scale(loss).backward()
        precision.scaler.unscale_(optimizer)
        for name, p in net.named_parameters():
            if not p.requires_grad and p.grad is not None:
                raise RuntimeError('Frozen encoder has gradient: ' + name)
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient: ' + name)
                group = '.'.join(name.split('.')[:2])
                gradient_max[group] = max(gradient_max.get(group, 0.), float(p.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                      config['gradient_clip'], error_if_nonfinite=True)
        previous_scale = precision.scaler.get_scale()
        precision.scaler.step(optimizer)
        precision.scaler.update()
        if precision.scaler.get_scale() < previous_scale:
            raise RuntimeError('AMP skipped an episode update')
        successful_updates += 1
        history.append(dict(step=step, query=qid, support=sid, loss=float(loss.detach()),
                            terms={k: float(v.detach()) for k, v in terms.items()}, gamma=float(net.dmsf.beta.sigmoid())))
        del q, s, out, loss, terms
        if step % 20 == 0:
            write_json(output / 'episode_training.json', history)
            print(json.dumps(dict(phase='episode', step=step, loss=history[-1]['loss'], gamma=history[-1]['gamma'])), flush=True)
        if step % config['selection_every'] == 0:
            selected = calibrate(step)
            selections.append(selected)
            if selected['mean_calibration_iou'] > best:
                best = selected['mean_calibration_iou']
                checkpoint.save(output / 'selected.pt', net, optimizer, precision, config, 0, step, history=history)
            write_json(output / 'checkpoint_selection.json', dict(curve=selections, query_labels_used=False))
            print(json.dumps(dict(phase='calibration', step=step, iou=selected['mean_calibration_iou'])), flush=True)
    del optimizer
    learned = dict(successful_episode_updates=successful_updates, maximum_abs_gradient_by_module=gradient_max,
        maximum_parameter_delta_by_module={}, elapsed_seconds=time.monotonic() - started)
    for name, p in net.named_parameters():
        if p.requires_grad:
            group = '.'.join(name.split('.')[:2])
            delta = float((p.detach().cpu() - initial[name]).abs().max())
            learned['maximum_parameter_delta_by_module'][group] = max(delta, learned['maximum_parameter_delta_by_module'].get(group, 0.))
    for group in ('dmsf.visual', 'dmsf.text_to_dino', 'dmsf.text_hidden', 'dmsf.beta', 'dcma.pseudo'):
        if not gradient_max.get(group, 0.) > 0 or not learned['maximum_parameter_delta_by_module'].get(group, 0.) > 0:
            raise RuntimeError('No learning in required path: ' + group)
    write_json(output / 'learning_audit.json', learned)
    del initial
    chosen = max(selections, key=lambda x: (x['mean_calibration_iou'], -x['step']))
    checkpoint.load(output / 'selected.pt', net)
    net.eval()
    write_json(output / 'thresholds.json', dict(selected_step=chosen['step'], methods=chosen['thresholds'],
        calibration_ids=[r['sample_id'] for r in calibration], query_labels_used=False))
    # Keep only three supports; transient caches must not become large outputs.
    for row in train + calibration:
        (cache / (row['sample_id'] + '.pt')).unlink()
    write_json(output / 'selection_locked.json', dict(selected_step=chosen['step'], criterion=config['checkpoint_selection'], query_images_loaded=False, query_labels_loaded=False))
    for row in queries + previews:
        sid = row['sample_id']
        if any(sha256(root / row[k]) != row[k + '_sha256'] for k in ('image', 'label')):
            raise ValueError('Input hash mismatch: ' + sid)
        items[sid] = PV896Dataset(root, [row], [0, 0, 0], [1, 1, 1])[0]
        n = int(items[sid]['mask'].sum())
        if sid in expected and n != expected[sid]:
            raise ValueError('Historical label pixels changed')
        observed.append(dict(sample_id=sid, foreground_pixels=n))
    write_json(output / 'data_observed.json', observed)
    support_features = [read_features(r['sample_id']) for r in supports]
    prototypes = [foreground_background_prototypes(s['dino'], items[r['sample_id']]['mask'][None].cuda())
                  for s, r in zip(support_features, supports)]
    threshold2 = references['A2']['thresholds']['methods']
    reference2 = {r['sample_id']: r for r in references['A2']['per_image']}
    supplied = {r['sample_id']: r for r in selection['records']}
    query_ids = {r['sample_id'] for r in queries}
    records, histograms = [], {}
    with torch.no_grad():
        for i, row in enumerate(queries + previews):
            sid = row['sample_id']
            q = encode(sid)
            target = items[sid]['mask'][0].numpy().astype(bool)
            record = dict(sample_id=sid, included_in_summary=sid in query_ids, source_scene=row['source_scene'],
                          annotation=supplied[sid], foreground_pixels=int(target.sum()), methods={})
            panel_maps = {}
            for seed, support, s, (fg, bg) in zip(selection['support_seeds'], supports, support_features, prototypes):
                with precision.context():
                    out = forward(q, s, support['sample_id'])
                maps = {name: resize(out[key].float(), (896, 896))[0, 0].cpu().numpy()
                        for name, key in [('spatial', 'spatial_cam'), ('semantic', 'semantic_cam'), ('fused', 'prior_cam')]}
                maps['final'] = out['logits'].float().softmax(1)[0, 1].cpu().numpy()
                maps['meta'] = out['meta_logits'].float().softmax(1)[0, 1].cpu().numpy()
                maps['base_fixed'] = out['base_logits'].float().softmax(1)[0, 1].cpu().numpy()
                maps['A2'] = foreground_background_response(q['dino'], fg, bg, (896, 896))[0, 0].cpu().numpy()
                for name, score in maps.items():
                    method = f'{name}_{seed}'
                    threshold = 0.5 if name == 'base_fixed' else threshold2[method]['threshold'] if name == 'A2' else chosen['thresholds'][method]['threshold']
                    counts = confusion(score, target, threshold)
                    if name == 'A2' and any(counts[k] != reference2[sid]['methods'][method][k] for k in counts):
                        raise RuntimeError('A2 exact reproduction failed: ' + sid + '/' + method)
                    record['methods'][method] = dict(metrics(counts), **ranking_metrics(score_histogram(score, target)))
                    if sid in query_ids:
                        histograms[method] = histograms.get(method, np.zeros((2, 4096), np.int64)) + score_histogram(score, target)
                    if seed == selection['support_seeds'][0]:
                        panel_maps[name] = score
                record['methods'][f'meta_fixed_{seed}'] = metrics(confusion(maps['meta'], target, 0.5))
                record['methods'][f'final_fixed_{seed}'] = metrics(confusion(maps['final'], target, config['final_threshold']))
                del out
            records.append(record)
            make_panel(output / 'predictions' / (sid + '.png'), load_pil(root, row), target, panel_maps,
                       chosen['thresholds'], threshold2, sid, chosen['step'], sid in query_ids)
            write_json(output / 'per_image.json', records)
            del q
            print(json.dumps(dict(phase='query_evaluation', done=i + 1, total=len(queries) + len(previews))), flush=True)
    entries = [r for r in records if r['included_in_summary']]

    def aggregate(members, method):
        counts = {k: sum(r['methods'][method][k] for r in members) for k in ('tp', 'fp', 'fn', 'tn')}
        return metrics(counts)

    summary = {}
    for method in entries[0]['methods']:
        result = aggregate(entries, method)
        if method in histograms:
            result.update(ranking_metrics(histograms[method]))
        negatives = [r for r in entries if r['foreground_pixels'] == 0]
        result['negative_pixel_false_positive_rate'] = aggregate(negatives, method)['false_positive_rate']
        result['negative_images_with_any_fp'] = sum(r['methods'][method]['fp'] > 0 for r in negatives)
        summary[method] = result
    means = {name: {k: (float(np.mean([summary[f'{name}_{seed}'][k] for seed in selection['support_seeds']]))
                        if all(summary[f'{name}_{seed}'][k] is not None for seed in selection['support_seeds']) else None)
                    for k in ('iou', 'precision', 'recall', 'false_positive_rate', 'negative_pixel_false_positive_rate')}
             for name in ('A2', 'spatial', 'semantic', 'fused', 'final', 'final_fixed', 'meta', 'meta_fixed', 'base_fixed')}
    groups = {field: {value: {m: aggregate([r for r in entries if r['annotation'][field] == value], m)
                             for m in summary} for value in sorted({r['annotation'][field] for r in entries})}
              for field in ('difficulty', 'area', 'morphology')}
    write_json(output / 'summary.json', dict(methods=summary, means_over_support_choices=means,
        groups=groups, selected_step=chosen['step'], selected_gamma=float(net.dmsf.beta.sigmoid()),
        query_count=len(queries), preview_count=len(previews), scope=config['scope'],
        interpretation='Descriptive comparison; additional supervised training vs frozen A2; component CAMs are jointly trained, not independent ablations'))
    for p in cache.iterdir():
        p.unlink()
    cache.rmdir()
    write_json(output / 'completion.json', dict(status='complete', source_commit=env['source_commit'],
        reused_base_updates=config['base_steps'], base_updates_this_run=0, episode_updates=successful_updates, selected_step=chosen['step'],
        query_labels_used_for_selection=False, exact_A2_reproduction=True, memory=memory(),
        claim='Training/evaluation completed; improvement is determined by recorded metrics, not this status'))


def make_panel(path, rgb, target, maps, thresholds, threshold2, sid, step, query):
    import numpy as np
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    mask = lambda x: Image.fromarray(np.asarray(x, dtype='uint8') * 255).convert('RGB')
    fixed = maps['final'] >= .5
    err = np.zeros((*target.shape, 3), dtype=np.uint8)
    err[fixed & target] = [60, 180, 90]
    err[fixed & ~target] = [240, 70, 70]
    err[~fixed & target] = [50, 140, 250]
    heat = Image.fromarray((colormaps['turbo'](np.clip(maps['fused'], 0, 1))[..., :3] * 255).astype('uint8'))
    pictures = [rgb, mask(target), mask(maps['A2'] >= threshold2['A2_42']['threshold']),
        mask(maps['base_fixed'] >= .5), mask(maps['meta'] >= .5), mask(fixed),
        mask(maps['final'] >= thresholds['final_42']['threshold']), Image.fromarray(err), heat]
    labels = ['RGB', 'Ground truth', 'Frozen A2', 'Frozen PSPNet base (0.5)', 'Meta head (0.5)',
        'Full network (0.5)', 'Full network (calibrated)', 'TP green / FP red / FN blue', 'Fused PRIOR [0,1]']
    canvas = Image.new('RGB', (1008, 1100), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(pictures, labels)):
        x, y = (i % 3) * 336, (i // 3) * 358
        draw.text((x + 5, y + 6), label, fill='black')
        canvas.paste(im.resize((336, 336), Image.Resampling.BILINEAR if i in (0, 8) else Image.Resampling.NEAREST), (x, y + 22))
    draw.text((6, 1078), f'{sid} | {"QUERY" if query else "PREVIEW ONLY"} | support pv001183 | step {step} | native896', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/a3_repaired'))
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', dict(traceback=traceback.format_exc()))
        raise
