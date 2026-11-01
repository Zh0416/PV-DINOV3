"""Supervised PV adaptation with entire geographic groups held out.

The 24 historical development queries become a development training pool.
Each reported tile is predicted by a model trained on the other geographic
groups. Calibration retains the original separate group. No independent-test
or strict novel-class few-shot claim is made.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pvseg.dynamic_brightness_fusion import CAMWeight, convex_fusion, gate_inputs
from pvseg.frozen_baselines import calibrate_threshold, confusion, metrics, score_histogram
from pvseg.hard_negative_gate import ResidualFusionGate, residual_fusion, gate_regularization
from pvseg.local_texture import FEATURE_NAMES, texture_features
from pvseg.losses import segmentation_loss
from pvseg.spatial import bounds, overlap, scene_groups

SEEDS = (42, 43, 44)  # Three support choices, not training seeds.
COUNT_KEYS = ('tp', 'fp', 'fn', 'tn')
ARMS = {
    'alpha_retrained': dict(channels=8, hard_weight=0., alpha_only=True),
    'residual_balanced': dict(channels=8, hard_weight=0.),
    'residual_hard': dict(channels=8, hard_weight=.5),
    'residual_spectral': dict(channels=10, hard_weight=.5),
    'residual_texture': dict(channels=17, hard_weight=.5),
    # Matched capacity/init/schedule; only FFT channels zeroed for this control.
    'residual_no_fft': dict(channels=17, hard_weight=.5, zero_fft=True),
    'residual_texture_balanced': dict(channels=17, hard_weight=0.),
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_one(root, name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {name}, got {matches}')
    return matches[0]


def locations(args):
    if Path('/kaggle/input').is_dir():
        input_root = Path('/kaggle/input')
        manifests = [p for p in input_root.rglob('manifest.csv') if all(
            (p.parent / cat / 'images').is_dir() for cat in ('正样本', '负样本', '困难正样本'))]
        if len(manifests) != 1:
            raise RuntimeError(f'Ambiguous PV dataset: {manifests}')
        root = manifests[0].parent
        score_path = find_one(input_root, 'score_grids.npz')
        weight_path = find_one(input_root, 'weight_grids.npz')
        output = Path('/kaggle/working/outputs/hard_negative_rejection_20261101')
    else:
        root = args.root
        score_path = REPO / 'outputs/greenhouse_pilot_20261030/greenhouse_pilot/score_grids.npz'
        weight_path = REPO / 'outputs/dynamic_brightness_20261030/dynamic_brightness/weight_grids.npz'
        output = REPO / 'outputs/hard_negative_rejection_20261101/local'
    return root, score_path, weight_path, args.output or output


def hard_pixel_loss(logits, target, fraction=.15):
    """V1/V2 auxiliary objective: hardest 15% per majority patch class.

    The main BCE+Dice keeps fractional patch targets; this auxiliary term
    treats >50% PV patches as positive and <50% as negative. Mixed patches
    near tiny PV objects can be harmed; the balanced-only arm measures this.
    """
    losses = []
    for selector, values in ((target > .5, F.softplus(-logits)),
                             (target < .5, F.softplus(logits))):
        if selector.any():
            selected = values[selector]
            losses.append(selected.topk(max(1, int(selected.numel() * fraction))).values.mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def make_cache(root, rows, ids, score, weight):
    cache, hashes = {}, {}
    for sid in ids:
        row = rows[sid]
        hashes[sid] = {key: digest(root / row[key]) for key in ('image', 'label')}
        assert all(hashes[sid][key] == row[key + '_sha256'] for key in hashes[sid]), sid
        with Image.open(root / row['image']) as im:
            texture = torch.from_numpy(texture_features(np.asarray(im.convert('RGB'))))[None]
        with Image.open(root / row['label']) as im:
            raw = np.asarray(im)
        assert raw.shape == (896, 896) and np.isin(raw, (0, 255)).all(), sid
        target = (raw == 255)
        area = target.astype(np.float32).reshape(56, 16, 56, 16).mean(axis=(1, 3))
        tensor = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]
        for support in SEEDS:
            prefix = f'{sid}_{support}_'
            d = tensor(score[prefix + 'spatial'])
            penalty = score[prefix + 'semantic_penalty'] * weight[sid + '_gate']
            s = tensor(np.clip(score[prefix + 'pv_semantic'] - penalty, -1, 1))
            inputs = gate_inputs(d, s, tensor(weight[sid + '_z']),
                                 tensor(weight[sid + '_gate']), tensor(penalty))
            cache[sid, support] = dict(d=d, s=s, x=torch.cat((inputs, texture), 1),
                                      target=tensor(area), full_target=target)
    return cache, hashes


def model_inputs(items, cfg, device):
    x = torch.cat([item['x'][:, :cfg['channels']] for item in items]).to(device)
    if cfg.get('zero_fft'):
        x[:, 8:10] = 0
    return x


def model_outputs(model, x, cfg):
    if cfg.get('alpha_only'):
        alpha = model(x)
        return alpha, torch.zeros_like(alpha)
    return model(x)


def train_model(cache, train_ids, cfg, steps, device):
    torch.manual_seed(42)
    model = (CAMWeight('local', hidden=16) if cfg.get('alpha_only') else
             ResidualFusionGate(cfg['channels'], hidden=16, max_residual=.15)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    items = [(sid, support) for sid in train_ids for support in SEEDS]
    generator = torch.Generator().manual_seed(42)
    history = []
    for step in range(1, steps + 1):
        chosen = [items[int(i)] for i in torch.randint(len(items), (8,), generator=generator)]
        batch = [cache[k] for k in chosen]
        d, s, target = [torch.cat([item[key] for item in batch]).to(device)
                        for key in ('d', 's', 'target')]
        alpha, residual = model_outputs(model, model_inputs(batch, cfg, device), cfg)
        score = residual_fusion(d, s, alpha, residual)
        logits = 20 * (score - .19645)
        seg = segmentation_loss(logits, target)
        hard = hard_pixel_loss(logits, target) if cfg['hard_weight'] else logits.new_zeros(())
        anchor, regular = gate_regularization(alpha, residual, residual_weight=1., tv_weight=.25)
        loss = seg + cfg['hard_weight'] * hard + .01 * anchor + .001 * regular
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 40 == 0:
            history.append(dict(step=step, loss=float(loss.detach()), seg=float(seg.detach()),
                                hard=float(hard.detach()), grad_norm=float(norm)))
    model.eval()
    return model, history


@torch.no_grad()
def predict(model, item, cfg, device):
    a, r = model_outputs(model, model_inputs([item], cfg, device), cfg)
    a, r = a.cpu(), r.cpu()
    # CPU FP32 interpolation gives the same reconstruction on the host.
    score = residual_fusion(item['d'], item['s'], a, r, (896, 896))[0, 0].numpy()
    return score, a[0, 0].numpy(), r[0, 0].numpy()


def calibrate(model, cache, calibration_ids, cfg, device):
    result = {}
    for support in SEEDS:
        hist = np.zeros((2, 4096), dtype=np.int64)
        for sid in calibration_ids:
            item = cache[sid, support]
            score, _, _ = predict(model, item, cfg, device)
            hist += score_histogram(score, item['full_target'])
        result[str(support)] = calibrate_threshold(hist)
    return result


def summarize(records, ids):
    by_support = {}
    for support in SEEDS:
        counts = {key: sum(records[sid][str(support)][key] for sid in ids) for key in COUNT_KEYS}
        by_support[str(support)] = metrics(counts)
    result = dict(by_support=by_support)
    for key in by_support['42']:
        values = [row[key] for row in by_support.values() if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def main(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root, score_path, weight_path, output = locations(args)
    output.mkdir(parents=True, exist_ok=False)
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    rows = {row['sample_id']: row for row in all_rows}
    old = json.loads(weight_path.with_name('protocol.json').read_text(encoding='utf-8'))
    ids, cal = old['query_ids'], old['calibration_ids']
    support_ids = [r['support_id'] for r in old['support_runs']]
    groups, edges = scene_groups(all_rows)
    group_for_scene = {s: g for g, scenes in groups.items() for s in scenes}
    group_for = {sid: group_for_scene[row['source_scene']] for sid, row in rows.items()}
    assert not {group_for[s] for s in ids} & {group_for[s] for s in cal + support_ids}
    selection = json.loads((REPO / 'configs/a0a1_selected_20261029.json').read_text(encoding='utf-8'))
    annotations = {r['sample_id']: r for r in selection['records']}
    folds = []
    for group in sorted({group_for[s] for s in ids}):
        held = [s for s in ids if group_for[s] == group]
        train = [s for s in ids if group_for[s] != group]
        assert not any(overlap(bounds(rows[a]), bounds(rows[b])) for a in held for b in train + cal + support_ids)
        folds.append(dict(group=group, train_ids=train, holdout_ids=held))
    if args.fold_limit:
        folds = folds[:args.fold_limit]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol = dict(version=3, ids=ids, calibration_ids=cal, support_ids=support_ids,
        groups=groups, cross_scene_overlap_edges=edges, folds=folds, arms=ARMS,
        steps=args.steps, batch_size=8, hidden=16, learning_rate=.002, seed=42,
        loss='Area-label balanced BCE + .5 Dice; optional .5 hardest-15%-majority-patch BCE',
        loss_reference=.19645, residual_bound=.15,
        threshold_selection='Per-fold/per-arm/per-support maximize micro-IoU on original 12 calibration masks; 4096 bins',
        checkpoint_selection='Fixed 160 steps, no fold-heldout checkpoint selection',
        training_pool='Historical 24 development queries now explicitly used as supervised adaptation pool; leave one geographic group out',
        supervision='Binary PV masks trusted; outside PV is background; no greenhouse pixel labels',
        limitation='Repeatedly inspected development data, new training supervision and fitting to three known support choices; not independent/novel-class FSS',
        texture_features=list(FEATURE_NAMES), device=device, torch=torch.__version__, numpy=np.__version__,
        tf32=False, score_sha256=digest(score_path), alpha_sha256=digest(weight_path),
        source_sha256={str(p.relative_to(REPO)): digest(p) for p in
                       [Path(__file__), REPO/'pvseg/hard_negative_gate.py', REPO/'pvseg/local_texture.py']})
    write_json(output / 'protocol_before_training.json', protocol)
    scores, weights = np.load(score_path), np.load(weight_path)
    cache, hashes = make_cache(root, rows, ids + cal, scores, weights)
    write_json(output / 'input_hashes.json', hashes)
    records = {arm: {} for arm in ('fixed', 'local', *ARMS)}
    grids, fold_reports, candidates = {}, {}, {}
    # Verify every historical baseline count, rather than just rounded IoU.
    old_records = json.loads(weight_path.with_name('metrics.json').read_text(encoding='utf-8'))['records']
    old_by_id = {r['sample_id']: r for r in old_records}
    for arm in ('fixed', 'local'):
        calibration = old['fixed_baseline_calibration'] if arm == 'fixed' else old['arms']['local']['calibration']
        for sid in ids:
            records[arm][sid] = {}
            for support in SEEDS:
                item = cache[sid, support]
                a = torch.full_like(item['d'], .7) if arm == 'fixed' else torch.from_numpy(
                    weights[f'{sid}_{support}_local_alpha'])[None, None]
                t = calibration['by_seed'][str(support)]['threshold']
                score = convex_fusion(item['d'], item['s'], a, (896, 896))[0, 0].numpy()
                counts = confusion(score, item['full_target'], t)
                reference = old_by_id[sid]['methods'][str(support)]['recalibrated'][arm]
                assert all(counts[k] == reference[k] for k in COUNT_KEYS), (arm, sid, support)
                records[arm][sid][str(support)] = dict(**counts, threshold=t)
    for fold in folds:
        group, train_ids, held = fold['group'], fold['train_ids'], fold['holdout_ids']
        fold_reports[group] = dict(**fold, models={})
        train_cache = {k: v for k, v in cache.items() if k[0] in train_ids}
        for arm, cfg in ARMS.items():
            model, history = train_model(train_cache, train_ids, cfg, args.steps, device)
            calibration = calibrate(model, cache, cal, cfg, device)
            name = f'{group}__{arm}'
            checkpoint = output / 'folds' / (name + '.pt')
            checkpoint.parent.mkdir(exist_ok=True)
            saved = dict(model={k: v.cpu() for k, v in model.state_dict().items()}, arm=arm,
                         config=cfg, steps=args.steps, training_ids=train_ids, holdout_ids=held,
                         calibration=calibration, history=history)
            torch.save(saved, checkpoint)
            fold_reports[group]['models'][arm] = dict(checkpoint=str(checkpoint.relative_to(output)),
                sha256=digest(checkpoint), calibration=calibration, history=history)
            for sid in held:
                records[arm][sid] = {}
                for support in SEEDS:
                    score, a, r = predict(model, cache[sid, support], cfg, device)
                    t = calibration[str(support)]['threshold']
                    counts = confusion(score, cache[sid, support]['full_target'], t)
                    records[arm][sid][str(support)] = dict(**counts, threshold=t)
                    grids[f'{sid}_{support}_{arm}_alpha'] = a
                    grids[f'{sid}_{support}_{arm}_residual'] = r
            compact = summarize(records[arm], held)
            print(json.dumps(dict(group=group, arm=arm, iou=compact['iou'], recall=compact['recall'],
                                  fp=compact['fp'])), flush=True)
    evaluated = [sid for fold in folds for sid in fold['holdout_ids']]
    subset_ids = dict(all=evaluated, greenhouse=[s for s in evaluated if annotations[s]['difficulty']=='大棚干扰'],
                      pure_negative=[s for s in evaluated if not annotations[s]['pv_present']])
    summary = dict(protocol=protocol, folds=fold_reports, subset_ids=subset_ids, records=records,
                   groups={key: {arm: summarize(r, subset) for arm, r in records.items()}
                           for key, subset in subset_ids.items()})
    # Fixed-step full-pool candidates are saved for future regions. Their scores
    # on the 24 training images are intentionally not used as validation.
    if args.fold_limit is None:
        for arm, cfg in ARMS.items():
            model, history = train_model(cache, ids, cfg, args.steps, device)
            calibration = calibrate(model, cache, cal, cfg, device)
            checkpoint = output / 'candidates' / (arm + '.pt')
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()}, arm=arm,
                            config=cfg, steps=args.steps, training_ids=ids, calibration_ids=cal,
                            calibration=calibration, history=history, texture_features=list(FEATURE_NAMES),
                            evaluation='Requires unseen spatial groups; all 24 images used for training.'), checkpoint)
            candidates[arm] = dict(path=str(checkpoint.relative_to(output)), sha256=digest(checkpoint),
                                   calibration=calibration)
    summary['candidates'] = candidates
    np.savez_compressed(output / 'holdout_grids.npz', **grids)
    write_json(output / 'summary.json', summary)
    write_json(output / 'completion.json', dict(status='complete', geographic_folds=len(folds),
        heldout_images=len(evaluated), arms=len(ARMS), historical_count_checks=len(ids)*3*2,
        grids_sha256=digest(output/'holdout_grids.npz')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(r'G:\PV-GF\PV-896'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--fold-limit', type=int)
    main(parser.parse_args())
