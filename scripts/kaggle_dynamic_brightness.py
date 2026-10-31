"""Private pilot: retain brightness suppression; train only convex CAM weights."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kaggle_a0a1 import load_pil, plan_samples, write_json


def main(output):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Private Kaggle assets required')
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    import open_clip
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from pvseg.brightness_gate import (apply_brightness_gate, brightness_gate,
        brightness_grid, fit_brightness_reference)
    from pvseg.data import discover_root, read_manifest
    from pvseg.dynamic_brightness_fusion import CAMWeight, convex_fusion, gate_inputs, weight_regularization
    from pvseg.losses import segmentation_loss
    from pvseg.frozen_baselines import (aligned_crop_features, calibrate_threshold, confusion,
        foreground_background_prototypes, foreground_background_response, metrics, score_histogram)
    from pvseg.greenhouse_prior import DEFAULT_CONCEPT_PROMPTS, GreenhouseContrastPrior, encode_concepts
    from pvseg.sspnet.encoders import FrozenSAT, GEO_SHA, SAT_SHA, sha256
    from pvseg.sspnet.learned_protocol import episode_schedule, training_plan

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with (output / 'tests.txt').open('w') as stream:
        for pattern in ('test_dynamic_brightness_fusion.py', 'test_brightness_gate.py',
                        'test_greenhouse_prior.py', 'test_segmentation_loss.py'):
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                           cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    config = json.loads((REPO / 'configs/dynamic_brightness_20261030.json').read_text())
    selection = json.loads((REPO / 'configs/a0a1_selected_20261029.json').read_text(encoding='utf-8'))
    root = discover_root('/kaggle/input')
    rows = read_manifest(root)
    supports, calibration, all_queries, _, audit = plan_samples(rows, selection)
    train, allocation = training_plan(rows, selection, audit)
    assert len(train) == 22 and len(allocation['positive_query_ids']) == 4
    by_id = {r['sample_id']: r for r in rows}
    queries = [by_id[s] for s in config['query_ids']]
    assert set(config['query_ids']) <= {r['sample_id'] for r in all_queries}
    dirs = [p.parent for p in Path('/kaggle/input').rglob('score_grids.npz') if p.parent.name == 'greenhouse_pilot']
    if len(dirs) != 1:
        raise RuntimeError('Mount exactly greenhouse pilot v1')
    old_dir = dirs[0]
    oldp = json.loads((old_dir / 'protocol.json').read_text())
    assert oldp['query_ids'] == config['query_ids'] and oldp['support_runs'] == audit['support_runs']
    assert oldp['calibration_ids'] == [r['sample_id'] for r in calibration]
    old = np.load(old_dir / 'score_grids.npz')
    seeds = [str(x['seed_label']) for x in audit['support_runs']]
    schedule = episode_schedule(allocation, config['steps'], config['seed'])
    protocol = dict(config=config, training=allocation, calibration_ids=oldp['calibration_ids'],
        query_ids=config['query_ids'], support_runs=audit['support_runs'], selection_audit=audit,
        source=json.loads((REPO / 'pilot_source_manifest.json').read_text()),
        cache_sha256=sha256(old_dir / 'score_grids.npz'), encoder_hashes=dict(sat=SAT_SHA, geo=GEO_SHA),
        formula='S_b=clip(S-P*g,-1,1); CAM=(1-alpha)*upsample(D)+alpha*upsample(S_b); alpha=upsample(sigmoid(logit(.7)+gate(x)))',
        frozen_components=['DINO-SAT', 'GeoRSCLIP', 'semantic greenhouse penalty', 'brightness g'],
        learned_components='Only alpha; scalar global vs local 8->8(3x3/GELU)->1(1x1), no classifier residual',
        gate_inputs=['D', 'S_b', 'clamp(z,-1,2)', 'g', 'P*g', 'S_b-D', 'local_std(D)', 'local_std(S_b)'],
        interpolation='D,S_b,alpha individually bilinear align_corners=False to896 BEFORE fusion',
        supervision='22 training masks + 12 calibration masks + 3 held-out inference support masks',
        pixel_labels='Binary PV only; FP is background false positive, not greenhouse-labelled pixel error',
        scope='Previously observed 24-image development subset; support choices are not training random seeds',
        query_masks_used_in_training_or_selection=False,
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), precision='FP32; TF32 off'))
    write_json(output / 'protocol_before_training.json', protocol)
    write_json(output / 'episode_schedule.json', [dict(query=q, support=s) for q, s in schedule])
    active = train + supports + calibration + queries
    assert len(active) == len({r['sample_id'] for r in active})
    for r in active:
        if any(sha256(root / r[k]) != r[k + '_sha256'] for k in ('image', 'label')):
            raise RuntimeError('Data hash changed: ' + r['sample_id'])
    def read_mask(sid):
        values = np.asarray(Image.open(root / by_id[sid]['label']))
        assert values.shape == (896, 896) and np.isin(values, [0, 255]).all()
        return torch.from_numpy((values == 255).astype(np.float32))[None, None]
    masks = {r['sample_id']: read_mask(r['sample_id']) for r in train + calibration}
    brightness = {}
    for r in train + calibration + queries:
        sid = r['sample_id']
        raw, z, stats = brightness_grid(np.array(load_pil(root, r)))
        brightness[sid] = dict(raw=raw, z=z, stats=stats)
    ref = fit_brightness_reference([brightness[r['sample_id']]['z'] for r in calibration],
        [F.avg_pool2d(masks[r['sample_id']], 16)[0, 0].numpy() for r in calibration])
    assert ref['lower'] == .6029685139656067 and ref['upper'] == .8076503276824951
    protocol['brightness_reference'] = ref
    for item in brightness.values():
        item['gate'] = brightness_gate(item['z'], ref)

    # Reuse historical scalar grids for all calibration/query inference.
    frozen, learned = {}, {}
    def as_tensor(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]
    def build_pair(sid, d, pv, penalty):
        b = brightness[sid]
        s, effective = apply_brightness_gate(pv, penalty, b['gate'])
        d, s = as_tensor(d), as_tensor(s)
        x = gate_inputs(d, s, as_tensor(b['z']), as_tensor(b['gate']), as_tensor(effective))
        return d, s, x.cuda()
    for row in calibration + queries:
        qid = row['sample_id']
        for seed in seeds:
            prefix = qid + '_' + seed + '_'
            frozen[qid, seed] = build_pair(qid, old[prefix + 'spatial'], old[prefix + 'pv_semantic'],
                                          old[prefix + 'semantic_penalty'])
    @torch.no_grad()
    def score_for(model, qid, seed, save=False, arm=None):
        d, s, x = frozen[qid, seed]
        a = model(x).cpu()
        score = convex_fusion(d, s, a, (896, 896))[0, 0].numpy()
        if save:
            learned[qid + '_' + seed + '_' + arm + '_alpha'] = a[0, 0].numpy()
        return score
    @torch.no_grad()
    def calibrate(model):
        model.eval()
        result = {}
        for seed in seeds:
            hist = np.zeros((2, 4096), np.int64)
            for row in calibration:
                qid = row['sample_id']
                hist += score_histogram(score_for(model, qid, seed), masks[qid][0, 0].numpy())
            result[seed] = calibrate_threshold(hist)
        return dict(by_seed=result, mean_iou=float(np.mean([v['calibration_micro_iou'] for v in result.values()])))
    fixed = CAMWeight('fixed').cuda().eval()
    baseline_cal = calibrate(fixed)
    assert {s: baseline_cal['by_seed'][s]['threshold'] for s in seeds} == config['loss_reference_thresholds']
    protocol['fixed_baseline_calibration'] = baseline_cal
    print(json.dumps(dict(phase='baseline_verified', calibration=baseline_cal, brightness_reference=ref)), flush=True)

    geo_paths = list(Path('/kaggle/input').rglob('RS5M_ViT-B-32.pt'))
    sat_paths = [p for p in Path('/kaggle/input').rglob('model.safetensors') if p.parent.name == 'verified_sat']
    assert len(geo_paths) == len(sat_paths) == 1
    assert sha256(geo_paths[0]) == GEO_SHA and sha256(sat_paths[0]) == SAT_SHA
    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, force_quick_gelu=True)
    state = torch.load(geo_paths[0], map_location='cpu', weights_only=True)
    clip.load_state_dict({k.removeprefix('module.'): v for k, v in state.get('state_dict', state).items()}, strict=True)
    del state
    clip.requires_grad_(False).cuda().eval()
    banks = encode_concepts(clip, open_clip.get_tokenizer('ViT-B-32'), device='cuda')
    with torch.no_grad():
        banks['pv'] = F.normalize(clip.encode_text(open_clip.get_tokenizer('ViT-B-32')(
            list(DEFAULT_CONCEPT_PROMPTS['pv'])).cuda()).float(), dim=-1)
    contrast = GreenhouseContrastPrior(banks).cuda().eval()
    boxes = [(x, y, x+224, y+224) for y in range(0, 673, 112) for x in range(0, 673, 112)]
    train_cache = {}
    with torch.no_grad():
        for i, row in enumerate(train):
            sid = row['sample_id']
            image = load_pil(root, row)
            crops = torch.stack([preprocess(image.crop(b)) for b in boxes])
            vectors = torch.cat([F.normalize(clip.encode_image(b.cuda()).float(), dim=-1) for b in crops.split(32)])
            geo = aligned_crop_features(vectors, boxes, (896, 896), (56, 56))
            maps = contrast(geo, torch.zeros(1, 1, 56, 56, device='cuda'))
            train_cache[sid] = {k: maps[k][0, 0].cpu().numpy() for k in ('pv_semantic', 'semantic_penalty')}
            print(json.dumps(dict(phase='extract_train_geo', done=i+1, total=len(train), sample_id=sid)), flush=True)
    del clip, crops, vectors, geo, maps, contrast, banks
    gc.collect()
    torch.cuda.empty_cache()
    sat = FrozenSAT(sat_paths[0], img_size=896).cuda().eval()
    with torch.no_grad():
        for i, row in enumerate(train):
            sid = row['sample_id']
            rgb = torch.from_numpy(np.array(load_pil(root, row))).permute(2, 0, 1)[None].float()/255
            train_cache[sid]['dino'] = sat(rgb.cuda()).cpu()
            print(json.dumps(dict(phase='extract_train_dino', done=i+1, total=len(train), sample_id=sid)), flush=True)
    del sat
    gc.collect()
    torch.cuda.empty_cache()
    prototypes = {sid: foreground_background_prototypes(train_cache[sid]['dino'], masks[sid])
                  for sid in allocation['positive_query_ids']}
    train_pairs = {}
    for qid, candidates in allocation['support_candidates'].items():
        for sid in candidates:
            q = train_cache[qid]
            d = foreground_background_response(q['dino'], *prototypes[sid], (56, 56))[0, 0].numpy()
            dt, st, x = build_pair(qid, d, q['pv_semantic'], q['semantic_penalty'])
            train_pairs[qid, sid] = dt.cuda(), st.cuda(), x
    del train_cache, prototypes
    gc.collect()
    results = {'fixed': dict(step=0, parameters=0, calibration=baseline_cal)}
    reference = float(np.mean(list(config['loss_reference_thresholds'].values())))
    for arm in config['arms']:
        torch.manual_seed(config['seed'])
        model = CAMWeight(arm, config['initial_alpha'], config['hidden']).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        start = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        best, history, first_grad = -1., [], None
        path = output / (arm + '_selected.pt')
        for step, (qid, sid) in enumerate(schedule, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            d, s, x = train_pairs[qid, sid]
            alpha = model(x)
            score = convex_fusion(d, s, alpha, (896, 896))
            seg = segmentation_loss(config['loss_scale']*(score-reference), masks[qid].cuda())
            anchor, tv = weight_regularization(alpha, config['initial_alpha'])
            loss = seg + config['anchor_weight']*anchor + config['tv_weight']*tv
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if step == 1:
                first_grad = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters()}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step, query=qid, support=sid, loss=float(loss.detach()),
                segmentation=float(seg.detach()), anchor=float(anchor.detach()), tv=float(tv.detach()),
                grad_norm=float(norm), alpha_mean=float(alpha.detach().mean())))
            if step % 40 == 0:
                print(json.dumps(dict(phase='training', arm=arm, **history[-1])), flush=True)
            if step % config['selection_every'] == 0:
                cal = calibrate(model)
                history[-1]['calibration'] = cal
                if cal['mean_iou'] > best:
                    best = cal['mean_iou']
                    torch.save(dict(model=model.state_dict(), arm=arm, step=step, calibration=cal), path)
                write_json(output / (arm + '_history.json'), history)
                print(json.dumps(dict(phase='checkpoint', arm=arm, step=step, calibration=cal, best=best)), flush=True)
        selected = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'])
        model.eval()
        results[arm] = dict(step=selected['step'], calibration=selected['calibration'],
            parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha256(path),
            gradient_first_step=first_grad,
            parameter_delta_l2={n: float((p.detach().cpu()-start[n]).norm()) for n, p in model.named_parameters()})
        for row in calibration + queries:
            for seed in seeds:
                score_for(model, row['sample_id'], seed, save=True, arm=arm)
        del model, optimizer
    # Both models and every threshold are locked before reading any query mask.
    protocol['arms'] = results
    write_json(output / 'models_locked_before_query_masks.json', dict(arms=results, config=config,
        source=protocol['source'], query_masks_parsed=False))
    for sid, item in brightness.items():
        if sid in protocol['calibration_ids'] + protocol['query_ids']:
            for name in ('raw', 'z', 'gate'):
                learned[sid + '_' + name] = item[name]
    np.savez_compressed(output / 'weight_grids.npz', **learned)
    records = []
    annotations = {r['sample_id']: r for r in selection['records']}
    for row in queries:
        qid = row['sample_id']
        target = read_mask(qid)[0, 0].numpy()
        record = dict(sample_id=qid, annotation=annotations[qid], source_scene=row['source_scene'],
                      foreground_pixels=int(target.sum()), methods={}, alpha={})
        for seed in seeds:
            d, s, _ = frozen[qid, seed]
            record['methods'][seed] = {mode: {} for mode in ('fixed', 'recalibrated')}
            record['alpha'][seed] = {}
            for arm in ('fixed', 'global', 'local'):
                a = np.full((56, 56), .7, np.float32) if arm == 'fixed' else learned[qid+'_'+seed+'_'+arm+'_alpha']
                score = convex_fusion(d, s, as_tensor(a), (896, 896))[0, 0].numpy()
                record['alpha'][seed][arm] = dict(mean=float(a.mean()), std=float(a.std()),
                    min=float(a.min()), max=float(a.max()), q10=float(np.quantile(a,.1)), q90=float(np.quantile(a,.9)))
                for mode in ('fixed', 'recalibrated'):
                    t = results['fixed' if mode == 'fixed' else arm]['calibration']['by_seed'][seed]['threshold']
                    record['methods'][seed][mode][arm] = dict(metrics(confusion(score, target, t)), threshold=t)
        records.append(record)
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'metrics.json', dict(protocol=protocol, records=records))
    write_json(output / 'completion.json', dict(status='complete', queries=len(records), training_images=len(train),
        query_masks_parsed_after_selection=True, arms=list(results)))
    print(json.dumps(dict(phase='COMPLETE', arms=results)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/dynamic_brightness'))
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
