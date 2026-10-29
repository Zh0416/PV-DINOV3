"""Training-only paired AMP/FP32 collapse diagnosis using the learned-A3 base.

No calibration/query image or label is read. All original assets stay on Kaggle.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(args):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run on Kaggle; no local data/weight download')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29'], check=True)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from pvseg.data import PV896Dataset, discover_root, read_manifest
    from pvseg.runtime import Precision, environment, seed_everything, write_json
    from pvseg.sspnet import SSPNet, checkpoint
    from pvseg.sspnet.diagnostics import binary_report, tensor_stats, pooled_reports, gradient_norms
    from pvseg.sspnet.encoders import PSPNet, FrozenSAT, load_text_cache, sha256
    from pvseg.sspnet.learned_protocol import training_plan
    from pvseg.sspnet.losses import objective as paper_objective
    from scripts.kaggle_a0a1 import plan_samples

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    seed_everything(42)
    precision = Precision('cuda')
    with (out / 'tests.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def find(name, parent=None):
        found = [p for p in Path('/kaggle/input').rglob(name) if parent is None or p.parent.name == parent]
        if len(found) != 1:
            raise ValueError(f'Expected unique {parent}/{name}, found {len(found)}')
        return found[0]

    previous = find('completion.json', 'a3_learned').parent
    previous_env = json.loads((previous / 'environment.json').read_text())
    if previous_env['source_commit'] != 'a6733794eb2fc2cf43c762ecf62952c5b1e4b15c':
        raise ValueError('Mount learned A3 version 2')
    rows = read_manifest(discover_root('/kaggle/input'))
    selection = json.loads((REPO / 'configs/a0a1_selected_20261029.json').read_text(encoding='utf-8'))
    _, _, _, _, audit = plan_samples(rows, selection)
    training, allocation = training_plan(rows, selection, audit)
    old_allocation = json.loads((previous / 'training_allocation.json').read_text())
    if allocation != old_allocation:
        raise ValueError('Prior training allocation differs')
    query_ids = allocation['positive_query_ids'] + allocation['negative_query_ids'][:1]
    pairs = [(qid, allocation['support_candidates'][qid][0]) for qid in query_ids]
    active_ids = set(query_ids) | {sid for _, sid in pairs}
    active = [r for r in training if r['sample_id'] in active_ids]
    root = discover_root('/kaggle/input')
    dataset = PV896Dataset(root, active, [0, 0, 0], [1, 1, 1])
    items = {r['sample_id']: dataset[i] for i, r in enumerate(active)}
    for r in active:
        for kind in ('image', 'label'):
            if sha256(root / r[kind]) != r[kind + '_sha256']:
                raise ValueError('Hash mismatch: ' + r['sample_id'])
    # Round-robin positives and the fixed negative, identical in both arms.
    positive_pairs, negative_pair = pairs[:-1], pairs[-1]
    schedule = [positive_pairs[(i // 2) % len(positive_pairs)] if i % 2 == 0 else negative_pair
                for i in range(args.steps)]
    arms = [('balanced_meta_ce_fp32', False)] if args.meta_ce_weight else [
        ('original_amp', True), ('original_fp32', False)]
    reference = None
    if args.meta_ce_weight:
        reference = find('summary.json', 'a3_diagnosis').parent
        old_protocol = json.loads((reference / 'protocol.json').read_text())
        old_summary = json.loads((reference / 'summary.json').read_text())
        if old_summary['source_commit'] != 'd399b38f64aa3aec98e1990ffb75ca6aceae5fa3':
            raise ValueError('Expected the completed original paired diagnosis v1')
        if old_protocol['pairs'] != [list(pair) for pair in pairs] or old_protocol['schedule'] != [
                list(pair) for pair in schedule[:old_protocol['steps']]]:
            raise ValueError('Repair and original diagnosis pairs/schedule differ')

    def objective(result, mask):
        return paper_objective(result, mask, meta_ce_weight=args.meta_ce_weight)

    write_json(out / 'protocol.json', dict(scope='TRAINING-ONLY engineering diagnosis; NOT validation/test',
        input_size=896, pairs=pairs, schedule=schedule, split_sha256=allocation['split_sha256'],
        steps=args.steps, lr=.0001, optimizer='AdamW', weight_decay=.0001, clip=10.,
        arms=[arm for arm, _ in arms], same_initial_state=True, same_rng=True,
        model_and_objective=('unchanged learned A3' if not args.meta_ce_weight else
            'unchanged model; paper objective + explicit balanced raw-meta CE adaptation'),
        balanced_meta_ce_weight=args.meta_ce_weight, query_or_calibration_images_read=False,
        previous_source=previous_env['source_commit']))
    write_json(out / 'environment.json', environment())
    base = PSPNet(imagenet=False).cuda()
    checkpoint.load(previous / 'base.pt', base)
    dino = FrozenSAT(find('model.safetensors', 'verified_sat'), img_size=896).cuda()
    text, text_meta = load_text_cache(find('georsclip_pv_text.pt'))
    seed_everything(42)
    net = SSPNet(base, dino, text, text_meta).cuda()
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()
               if not k.startswith(('cnn.', 'dino.'))}
    initial_rng = checkpoint.rng_state()
    cache_dir = out / 'feature_cache'
    cache_dir.mkdir()
    move = lambda x, device: x.to(device) if isinstance(x, torch.Tensor) else {
        k: move(v, device) for k, v in x.items()} if isinstance(x, dict) else tuple(move(v, device) for v in x)
    net.eval()
    with torch.no_grad():
        for sid in sorted(active_ids):
            rgb = items[sid]['image'][None].cuda()
            feature = dict(cnn=base.features(rgb), dino=dino(rgb))
            torch.save(move(feature, 'cpu'), cache_dir / (sid + '.pt'))
            print(json.dumps(dict(phase='cache', sample=sid)), flush=True)
    del feature, rgb
    # Five pairs reuse five feature maps. Keep them in host RAM; load one pair per step.
    features = {sid: torch.load(cache_dir / (sid + '.pt'), weights_only=True) for sid in active_ids}
    write_json(out / 'feature_stats.json', {sid: dict(mid=tensor_stats(f['cnn']['mid']),
        base_logits=tensor_stats(f['cnn']['base_logits']), dino=tensor_stats(f['dino'])) for sid, f in features.items()})

    def forward(pair):
        qid, sid = pair
        q, s = move(features[qid], 'cuda'), move(features[sid], 'cuda')
        return net.forward_from_features(q['cnn'], q['dino'], [s['cnn']], [s['dino']],
            [items[sid]['mask'][None].cuda()], (896, 896))

    activations, handles = {}, []
    for name in ('down_q', 'down_s', 'query_merge', 'support_merge', 'dcma', 'decoder', 'meta_head'):
        def hook(module, inputs, result, name=name):
            if recording[0]:
                activations[name] = tensor_stats(result[0] if isinstance(result, tuple) else result)
        handles.append(getattr(net, name).register_forward_hook(hook))
    recording = [False]

    def panel(label, qid, output):
        rgb = (items[qid]['image'].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
        truth = items[qid]['mask'][0].numpy() > .5
        arrays = [rgb, truth.astype('uint8') * 255]
        names = ['RGB', 'Training label']
        for key, title in [('base_logits', 'Frozen base'), ('meta_logits', 'Meta argmax'), ('logits', 'Final argmax')]:
            arrays.append((output[key].argmax(1)[0].detach().cpu().numpy().astype('uint8') * 255))
            names.append(title)
        canvas = Image.new('RGB', (1120, 260), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (array, title) in enumerate(zip(arrays, names)):
            canvas.paste(Image.fromarray(array).convert('RGB').resize((224, 224),
                Image.Resampling.BILINEAR if i == 0 else Image.Resampling.NEAREST), (224 * i, 20))
            draw.text((224 * i + 3, 4), title, fill='black')
        draw.text((3, 247), f'{label} | {qid} | TRAIN DIAGNOSIS ONLY | native896', fill='black')
        folder = out / 'predictions'
        folder.mkdir(exist_ok=True)
        canvas.save(folder / f'{label}_{qid}.png')

    def measure(label, amp, save_panels=False):
        net.eval()
        records = []
        with torch.no_grad():
            for pair in pairs:
                recording[0] = True
                activations.clear()
                context = precision.context() if amp else nullcontext()
                with context:
                    result = forward(pair)
                    loss, terms = objective(result, items[pair[0]]['mask'][None].cuda())
                recording[0] = False
                target = items[pair[0]]['mask'][None].cuda()
                record = dict(query=pair[0], support=pair[1], terms={k: float(v) for k, v in terms.items()},
                    activations=dict(activations), outputs={name: binary_report(result[key], target)
                    for name, key in [('base', 'base_logits'), ('meta', 'meta_logits'), ('final', 'logits')]},
                    pseudo=tensor_stats(result['pseudo_logits']), gamma=float(result['gamma']))
                records.append(record)
                if save_panels:
                    panel(label, pair[0], result)
        pooled = {name: pooled_reports([r['outputs'][name] for r in records]) for name in ('base', 'meta', 'final')}
        report = dict(label=label, per_image=records, pooled=pooled,
                      loss=float(np.mean([r['terms']['total'] for r in records])))
        write_json(out / (label + '.json'), report)
        print(json.dumps(dict(phase='measurement', label=label, pooled=pooled, loss=report['loss'])), flush=True)
        return report

    def loss_gradients(label, amp):
        net.eval()
        optimizer_params = dict(meta_head=net.meta_head[-1].weight, decoder=net.decoder.reduce[0].weight,
            attention_q=net.dcma.q.weight, attention_v=net.dcma.v.weight,
            visual=net.dmsf.visual.net[0].weight, text=net.dmsf.text_to_dino[0].weight, beta=net.dmsf.beta)
        with precision.context() if amp else nullcontext():
            result = forward(pairs[0])
            _, terms = objective(result, items[pairs[0][0]]['mask'][None].cuda())
        report = {}
        for term, value in terms.items():
            gradients = torch.autograd.grad(value, list(optimizer_params.values()), retain_graph=True, allow_unused=True)
            report[term] = {key: None if g is None else tensor_stats(g)
                           for key, g in zip(optimizer_params, gradients)}
        write_json(out / (label + '_loss_gradients.json'), report)

    # Inspect the saved failed checkpoint in both numerical modes, then restart
    # from identical random downstream weights for the paired learning probe.
    checkpoint.load(previous / 'selected.pt', net)
    measure('selected_amp', True, True)
    loss_gradients('selected_amp', True)
    measure('selected_fp32', False)
    loss_gradients('selected_fp32', False)
    all_history, conclusions = {}, {}
    for arm, amp in arms:
        net.load_state_dict(initial, strict=False)
        checkpoint.restore_rng(initial_rng)
        scaler = torch.amp.GradScaler('cuda', enabled=amp and precision.scaler.is_enabled())
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.0001, weight_decay=.0001)
        history, measurements = [], [measure(arm + '_step000', amp, True)]
        if reference:
            # Verify the initialization against saved measurements, before any
            # repair update. The comparison ignores losses because they differ.
            old = json.loads((reference / 'original_fp32_step000.json').read_text())
            differences = []
            for before, current in zip(old['per_image'], measurements[0]['per_image']):
                if (before['query'], before['support']) != (current['query'], current['support']):
                    raise ValueError('Initialization pair ordering differs')
                for name in ('base', 'meta', 'final'):
                    for key in ('mean', 'std', 'minimum', 'maximum'):
                        a, b = before['outputs'][name]['logit_margin'][key], current['outputs'][name]['logit_margin'][key]
                        differences.append(abs(a - b))
                        if not np.isclose(a, b, rtol=1e-4, atol=2e-5):
                            raise ValueError('Original/repair initialization mismatch')
            write_json(out / 'initialization_comparison.json', dict(
                max_statistic_absolute_difference=max(differences), reference_source=old_summary['source_commit'],
                same_pairs_and_schedule=True, tolerance=dict(rtol=1e-4, atol=2e-5)))
        loss_gradients(arm + '_step000', amp)
        started = time.monotonic()
        for step, pair in enumerate(schedule, 1):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            with precision.context() if amp else nullcontext():
                result = forward(pair)
                loss, terms = objective(result, items[pair[0]]['mask'][None].cuda())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = gradient_norms(net)
            norm = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 10., error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise RuntimeError('Skipped AMP update')
            history.append(dict(step=step, query=pair[0], support=pair[1], terms={k: float(v.detach()) for k, v in terms.items()},
                gradients=gradients, norm_before_clip=float(norm), meta=binary_report(result['meta_logits'], items[pair[0]]['mask'][None].cuda()),
                final=binary_report(result['logits'], items[pair[0]]['mask'][None].cuda())))
            del result, loss, terms
            if step in (1, 5, 10, 20, 40, 80, 120, 160, args.steps):
                rng = checkpoint.rng_state()
                measurements.append(measure(f'{arm}_step{step:03d}', amp, step == args.steps))
                if step in (20, args.steps):
                    loss_gradients(f'{arm}_step{step:03d}', amp)
                checkpoint.restore_rng(rng)
                write_json(out / (arm + '_history.json'), history)
        conclusions[arm] = dict(initial=measurements[0], final=measurements[-1], elapsed=time.monotonic() - started)
        all_history[arm] = history
        checkpoint.save(out / (arm + '.pt'), net, optimizer, SimpleNamespace(scaler=scaler),
            dict(diagnostic_arm=arm, meta_ce_weight=args.meta_ce_weight, amp=amp), 0, args.steps, history=history)
        del optimizer
        net.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    for h in handles:
        h.remove()
    write_json(out / 'summary.json', dict(arms=conclusions, scope='training-only diagnosis',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()))
    del features
    for path in cache_dir.iterdir():
        path.unlink()
    cache_dir.rmdir()
    write_json(out / 'completion.json', dict(status='complete', steps_per_arm=args.steps,
        calibration_or_query_used=False, claim='Diagnosis finished; metrics determine whether training succeeded'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/a3_diagnosis'))
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--meta-ce-weight', type=float, default=0.,
        help='Explicit PV adaptation; >0 runs only FP32 against mounted original diagnosis')
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.meta_ce_weight < 0:
        parser.error('steps must be positive; meta-ce-weight must be nonnegative')
    try:
        main(arguments)
    except Exception:
        if arguments.output.is_dir():
            (arguments.output / 'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc())), encoding='utf-8')
        raise
