"""Bounded training-set overfit diagnostic, NOT an accuracy experiment.

Use only on Kaggle. Real data/weights/checkpoints stay there. Original 896px
tiles are resized to 512px, with no augmentation or change to objective/model.
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/kaggle/working/sspnet_convergence')
    p.add_argument('--base-steps', type=int, default=320)
    p.add_argument('--episode-steps', type=int, default=480)
    args = p.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run on Kaggle; do not download data or weights locally')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'], check=True)
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torch.nn import functional as F
    from pvseg.data import CATEGORIES, discover_root, read_manifest
    from pvseg.metrics import BinaryCounts
    from pvseg.runtime import Precision, environment, memory, seed_everything, write_json
    from pvseg.sspnet import SSPNet, checkpoint
    from pvseg.sspnet.data import Episodes, ResizedPV, make_split
    from pvseg.sspnet.encoders import FrozenSAT, PSPNet, freeze_bn, load_text_cache, sha256
    from pvseg.sspnet.losses import objective

    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required')
    torch.set_num_threads(2)
    seed_everything(42)
    precision = Precision('cuda')
    write_json(output / 'environment.json', environment())
    with (output / 'tests.txt').open('w') as stream:
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                              cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    print((output / 'tests.txt').read_text(), flush=True)
    test.check_returncode()

    def find_asset(name, parent=None):
        matches = sorted(x for x in Path('/kaggle/input').rglob(name)
                         if parent is None or x.parent.name == parent)
        if not matches:
            raise FileNotFoundError(f'Mount SSPNet v3 outputs; asset missing: {name}')
        # The same verified asset may be present in multiple attached notebooks.
        return matches[0]

    weights = find_asset('model.safetensors', 'verified_sat')
    text_cache = find_asset('georsclip_pv_text.pt')
    root = discover_root('/kaggle/input')
    train_rows, _, split = make_split(read_manifest(root), pilot=True, seed=42)
    dataset = ResizedPV(root, train_rows, 512)
    by_id = {r['sample_id']: i for i, r in enumerate(train_rows)}
    selected, audit = [], []
    # Two ordinary positives, one small/hard positive and one negative, all from
    # the existing training split. No search using validation performance.
    for category, count in [('positive', 2), ('hard_positive', 1), ('negative', 1)]:
        pool = sorted((r for r in train_rows if CATEGORIES[r['category']] == category),
                      key=lambda r: r['sample_id'])
        for row in pool:
            item = dataset[by_id[row['sample_id']]]
            patch = F.interpolate(item['mask'][None], (32, 32), mode='nearest')
            if category != 'negative' and not patch.any():
                continue
            selected.append(row)
            raw = dataset.raw[by_id[row['sample_id']]]
            audit.append({'id': row['sample_id'], 'category': category, 'source_scene': row['source_scene'],
                'raw_pixels_896': int(raw['mask'].sum()), 'pixels_512': int(item['mask'].sum()),
                'positive_patches_32': int(patch.sum()),
                'raw_foreground_fraction': float(raw['mask'].mean()),
                'resized_foreground_fraction': float(item['mask'].mean())})
            count -= 1
            if not count:
                break
        if count:
            raise ValueError(f'Too few eligible diagnostic queries: {category}')
    queries = ResizedPV(root, selected, 512)
    episodes = Episodes(queries, dataset, split['spatial_groups'], shots=1, seed=42)
    items = [queries[i] for i in range(len(queries))]
    episode_items = [episodes[i] for i in range(len(episodes))]
    assignments = [{'query': e['query_id'], 'supports': e['support_ids'].split('|')} for e in episode_items]
    selected_ids = {r['sample_id'] for r in selected}
    assert selected_ids <= set(split['train_ids']) and not selected_ids & set(split['validation_ids'])
    write_json(output / 'data_audit.json', {'scope': 'training-set diagnostic; NOT validation/test accuracy',
        'source_size': 896, 'model_size': 512, 'augment': False, 'queries': audit,
        'fixed_support_assignments': assignments, 'original_split': split})

    def cuda_batch(item):
        return {k: v[None].cuda() for k, v in item.items() if isinstance(v, torch.Tensor)}

    base_batches = [cuda_batch(x) for x in items]
    epi_batches = [cuda_batch(x) for x in episode_items]

    def prediction_panels(stage, batches, predictions):
        folder = output / 'predictions'
        folder.mkdir(exist_ok=True)
        for row, batch, logits in zip(selected, batches, predictions):
            image = batch.get('query', batch.get('image'))[0]
            truth = batch.get('target', batch.get('mask'))[0, 0]
            rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
            pred = logits.argmax(1)[0].cpu().numpy().astype(bool)
            overlay = rgb.copy()
            overlay[pred] = (overlay[pred] * .5 + np.array([255, 30, 30]) * .5).astype('uint8')
            canvas = Image.new('RGB', (1024, 292), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), f'{stage} | {row["sample_id"]} | TRAIN overfit diagnostic ONLY', fill='black')
            arrays = [rgb, (truth.cpu().numpy() * 255).astype('uint8'), pred.astype('uint8') * 255, overlay]
            for col, (array, title) in enumerate(zip(arrays, ['RGB', 'Ground truth', 'Prediction', 'Overlay'])):
                mode = Image.Resampling.NEAREST if col in (1, 2) else Image.Resampling.LANCZOS
                canvas.paste(Image.fromarray(array).convert('RGB').resize((256, 256), mode), (256 * col, 36))
                draw.text((256 * col + 4, 20), title, fill='black')
            canvas.save(folder / f'{stage}_{row["sample_id"]}.png')

    @torch.no_grad()
    def measure(model, batches, episodic, label):
        model.eval()
        counts, meta_counts = BinaryCounts(), BinaryCounts()
        losses, per_query, predictions = [], [], []
        routes = {'near_tie': 0, 'total': 0}
        for row, batch in zip(selected, batches):
            with precision.context():
                if episodic:
                    result = model(batch['query'], batch['supports'], batch['support_masks'])
                    loss, terms = objective(result, batch['target'])
                    logits, target = result['logits'], batch['target']
                    meta_counts.update(result['meta_logits'][:, 1:2].float() - result['meta_logits'][:, :1].float(), target)
                    routes['near_tie'] += int(result['routing']['near_tie'].sum())
                    routes['total'] += result['routing']['near_tie'].numel()
                else:
                    logits, target = model(batch['image']), batch['mask']
                    loss = F.cross_entropy(logits.float(), target[:, 0].long())
                    terms = {'main_ce': loss}
            difference = logits[:, 1:2].float() - logits[:, :1].float()
            one = BinaryCounts()
            one.update(difference, target)
            counts.update(difference, target)
            per_query.append({'id': row['sample_id'], 'metrics': one.report(),
                              'terms': {k: float(v) for k, v in terms.items()}})
            losses.append(float(loss))
            predictions.append(logits.detach().float())
        report = {'label': label, 'loss': sum(losses) / len(losses), 'metrics': counts.report(),
                  'per_query': per_query, 'routing': routes}
        if episodic:
            report['meta_metrics'] = meta_counts.report()
        if label in ('before', 'after'):
            prediction_panels(('episodic_' if episodic else 'base_') + label, batches, predictions)
        return report

    def gradients(model, episodic):
        result = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise RuntimeError(f'Frozen parameter has a gradient: {name}')
                continue
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f'Nonfinite gradient: {name}')
                group = '.'.join(name.split('.')[:2]) if episodic else 'base'
                result[group] = result.get(group, 0.) + float(parameter.grad.float().square().sum())
        return {k: v ** .5 for k, v in result.items()}

    def update(model, batch, optimizer, episodic):
        model.train()
        if not episodic:
            freeze_bn(model)
        optimizer.zero_grad(set_to_none=True)
        with precision.context():
            if episodic:
                result = model(batch['query'], batch['supports'], batch['support_masks'])
                loss, terms = objective(result, batch['target'])
            else:
                loss = F.cross_entropy(model(batch['image']).float(), batch['mask'][:, 0].long())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        precision.scaler.scale(loss).backward()
        precision.scaler.unscale_(optimizer)
        audit = gradients(model, episodic)
        torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 10., error_if_nonfinite=True)
        precision.scaler.step(optimizer)
        precision.scaler.update()
        return float(loss.detach()), audit

    def resume_probe(model, optimizer, batch, episodic, step, history):
        path = output / ('episodic_resume_probe.pt' if episodic else 'base_resume_probe.pt')
        checkpoint.save(path, model, optimizer, precision, {'diagnostic': True}, 0, step, history=history)
        loss_a, _ = update(model, batch, optimizer, episodic)
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    if not (episodic and (k.startswith('dino.') or k.startswith('cnn.')))}
        scaler_a = precision.scaler.state_dict()
        rng_a = checkpoint.rng_state()
        restored = checkpoint.load(path, model, optimizer, precision, restore_random=True)
        if restored['history'] != history:
            raise RuntimeError('GPU resume history changed')
        del restored
        loss_b, _ = update(model, batch, optimizer, episodic)
        differences = {k: float((v.float() - model.state_dict()[k].detach().cpu().float()).abs().max())
                       for k, v in expected.items()}
        maximum = max(differences.values(), default=0.)
        rng_b = checkpoint.rng_state()
        same_rng = torch.equal(rng_a['torch'], rng_b['torch']) and all(
            torch.equal(a, b) for a, b in zip(rng_a['cuda'], rng_b['cuda']))
        report = {'loss_a': loss_a, 'loss_b': loss_b, 'max_abs_parameter_difference': maximum,
                  'exact_parameters': maximum == 0, 'same_scaler': scaler_a == precision.scaler.state_dict(),
                  'same_torch_rng': same_rng, 'history_preserved': True}
        report['passed'] = maximum <= 1e-6 and abs(loss_a - loss_b) <= 1e-6 and report['same_scaler'] and same_rng
        write_json(path.with_suffix('.json'), report)
        if not report['passed']:
            raise RuntimeError(f'GPU resume equivalence failed: {report}')
        path.unlink()
        return report, loss_b

    def run_stage(model, batches, episodic, maximum):
        stage = 'episodic' if episodic else 'base'
        optimizer = torch.optim.SGD([x for x in model.parameters() if x.requires_grad],
                                    lr=.005, momentum=.9, weight_decay=1e-4)
        reports = [measure(model, batches, episodic, 'before')]
        history, gradient_audits = [], []
        started = time.perf_counter()
        probe = None
        for step in range(1, maximum + 1):
            batch = batches[(step - 1) % len(batches)]
            if step == 21:
                probe, loss = resume_probe(model, optimizer, batch, episodic, step - 1, history)
                audit = {}
            else:
                loss, audit = update(model, batch, optimizer, episodic)
            history.append({'update': step, 'query': selected[(step - 1) % len(batches)]['sample_id'], 'loss': loss})
            if step in (1, 2, 40) or step == maximum:
                gradient_audits.append({'update': step, 'norms': audit})
            if step % 40 == 0 or step == maximum:
                report = measure(model, batches, episodic, f'update_{step}')
                report['update'] = step
                reports.append(report)
                print(json.dumps({'stage': stage, 'update': step, 'train_loss': report['loss'],
                    'train_iou': report['metrics']['iou'], 'meta_iou': report.get('meta_metrics', {}).get('iou')}), flush=True)
                write_json(output / f'{stage}_progress.json', {'reports': reports, 'gradient_audits': gradient_audits})
                iou = report['metrics']['iou'] or 0.
                meta_iou = report.get('meta_metrics', {}).get('iou') or 0.
                if step >= 120 and iou >= .70 and (not episodic or meta_iou >= .70) and \
                        report['metrics']['positive_images_without_true_positive_pixels'] == 0:
                    break
        final = measure(model, batches, episodic, 'after')
        reports.append(final)
        before, after = reports[0], final
        learned = (after['metrics']['iou'] or 0.) >= .60 and after['loss'] < before['loss'] and \
                  after['metrics']['positive_images_without_true_positive_pixels'] == 0
        if episodic:
            learned = learned and (after['meta_metrics']['iou'] or 0.) >= .60
        report = {'stage': stage, 'scope': 'fixed training-set overfit diagnostic only',
            'updates': step, 'maximum_updates': maximum, 'optimizer': 'SGD lr=.005 momentum=.9 wd=1e-4 constant LR',
            'batch_size': 1, 'augmentation': False, 'passed': learned, 'seconds': time.perf_counter() - started,
            'resume_probe': probe, 'reports': reports, 'training_history': history,
            'gradient_audits': gradient_audits, 'memory': memory()}
        if not episodic:
            model.provenance.update(segmentation_pretrained=True, engineering_only=True,
                diagnostic_only=True, split_sha256=split['split_sha256'], bn_policy='running_statistics_frozen')
        checkpoint.save(output / f'{stage}_diagnostic.pt', model, optimizer, precision,
            {'diagnostic': True, 'split_sha256': split['split_sha256'], 'size': 512}, 0, step, history=history)
        write_json(output / f'{stage}_report.json', report)
        return report

    cnn = PSPNet(True).cuda()
    base_report = run_stage(cnn, base_batches, False, args.base_steps)
    # Never label this diagnostic checkpoint as a fully trained stage-one model.
    del cnn
    gc.collect()
    torch.cuda.empty_cache()
    cnn = PSPNet(False)
    checkpoint.load(output / 'base_diagnostic.pt', cnn)
    cnn.provenance = dict(cnn.provenance, checkpoint_sha256=sha256(output / 'base_diagnostic.pt'))
    text, metadata = load_text_cache(text_cache)
    model = SSPNet(cnn, FrozenSAT(weights, 512), text, metadata).cuda()
    episodic_report = run_stage(model, epi_batches, True, args.episode_steps)
    del model, cnn
    gc.collect()
    torch.cuda.empty_cache()
    # Native-size compatibility is checked separately from 512px convergence.
    # This single update is NOT a native-size training/accuracy result.
    native_data = Episodes(ResizedPV(root, selected[:1], 896), ResizedPV(root, train_rows, 896),
                           split['spatial_groups'], shots=1, seed=42)
    native_item = native_data[0]
    native_batch = cuda_batch(native_item)
    cnn = PSPNet(False)
    checkpoint.load(output / 'base_diagnostic.pt', cnn)
    cnn.provenance = dict(cnn.provenance, checkpoint_sha256=sha256(output / 'base_diagnostic.pt'))
    native_model = SSPNet(cnn, FrozenSAT(weights, 896), text, metadata).cuda()
    checkpoint.load(output / 'episodic_diagnostic.pt', native_model)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD([p for p in native_model.parameters() if p.requires_grad], lr=0.)
    native_loss, native_gradients = update(native_model, native_batch, optimizer, True)
    native_report = {'passed': True, 'size': 896, 'query': native_item['query_id'],
        'supports': native_item['support_ids'].split('|'), 'loss': native_loss,
        'gradient_norms': native_gradients, 'memory': memory(),
        'scope': 'single real-weight native-size forward/backward with lr=0; NOT accuracy validation'}
    write_json(output / 'native_896_probe.json', native_report)
    completion = {'passed': base_report['passed'] and episodic_report['passed'],
        'base_passed': base_report['passed'], 'episodic_passed': episodic_report['passed'],
        'base_before': base_report['reports'][0], 'base_after': base_report['reports'][-1],
        'episodic_before': episodic_report['reports'][0], 'episodic_after': episodic_report['reports'][-1],
        'scope': 'memorization/gradient/resume diagnostic; NOT validation/test or paper performance',
        'size': 512, 'source_size': 896, 'native_896_probe': native_report, 'memory': memory()}
    write_json(output / 'completion.json', completion)
    print('SSPNET CONVERGENCE DIAGNOSTIC ' + ('PASSED' if completion['passed'] else 'NEEDS INVESTIGATION'), flush=True)


if __name__ == '__main__':
    main()
