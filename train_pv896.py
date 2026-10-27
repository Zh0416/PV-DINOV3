"""Authorized pilot only: at most 60 pairs, 3 epochs, decoder training."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import time
import traceback

import torch
from torch.utils.data import DataLoader

from pvseg.data import CATEGORIES, PV896Dataset, discover_root, read_manifest
from pvseg.losses import binary_loss
from pvseg.metrics import BinaryCounts
from pvseg.model import FrozenSATSegDINO, resolve_weights, HF_REPOSITORY, HF_REVISION
from pvseg.spatial import spatial_pilot_split
from pvseg.runtime import (Precision, decoder_optimizer, environment, load_checkpoint, memory,
                           save_checkpoint, save_prediction, seed_everything, sync, timed, write_json)


class Scores:
    def __init__(self):
        self.overall = BinaryCounts()
        self.categories = {c: BinaryCounts() for c in CATEGORIES.values()}
        self.losses = {c: [] for c in CATEGORIES.values()}
        self.samples = []

    def update(self, batch, logits, targets, loss):
        self.overall.update(logits, targets)
        # The guarded batch size is one, so combined loss is sample-specific.
        for i, category in enumerate(batch['category']):
            self.categories[category].update(logits[i:i+1], targets[i:i+1])
            self.losses[category].append(float(loss.detach()))
            single = BinaryCounts()
            single.update(logits[i:i+1], targets[i:i+1])
            self.samples.append({'sample_id': batch['sample_id'][i], 'category': category,
                'source_scene': batch['source_scene'][i],
                'source_label_uncertain_filename': bool(batch['uncertain_source'][i]),
                'loss': float(loss.detach()), **single.report()})

    def report(self):
        losses = [v for values in self.losses.values() for v in values]
        return {'loss': sum(losses)/len(losses), 'overall': self.overall.report(),
            'per_category': {c: {**m.report(), 'loss': sum(self.losses[c])/len(self.losses[c]) if self.losses[c] else None}
                             for c, m in self.categories.items()}, 'samples': self.samples}


def evaluate(model, loader, device, precision, output):
    model.eval()
    scores = Scores()
    with torch.no_grad():
        for batch in loader:
            targets = batch['mask'].to(device)
            with precision.context():
                logits = model(batch['image'].to(device))
                loss = binary_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite validation loss')
            scores.update(batch, logits, targets, loss)
            save_prediction(output, batch, logits, model.metadata['normalization'], smoke=False)
    return scores.report()


def run(args):
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot requires an allocated CUDA GPU')
    device = torch.device('cuda:0')
    software = environment()
    write_json(args.output_dir/'environment.json', software)
    root = args.data_root or discover_root(args.input_root)
    train_rows, val_rows, split_report = spatial_pilot_split(read_manifest(root), CATEGORIES, args.seed)
    selected = train_rows + val_rows
    if len(selected) > 60 or any(n > 20 for n in Counter(r['category'] for r in selected).values()):
        raise AssertionError('Pilot sample cap exceeded')
    config.update(data_root=str(root), splits={'train': train_rows, 'validation': val_rows},
        counts={s: dict(Counter(CATEGORIES[r['category']] for r in rows))
                for s, rows in [('train', train_rows), ('validation', val_rows)]},
        decoder_initialization='Fresh seed=42 decoder, not resumed from one-step Smoke decoder',
        loss={'BCEWithLogitsLoss': .5, 'soft_Dice': .5},
        source_uncertainty_policy='Preserve masks; source SHP filename flag only, never an ignore label',
        metrics=('sigmoid>=0.5; foreground micro metrics from cumulative TP/FP/FN; zero denominators -> null. '
                 'Empty truth and prediction does not receive perfect IoU. Negative FP contributes to global FP. '
                 'Missed positive image: nonempty truth with zero true-positive pixels. '
                 'Training metrics use augmented pre-step predictions; validation has no augmentation.'))
    config['missing_categories'] = {s: sorted(set(CATEGORIES.values())-set(counts)) for s, counts in config['counts'].items()}
    write_json(args.output_dir/'spatial_split.json', split_report)
    write_json(args.output_dir/'selected_sample_ids.json', {s: [r['sample_id'] for r in rows] for s, rows in config['splits'].items()})
    for split, rows in config['splits'].items():
        with (args.output_dir/f'{split}_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['split', *rows[0]])
            writer.writeheader()
            writer.writerows({'split': split, **r} for r in rows)
    write_json(args.output_dir/'config.json', config)
    (weights, weight_config, source), weight_seconds = timed(lambda: resolve_weights(args.weights, args.download_weights, str(args.weight_cache)))
    if source['sha256'] != '8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286':
        raise ValueError('Weight SHA256 differs from verified Smoke Test')
    source.update(verified_repository=HF_REPOSITORY, verified_revision=HF_REVISION)
    if weight_config['pretrained_cfg']['mean'] != [.430, .411, .296] or weight_config['pretrained_cfg']['std'] != [.213, .156, .143]:
        raise ValueError('Normalization differs from verified Smoke Test')
    model, load_seconds = timed(lambda: FrozenSATSegDINO(weights, weight_config, source, args.img_size).to(device))
    options = dict(root=root, img_size=args.img_size, image_dir=args.image_dir, mask_dir=args.mask_dir, **model.metadata['normalization'])
    audit_dataset = PV896Dataset(rows=selected, augment=False, **options)
    audit = []
    for index, row in enumerate(selected):
        sample = audit_dataset[index]
        audit.append({'sample_id': row['sample_id'], 'category': sample['category'], 'source_scene': row['source_scene'],
                      'source_label_uncertain_filename': row['source_label_uncertain_filename'],
                      'image_shape': list(sample['image'].shape), 'mask_shape': list(sample['mask'].shape),
                      'foreground_fraction': float(sample['mask'].mean()), 'status': 'PASS'})
    write_json(args.output_dir/'selected_pixel_audit.json', audit)
    train_loader = DataLoader(PV896Dataset(rows=train_rows, augment=True, **options), batch_size=1, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(PV896Dataset(rows=val_rows, augment=False, **options), batch_size=1, shuffle=False, num_workers=0)
    optimizer = decoder_optimizer(model, args.lr)
    precision = Precision(device)
    config.update(precision=precision.describe(), model=model.metadata, optimizer={'name': 'AdamW', 'lr': args.lr, 'weight_decay': 1e-4})
    write_json(args.output_dir/'config.json', config)
    backbone_versions = [p._version for p in model.backbone.parameters()]
    torch.cuda.reset_peak_memory_stats()
    best_iou, best_epoch, history, steps = -1, None, [], 0
    for epoch in range(1, args.epochs+1):
        sync()
        epoch_started = time.perf_counter()
        scores = Scores()
        model.train()
        if model.backbone.training:
            raise AssertionError('Backbone must remain eval')
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            targets = batch['mask'].to(device)
            with precision.context():
                logits = model(batch['image'].to(device))
                loss = binary_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            precision.scaler.scale(loss).backward()
            precision.scaler.unscale_(optimizer)
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise AssertionError('Frozen backbone acquired gradients')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.decoder.parameters()):
                raise RuntimeError('Missing or nonfinite decoder gradient')
            old_scale = precision.scaler.get_scale()
            precision.scaler.step(optimizer)
            precision.scaler.update()
            if precision.scaler.get_scale() < old_scale:
                raise RuntimeError('AMP skipped optimizer step')
            scores.update(batch, logits, targets, loss)
            steps += 1
        sync()
        train_seconds = time.perf_counter()-epoch_started
        validation, val_seconds = timed(lambda: evaluate(model, val_loader, device, precision, args.output_dir/f'epoch_{epoch}_predictions'))
        if backbone_versions != [p._version for p in model.backbone.parameters()]:
            raise AssertionError('Backbone parameters changed')
        train = scores.report()
        record = {'epoch': epoch, 'train_loss': train['loss'], 'train': train, 'validation': validation,
            'train_seconds': train_seconds, 'validation_seconds': val_seconds, 'epoch_seconds': time.perf_counter()-epoch_started,
            'optimizer_steps_cumulative': steps, 'gpu_memory': memory(), 'backbone_parameters_unchanged': True,
            'backbone_gradients': 0, 'backbone_eval_during_training': True}
        history.append(record)
        write_json(args.output_dir/'history.json', history)
        with (args.output_dir/'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'train_loss', 'validation_loss', 'iou', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn'])
            for item in history:
                writer.writerow([item['epoch'], item['train_loss'], item['validation']['loss'],
                                 *[item['validation']['overall'][k] for k in ['iou', 'f1', 'precision', 'recall', 'tp', 'fp', 'fn']]])
        iou = validation['overall']['iou']
        if iou is not None and iou > best_iou:
            best_iou, best_epoch = iou, epoch
            save_checkpoint(args.output_dir/'best_decoder.pt', model, optimizer, config, software, epoch=epoch,
                best_validation_iou=best_iou, validation=validation, spatial_split=split_report, scaler_state=precision.scaler.state_dict())
            write_json(args.output_dir/'best_checkpoint_metadata.json', {'epoch': epoch, 'validation': validation,
                       'model': model.metadata, 'path': str(args.output_dir/'best_decoder.pt')})
        print({'epoch': epoch, 'train_loss': train['loss'], 'validation_loss': validation['loss'],
               'metrics': validation['overall'], 'epoch_seconds': record['epoch_seconds']}, flush=True)
    if best_epoch is None:
        raise RuntimeError('No defined validation IoU; no best checkpoint')
    load_checkpoint(args.output_dir/'best_decoder.pt', model)
    best_validation = evaluate(model, val_loader, device, precision, args.output_dir/'best_predictions')
    if best_validation != history[best_epoch-1]['validation']:
        raise AssertionError('Reloaded best-checkpoint metrics differ from saved epoch')
    write_json(args.output_dir/'completion.json', {'status': 'COMPLETE', 'epochs': args.epochs, 'total_unique_pairs': len(selected),
        'train_pairs': len(train_rows), 'validation_pairs': len(val_rows), 'optimizer_steps': steps,
        'best_epoch': best_epoch, 'best_validation': best_validation, 'best_checkpoint_reload_verified': True,
        'weights_seconds': weight_seconds, 'model_load_seconds': load_seconds, 'total_seconds': time.perf_counter()-started,
        'gpu_memory': memory(), 'software': software, 'weight_source': source, 'failures': [],
        'stop_reason': 'Authorized pilot finished; no expansion or full training without separate user confirmation'})
    print('STOP: bounded pilot complete; await user review before further training.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow_pilot', action='store_true')
    parser.add_argument('--data_root', type=Path)
    parser.add_argument('--input_root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--image_dir', type=Path)
    parser.add_argument('--mask_dir', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--download_weights', action='store_true')
    parser.add_argument('--weight_cache', type=Path, default=Path('/kaggle/working/model_cache'))
    parser.add_argument('--img_size', type=int, default=896)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=Path, default=Path('/kaggle/working/PV-DINOV3/outputs/pilot'))
    args = parser.parse_args()
    if not args.allow_pilot or not 1 <= args.epochs <= 3 or args.batch_size != 1 or args.img_size != 896 or args.seed != 42 or args.lr != 1e-4:
        parser.error('Pilot guard: --allow_pilot, epochs 1..3, batch_size=1, img_size=896, seed=42, lr=1e-4')
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir/'failure.json', {'status': 'FAILED', 'error': str(error), 'traceback': traceback.format_exc(), 'gpu_memory': memory()})
        raise


if __name__ == '__main__':
    main()
