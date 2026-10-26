"""Explicitly authorized decoder-only pilot: <=60 pairs, <=3 epochs. Never full training."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pvseg.data import CATEGORIES, PV896Dataset, discover_root, pilot_split, read_manifest
from pvseg.losses import binary_loss
from pvseg.metrics import BinaryCounts
from pvseg.model import FrozenSATSegDINO, resolve_weights
from pvseg.runtime import (Precision, decoder_optimizer, environment, save_checkpoint,
                           save_prediction, seed_everything, write_json)


def evaluate(model, loader, device, precision):
    model.eval()
    overall = BinaryCounts()
    per_category = {c: BinaryCounts() for c in CATEGORIES.values()}
    losses = []
    with torch.no_grad():
        for batch in loader:
            target = batch['mask'].to(device)
            with precision.context():
                logits = model(batch['image'].to(device))
                loss = binary_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite validation loss')
            losses.extend([float(loss)] * len(target))
            overall.update(logits, target)
            for i, category in enumerate(batch['category']):
                per_category[category].update(logits[i:i + 1], target[i:i + 1])
    return {'loss': sum(losses) / len(losses), 'overall': overall.report(),
            'per_category': {c: counts.report() for c, counts in per_category.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow_pilot', action='store_true', help='Use only after user confirms the pilot phase')
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
    if not args.allow_pilot:
        parser.error('Pilot requires explicit user confirmation and --allow_pilot; no training started')
    if not 1 <= args.epochs <= 3 or args.batch_size != 1 or args.img_size != 896:
        parser.error('Pilot guard: epochs 1..3, batch_size=1, img_size=896')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot requires an allocated CUDA GPU')
    device = torch.device('cuda:0')
    software = environment()
    root = args.data_root or discover_root(args.input_root)
    train_rows, val_rows = pilot_split(read_manifest(root), args.seed)
    if len(train_rows) + len(val_rows) > 60:
        raise AssertionError('Pilot cap exceeded')
    config['splits'] = {'train': train_rows, 'validation': val_rows}
    config['counts'] = {name: dict(Counter(CATEGORIES[r['category']] for r in rows))
                        for name, rows in [('train', train_rows), ('validation', val_rows)]}
    config['missing_categories'] = {name: sorted(set(CATEGORIES.values()) - set(counts))
                                    for name, counts in config['counts'].items()}
    config['split_limitation'] = 'Scene groups disjoint; geographic overlap between different scenes not excluded'
    write_json(args.output_dir / 'config.json', config)
    weights, weight_config, weight_source = resolve_weights(args.weights, args.download_weights, str(args.weight_cache))
    model = FrozenSATSegDINO(weights, weight_config, weight_source, args.img_size).to(device)
    normalization = model.metadata['normalization']
    options = dict(root=root, img_size=args.img_size, image_dir=args.image_dir, mask_dir=args.mask_dir, **normalization)
    train_loader = DataLoader(PV896Dataset(rows=train_rows, augment=True, **options),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(PV896Dataset(rows=val_rows, augment=False, **options),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = decoder_optimizer(model, args.lr)
    precision = Precision(device)
    config['precision'] = precision.describe()
    config['metrics'] = ('Threshold sigmoid>=0.5; cumulative TP/FP/FN foreground micro metrics. '
                         'Zero denominator -> null, including empty-truth/empty-prediction union. '
                         'Negative FP contributes to overall counts; report negative FP pixels and images separately.')
    write_json(args.output_dir / 'config.json', config)
    best_iou = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if model.backbone.training:
            raise AssertionError('Backbone must remain eval')
        total, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with precision.context():
                logits = model(batch['image'].to(device))
                loss = binary_loss(logits, batch['mask'].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            precision.scaler.scale(loss).backward()
            precision.scaler.unscale_(optimizer)
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise AssertionError('Frozen backbone acquired gradients')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.decoder.parameters()):
                raise RuntimeError('Missing or nonfinite decoder gradient')
            precision.scaler.step(optimizer)
            precision.scaler.update()
            total += float(loss.detach()) * len(batch['image'])
            count += len(batch['image'])
        validation = evaluate(model, val_loader, device, precision)
        row = {'epoch': epoch, 'train_loss': total / count, 'validation': validation}
        history.append(row)
        write_json(args.output_dir / 'history.json', history)
        print(row, flush=True)
        iou = validation['overall']['iou']
        if iou is not None and iou > best_iou:
            best_iou = iou
            save_checkpoint(args.output_dir / 'best_decoder.pt', model, optimizer, config, software,
                            epoch=epoch, best_validation_iou=best_iou)
            for index, batch in enumerate(val_loader):
                with torch.no_grad(), precision.context():
                    logits = model(batch['image'].to(device))
                save_prediction(args.output_dir / f'best_epoch_{epoch}_predictions', batch, logits, normalization, smoke=False)
                if index >= 5:
                    break
    write_json(args.output_dir / 'completion.json', {
        'status': 'COMPLETE', 'epochs': args.epochs, 'total_unique_pairs': len(train_rows) + len(val_rows),
        'best_validation_iou': best_iou if best_iou >= 0 else None,
        'stop_reason': 'Pilot finished; full-dataset training requires separate user confirmation'})


if __name__ == '__main__':
    main()
