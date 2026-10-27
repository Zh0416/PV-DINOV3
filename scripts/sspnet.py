"""Two-stage training/evaluation for the supervised single-class SSPNet PV adaptation."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pvseg.data import discover_root, read_manifest
from pvseg.metrics import BinaryCounts
from pvseg.runtime import Precision, environment, memory, seed_everything, write_json
from pvseg.sspnet import SSPNet
from pvseg.sspnet import checkpoint
from pvseg.sspnet.data import Episodes, ResizedPV, make_split
from pvseg.sspnet.encoders import FrozenSAT, PSPNet, freeze_bn, load_text_cache, sha256
from pvseg.sspnet.losses import objective


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['base', 'episodic', 'evaluate'])
    p.add_argument('--data-root', default='/kaggle/input')
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--shots', type=int, choices=[1, 5], default=1)
    p.add_argument('--weights', help='Existing verified DINOv3 SAT safetensors')
    p.add_argument('--text-cache', help='Validated GeoRSCLIP frozen raw embedding')
    p.add_argument('--base-checkpoint', help='Stage 1 segmentation-pretrained PSPNet')
    p.add_argument('--resume', help='Resume stage checkpoint at an epoch boundary')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--query-chunk', type=int, default=256)
    p.add_argument('--routing', choices=['literal', 'raw_max'], default='literal')
    p.add_argument('--pilot', action='store_true', help='Reuse the previous <=60-pair development split')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0, help='Bounded engineering check; 0 means full epoch')
    p.add_argument('--allow-untrained-base', action='store_true', help='Smoke ONLY, no accuracy claim')
    p.add_argument('--device', default='cuda')
    return p


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward_episode(model, batch):
    return model(batch['query'], batch['supports'], batch['support_masks'])


@torch.no_grad()
def evaluate(model, loader, precision, device, episodic, limit=0):
    model.eval()
    counts, loss_sum, count = BinaryCounts(), 0., 0
    categories = {c: BinaryCounts() for c in ['positive', 'hard_positive', 'negative']}
    assignments, routing = [], {'base_selected': 0, 'near_tie': 0, 'total': 0}
    for index, raw in enumerate(loader):
        if limit and index >= limit:
            break
        batch = move(raw, device)
        with precision.context():
            if episodic:
                result = forward_episode(model, batch)
                loss, _ = objective(result, batch['target'])
                logits, target = result['logits'], batch['target']
                assignments.extend({'query': q, 'supports': s.split('|')}
                                   for q, s in zip(batch['query_id'], batch['support_ids']))
                routing['base_selected'] += int(result['routing']['base_selected'].sum())
                routing['near_tie'] += int(result['routing']['near_tie'].sum())
                routing['total'] += result['routing']['near_tie'].numel()
            else:
                logits, target = model(batch['image']), batch['mask']
                loss = F.cross_entropy(logits.float(), target[:, 0].long())
        # Foreground-vs-background logit difference equals the 2-class softmax threshold.
        difference = logits[:, 1:2].float() - logits[:, :1].float()
        counts.update(difference, target)
        for i, category in enumerate(batch['category']):
            categories[category].update(difference[i:i+1], target[i:i+1])
        loss_sum += float(loss) * len(target)
        count += len(target)
    if not count:
        raise ValueError('Empty evaluation loader')
    return {'loss': loss_sum / count, 'metrics': counts.report(),
            'by_category': {k: v.report() for k, v in categories.items()},
            'support_assignments': assignments, 'routing': routing}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError('epochs/batch-size/accumulation must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completion.json').exists() and args.stage != 'evaluate':
        raise RuntimeError('Completed output exists; use a new output directory')
    seed_everything(args.seed)
    device = torch.device(args.device)
    precision = Precision(device)
    root = Path(args.data_root)
    if not (root/'manifest.csv').is_file():
        root = discover_root(root)
    train_rows, val_rows, split = make_split(read_manifest(root), args.pilot, args.seed)
    write_json(output/'split.json', split)
    write_json(output/'environment.json', environment())
    config = vars(args).copy()
    config.update(split_sha256=split['split_sha256'], precision=precision.describe(),
                  supervision='supervised_known_class_PV', raw_source_size=896)
    write_json(output/'config.json', config)
    episodic = args.stage != 'base'
    cnn = PSPNet(imagenet=not episodic and not args.resume)
    base_state = None
    if episodic:
        if not args.weights or not args.text_cache:
            raise ValueError('Episodic stage requires SAT weights and GeoRSCLIP text cache')
        if args.base_checkpoint:
            base_state = checkpoint.load(args.base_checkpoint, cnn)
            if base_state['config']['split_sha256'] != split['split_sha256']:
                raise ValueError('PSP pretraining split differs from episodic split')
            cnn.provenance = dict(cnn.provenance, checkpoint_sha256=sha256(args.base_checkpoint))
        if (not cnn.provenance.get('segmentation_pretrained') or cnn.provenance.get('engineering_only')) and not args.allow_untrained_base:
            raise ValueError('Stage 1 PSP segmentation pretraining required; random-base use is smoke-only')
        dino = FrozenSAT(args.weights, args.size)
        text, text_meta = load_text_cache(args.text_cache)
        model = SSPNet(cnn, dino, text, text_meta, shots=args.shots,
                       query_chunk=args.query_chunk, routing=args.routing)
        train_data = Episodes(ResizedPV(root, train_rows, args.size, True),
                              ResizedPV(root, train_rows, args.size, True), split['spatial_groups'], args.shots, args.seed)
        val_data = Episodes(ResizedPV(root, val_rows, args.size),
                            ResizedPV(root, train_rows, args.size), split['spatial_groups'], args.shots, args.seed)
    else:
        model = cnn
        train_data, val_data = ResizedPV(root, train_rows, args.size, True), ResizedPV(root, val_rows, args.size)
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=.9, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    start_epoch, global_step = 0, 0
    if args.resume:
        restored = checkpoint.load(args.resume, model, optimizer, precision, True, generator)
        keys = ['split_sha256', 'size', 'shots'] if args.stage == 'evaluate' else [
            'split_sha256', 'size', 'shots', 'lr', 'accumulation', 'batch_size', 'epochs']
        for key in keys:
            if restored['config'][key] != config[key]:
                raise ValueError(f'Resume config differs: {key}')
        start_epoch, global_step = restored['epoch'] + 1, restored['step']
    if args.stage == 'evaluate':
        if not args.resume:
            raise ValueError('evaluate requires --resume episodic checkpoint')
        report = evaluate(model, val_loader, precision, device, True, args.max_steps)
        write_json(output/'evaluation.json', report)
        return report
    steps_per_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    total_updates = ((steps_per_epoch + args.accumulation - 1) // args.accumulation) * args.epochs
    history = []
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if not episodic:
            freeze_bn(model)  # batch=1 + pooled 1x1 features; record this deviation.
        else:
            train_data.epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        losses, training_assignments = [], []
        for index, raw in enumerate(train_loader):
            if index >= steps_per_epoch:
                break
            batch = move(raw, device)
            group_start = (index // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, steps_per_epoch - group_start)
            with precision.context():
                if episodic:
                    result = forward_episode(model, batch)
                    loss, terms = objective(result, batch['target'])
                    training_assignments.extend({'query': q, 'supports': s.split('|')}
                        for q, s in zip(batch['query_id'], batch['support_ids']))
                else:
                    loss = F.cross_entropy(model(batch['image']).float(), batch['mask'][:, 0].long())
                    terms = {'main_ce': loss}
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            precision.scaler.scale(loss / group_size).backward()
            if (index + 1) % args.accumulation == 0 or index + 1 == steps_per_epoch:
                precision.scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10., error_if_nonfinite=True)
                lr = args.lr * max(0., 1 - global_step / total_updates) ** .9
                for group in optimizer.param_groups:
                    group['lr'] = lr
                precision.scaler.step(optimizer)
                precision.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append({k: float(v.detach()) for k, v in terms.items()})
            print(json.dumps({'stage': args.stage, 'epoch': epoch, 'batch': index,
                              'loss': float(loss.detach()), 'updates': global_step}), flush=True)
        if not episodic:
            model.provenance.update(segmentation_pretrained=True, completed_epochs=epoch + 1,
                engineering_only=bool(args.max_steps), split_sha256=split['split_sha256'], bn_policy='running_statistics_frozen')
        validation = evaluate(model, val_loader, precision, device, episodic,
                              min(3, args.max_steps) if args.max_steps else 0)
        history.append({'epoch': epoch, 'training_losses': losses, 'validation': validation,
                        'training_support_assignments': training_assignments})
        checkpoint.save(output/'last.pt', model, optimizer, precision, config, epoch, global_step, generator)
        write_json(output/'history.json', history)
    # Round trip checks all saved tensors, not only that torch.load succeeded.
    saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
    for key, value in saved['state_dict'].items():
        if not torch.equal(value, model.state_dict()[key].cpu()):
            raise RuntimeError(f'Checkpoint round trip differs: {key}')
    completion = {'stage': args.stage, 'epochs_completed': args.epochs,
        'optimizer_updates': global_step, 'engineering_only': bool(args.max_steps or args.allow_untrained_base),
        'seconds': time.perf_counter() - started, 'memory': memory(),
        'checkpoint_tensor_round_trip': True, 'train_pairs': len(train_rows), 'val_pairs': len(val_rows),
        'accuracy_scope': 'development validation only; no independent held-out test; no novel-class claim',
        'final_validation': history[-1]['validation'] if history else None}
    write_json(output/'completion.json', completion)
    return completion


if __name__ == '__main__':
    main()
