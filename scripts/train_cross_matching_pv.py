"""Train and evaluate the ResNet50 Cross-Matching PV-896 baseline."""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet50_Weights

from pvseg.cross_matching import CrossMatchingNet
from pvseg.cross_matching_data import PVEpisodes, spatial_partition, tile_origins
from pvseg.data import read_manifest
from pvseg.metrics import BinaryCounts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--imagenet-weights', type=Path)
    weights.add_argument('--download-imagenet', action='store_true')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--eval-shots', type=int, choices=(1, 5), default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=9e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-query-limit', type=int, default=0)
    parser.add_argument('--validation-query-limit', type=int, default=0)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def select_queries(rows, limit, seed):
    if limit <= 0 or limit >= len(rows):
        return sorted(rows, key=lambda row: row['sample_id'])
    rng = random.Random(seed)
    categories = ('正样本', '困难正样本', '负样本')
    selected = []
    for category_index, category in enumerate(categories):
        pool = [row for row in rows if row['category'] == category]
        quota = limit // len(categories) + (category_index < limit % len(categories))
        chosen = rng.sample(pool, min(len(pool), quota))
        selected.extend(chosen)
    selected_ids = {chosen['sample_id'] for chosen in selected}
    remainder = [row for row in rows if row['sample_id'] not in selected_ids]
    selected.extend(rng.sample(remainder, min(limit - len(selected), len(remainder))))
    return sorted(selected, key=lambda row: row['sample_id'])


def load_backbone(model, args):
    if args.imagenet_weights:
        state = torch.load(args.imagenet_weights, map_location='cpu', weights_only=True)
        model.encoder.backbone.load_state_dict(state, strict=True)
        return {'kind': 'local_imagenet_weights', 'path': str(args.imagenet_weights.resolve()),
                'sha256': digest(args.imagenet_weights)}
    weight_enum = ResNet50_Weights.IMAGENET1K_V1
    model.encoder.backbone.load_state_dict(weight_enum.get_state_dict(progress=True, check_hash=True))
    return {'kind': 'torchvision', 'name': 'IMAGENET1K_V1', 'url': weight_enum.url}


@torch.no_grad()
def evaluate(model, episodes, device, output_dir=None):
    model.eval()
    overall = BinaryCounts()
    categories = {category: BinaryCounts() for category in ('正样本', '困难正样本', '负样本')}
    details = []
    origins = tile_origins(896, episodes.crop_size)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(episodes.query_rows):
        query, target = episodes.load(row)
        support, support_mask, support_ids = episodes.support_patches(
            row, random.Random(episodes.seed + index))
        support = support.unsqueeze(0).to(device)
        support_mask = support_mask.unsqueeze(0).to(device)
        sums = torch.zeros(896, 896, device=device)
        coverage = torch.zeros_like(sums)
        inference_profile = None
        for tile_index, (top, left) in enumerate(origins):
            patch = query[:, top:top + episodes.crop_size,
                          left:left + episodes.crop_size].unsqueeze(0).to(device)
            if episodes.shots == 1:
                logits = model(patch, support[:, 0], support_mask[:, 0],
                               profile=tile_index == 0)
                score = F.softmax(logits.float(), dim=1)[0, 1]
                if tile_index == 0:
                    inference_profile = [model.last_forward_profile]
            else:
                score = model.predict_nshot(patch, support, support_mask,
                                           profile=tile_index == 0).float()[0]
                if tile_index == 0:
                    inference_profile = model.last_nshot_profiles
            sums[top:top + episodes.crop_size, left:left + episodes.crop_size] += score
            coverage[top:top + episodes.crop_size, left:left + episodes.crop_size] += 1
        prediction = sums / coverage >= 0.5
        truth = target[0].to(device) >= 0.5
        predicted_logits = torch.where(prediction, 20.0, -20.0)[None, None]
        target_tensor = truth.float()[None, None]
        overall.update(predicted_logits, target_tensor)
        categories[row['category']].update(predicted_logits, target_tensor)
        true_positive = int((prediction & truth).sum().item())
        false_positive = int((prediction & ~truth).sum().item())
        false_negative = int((~prediction & truth).sum().item())
        details.append({'query_id': row['sample_id'], 'support_ids': support_ids,
                        'category': row['category'], 'tp': true_positive,
                        'fp': false_positive, 'fn': false_negative,
                        'first_tile_forward_profiles': inference_profile})
        if output_dir:
            Image.fromarray(prediction.byte().cpu().numpy() * 255).save(
                output_dir / f"{row['sample_id']}.png")
    return {'overall': overall.report(),
            'by_category': {key: counts.report() for key, counts in categories.items()},
            'per_query': details, 'tiles_per_query': len(origins)}


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError('Epochs, batch size, and learning rate must be positive')
    if args.patch_size != 256:
        raise ValueError('This first protocol uses native-resolution 256-pixel crops only')
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('AMP requires CUDA')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    root = args.data_root.resolve()
    rows = read_manifest(root)
    train_rows, validation_rows, scene_group, split = spatial_partition(
        rows, seed=args.seed, validation_fraction=args.validation_fraction)
    train_queries = select_queries(train_rows, args.train_query_limit, args.seed)
    validation_queries = select_queries(validation_rows, args.validation_query_limit, args.seed + 1)
    if not train_queries or not validation_queries:
        raise ValueError('Training and validation each require at least one query')
    train_episodes = PVEpisodes(root, train_queries, train_rows, scene_group,
                                shots=1, crop_size=args.patch_size, seed=args.seed, training=True)
    validation_episodes = PVEpisodes(root, validation_queries, train_rows, scene_group,
                                     shots=args.eval_shots, crop_size=args.patch_size,
                                     seed=args.seed, training=False)
    model = CrossMatchingNet()
    backbone_source = load_backbone(model, args)
    model.to(device)
    head_weights = [parameter for name, parameter in model.matcher.named_parameters()
                    if name.endswith('weight')]
    head_biases = [parameter for name, parameter in model.matcher.named_parameters()
                   if name.endswith('bias')]
    optimizer = torch.optim.SGD([{'params': head_weights, 'lr': args.learning_rate * 10},
                                 {'params': head_biases, 'lr': args.learning_rate * 20}],
                                momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    counts = Counter(row['category'] for row in train_queries)
    sample_weights = [1 / counts[row['category']] for row in train_queries]
    sampler = WeightedRandomSampler(sample_weights, len(train_queries), replacement=True)
    loader = DataLoader(train_episodes, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0)
    total_steps = args.epochs * len(loader)

    args.output_dir.mkdir(parents=True)
    config = {'protocol': 'known-class PV support-query episodic training',
              'architecture': 'ResNet50 + SCCNet Cross-Matching only',
              'source_code_reference': 'linhanwang/SCCNet@741cbade237c803836c2b4192bfa91441964f461',
              'data_root': str(root), 'manifest_sha256': digest(root / 'manifest.csv'),
              'backbone_source': backbone_source, 'input_original_size': 896,
              'patch_size': args.patch_size, 'train_shots': 1,
              'eval_shots': args.eval_shots, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'base_learning_rate': args.learning_rate,
              'head_weight_learning_rate_multiplier': 10,
              'head_bias_learning_rate_multiplier': 20,
              'seed': args.seed, 'device': str(device), 'amp': args.amp,
              'train_query_ids': [row['sample_id'] for row in train_queries],
              'validation_query_ids': [row['sample_id'] for row in validation_queries]}
    save_json(args.output_dir / 'config.json', config)
    save_json(args.output_dir / 'spatial_split.json', split)
    best_iou = -1.0
    history = []
    global_step = 0
    with (args.output_dir / 'forward_profiles.jsonl').open('w', encoding='utf-8') as profile_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for batch in loader:
                current_learning_rate = args.learning_rate * (1 - global_step / total_steps) ** 0.9
                optimizer.param_groups[0]['lr'] = current_learning_rate * 10
                optimizer.param_groups[1]['lr'] = current_learning_rate * 20
                query = batch['query'].to(device)
                support = batch['supports'][:, 0].to(device)
                support_mask = batch['support_masks'][:, 0].to(device)
                target = batch['target'].to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                    logits = model(query, support, support_mask, profile=True)
                    loss = F.cross_entropy(logits.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite cross-matching loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                global_step += 1
                loss_sum += float(loss.detach())
                profile = {'epoch': epoch, 'step': global_step,
                           'query_ids': list(batch['query_id']),
                           'support_ids': list(batch['support_ids'][0]),
                           'base_learning_rate': current_learning_rate,
                           'head_weight_learning_rate': optimizer.param_groups[0]['lr'],
                           'head_bias_learning_rate': optimizer.param_groups[1]['lr'],
                           **model.last_forward_profile,
                           'step_seconds': time.perf_counter() - step_started,
                           'step_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)
                           if device.type == 'cuda' else None}
                profile_file.write(json.dumps(profile, ensure_ascii=False) + '\n')
                profile_file.flush()
            validation = evaluate(model, validation_episodes, device)
            summary = {'epoch': epoch, 'train_loss': loss_sum / len(loader),
                       'optimizer_steps': global_step, 'validation': validation}
            history.append(summary)
            save_json(args.output_dir / 'history.json', history)
            iou = validation['overall']['iou']
            if iou is not None and iou > best_iou:
                best_iou = iou
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'epoch': epoch, 'validation_iou': iou, 'config': config}
                temporary = args.output_dir / 'best_checkpoint.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(args.output_dir / 'best_checkpoint.pt')
            print(json.dumps({'epoch': epoch, 'train_loss': summary['train_loss'],
                              'validation_iou': iou, 'steps': global_step}, ensure_ascii=False), flush=True)
    if best_iou < 0:
        raise RuntimeError('Validation foreground IoU is undefined; no checkpoint selected')
    save_json(args.output_dir / 'completion.json', {'best_validation_iou': best_iou,
                                                    'epochs': args.epochs, 'steps': global_step})


if __name__ == '__main__':
    run(parse_args())
