"""Run a gated one-support smoke test and paired development ablations."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = Path('/kaggle/working/pv_dual_prior_pilot')
GEO_SHA256 = '129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def metrics(prediction, target):
    import numpy as np
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                specificity=ratio(tn, tn + fp))


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.29',
                    'open_clip_torch==3.2.0', 'ftfy', 'huggingface-hub>=0.34'],
                   check=True)
    import numpy as np
    from PIL import Image
    import open_clip
    import torch
    from torch.nn import functional as F
    from pvseg.dinov3_foundation import FrozenDINOv3SAT, MODEL_SHA256
    from pvseg.dual_prior import DualPriorSegmenter, segmentation_loss
    from pvseg.frozen_baselines import aligned_crop_boxes, aligned_crop_features
    from pvseg.support_haar_fusion import haar_features

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU required')
    device = torch.device('cuda')
    protocol = json.loads((ROOT / 'assets/dual_prior_protocol.json').read_text(encoding='utf-8'))
    manifests = [path for path in Path('/kaggle/input').rglob('manifest.csv')
                 if digest(path) == protocol['manifest_sha256']]
    if len(manifests) != 1:
        raise ValueError('Expected exactly one matching PV-896 manifest')
    data = manifests[0].parent
    with manifests[0].open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['sample_id']: row for row in csv.DictReader(stream)}
    if (protocol['support_id'] in protocol['query_ids'] or
            len(protocol['query_ids']) != 24 or len(set(protocol['query_ids'])) != 24 or
            set(protocol['train_query_ids']) & set(protocol['query_ids']) or
            set(protocol['smoke_query_ids']) - set(protocol['train_query_ids'])):
        raise ValueError('Support and paired query selection must be disjoint')
    sat_paths = [path for path in Path('/kaggle/input').rglob('model.safetensors')
                 if digest(path) == MODEL_SHA256]
    geo_paths = [path for path in Path('/kaggle/input').rglob('RS5M_ViT-B-32.pt')
                 if digest(path) == GEO_SHA256]
    if len(sat_paths) != 1 or len(geo_paths) != 1:
        raise ValueError('Expected pinned DINO and Geo checkpoints')
    sat = FrozenDINOv3SAT(sat_paths[0], image_size=896).to(device).eval()
    clip, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained=None, force_quick_gelu=True)
    saved = torch.load(geo_paths[0], map_location='cpu', weights_only=True)
    clip.load_state_dict({key.removeprefix('module.'): value
                          for key, value in saved.get('state_dict', saved).items()},
                         strict=True)
    del saved
    clip.requires_grad_(False).to(device).eval()
    boxes = aligned_crop_boxes((896, 896))
    if len(boxes) != 49:
        raise RuntimeError('Unexpected Geo crop count')

    def read_rgb(sample_id):
        with Image.open(data / rows[sample_id]['image']) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        if rgb.shape != (896, 896, 3):
            raise ValueError(f'Unexpected image dimensions: {sample_id}')
        return rgb

    def encode(rgb):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        image = Image.fromarray(rgb)
        crops = torch.stack([preprocess(image.crop(box)) for box in boxes])
        with torch.no_grad():
            dino = sat(tensor).cpu()
            vectors = torch.cat([F.normalize(clip.encode_image(batch.to(device)).float(),
                                             dim=-1) for batch in crops.split(32)])
            geo = aligned_crop_features(vectors, boxes, (896, 896), (56, 56)).cpu()
        if dino.shape != (1, 1024, 56, 56) or geo.shape != (1, 512, 56, 56):
            raise RuntimeError('Unexpected frozen feature dimensions')
        return dino, geo

    OUTPUT.mkdir(parents=True, exist_ok=False)
    save_json(OUTPUT / 'protocol.json', protocol)
    support_rgb = read_rgb(protocol['support_id'])
    with Image.open(data / rows[protocol['support_id']]['label']) as image:
        support_target = torch.from_numpy((np.asarray(image.convert('L')) == 255).copy())
    support_target = support_target.float()[None, None].to(device)
    support_dino, support_geo = encode(support_rgb)
    support_dino, support_geo = support_dino.to(device), support_geo.to(device)

    training = {}
    for sample_id in protocol['train_query_ids']:
        rgb = read_rgb(sample_id)
        dino, geo = encode(rgb)
        with Image.open(data / rows[sample_id]['label']) as image:
            target = torch.from_numpy((np.asarray(image.convert('L')) == 255).copy())
        training[sample_id] = (dino, geo, haar_features(rgb),
                               target.float()[None, None])

    def train_variant(name, spec, steps, episode_ids):
        torch.manual_seed(protocol['seed'])
        model = DualPriorSegmenter(channels=protocol['channels'], **spec).to(device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=protocol['learning_rate'],
                                      weight_decay=protocol['weight_decay'])
        expected = ['decoder']
        if spec['branch'] in ('dino', 'both'):
            expected.append('dino_projection')
        if spec['branch'] in ('geo', 'both'):
            expected.append('geo_projection')
        if spec['branch'] == 'both' and spec['fusion'] == 'dynamic':
            expected.append('feature_gate')
        if spec['frequency'] == 'feature':
            expected.append('feature_modulation')
        if spec['frequency'] == 'score':
            expected.append('score_correction')
        history, gradient_norms = [], {}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(steps):
            sample_id = episode_ids[step % len(episode_ids)]
            query_dino, query_geo, query_haar, query_target = training[sample_id]
            optimizer.zero_grad(set_to_none=True)
            output = model(support_dino, support_geo, support_target,
                           query_dino.to(device), query_geo.to(device), (896, 896),
                           query_haar.to(device) if spec['frequency'] != 'none' else None)
            loss = segmentation_loss(output['logits'], query_target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}: nonfinite loss')
            loss.backward()
            if step == 0:
                gradient_norms = {
                    group: sum(float(parameter.grad.abs().sum())
                               for parameter in getattr(model, group).parameters()
                               if parameter.grad is not None)
                    for group in expected
                }
                if not all(value > 0 for value in gradient_norms.values()):
                    raise RuntimeError(f'{name}: missing expected gradient {gradient_norms}')
            torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step + 1, query_id=sample_id,
                                loss=float(loss.detach())))
        torch.cuda.synchronize()
        return model.eval(), dict(history=history, gradient_norms=gradient_norms,
                                  train_seconds=time.perf_counter() - start,
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                  parameter_count=sum(parameter.numel() for parameter in parameters),
                                  last_train_foreground_fraction=float(
                                      (output['logits'].detach().sigmoid() >= .5).float().mean()))

    def evaluate(model, selected, spec, cached=None, training_targets=False):
        records = {}
        for sample_id in selected:
            if cached is None:
                rgb = read_rgb(sample_id)
                dino, geo = encode(rgb)
                haar = haar_features(rgb)
            else:
                dino, geo, haar = cached[sample_id][:3]
            with torch.no_grad():
                score = model(support_dino, support_geo, support_target,
                              dino.to(device), geo.to(device), (896, 896),
                              haar.to(device) if spec['frequency'] != 'none' else None
                              )['logits'].sigmoid()[0, 0].cpu().numpy()
            prediction = score >= .5
            if training_targets:
                target = cached[sample_id][3][0, 0].numpy().astype(bool)
            else:
                with Image.open(data / rows[sample_id]['label']) as image:
                    target = np.asarray(image.convert('L')) == 255
            records[sample_id] = dict(**metrics(prediction, target),
                                      category=rows[sample_id]['category'],
                                      predicted_fraction=float(prediction.mean()))
        totals = {key: sum(record[key] for record in records.values())
                  for key in ('tp', 'fp', 'fn', 'tn')}
        negatives = [record for record in records.values() if record['category'] == '负样本']
        return dict(summary=metrics_from_counts(totals),
                    pure_negative_fp=sum(record['fp'] for record in negatives),
                    pure_negative_predicted=sum(record['fp'] > 0 for record in negatives),
                    per_image=records)

    def metrics_from_counts(counts):
        tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        return dict(**counts, iou=ratio(tp, tp + fp + fn),
                    precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                    f1=ratio(2 * tp, 2 * tp + fp + fn),
                    specificity=ratio(tn, tn + fp))

    full_spec = dict(branch='both', fusion='dynamic', frequency='feature',
                     cam_guidance=True)
    smoke_id = protocol['smoke_query_ids'][0]
    smoke_model, smoke_fit = train_variant('smoke', full_spec,
                                          protocol['smoke_steps'], [smoke_id])
    smoke_eval = evaluate(smoke_model, protocol['smoke_query_ids'], full_spec,
                          training, training_targets=True)
    smoke_pass = (smoke_fit['history'][-1]['loss'] < smoke_fit['history'][0]['loss']
                  and 0 < smoke_fit['last_train_foreground_fraction'] < 1)
    save_json(OUTPUT / 'smoke.json', dict(passed=smoke_pass, fit=smoke_fit,
                                         evaluation=smoke_eval))
    if not smoke_pass:
        raise RuntimeError('Smoke gate failed; formal ablation not started')
    del smoke_model
    torch.cuda.empty_cache()
    cached = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    encoding_start = time.perf_counter()
    for sample_id in protocol['query_ids']:
        rgb = read_rgb(sample_id)
        dino, geo = encode(rgb)
        cached[sample_id] = (dino, geo, haar_features(rgb))
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encoding_start
    encoding_peak_gpu_bytes = torch.cuda.max_memory_allocated()

    variants = {
        'A_dino_only': dict(branch='dino', fusion='fixed', frequency='none',
                            cam_guidance=False),
        'B_geo_only': dict(branch='geo', fusion='fixed', frequency='none',
                           cam_guidance=False),
        'C_fixed': dict(branch='both', fusion='fixed', frequency='none',
                        cam_guidance=False),
        'D_dynamic': dict(branch='both', fusion='dynamic', frequency='none',
                          cam_guidance=False),
        'E_old_score_haar': dict(branch='both', fusion='dynamic', frequency='score',
                                  cam_guidance=False),
        'F_feature_haar': dict(branch='both', fusion='dynamic', frequency='feature',
                                cam_guidance=False),
        'G_feature_haar_cam_decoder': full_spec,
    }
    results = {}
    for name, spec in variants.items():
        model, fit = train_variant(name, spec, protocol['ablation_steps'],
                                   protocol['train_query_ids'])
        torch.cuda.synchronize()
        started = time.perf_counter()
        evaluation = evaluate(model, protocol['query_ids'], spec, cached)
        torch.cuda.synchronize()
        results[name] = dict(config=spec, fit=fit, evaluation=evaluation,
                             inference_seconds=time.perf_counter() - started)
        save_json(OUTPUT / 'results_partial.json',
                  dict(status='partial', protocol=protocol, variants=results))
        torch.save(model.cpu().state_dict(), OUTPUT / f'{name}.pt')
        del model
        torch.cuda.empty_cache()
        print(json.dumps(dict(variant=name, iou=evaluation['summary']['iou'])), flush=True)
    save_json(OUTPUT / 'results.json',
              dict(status='complete', protocol=protocol, smoke=smoke_fit,
                   encoding_seconds=encoding_seconds,
                   encoding_peak_gpu_bytes=encoding_peak_gpu_bytes,
                   variants=results))


if __name__ == '__main__':
    main()
