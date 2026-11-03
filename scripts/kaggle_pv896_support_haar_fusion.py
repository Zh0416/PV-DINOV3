"""Evaluate support-trained dynamic fusion and Haar correction on PV-896."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = Path('/kaggle/working/pv896_support_haar_fusion')
GEO_SHA256 = '129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def counts(prediction, target):
    import numpy as np
    return dict(tp=int(np.count_nonzero(prediction & target)),
                fp=int(np.count_nonzero(prediction & ~target)),
                fn=int(np.count_nonzero(~prediction & target)),
                tn=int(np.count_nonzero(~prediction & ~target)))


def metrics(values):
    tp, fp, fn, tn = (values[key] for key in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda top, bottom: top / bottom if bottom else None
    return dict(**values, iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                false_positive_rate=ratio(fp, fp + tn))


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
    from pvseg.frozen_baselines import aligned_crop_boxes, aligned_crop_features
    from pvseg.support_haar_fusion import fit_from_support, predict

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU is required for the generic encoders')
    device = torch.device('cuda')
    protocol = json.loads((ROOT / 'assets/support_haar_protocol.json').read_text(encoding='utf-8'))
    manifests = [path for path in Path('/kaggle/input').rglob('manifest.csv')
                 if digest(path) == protocol['manifest_sha256']]
    if len(manifests) != 1:
        raise ValueError('Expected one matching PV-896 manifest')
    data = manifests[0].parent
    with manifests[0].open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['sample_id']: row for row in csv.DictReader(stream)}
    support_id = protocol['support_id']
    categories = protocol['query_ids']
    query_ids = [sample_id for members in categories.values() for sample_id in members]
    if (len(query_ids) != 300 or len(set(query_ids)) != 300 or
            support_id in query_ids or rows[support_id]['category'] != '正样本' or
            any(len(members) != 100 or any(rows[sample_id]['category'] != category
                                          for sample_id in members)
                for category, members in categories.items())):
        raise ValueError('Support or queries differ from the locked selection')
    sat_paths = [path for path in Path('/kaggle/input').rglob('model.safetensors')
                 if digest(path) == MODEL_SHA256]
    geo_paths = [path for path in Path('/kaggle/input').rglob('RS5M_ViT-B-32.pt')
                 if digest(path) == GEO_SHA256]
    if len(sat_paths) != 1 or len(geo_paths) != 1:
        raise ValueError('Expected one verified checkpoint per generic encoder')
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

    def encode(rgb):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        image = Image.fromarray(rgb)
        crops = torch.stack([preprocess(image.crop(box)) for box in boxes])
        with torch.no_grad():
            dino = sat(tensor).cpu()
            vectors = torch.cat([F.normalize(clip.encode_image(batch.to(device)).float(),
                                              dim=-1) for batch in crops.split(32)])
            geo = aligned_crop_features(vectors, boxes, (896, 896), (56, 56)).cpu()
        return dino, geo

    support_row = rows[support_id]
    with Image.open(data / support_row['image']) as image:
        support_rgb = np.asarray(image.convert('RGB')).copy()
    with Image.open(data / support_row['label']) as image:
        support_mask = np.asarray(image.convert('L')) == 255
    if support_rgb.shape != (896, 896, 3) or support_mask.shape != (896, 896):
        raise ValueError('Unexpected support dimensions')
    support_dino, support_geo = encode(support_rgb)
    mask = torch.from_numpy(support_mask.copy()).float()[None, None]
    task = fit_from_support(support_rgb, support_dino, support_geo, mask,
                            steps=protocol['steps'], learning_rate=protocol['learning_rate'],
                            seed=protocol['seed'])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'masks').mkdir()
    write_json(OUTPUT / 'protocol_before_prediction.json', protocol)
    torch.save(task, OUTPUT / 'support_derived_task.pt')
    write_json(OUTPUT / 'support_fit.json',
               {key: value for key, value in task.items() if key != 'models'} |
               dict(parameter_updates=[
                   dict(fusion=state['updated_fusion_tensors'],
                        haar=state['updated_haar_tensors'])
                   for state in task['models']]))
    del support_rgb, support_mask, support_dino, support_geo, mask

    predictions = {}
    for index, sample_id in enumerate(query_ids, 1):
        with Image.open(data / rows[sample_id]['image']) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        if rgb.shape != (896, 896, 3):
            raise ValueError(f'Unexpected query dimensions: {sample_id}')
        dino, geo = encode(rgb)
        probability, prediction = predict(task, rgb, dino, geo)
        Image.fromarray(prediction.astype(np.uint8) * 255).save(
            OUTPUT / 'masks' / f'{sample_id}.png')
        predictions[sample_id] = dict(predicted_present=bool(prediction.any()),
                                      predicted_pixels=int(prediction.sum()),
                                      max_probability=float(probability.max()),
                                      mask=f'masks/{sample_id}.png')
        if index % 10 == 0:
            write_json(OUTPUT / 'prediction_progress.json',
                       dict(completed=index, total=len(query_ids)))
            print(json.dumps(dict(predicted=index, total=len(query_ids))), flush=True)

    records = {}
    for sample_id in query_ids:
        with Image.open(OUTPUT / predictions[sample_id]['mask']) as image:
            prediction = np.asarray(image) == 255
        with Image.open(data / rows[sample_id]['label']) as image:
            target = np.asarray(image) == 255
        if target.shape != (896, 896):
            raise ValueError(f'Unexpected target dimensions: {sample_id}')
        records[sample_id] = dict(**predictions[sample_id], **metrics(counts(prediction, target)),
                                  actual_present=bool(target.any()))
    summary = {}
    for category, members in list(categories.items()) + [('全部300张', query_ids)]:
        total = {key: sum(records[sample_id][key] for sample_id in members)
                 for key in ('tp', 'fp', 'fn', 'tn')}
        summary[category] = dict(images=len(members), **metrics(total),
                                 predicted_present_images=sum(records[sample_id]['predicted_present']
                                                              for sample_id in members),
                                 correct_presence=sum(records[sample_id]['predicted_present'] ==
                                                      records[sample_id]['actual_present']
                                                      for sample_id in members))
    result = dict(status='complete', protocol=protocol,
                  foundation_sha256=dict(dino=MODEL_SHA256, geo=GEO_SHA256),
                  support_fit={key: value for key, value in task.items() if key != 'models'},
                  summary=summary, per_image=records)
    write_json(OUTPUT / 'results.json', result)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
