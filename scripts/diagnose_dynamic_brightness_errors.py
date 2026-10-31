"""Post-hoc error attribution with locked CAMs, weights, and thresholds.

The oracle uses query labels for diagnosis only. It allows an independent
alpha per native pixel, so its IoU is a relaxed upper bound, not a model score.
No training, threshold selection, or checkpoint selection happens here.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pvseg.data import read_manifest
from pvseg.dynamic_brightness_fusion import convex_fusion
from pvseg.losses import segmentation_loss
from pvseg.frozen_baselines import confusion


def up(grid):
    return F.interpolate(torch.from_numpy(grid)[None, None], (896, 896),
                         mode='bilinear', align_corners=False)[0, 0].numpy()


def attribute(dino, semantic, score, target, threshold):
    prediction = score >= threshold
    fp = prediction & ~target
    fn = ~prediction & target
    both_high = np.minimum(dino, semantic) >= threshold
    both_low = np.maximum(dino, semantic) < threshold
    values = {
        'tp': int((prediction & target).sum()),
        'fp': int(fp.sum()), 'fn': int(fn.sum()),
        'tn': int((~prediction & ~target).sum()),
        'fp_both_branches_high': int((fp & both_high).sum()),
        'fp_branches_disagree': int((fp & ~both_high).sum()),
        'fn_both_branches_low': int((fn & both_low).sum()),
        'fn_branches_disagree': int((fn & ~both_low).sum()),
        'oracle_tp': int((target & ~both_low).sum()),
        'oracle_fn': int((target & both_low).sum()),
        'oracle_fp': int((~target & both_high).sum()),
    }
    # These also detect any floating point threshold-edge violation of the
    # assumed convex interval in the actual saved predictions.
    assert values['fp_both_branches_high'] == values['oracle_fp']
    assert values['fn_both_branches_low'] == values['oracle_fn']
    assert values['fp'] == values['fp_both_branches_high'] + values['fp_branches_disagree']
    assert values['fn'] == values['fn_both_branches_low'] + values['fn_branches_disagree']
    return values


def rates(values):
    result = dict(values)
    def ratio(a, b):
        return a / b if b else None
    result.update(
        iou=ratio(values['tp'], values['tp'] + values['fp'] + values['fn']),
        recall=ratio(values['tp'], values['tp'] + values['fn']),
        fp_unavoidable_fraction=ratio(values['fp_both_branches_high'], values['fp']),
        fn_unavoidable_fraction=ratio(values['fn_both_branches_low'], values['fn']),
        oracle_iou_upper_bound=ratio(values['oracle_tp'], values['oracle_tp'] + values['oracle_fp'] + values['oracle_fn']),
    )
    return result


def gradient_check():
    """Check gradient descent lowers a false positive and raises a false negative."""
    dino = torch.full((1, 1, 1, 2), .1, requires_grad=True)
    semantic = torch.full((1, 1, 1, 2), .4, requires_grad=True)
    alpha = torch.tensor([[[[.7, .2]]]], requires_grad=True)
    target = torch.tensor([[[[0., 1.]]]])
    score = convex_fusion(dino, semantic, alpha)
    probability = (20 * (score - .2)).sigmoid()
    loss = segmentation_loss(20 * (score - .2), target)
    loss.backward()
    fp_gradient, fn_gradient = alpha.grad.flatten().tolist()
    assert fp_gradient > 0 and fn_gradient < 0
    assert dino.grad is None and semantic.grad is None
    return dict(loss=float(loss.detach()),
                false_positive_probability=float(probability.flatten()[0].detach()),
                false_negative_probability=float(probability.flatten()[1].detach()),
                fp_alpha_gradient=fp_gradient, fn_alpha_gradient=fn_gradient,
                descent_reduces_fp_and_increases_fn_scores=True, frozen_cams_have_no_gradient=True)


def main(args):
    torch.set_num_threads(2)
    raw = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    rows = {r['sample_id']: r for r in read_manifest(args.data)}
    seeds, arms = ('42', '43', '44'), ('fixed', 'global', 'local')
    records, checked = [], 0
    with np.load(args.cache / 'score_grids.npz') as cams, np.load(args.run / 'weight_grids.npz') as grids:
        for record in raw['records']:
            sid = record['sample_id']
            with Image.open(args.data / rows[sid]['label']) as im:
                label = np.asarray(im)
                assert label.shape == (896, 896) and np.isin(label, [0, 255]).all()
                target = label == 255
            output = dict(sample_id=sid, greenhouse_scene=record['annotation']['difficulty'] == '大棚干扰', methods={})
            for seed in seeds:
                prefix = sid + '_' + seed + '_'
                dino = up(cams[prefix + 'spatial'])
                semantic = up(np.clip(cams[prefix + 'pv_semantic'] - cams[prefix + 'semantic_penalty'] * grids[sid + '_gate'], -1, 1))
                output['methods'][seed] = {mode: {} for mode in ('fixed', 'recalibrated')}
                for arm in arms:
                    alpha = np.full((56, 56), .7, np.float32) if arm == 'fixed' else grids[prefix + arm + '_alpha']
                    alpha = up(alpha - .7) + .7
                    score = (1 - alpha) * dino + alpha * semantic
                    for mode in ('fixed', 'recalibrated'):
                        saved = record['methods'][seed][mode][arm]
                        values = attribute(dino, semantic, score, target, saved['threshold'])
                        expected = confusion(score, target, saved['threshold'])
                        assert all(saved[k] == values[k] == expected[k] for k in ('tp', 'fp', 'fn', 'tn'))
                        output['methods'][seed][mode][arm] = dict(threshold=saved['threshold'], **values)
                        checked += 1
            records.append(output)

    groups = dict(all=records, greenhouse_scene=[r for r in records if r['greenhouse_scene']],
                  other_scene=[r for r in records if not r['greenhouse_scene']])
    summary = {}
    for group, selected in groups.items():
        summary[group] = {}
        for mode in ('fixed', 'recalibrated'):
            summary[group][mode] = {}
            for arm in arms:
                by_seed = {}
                for seed in seeds:
                    keys = set(selected[0]['methods'][seed][mode][arm]) - {'threshold'}
                    by_seed[seed] = rates({key: sum(r['methods'][seed][mode][arm][key] for r in selected) for key in keys})
                mean = {key: float(np.mean([by_seed[s][key] for s in seeds]))
                        if all(by_seed[s][key] is not None for s in seeds) else None
                        for key in by_seed[seeds[0]]}
                summary[group][mode][arm] = dict(images=len(selected), by_support=by_seed, mean=mean)

    result = dict(
        scope='Post-hoc diagnosis on the existing 24-image development subset; no new independent validation',
        hypothesis='At a fixed threshold, convex fusion cannot correct a background pixel when both CAMs are high or a PV pixel when both CAMs are low',
        oracle='Ground-truth-assisted independent alpha per native pixel; relaxed upper bound beyond the actual 56x56 smooth alpha network',
        labels='Binary PV masks only; greenhouse_scene is an image-level tag. FP includes all non-PV background in these scenes.',
        support_aggregation='Counts and rates are computed per support then averaged over three support choices; not independent training seeds',
        model_or_threshold_changes=False, reconstructed_confusion_tables=checked,
        gradient_sign_check=gradient_check(),
        summary=summary, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(reconstructed_confusion_tables=checked, gradient_sign_check=result['gradient_sign_check'],
                         local_recalibrated={group: value['recalibrated']['local']['mean'] for group, value in summary.items()}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=REPO / 'outputs/dynamic_brightness_20261030/dynamic_brightness')
    parser.add_argument('--cache', type=Path, default=REPO / 'outputs/greenhouse_pilot_20261030/greenhouse_pilot')
    parser.add_argument('--data', type=Path, default=Path('G:/PV-GF/PV-896'))
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/dynamic_brightness_20261030/error_diagnosis.json')
    main(parser.parse_args())
