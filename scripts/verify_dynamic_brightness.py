"""Reconstruct saved alpha fusion independently; compare historical control."""
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
from pvseg.frozen_baselines import confusion, metrics


def main(args):
    torch.set_num_threads(2)
    raw = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    previous = json.loads((args.brightness / 'metrics.json').read_text(encoding='utf-8'))
    p = raw['protocol']
    grids = np.load(args.run / 'weight_grids.npz')
    old = np.load(args.cache / 'score_grids.npz')
    bright = np.load(args.brightness / 'brightness_grids.npz')
    rows = {r['sample_id']: r for r in read_manifest(args.data)}
    refs = {r['sample_id']: r for r in previous['records']}
    seeds, arms, count = ('42', '43', '44'), ('fixed', 'global', 'local'), 0
    assert not set(p['training']['training_ids']) & set(p['calibration_ids'] + p['query_ids'])
    lock = json.loads((args.run / 'models_locked_before_query_masks.json').read_text())
    assert lock['query_masks_parsed'] is False and lock['arms'] == p['arms']
    def up(a):
        return F.interpolate(torch.from_numpy(a)[None, None], (896, 896), mode='bilinear', align_corners=False)[0, 0].numpy()
    hist = {(s, a): np.zeros((2, 4096), np.int64) for s in seeds for a in arms}
    for sid in p['calibration_ids'] + p['query_ids']:
        for key in ('raw', 'z', 'gate'):
            np.testing.assert_array_equal(grids[sid+'_'+key], bright[sid+'_'+key])
        target = np.asarray(Image.open(args.data / rows[sid]['label'])) == 255
        record = next((r for r in raw['records'] if r['sample_id'] == sid), None)
        for seed in seeds:
            prefix = sid+'_'+seed+'_'
            corrected = np.clip(old[prefix+'pv_semantic'] - old[prefix+'semantic_penalty']*grids[sid+'_gate'], -1, 1)
            if prefix+'semantic_brightness' in bright:
                np.testing.assert_array_equal(corrected, bright[prefix+'semantic_brightness'])
            d, s = up(old[prefix+'spatial']), up(corrected)
            for arm in arms:
                a = np.full((56, 56), .7, np.float32) if arm == 'fixed' else grids[prefix+arm+'_alpha']
                assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
                if arm == 'global':
                    assert a.max() == a.min()
                alpha = up(a-.7)+.7
                score = (1-alpha)*d+alpha*s
                assert (score >= np.minimum(d,s)-1e-7).all() and (score <= np.maximum(d,s)+1e-7).all()
                if record is None:
                    hist[seed, arm] += np.stack([np.histogram(score[target], 4096, (-1,1))[0],
                                                np.histogram(score[~target], 4096, (-1,1))[0]])
                    continue
                for mode in ('fixed', 'recalibrated'):
                    v = record['methods'][seed][mode][arm]
                    recomputed = metrics(confusion(score, target, v['threshold']))
                    assert all(v[k] == recomputed[k] for k in recomputed), (sid,seed,mode,arm)
                    if arm == 'fixed':
                        historical = refs[sid]['methods'][seed]['recalibrated']['brightness']
                        assert all(v[k] == historical[k] for k in recomputed), (sid,seed,mode)
                    count += 1
    for (seed, arm), h in hist.items():
        tp, fp = h[:, ::-1].cumsum(1)[:, ::-1]
        curve = tp/np.maximum(h[0].sum()+fp,1)
        best = np.flatnonzero(curve == curve.max())[-1]
        cal = p['arms'][arm]['calibration']['by_seed'][seed]
        assert cal['threshold'] == -1+2*best/4096
        assert cal['calibration_micro_iou'] == float(curve[best])
    result = dict(status='passed', exact_reconstructed_confusion_tables=count,
        exact_reconstructed_thresholds=len(hist), historical_brightness_counts_exact=True,
        historical_brightness_maps_exact=True, unchanged_brightness_gate=True,
        convex_fusion_bounds=True, query_masks_excluded_from_selection=True,
        global_alpha_spatially_constant=True)
    (args.output / 'independent_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('run', 'brightness', 'cache', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
