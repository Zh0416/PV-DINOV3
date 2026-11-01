"""Rebuild all CV counts from saved grids, verify weights, and summarize groups."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pvseg.spatial import scene_groups, bounds, overlap

KEYS = ('tp','fp','fn','tn')
NAMES = {
    'fixed':'原固定权重', 'local':'原动态 alpha', 'alpha_retrained':'同数据重训 alpha',
    'residual_balanced':'残差校正／普通损失', 'residual_hard':'残差校正／困难损失',
    'residual_spectral':'加入两路 FFT 描述', 'residual_texture':'局部纹理＋FFT／困难损失',
    'residual_no_fft':'局部纹理／FFT 通道置零',
    'residual_texture_balanced':'局部纹理＋FFT／普通损失',
}


def reconstruct(d, s, alpha, residual):
    tensor = lambda a: torch.from_numpy(a.astype(np.float32))[None,None]
    up = lambda a: F.interpolate(a, size=(896,896), mode='bilinear', align_corners=False)[0,0].numpy()
    a = up(tensor(alpha)-.7)+.7
    return (1-a)*up(tensor(d)) + a*up(tensor(s)) + up(tensor(residual))


def count(mask, target):
    return dict(tp=int(np.count_nonzero(mask & target)), fp=int(np.count_nonzero(mask & ~target)),
                fn=int(np.count_nonzero(~mask & target)), tn=int(np.count_nonzero(~mask & ~target)))


def stats(counts):
    tp,fp,fn,tn = (counts[k] for k in KEYS)
    divide = lambda n,d: n/d if d else None
    return dict(**counts, iou=divide(tp,tp+fp+fn), recall=divide(tp,tp+fn),
                precision=divide(tp,tp+fp), fpr=divide(fp,fp+tn))


def group_stats(records, ids):
    support = {str(seed): stats({k:sum(records[sid][str(seed)][k] for sid in ids) for k in KEYS})
               for seed in (42,43,44)}
    avg = {}
    for key in support['42']:
        values = [r[key] for r in support.values() if r[key] is not None]
        avg[key] = float(np.mean(values)) if values else None
    return dict(**avg, by_support=support)


def verify(data, root):
    summary = json.loads((data/'summary.json').read_text(encoding='utf-8'))
    grids = np.load(data/'holdout_grids.npz')
    cached = REPO/'outputs'
    old_scores = np.load(cached/'greenhouse_pilot_20261030/greenhouse_pilot/score_grids.npz')
    old_alpha = np.load(cached/'dynamic_brightness_20261030/dynamic_brightness/weight_grids.npz')
    with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {r['sample_id']:r for r in rows}
    groups, _ = scene_groups(rows)
    assert groups == summary['protocol']['groups']
    group_for = {s:g for g,ss in groups.items() for s in ss}
    assigned=[]
    for fold in summary['protocol']['folds']:
        held,train=fold['holdout_ids'],fold['train_ids']
        assert not {group_for[by_id[s]['source_scene']] for s in held} & {group_for[by_id[s]['source_scene']] for s in train}
        assert not any(overlap(bounds(by_id[a]),bounds(by_id[b])) for a in held for b in train)
        assigned.extend(held)
    assert sorted(assigned)==sorted(summary['subset_ids']['all']) and len(assigned)==len(set(assigned))
    verified_hashes=0
    for entry in list(summary['candidates'].values()) + [v for fold in summary['folds'].values() for v in fold['models'].values()]:
        path=data/entry.get('path',entry.get('checkpoint'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
        verified_hashes+=1
    compared=0
    for sid in assigned:
        with Image.open(root/by_id[sid]['label']) as im:
            target=np.asarray(im)==255
        for seed in (42,43,44):
            prefix=f'{sid}_{seed}_'
            d=old_scores[prefix+'spatial']
            s=np.clip(old_scores[prefix+'pv_semantic']-old_scores[prefix+'semantic_penalty']*old_alpha[sid+'_gate'],-1,1)
            for arm,records in summary['records'].items():
                a=(np.full((56,56),.7,np.float32) if arm=='fixed' else
                   old_alpha[prefix+'local_alpha'] if arm=='local' else grids[prefix+arm+'_alpha'])
                r=np.zeros_like(a) if arm in ('fixed','local') else grids[prefix+arm+'_residual']
                stored=records[sid][str(seed)]
                actual=count(reconstruct(d,s,a,r)>=stored['threshold'],target)
                assert all(actual[k]==stored[k] for k in KEYS),(sid,seed,arm,actual,stored)
                compared+=1
    computed={group:{arm:group_stats(records,ids) for arm,records in summary['records'].items()}
              for group,ids in summary['subset_ids'].items()}
    for group,arms in computed.items():
        for arm,row in arms.items():
            expected=summary['groups'][group][arm]
            for key in (*KEYS,'iou','recall','precision'):
                assert row[key]==expected[key],(group,arm,key,row[key],expected[key])
    baseline=computed['greenhouse']['local']
    eligible=[arm for arm in summary['candidates'] if
              computed['greenhouse'][arm]['recall']>=baseline['recall'] and
              computed['greenhouse'][arm]['fp']<baseline['fp'] and
              computed['all'][arm]['iou']>computed['all']['local']['iou']]
    recommended=max(eligible,key=lambda arm:computed['greenhouse'][arm]['iou']) if eligible else 'local'
    best_iou=max(summary['candidates'],key=lambda arm:computed['all'][arm]['iou'])
    audit=dict(status='passed', count_comparisons=compared, checkpoint_hashes=verified_hashes,
        geographic_folds=len(summary['protocol']['folds']), groups=computed,
        recommended=recommended, best_development_iou=best_iou,
        candidate_selection='Development-CV selection: maximize greenhouse IoU among models maintaining historical greenhouse recall and improving FP/all-IoU; not independent test selection',
        final_candidate_independent_evaluation=False)
    (data/'independent_reconstruction.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(r'G:\PV-GF\PV-896'))
    args=p.parse_args()
    torch.set_num_threads(2)
    audit=verify(args.data,args.root)
    print(json.dumps({k:v for k,v in audit.items() if k!='groups'},ensure_ascii=False,indent=2))
    for group,arms in audit['groups'].items():
        print(group)
        for arm,row in arms.items():
            print(arm,{k:round(row[k],6) if row[k] is not None else None for k in ('iou','recall','fp','fn')})
