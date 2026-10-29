"""Rebuild the training-only comparison using archived reports; no assets needed."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read(folder, name):
    return json.loads((ROOT / folder / name).read_text(encoding='utf-8'))


def check_finite(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'finite':
                assert item is True
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


old_protocol = read('original', 'protocol.json')
new_protocol = read('balanced_ce', 'protocol.json')
assert old_protocol['pairs'] == new_protocol['pairs']
assert old_protocol['schedule'] == new_protocol['schedule'][:120]
assert read('balanced_ce', 'completion.json')['status'] == 'complete'
assert read('balanced_ce', 'initialization_comparison.json')['max_statistic_absolute_difference'] == 0
history = read('balanced_ce', 'balanced_meta_ce_fp32_history.json')
assert [r['step'] for r in history] == list(range(1, 241))
check_finite(history)
rows = []
curves = [('Original FP32', 'original', 'original_fp32', '#b54334'),
          ('Balanced raw-meta CE', 'balanced_ce', 'balanced_meta_ce_fp32', '#137e9a')]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
for label, folder, prefix, color in curves:
    paths = sorted((ROOT / folder).glob(prefix + '_step[0-9][0-9][0-9].json'))
    data = [read(folder, p.name) for p in paths]
    steps = [int(p.stem[-3:]) for p in paths]
    check_finite(data)
    axes[0].plot(steps, [100*r['pooled']['meta']['iou'] for r in data],
                 '-o', color=color, label=label+' / meta', ms=3)
    axes[0].plot(steps, [100*r['pooled']['final']['iou'] for r in data],
                 '--', color=color, label=label+' / final')
    axes[1].plot(steps, [100*sum(a['outputs']['meta']['saturated_fraction'] for a in r['per_image'])/5
                        for r in data], '-o', color=color, label=label, ms=3)
    axes[2].plot(steps, [sum(a['activations']['decoder']['rms'] for a in r['per_image'])/5
                        for r in data], '-o', color=color, label=label, ms=3)
    for step, r in zip(steps, data):
        positives = [a for a in r['per_image'] if a['outputs']['final']['tp']+a['outputs']['final']['fn'] > 0]
        negatives = [a for a in r['per_image'] if a not in positives]
        gate = (r['pooled']['meta']['iou'] >= .6 and r['pooled']['final']['iou'] >= .6
                and all(a['outputs']['meta']['tp'] > 0 and a['outputs']['final']['tp'] > 0 for a in positives)
                and r['loss'] < data[0]['loss'])
        rows.append(dict(arm=prefix, step=step, pooled=r['pooled'], mean_objective=r['loss'],
                         engineering_gate_passed=gate,
                         negative_reports=[dict(query=a['query'], final=a['outputs']['final']) for a in negatives]))
result = dict(scope='TRAINING fit only: 4 positive + 1 negative native896; not validation/test',
              equal_update_comparison=120, extended_repair_updates=240,
              initialization_max_statistic_difference=0.,
              repair_all_240_updates_finite=True, rows=rows)
(ROOT/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
axes[0].axhline(60, color='#6b7280', ls=':', lw=1, label='Engineering gate (60%)')
axes[0].set(ylim=(-2, 102), ylabel='Foreground micro-IoU (%)', title='Actual two-class argmax')
axes[1].set(ylim=(-2, 102), ylabel='Saturated pixels (%)', title='Meta p < 1e-6 or p > 1-1e-6')
axes[2].set(yscale='log', ylabel='Decoder activation RMS (log scale)', title='ASPP output scale')
for ax in axes:
    ax.set_xlabel('Optimizer updates')
    ax.axvline(120, color='#9ca3af', lw=.8)
    ax.axvspan(120, 240, color='#e5e7eb', alpha=.35)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(fontsize=7, loc='upper left')
axes[1].legend(fontsize=8)
fig.suptitle('Fixed training diagnosis: 4 positive + 1 negative PV tiles, native 896 x 896\n'
             'Equal updates through 120; shaded region = extended repair only', fontsize=12)
fig.supxlabel('No calibration/query images used. Confident saturation can include correct pixels.', fontsize=9)
fig.savefig(ROOT/'head_diagnosis.png', dpi=180)
print('Verified 240 finite repair updates and archived reports; rebuilt comparison.json and head_diagnosis.png')
