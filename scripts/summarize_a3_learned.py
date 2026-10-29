"""Validate reports/previews only; never download or load model checkpoints."""
import argparse
from html import escape
import json
from pathlib import Path
import statistics


def main(root, reference):
    from PIL import Image
    from pvseg.frozen_baselines import metrics

    read = lambda folder, name: json.loads((folder / (name + '.json')).read_text(encoding='utf-8'))
    records, summary = read(root, 'per_image'), read(root, 'summary')
    completion, learning = read(root, 'completion'), read(root, 'learning_audit')
    allocation, selection = read(root, 'training_allocation'), read(root, 'checkpoint_selection')
    thresholds = read(root, 'thresholds')
    reference_records = {r['sample_id']: r for r in read(reference, 'per_image')}
    entries = [r for r in records if r['included_in_summary']]
    if len(records) != 60 or len(entries) != 58 or len({r['sample_id'] for r in records}) != 60:
        raise ValueError('Unexpected query/preview IDs')
    if completion['status'] != 'complete' or completion['episode_updates'] != 320 or learning['successful_episode_updates'] != 320:
        raise ValueError('Training/evaluation did not complete')
    chosen = max(selection['curve'], key=lambda x: (x['mean_calibration_iou'], -x['step']))
    if chosen['step'] != completion['selected_step'] or thresholds['methods'] != chosen['thresholds']:
        raise ValueError('Selected checkpoint/threshold mismatch')
    if selection['query_labels_used'] or thresholds['query_labels_used'] or completion['query_labels_used_for_selection']:
        raise ValueError('Query labels used for selection')
    if len(allocation['training_ids']) != 22 or len(allocation['positive_query_ids']) != 4:
        raise ValueError('Unexpected training allocation')
    if set(allocation['training_ids']) & (set(reference_records) | set(allocation['reserved_inference_support_ids']) | set(thresholds['calibration_ids'])):
        raise ValueError('Training role overlap')
    for module in ('dmsf.visual', 'dmsf.text_to_dino', 'dmsf.text_hidden', 'dmsf.beta', 'dcma.pseudo'):
        if learning['maximum_abs_gradient_by_module'].get(module, 0) <= 0 or learning['maximum_parameter_delta_by_module'].get(module, 0) <= 0:
            raise ValueError('No gradient/update: ' + module)

    def aggregate(members, name):
        return metrics({k: sum(r['methods'][name][k] for r in members) for k in ('tp', 'fp', 'fn', 'tn')})

    methods = list(summary['methods'])
    for record in records:
        old = reference_records[record['sample_id']]
        for key in ('included_in_summary', 'foreground_pixels', 'annotation', 'source_scene'):
            if old[key] != record[key]:
                raise ValueError('Historical metadata mismatch: ' + record['sample_id'])
        for name in methods:
            m = record['methods'][name]
            if sum(m[k] for k in ('tp', 'fp', 'fn', 'tn')) != 896 ** 2 or m['tp'] + m['fn'] != record['foreground_pixels']:
                raise ValueError('Confusion pixel mismatch')
            if name.startswith('A2_') and any(m[k] != old['methods'][name][k] for k in ('tp', 'fp', 'fn', 'tn')):
                raise ValueError('Historical A2 changed')
        with Image.open(root / 'predictions' / (record['sample_id'] + '.png')) as im:
            if im.size != (1792, 282):
                raise ValueError('Unexpected panel size')
            im.verify()
    for name in methods:
        result = aggregate(entries, name)
        for k, value in result.items():
            if summary['methods'][name][k] != value:
                raise ValueError('Pooled summary mismatch: ' + name + '/' + k)

    families = ('A2', 'spatial', 'semantic', 'fused', 'final', 'final_fixed')
    keys = ('iou', 'precision', 'recall', 'false_positive_rate')

    def mean_runs(members, family):
        runs = [aggregate(members, f'{family}_{seed}') for seed in (42, 43, 44)]
        return {k: statistics.mean(v[k] for v in runs) if all(v[k] is not None for v in runs) else None for k in keys}

    means = {f: mean_runs(entries, f) for f in families}
    # Match the historical difficulty breakdown (positive images only), while
    # retaining all-image groups as a separate diagnostic.
    positives = [r for r in entries if r['foreground_pixels']]
    groups = {field: {value: dict(count=len(members), means={f: mean_runs(members, f) for f in families})
                     for value in sorted({r['annotation'][field] for r in positives})
                     for members in [[r for r in positives if r['annotation'][field] == value]]}
              for field in ('difficulty', 'area', 'morphology')}
    negatives = [r for r in entries if not r['foreground_pixels']]
    fmt = lambda v: f'{v * 100:.2f}%' if v is not None else '未定义'
    labels = dict(A2='冻结 A2', spatial='联合训练空间 CAM', semantic='联合训练语义 CAM',
                  fused='学习融合先验', final='完整网络（校准阈值）', final_fixed='完整网络（固定 0.5）')
    lines = ['# A3 可学习先验实测', '',
             f"源码 `{completion['source_commit']}`；基础更新 {completion['base_updates']} 次，第二阶段更新 {completion['episode_updates']} 次，校准选择第 {chosen['step']} 次更新。",
             '', '22 张额外训练图（4 张含前景），12 张校准图，3 个独立支持选择。58 张统计查询、2 张仅预览。相对 A2 增加了监督训练，不能将变化单独归因于融合；单分支 CAM 是联合训练后的诊断。', '',
             '| 输出 | 前景 IoU | 精确率 | 召回率 | 背景误检率 |', '|---|---:|---:|---:|---:|']
    for family in families:
        lines.append('| ' + labels[family] + ' | ' + ' | '.join(fmt(means[family][k]) for k in keys) + ' |')
    lines += ['', f"所选 checkpoint 的空间权重 gamma={summary['selected_gamma']:.6f}。", '',
              'CAM 经过逐图归一化，热图不是概率。先验阈值化仅是诊断；完整网络的结果单独列出。精确率在完全不预测前景时未定义，不能将全背景解释为误检改善。', '', '## 背景分组（含光伏图，与历史分组口径一致）', '',
              '| 背景 | 张数 | A2 IoU / 背景误检率 | 融合先验 IoU / 背景误检率 | 完整网络 IoU / 背景误检率 |', '|---|---:|---|---|---|']
    for value, data in groups['difficulty'].items():
        row = [value or '(空)', str(data['count'])]
        for family in ('A2', 'fused', 'final'):
            row.append(' / '.join(fmt(data['means'][family][k]) for k in ('iou', 'false_positive_rate')))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## 负样本', '', '| 输出 | 负图背景误检率（三支持均值） |', '|---|---:|']
    for family in families:
        lines.append(f"| {labels[family]} | {fmt(mean_runs(negatives, family)['false_positive_rate'])} |")
    lines += ['', '## 核验', '', '历史 A2 的全部逐图混淆像素计数完全复现；输出像素总量、支持与数据角色、校准选择及 60 张预览校验通过。视觉投影、文本跨模态映射、语义投影、融合标量与 DCMA 辅助头均有非零梯度和实际参数变化。这证明训练路径接通，效果由上表决定。', '',
              '局限：仅一个训练空间组、4 张前景训练图、固定训练随机种子；训练步数有界，原文 SGD 改为 AdamW，DINOv3 替换 DINOv2，采用已知光伏类合并方式。当前用户挑选的开发查询已被多轮查看，不作为独立泛化结论。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    cards = []
    for r in records:
        delta = r['methods']['fused_42']['iou']
        cards.append(f'<article data-id="{escape(r["sample_id"])}"><h3>{escape(r["sample_id"])} — {escape(r["annotation"]["difficulty"])} — {"统计查询" if r["included_in_summary"] else "仅预览"}</h3><img loading="lazy" src="predictions/{escape(r["sample_id"])}.png"><p>A2 IoU {fmt(r["methods"]["A2_42"]["iou"])} · 融合先验 {fmt(delta)} · 完整网络 {fmt(r["methods"]["final_42"]["iou"])}</p></article>')
    table = ''.join(f'<tr><td>{escape(labels[f])}</td>' + ''.join(f'<td>{fmt(means[f][k])}</td>' for k in keys) + '</tr>' for f in families)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>A3 可学习先验实测</title><style>body{font-family:Arial,sans-serif;background:#f4f5f7;color:#182130;margin:24px}article,table{background:white;padding:12px;margin:16px 0;border-radius:8px}img{width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #ddd}input{padding:10px;min-width:240px}</style><h1>A3 可学习先验实测</h1><p>加入 22 张训练图；与冻结 A2 的比较含有监督差异。CAM 是先验，完整网络 mask 单独展示。</p><table><tr><th>输出</th><th>IoU</th><th>精确率</th><th>召回率</th><th>背景误检率</th></tr>' + table + '</table><p><a href="REPORT.md">实验报告</a> · <a href="learning_audit.json">学习核验</a> · <a href="checkpoint_selection.json">校准选择</a></p><input placeholder="按 ID 筛选" oninput="document.querySelectorAll(\'article\').forEach(e=>e.hidden=!e.dataset.id.includes(this.value))">' + ''.join(cards) + '</html>'
    (root / 'index.html').write_text(page, encoding='utf-8')
    integrity = dict(status='passed', query_count=58, preview_count=2, decoded_panels=60,
                     exact_historical_A2_counts=True, pooled_counts_verified=True,
                     required_learning_paths_updated=True, calibration_only_selection=True,
                     means=means, groups=groups)
    (root / 'result_integrity.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(means=means, groups=groups['difficulty']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reference)
